import json

import numpy as np
import pymc as pm

CONFIG = json.load(open("config.json"))
MODELCONFIG = CONFIG["model"]
SAMPLECONFIG = CONFIG["sampling"]


def run_model(model_name, observed_event_counts, n_bins, n_days, total_bins, time_bins):
    if model_name == "hyper_hyper":
        # Build the model and get variables
        model, tsleep, tawake, lambda_awake, lambda_sleep, alpha_lambda, beta_lambda = (
            build_hyper_hyper_model(
                observed_event_counts, n_bins, n_days, total_bins, time_bins
            )
        )

        with model:
            # Define step methods
            step_discrete = pm.Metropolis([tsleep, tawake])
            step_continuous = pm.NUTS(
                vars=[lambda_awake, lambda_sleep, alpha_lambda, beta_lambda]
            )

            # Sampling using MCMC with custom step methods
            trace = pm.sample(
                draws=SAMPLECONFIG["n_samples"],
                chains=SAMPLECONFIG["n_chains"],
                tune=SAMPLECONFIG["tune"],
                cores=SAMPLECONFIG["cores"],
                target_accept=SAMPLECONFIG["target_accept"],
                compute_convergence_checks=SAMPLECONFIG["compute_convergence_checks"],
                random_seed=SAMPLECONFIG["random_seed"],
                step=[step_discrete, step_continuous],
            )

            # **Move these operations inside the 'with model' context**
            posterior_predictive = pm.sample_posterior_predictive(
                trace, progressbar=SAMPLECONFIG["progressbar"]
            )
            log_likelihood = pm.compute_log_likelihood(trace)
        logp = trace.sample_stats["lp"]
    else:
        if model_name == "pooled_pooled":
            model = build_pooled_pooled_model(
                observed_event_counts, n_bins, n_days, total_bins, time_bins
            )
        elif model_name == "independent_pooled":
            model = build_independent_pooled_model(
                observed_event_counts, n_bins, n_days, total_bins, time_bins
            )
        elif model_name == "independent_independent":
            model = build_independent_independent_model(
                observed_event_counts, n_bins, n_days, total_bins, time_bins
            )
        elif model_name == "indipendent_hyper":
            model = build_indipendent_hyper_model(
                observed_event_counts, n_bins, n_days, total_bins, time_bins
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        with model:
            # Sampling without custom step methods
            trace = pm.sample(
                draws=SAMPLECONFIG["n_samples"],
                chains=SAMPLECONFIG["n_chains"],
                tune=SAMPLECONFIG["tune"],
                cores=SAMPLECONFIG["cores"],
                target_accept=SAMPLECONFIG["target_accept"],
                compute_convergence_checks=SAMPLECONFIG["compute_convergence_checks"],
                random_seed=SAMPLECONFIG["random_seed"],
            )

            # **Ensure these are inside the 'with model' context**
            posterior_predictive = pm.sample_posterior_predictive(
                trace, progressbar=SAMPLECONFIG["progressbar"]
            )
            log_likelihood = pm.compute_log_likelihood(trace)
        logp = trace.sample_stats["lp"]

    return trace, posterior_predictive, log_likelihood, logp


def build_pooled_pooled_model(
    observed_event_counts, n_bins, n_days, total_bins, time_bins
):
    with pm.Model() as model:
        # Priors for sleep and wake times (uniform between 0 and 96 bins)
        tsleep = pm.DiscreteUniform("tsleep", lower=0, upper=n_bins)
        tawake = pm.DiscreteUniform("tawake", lower=0, upper=n_bins)

        # Shared rate for awake periods
        lambda_awake = pm.Gamma(
            "lambda_awake",
            alpha=MODELCONFIG["alpha_awake"],
            beta=MODELCONFIG["beta_awake"],
        )

        # Very low rate for sleep periods (prior belief)
        lambda_sleep = pm.Exponential("lambda_sleep", lam=MODELCONFIG["lambda_sleep"])

        # Create the time bins (reshaped to match broadcasting requirements)
        # Expand tsleep and tawake to repeat each day’s value across 96 bins

        tsleep_expanded = np.repeat(tsleep, total_bins).reshape((total_bins, 1))
        tawake_expanded = np.repeat(tawake, total_bins).reshape((total_bins, 1))

        # Calculate the sleep indicator using the expanded tsleep and tawake
        sleep_indicator = is_asleep(time_bins, tsleep_expanded, tawake_expanded, n_bins)

        # Expected event rate
        event_rate = (
            sleep_indicator * lambda_sleep + (1 - sleep_indicator) * lambda_awake
        )

        # Likelihood
        pm.Poisson(
            "events", mu=event_rate, observed=observed_event_counts.reshape(-1, 1)
        )

    return model


def build_independent_pooled_model(
    observed_event_counts, n_bins, n_days, total_bins, time_bins
):
    with pm.Model() as model:
        # Priors for sleep and wake times (uniform between 0 and 96 bins)
        tsleep = pm.DiscreteUniform("tsleep", lower=0, upper=n_bins, shape=(n_days,))
        tawake = pm.DiscreteUniform("tawake", lower=0, upper=n_bins, shape=(n_days,))

        # Shared rate for awake periods
        lambda_awake = pm.Gamma(
            "lambda_awake",
            alpha=MODELCONFIG["alpha_awake"],
            beta=MODELCONFIG["beta_awake"],
        )

        # Very low rate for sleep periods (prior belief)
        lambda_sleep = pm.Exponential("lambda_sleep", lam=MODELCONFIG["lambda_sleep"])

        # Create the time bins (reshaped to match broadcasting requirements)
        # Expand tsleep and tawake to repeat each day’s value across 96 bins
        tsleep_expanded = np.repeat(tsleep, n_bins).reshape((total_bins, 1))
        tawake_expanded = np.repeat(tawake, n_bins).reshape((total_bins, 1))

        # Create the time bins (reshaped to match broadcasting requirements)
        time_bins = np.arange(total_bins).reshape((total_bins, 1)) % n_bins

        # Calculate the sleep indicator using the expanded tsleep and tawake
        sleep_indicator = is_asleep(time_bins, tsleep_expanded, tawake_expanded, n_bins)

        # Expected event rate
        event_rate = (
            sleep_indicator * lambda_sleep + (1 - sleep_indicator) * lambda_awake
        )

        # Likelihood
        pm.Poisson(
            "events", mu=event_rate, observed=observed_event_counts.reshape(-1, 1)
        )
    return model


def build_independent_independent_model(
    observed_event_counts, n_bins, n_days, total_bins, time_bins
):
    with pm.Model() as model:
        # Priors for sleep and wake times (independent for each day)
        tsleep = pm.DiscreteUniform("tsleep", lower=0, upper=n_bins, shape=(n_days,))
        tawake = pm.DiscreteUniform("tawake", lower=0, upper=n_bins, shape=(n_days,))

        # Day-specific awake rates drawn from the Gamma distribution with hyperparameters
        lambda_awake = pm.Gamma(
            "lambda_awake",
            alpha=MODELCONFIG["alpha_awake"],
            beta=MODELCONFIG["beta_awake"],
            shape=(n_days,),
        )

        # Very low rate for sleep periods (prior belief)
        lambda_sleep = pm.Exponential("lambda_sleep", lam=MODELCONFIG["lambda_sleep"])

        # Create the time bins (reshaped to match broadcasting requirements)
        # Expand tsleep and tawake to repeat each day’s value across 96 bins
        tsleep_expanded = np.repeat(tsleep, n_bins).reshape((total_bins, 1))
        tawake_expanded = np.repeat(tawake, n_bins).reshape((total_bins, 1))
        lambda_awake_expanded = np.repeat(lambda_awake, n_bins).reshape((total_bins, 1))

        # Calculate the sleep indicator using the expanded tsleep and tawake
        sleep_indicator = is_asleep(time_bins, tsleep_expanded, tawake_expanded, n_bins)

        # Expected event rate
        event_rate = (
            sleep_indicator * lambda_sleep
            + (1 - sleep_indicator) * lambda_awake_expanded
        )

        # Likelihood
        pm.Poisson(
            "events", mu=event_rate, observed=observed_event_counts.reshape(-1, 1)
        )
    return model


def build_indipendent_hyper_model(
    observed_event_counts, n_bins, n_days, total_bins, time_bins
):
    # Build the Independent-Hyper Model
    with pm.Model() as model:
        # Priors for sleep and wake times (independent for each day)
        tsleep = pm.DiscreteUniform("tsleep", lower=0, upper=n_bins, shape=(n_days,))
        tawake = pm.DiscreteUniform("tawake", lower=0, upper=n_bins, shape=(n_days,))

        # Day-specific awake rates drawn from the Gamma distribution with hyperparameters
        # Hyperpriors for the Gamma distribution governing λawake (αλ and βλ)
        alpha_lambda = pm.Exponential(
            "alpha_lambda", lam=MODELCONFIG["alpha_lambda"]
        )  # User-specific hyperprior for the shape parameter
        beta_lambda = pm.Exponential(
            "beta_lambda", lam=MODELCONFIG["beta_lambda"]
        )  # User-specific hyperprior for the rate parameter

        # Day-specific awake rates drawn from the Gamma distribution with hyperparameters
        lambda_awake = pm.Gamma(
            "lambda_awake", alpha=alpha_lambda, beta=beta_lambda, shape=(n_days,)
        )

        # Very low rate for sleep periods (prior belief)
        lambda_sleep = pm.Exponential("lambda_sleep", lam=MODELCONFIG["lambda_sleep"])

        # Create the time bins (reshaped to match broadcasting requirements)
        # Expand tsleep and tawake to repeat each day’s value across 96 bins
        tsleep_expanded = np.repeat(tsleep, n_bins).reshape((total_bins, 1))
        tawake_expanded = np.repeat(tawake, n_bins).reshape((total_bins, 1))
        lambda_awake_expanded = np.repeat(lambda_awake, n_bins).reshape((total_bins, 1))

        # Calculate the sleep indicator using the expanded tsleep and tawake
        sleep_indicator = is_asleep(time_bins, tsleep_expanded, tawake_expanded, n_bins)

        # Expected event rate
        event_rate = (
            sleep_indicator * lambda_sleep
            + (1 - sleep_indicator) * lambda_awake_expanded
        )

        # Likelihood
        pm.Poisson(
            "events", mu=event_rate, observed=observed_event_counts.reshape(-1, 1)
        )

    return model


def build_hyper_hyper_model(
    observed_event_counts, n_bins, n_days, total_bins, time_bins
):
    with pm.Model() as model:
        # Hyperpriors for the circadian rhythm
        alpha_t = pm.Exponential(
            "alpha_t", lam=MODELCONFIG["alpha_t"]
        )  # Hyperprior for circadian rhythm (shape)
        beta_t = pm.Exponential(
            "beta_t", lam=MODELCONFIG["beta_t"]
        )  # Hyperprior for circadian rhythm (rate)
        # Underlying circadian rhythm (Gamma distributed)
        tt = pm.Gamma("tt", alpha=alpha_t, beta=beta_t)

        # Priors for daily sleep and wake times (Normal distributions centered around circadian rhythm)
        musleep = 8 * n_bins / 24
        muawake = 15 * n_bins / 24
        tsleep = pm.Normal(
            "tsleep", mu=musleep, sigma=tt, shape=(n_days,)
        )  # Sleep time centered around bin 40 (~23:00)
        tawake = pm.Normal(
            "tawake", mu=muawake, sigma=tt, shape=(n_days,)
        )  # Wake time centered around bin 70 (~07:00)

        # Hyperparameters for lambda_awake (hierarchical Gamma distribution)
        # Hyperpriors for the Gamma distribution governing λawake (αλ and βλ)
        alpha_lambda = pm.Exponential(
            "alpha_lambda", lam=MODELCONFIG["alpha_lambda"]
        )  # User-specific hyperprior for the shape parameter
        beta_lambda = pm.Exponential(
            "beta_lambda", lam=MODELCONFIG["beta_lambda"]
        )  # User-specific hyperprior for the rate parameter

        # Day-specific awake rates drawn from the Gamma distribution with hyperparameters
        lambda_awake = pm.Gamma(
            "lambda_awake", alpha=alpha_lambda, beta=beta_lambda, shape=(n_days,)
        )

        # Shared lambda_sleep
        # Very low rate for sleep periods (prior belief)
        lambda_sleep = pm.Exponential("lambda_sleep", lam=MODELCONFIG["lambda_sleep"])

        # Create the time bins (reshaped to match broadcasting requirements)
        # Expand tsleep and tawake to repeat each day’s value across 96 bins
        tsleep_expanded = np.repeat(tsleep, n_bins).reshape((total_bins, 1))
        tawake_expanded = np.repeat(tawake, n_bins).reshape((total_bins, 1))
        lambda_awake_expanded = np.repeat(lambda_awake, n_bins).reshape((total_bins, 1))

        # Calculate the sleep indicator using the expanded tsleep and tawake
        sleep_indicator = is_asleep(time_bins, tsleep_expanded, tawake_expanded, n_bins)

        # Expected event rate
        event_rate = (
            sleep_indicator * lambda_sleep
            + (1 - sleep_indicator) * lambda_awake_expanded
        )

        # Likelihood
        pm.Poisson(
            "events", mu=event_rate, observed=observed_event_counts.reshape(-1, 1)
        )

    return model, tsleep, tawake, lambda_awake, lambda_sleep, alpha_lambda, beta_lambda


def create_time_bins(n_days: int) -> np.ndarray:
    """
    Create time bins for a specified number of bins and days.

    Parameters:
    ----------
    n_bins : int
        The number of bins representing a single day (e.g., 96 for 15-minute intervals).
    n_days : int
        The number of days for which to create time bins.

    Returns:
    -------
    np.ndarray
        An array of time bins, wrapped around the number of bins.
    """
    n_bins = MODELCONFIG["n_bins"]
    total_bins = n_bins * n_days
    time_bins = np.arange(total_bins).reshape((total_bins, 1)) % n_bins
    return n_bins, total_bins, time_bins


def is_asleep(t: float, tsleep: float, tawake: float, n_bins: int) -> np.ndarray:
    """
    Determine whether a given time falls within a specified sleep period.

    Parameters:
    ----------
    t : float
        The time to check, expected to be in a format that can be wrapped around a certain number of bins.
    tsleep : float
        The time at which sleep starts, wrapped around the number of bins.
    tawake : float
        The time at which sleep ends, wrapped around the number of bins.
    n_bins : int
        The total number of bins representing the time period (e.g., 96 for a full day with 15-minute intervals).

    Returns:
    -------
    np.ndarray
        An array of floats (0.0 or 1.0) indicating whether the time `t`
        is during the sleep period (1.0 for asleep, 0.0 for awake).
    """
    t = t % n_bins
    tsleep = tsleep % n_bins
    tawake = tawake % n_bins

    asleep = pm.math.switch(
        pm.math.le(
            tsleep, tawake
        ),  # Check if sleep period does not wrap around midnight
        (t >= tsleep) & (t < tawake),  # True value: t is between tsleep and tawake
        (t < tsleep) | (t >= tawake),  # False value: t is outside the sleep period
    )

    return asleep.astype(float)  # Convert boolean to float


def calculate_DIC(map_models) -> tuple:
    """
    Function to calculate the DIC relative to a reference model for each model.

    Parameters:
    map_models (Dict[str, xr.Dataset]): Dictionary mapping model names
    to their xarray Dataset containing posterior samples.

    Returns:
    tuple: (DIC value, DIC standard error)
    """
    reference_model = "pooled_pooled"
    DIC_values = []
    DIC_errors = []
    model_names = []

    for k, v in map_models.items():

        model_names.append(k)

        # Example log-likelihood data shape (chains, draws, events_dim_0, events_dim_1)
        log_likelihood_data = v["log_likelihood"]["events"]

        # Step 1: Compute the deviance per draw (summing over bins)
        deviances = -2 * log_likelihood_data.sum(
            axis=2
        )  # Sum over the actual bin dimensions

        # Step 2: Expected deviance E[D] (mean across chains and draws)
        expected_deviance = deviances.mean(dim=["chain", "draw"])

        # Step 3: Posterior mean log-likelihood across chains and draws
        posterior_mean_log_likelihood = log_likelihood_data.mean(
            axis=(0, 1, 2)
        )  # Shape (1,)
        deviance_posterior_mean = -2 * posterior_mean_log_likelihood

        # Step 4: Effective number of parameters p_D and DIC
        p_D = expected_deviance - deviance_posterior_mean
        DIC = expected_deviance + p_D

        # Step 5: Calculate the standard error of the DIC
        deviance_std = deviances.std(
            dim=["chain", "draw"]
        )  # Standard deviation of deviances
        DIC_error = deviance_std / np.sqrt(deviances.size)  # Standard error of DIC

        # Store the DIC value and error, adjusting for the reference model
        if k == reference_model:
            reference_DIC = DIC.values[0]
        else:
            DIC_values.append(DIC.values[0])  # Relative to reference model
            DIC_errors.append(DIC_error.values[0])

    return reference_DIC, DIC_values, DIC_errors, model_names
