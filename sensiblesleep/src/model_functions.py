import numpy as np
import pymc as pm

from config import MODELCONFIG


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
    n_bins = MODELCONFIG['n_bins']
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
        pm.math.le(tsleep, tawake),  # Check if sleep period does not wrap around midnight
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
        deviances = -2 * log_likelihood_data.sum(axis=2)  # Sum over the actual bin dimensions

        # Step 2: Expected deviance E[D] (mean across chains and draws)
        expected_deviance = deviances.mean(dim=["chain", "draw"])

        # Step 3: Posterior mean log-likelihood across chains and draws
        posterior_mean_log_likelihood = log_likelihood_data.mean(axis=(0, 1, 2))  # Shape (1,)
        deviance_posterior_mean = -2 * posterior_mean_log_likelihood

        # Step 4: Effective number of parameters p_D and DIC
        p_D = expected_deviance - deviance_posterior_mean
        DIC = expected_deviance + p_D

        # Step 5: Calculate the standard error of the DIC
        deviance_std = deviances.std(dim=["chain", "draw"])  # Standard deviation of deviances
        DIC_error = deviance_std / np.sqrt(deviances.size)  # Standard error of DIC

        # Store the DIC value and error, adjusting for the reference model
        if k == reference_model:
            reference_DIC = DIC.values[0]
        else:
            DIC_values.append(DIC.values[0])  # Relative to reference model
            DIC_errors.append(DIC_error.values[0])

    return reference_DIC, DIC_values, DIC_errors, model_names
