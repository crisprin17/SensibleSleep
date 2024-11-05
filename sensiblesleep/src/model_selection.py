# python model_selection.py

import json
import numpy as np
import pymc as pm

# Import your custom modules (adjust the import paths as necessary)
from model_functions import (
    build_hyper_hyper_model,
    build_independent_hyper_model,
    build_independent_independent_model,
    build_independent_pooled_model,
    build_pooled_pooled_model,
    calculate_DIC,
)
from plot_functions import plot_DIC, plot_logp

CONFIG = json.load(open("config.json"))
MODELCONFIG = CONFIG["model"]
SAMPLECONFIG = CONFIG["sampling"]


# Load your observed data (replace with your actual data loading code)
# For demonstration purposes, we'll create synthetic data
n_days = 7
n_bins = 96  # Assuming 15-minute bins in a day
total_bins = n_days * n_bins
observed_event_counts = np.random.poisson(lam=5, size=total_bins)
time_bins = np.tile(np.arange(n_bins), n_days)


# Function to run a model and return the trace and log posterior probabilities
def run_model(model_name, observed_event_counts, n_bins, n_days, total_bins, time_bins):
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
    elif model_name == "independent_hyper":
        model = build_independent_hyper_model(
            observed_event_counts, n_bins, n_days, total_bins, time_bins
        )
    elif model_name == "hyper_hyper":
        # Build the model and get variables
        (
            model,
            tsleep,
            tawake,
            lambda_awake,
            lambda_sleep,
            alpha_lambda,
            beta_lambda,
            mu_tsleep,
            sigma_tsleep,
            mu_tawake,
            sigma_tawake,
        ) = build_hyper_hyper_model(
            observed_event_counts, n_bins, n_days, total_bins, time_bins
        )
        with model:
            # Define step methods
            step_discrete = pm.Metropolis([tsleep, tawake])
            step_continuous = pm.NUTS(
                vars=[
                    lambda_awake,
                    lambda_sleep,
                    alpha_lambda,
                    beta_lambda,
                    mu_tsleep,
                    sigma_tsleep,
                    mu_tawake,
                    sigma_tawake,
                ]
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_name != "hyper_hyper":
        with model:
            trace = pm.sample(
                draws=SAMPLECONFIG["n_samples"],
                chains=SAMPLECONFIG["n_chains"],
                tune=SAMPLECONFIG["tune"],
                cores=SAMPLECONFIG["cores"],
                target_accept=SAMPLECONFIG["target_accept"],
                compute_convergence_checks=SAMPLECONFIG["compute_convergence_checks"],
                random_seed=SAMPLECONFIG["random_seed"],
            )

    with model:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, progressbar=SAMPLECONFIG["progressbar"]
        )
        log_likelihood = pm.compute_log_likelihood(trace)
        logp = trace.sample_stats["lp"]

    return trace, logp


# Run all models and collect traces and log probabilities
model_names = [
    "pooled_pooled",
    "independent_pooled",
    "independent_independent",
    "independent_hyper",
    "hyper_hyper",
]
traces = {}
logps = {}
for model_name in model_names:
    print(f"Running model: {model_name}")
    trace, logp = run_model(
        model_name, observed_event_counts, n_bins, n_days, total_bins, time_bins
    )
    traces[model_name] = trace
    logps[model_name] = logp

# Map model names to display names and colors for plotting
map_models = {
    "pooled pooled": logps["pooled_pooled"],
    "independent pooled": logps["independent_pooled"],
    "independent independent": logps["independent_independent"],
    "independent hyper": logps["independent_hyper"],
    "hyper hyper": logps["hyper_hyper"],
}

map_colors = {
    "pooled pooled": "purple",
    "independent pooled": "cyan",
    "independent independent": "red",
    "independent hyper": "green",
    "hyper hyper": "blue",
}

# Plot the log posterior probabilities
plot_logp(map_models, map_colors)

# Calculate DIC values for model comparison
reference_DIC, DIC_values, DIC_errors, model_display_names = calculate_DIC(traces)

# Plot DIC values
plot_DIC(reference_DIC, DIC_values, DIC_errors, model_display_names)
