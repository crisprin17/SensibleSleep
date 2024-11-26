"""
SensibleSleep Model Selection Script
==================================

This script implements the model selection process for the SensibleSleep algorithm,
which infers sleep patterns from smartphone screen events using Bayesian modeling.

Based on the paper:
Cuttone, A., Bækgaard, P., Sekara, V., Jonsson, H., Larsen, J. E., & Lehmann, S. (2017). 
"SensibleSleep: A Bayesian Model for Learning Sleep Patterns from Smartphone Events."
PLoS ONE, 12(1), e0169901.

Models Implemented:
-----------------
1. Pooled-Pooled: 
   - Simplest model
   - Single sleep/wake time for all days
   - Shared activity rates

2. Independent-Pooled:
   - Independent sleep/wake times per day
   - Shared activity rates

3. Independent-Independent:
   - Independent sleep/wake times per day
   - Independent wake activity rates
   - Shared sleep activity rate

4. Independent-Hyper:
   - Independent sleep/wake times
   - Hierarchical structure for activity rates

5. Hyper-Hyper:
   - Most complex model
   - Hierarchical structure for both timing and rates
   - Captures both individual variation and population patterns

Usage:
------
1. Ensure config.json is properly configured
2. Prepare input data in the correct format (15-minute bins of event counts)
3. Run: python model_selection.py

The script will:
- Run all models on the provided data
- Compare models using DIC and log posterior probabilities
- Generate comparison plots
- Save results and traces

Requirements:
------------
- PyMC
- NumPy
- Matplotlib (for plotting)
- JSON configuration file

Author: Cristina Principato, Joanna Kuc
Date: 2024-11-19
"""

# python model_selection.py

import json
import numpy as np
import pymc as pm
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR.parent / "config.json"
CONFIG = json.load(open(CONFIG_PATH))
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

    """
    Run a specified sleep detection model and return its trace and log probability.
    
    Parameters:
    ----------
    model_name : str
        Name of the model to run
    observed_event_counts : np.ndarray
        Array of event counts
    n_bins : int
        Number of bins per day
    n_days : int
        Number of days
    total_bins : int
        Total number of bins
    time_bins : np.ndarray
        Array of time bin indices
    
    Returns:
    -------
    tuple
        (trace, logp) containing the MCMC trace and log probability
    """

    # Print input shapes for debugging
    print(f"\nModel: {model_name}")
    print(f"observed_event_counts shape: {observed_event_counts.shape}")
    print(f"time_bins shape: {time_bins.shape}")
    print(f"n_bins: {n_bins}, n_days: {n_days}, total_bins: {total_bins}")

    # Build model
    if model_name == "hyper_hyper":

        # Build the model and get variables
        # !! TODO: this should be fixed - the number of variables doesn't match
        # the number of variables returned by the function
        (
            model,
            tsleep,
            tawake,
            lambda_awake,
            lambda_sleep,
            alpha_lambda,
            beta_lambda,
            mu_tsleep,  # this is not returned
            sigma_tsleep,  # this is not returned
            mu_tawake,  # this is not returned
            sigma_tawake,  # this is not returned
        ) = build_hyper_hyper_model(
            observed_event_counts, n_bins, n_days, total_bins, time_bins
        )
        with model:
            # Define step methods
            
            step = [
            pm.Metropolis([tsleep, tawake]),
            pm.NUTS(
                vars=[
                    lambda_awake,
                    lambda_sleep,
                    alpha_lambda,
                    beta_lambda,
                    mu_tsleep,  # this is not returned
                    sigma_tsleep,  # this is not returned
                    mu_tawake,  # this is not returned
                    sigma_tawake,  # this is not returned
                ]
                )
            ]
        
    else:
        model_builders = {
            "pooled_pooled": build_pooled_pooled_model,
            "independent_pooled": build_independent_pooled_model,
            "independent_independent": build_independent_independent_model,
            "independent_hyper": build_independent_hyper_model,
        }
        try:
            model = model_builders[model_name](
                observed_event_counts, n_bins, n_days, total_bins, time_bins
            )
            step = None
        except Exception as e:
            print(f"Error building model {model_name}:")
            print(f"Error message: {str(e)}")
            raise


    # Sample from the model
    with model:
        # Sampling
        try:
            trace = pm.sample(
                draws=SAMPLECONFIG["n_samples"],
            chains=SAMPLECONFIG["n_chains"],
            tune=SAMPLECONFIG["tune"],
            cores=SAMPLECONFIG["cores"],
            target_accept=SAMPLECONFIG["target_accept"],
            compute_convergence_checks=SAMPLECONFIG["compute_convergence_checks"],
            random_seed=SAMPLECONFIG["random_seed"],
                step=step
            )
            
            logp = trace.sample_stats["lp"]

        except Exception as e:
            print(f"Error sampling from model {model_name}:")
            print(f"Error message: {str(e)}")
            raise

    return model, trace, logp

def validate_model(model, trace, progressbar=True):
    """
    (Optional)Perform validation checks on a fitted model.
    
    Parameters:
    ----------
    model : pm.Model
        PyMC model object
    trace : pm.MultiTrace
        Trace from the fitted model
    observed_event_counts : np.ndarray
        Original observed data
    progressbar : bool, optional
        Whether to show progress bar for posterior predictive sampling
        
    Returns:
    -------
    dict
        Dictionary containing validation metrics:
        - posterior_predictive: Posterior predictive samples
        - log_likelihood: Log likelihood of the model
    """

    with model:
        posterior_predictive = pm.sample_posterior_predictive(
            trace, progressbar=SAMPLECONFIG["progressbar"]
        )
        log_likelihood = pm.compute_log_likelihood(trace)
        logp = trace.sample_stats["lp"]

    return posterior_predictive, log_likelihood, logp

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
validation_metrics = {} # Optional

for model_name in model_names:
    print(f"Running model: {model_name}")
    model, trace, logp = run_model(
        model_name, observed_event_counts, n_bins, n_days, total_bins, time_bins
    )
    traces[model_name] = trace
    logps[model_name] = logp

    # optional 
    validation_metrics[model_name] = validate_model(
        model, trace, observed_event_counts
    )

# Use validation_metrics for additional model comparison (optional)
for model_name, (posterior_pred, log_like, logp) in validation_metrics.items():
    print(f"\nValidation metrics for {model_name}:")
    print("Posterior predictive summary:")
    print("  Mean:", posterior_pred.posterior_predictive.mean().values)
    print("  Std:", posterior_pred.posterior_predictive.std().values)
    
    print("\nLog likelihood summary:")
    print("  Mean:", log_like.log_likelihood.mean().values)
    print("  Std:", log_like.log_likelihood.std().values)
    
    print("\nLog probability:", logp.mean())


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

# Create the map_models dictionary for DIC calculation
dic_models = {}
for model_name, trace in traces.items():
    print(f"\nModel: {model_name}")
    print("Log likelihood structure:")
    print(trace.log_likelihood)
    print("\nAvailable dimensions:")
    print(trace.log_likelihood.dims)
    
    # Convert trace to the expected format
    dic_models[model_name] = {
        "log_likelihood": {
            "events": trace.log_likelihood
        }
    }
    
# Calculate DIC values for model comparison if we have more than one model
if len(dic_models) > 1:
    reference_DIC, DIC_values, DIC_errors, model_display_names = calculate_DIC(dic_models)
    # Plot DIC values
    plot_DIC(reference_DIC, DIC_values, DIC_errors, model_display_names)
else:
    print("\nSkipping DIC calculation - need at least two models for comparison")
