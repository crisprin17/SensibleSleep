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
from model_functions import run_model, calculate_DIC
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
    model,trace, posterior_predictive, log_likelihood, logp = run_model(
        model_name, observed_event_counts, n_bins, n_days, total_bins, time_bins
    )
    traces[model_name] = trace
    logps[model_name] = logp
    # optional 
    validation_metrics[model_name] = validate_model(
        model, trace, observed_event_counts
    )

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
