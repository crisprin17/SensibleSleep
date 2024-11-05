# SensibleSleep

<div align="center">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/compasspathways/ds_utils/blob/main/.pre-commit-config.yaml)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

SensibleSleep is an open-source Python package that implements a Hierarchical Bayesian model for learning sleep patterns from smartphone screen-on events. 

The package allows you to generate synthetic data, build and run Bayesian models using PyMC, and analyze the inferred sleep patterns. This package is based on the methodology presented in the paper:

Cuttone A, Bækgaard P, Sekara V, Jonsson H, Larsen JE, Lehmann S. SensibleSleep: A Bayesian Model for Learning Sleep Patterns from Smartphone Events. PLoS One. 2017;12(1)

## Installation Steps

1. Clone the repository

```bash
git clone https://github.com/compasspathways/sensiblesleep.git
```

2. Set Up a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the dependencies

```bash
pip install -r requirements.txt
```

4. Install the package

```bash
pip install .
```

## Quick Start
Here's a quick example of how to generate synthetic data, build the model, and run it:


```bash
import pymc as pm
import arviz as az
az.style.use("arviz-doc")

# Import the generate_synthetic_data function
from synthetic_data.synthetic_data import generate_synthetic_data
from sensible_sleep. import build_model

# Generate the synthetic data
df = generate_synthetic_data()

# for one user
n_days, bin_hours, observed_event_counts = extract_user_data(df, user='user_001')
n_bins, total_bins, time_bins = create_time_bins(n_days=n_days)

# Build the model (choose the desired model type)
model_name = 'hyper_hyper'  
trace, posterior_predictive, log_likelihood, logp = run_model(
    model_name, observed_event_counts, n_bins, n_days, total_bins, time_bins
)

# Analyze the results
plot_trace(trace)
summarize_trace(trace)
```


## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Acknowledgments
Research Foundation: This package is based on the methodology and findings from the paper:

Cuttone A, Bækgaard P, Sekara V, Jonsson H, Larsen JE, Lehmann S. SensibleSleep: A Bayesian Model for Learning Sleep Patterns from Smartphone Events. PLoS One. 2017 Jan 11;12(1):e0169901. doi: 10.1371/journal.pone.0169901. PMID: 28076375; PMCID: PMC5226832.

Open-Source Community: Thanks to all the contributors and the open-source community for their invaluable tools and libraries that made this project possible.


## Need Help?
If you encounter any issues or have questions, feel free to:

Open an issue on the GitHub repository: https://github.com/crisprin17/SensibleSleep/issues
Contact the maintainer at cristiana.principato@gmail.com.
We're here to help you make the most out of SensibleSleep!