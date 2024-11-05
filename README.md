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

# Build the model (choose the desired model type)
model = build_model(df, model_type='hierarchical_hierarchical')

# Sample from the model
with model:
    trace = pm.sample(draws=1000, tune=1000, cores=2, target_accept=0.9, random_seed=42)

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