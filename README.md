# Forecasting Thermospheric Density with Transformers for Multi-Satellite Orbit Management

This is the code repository for the paper **Forecasting Thermospheric Density with Transformers for Multi-Satellite
Orbit Management** presented by Cedric Bös, Alessandro Bortotto, and Mohamed Khalil Ben-Larbi at CAMSAT 2025 in
Würzburg (_DOI is added as soon as it becomes available_).

## Structure

- `data`: Prepared dataset for directly running a custom training loop
- `models`: Trained models and data scalers 
- `notebooks`: Code notebooks for training the model (beware... it contains research code... :))
- `src`: General python code for running the model and for complex data aggregations

## What to do?

### Environment

For starting a new experiment, first initialize a new python environment with the python requirements defined in
`pyproject.toml`.
The easiest way is to use [uv](https://github.com/astral-sh/uv), a fast, easy, and modern python dependency manager.
For doing so, just execute `uv sync`.
This will automatically generate a new python virtual environment and install all necessary dependencies.
We tested this project with python 3.11 and 3.12, and the library versions defined in the `pyproject.toml`.

### Directly begin training

For training, we used `notebooks/train.ipynb`.
The data and the pretrained model are available through [git-lfs](https://git-lfs.com/) which is a dependency that must
be installed before the appropriate files can be downloaded.
The prepared datasets are then available in `data/`, while the models are available in `models/`.

### Start with the data

If you want to start from scratch, the original data from the
[MIT challenge](https://aeroastro.mit.edu/arclab/aichallenge/) is required.
The dataset can be found on the organizer's
[Dropbox](https://www.dropbox.com/scl/fo/ilxkfy9yla0z2ea97tfqv/AB9lngJ2yHvf9t5h2oQXaDc?rlkey=iju8q5b1kxol78kbt0b9tcfz3).
A more detailed description about the raw data is available in the
[documentation](https://2025-ai-challenge.readthedocs.io/en/latest/) of the challenge.

#### Data Analysis

We used `notebooks/anti_csv.ipynb` for converting the whole raw _.csv_ files to a more compact _.parquet_ file format
for further analysis.
This made the downstream tasks easier and quicker.
The `notebooks/*-exploration.ipynb` notebooks were used for a first analysis of the data.
Note that the GOES dataset was analyzed in an interactive python mode, so exploration notebook is available for this
dataset.

#### Dataset preparation

After a dataset analysis the datasets were prepared.
This involved all offline steps described in the original paper and in the presentation at CAMSAT 2025.
For this `notebooks/data-agg.ipynb` was used which produces the datascalers and the reduced dataset in _.npz_ format.
The dataset used for training our final models can also be found in `data/dataset-scaled-time.npz`
