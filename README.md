# RIXS: How well does LR-TDDFT perform?
A benchmark set to compare electronic RIXS spectra calculated using LR-TDDFT and ADC.

The Python modules required to run the Jupyter notebooks can be found in `rixs.yml` and easily installed using conda.

## Install miniconda

Download the appropriate installer for your operating system and follow the installation instructions on the [miniconda website](https://docs.anaconda.com/free/miniconda/miniconda-install/).

## Create the conda environment

Using the `rixs.yml` file to create a conda environment `rixs-env` will install all the needed packages to run the Jupyter notebooks in this repository.

```
conda env create -f rixs.yml
```

## Use the new environment

```
conda activate rixs-env
```

## Set the PYTHONPATH

```
export PYTHONPATH=$PYTHONPATH:/path/to/rixs-lr-tddft
```
