# ICA-FunctionalLungMRI-
Python simulation illustrating an ICA-based workflow for functional lung MRI ventilation and perfusion imaging, as used in our manuscript.

# ICA demo for functional lung MRI (ventilation/perfusion-weighted)

This repository contains a **simulation-based, minimal demonstration** of the workflow used in our manuscript on applying Independent Component Analysis (ICA) to functional lung MRI for obtaining ventilation- and perfusion-weighted images.

The code uses **synthetic data** to illustrate the method and processing steps. While the example runs entirely on simulated data, the workflow is designed to be adaptable to real functional lung MRI datasets.

## Contents

- `scripts/ica_pipeline.py`  
  Main script implementing the simulation and ICA-based analysis pipeline.

## Purpose of this repository

The goal of this repository is to:
- Provide a transparent and reproducible illustration of the ICA approach used in the paper  
- Allow readers to understand and test the core methodology  
- Serve as a starting point for adapting the method to real MRI data  

This is **not** a full clinical pipeline, but a compact and didactic implementation aligned with the methods described in the manuscript.

## Requirements

For running the simulation example:

- Python 3.x
- numpy
- scipy
- scikit-learn
- matplotlib

You can install the minimal dependencies with:

```bash
pip install -r requirements.txt
