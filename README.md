# SPAE

## Introduction

**SPAE** is a novel computational framework designed to analyze cell cycle dynamics and cell differentiation using single-cell RNA sequencing (scRNA-seq) data.   SPAE employs an autoencoder that integrates both nonlinear and piecewise linear components, it uniquely utilizes sine and cosine functions within the decoder to accurately fit the periodicity of the cell cycle, thus facilitating precise pseudotime estimation. Moreover, SPAE incorporates a piecewise linear regression model to predict differentiation pathways across various cell types. We rigorously assessed SPAE's efficacy in estimating cell cycle pseudotime and determining cell stage classifications. SPAE is applied to several downstream analyses and applications to demonstrate its ability to accurately estimate the cell cycle pseudotime and stages.

<img src="https://github.com/YaJahn/SPAE/blob/master/Fig1.png" width="500px">

###  Requirements
Ensure you have the following dependencies installed:

- **numpy** == 1.23.5
- **pandas** == 2.0.3
- **scikit-learn** == 1.3.0
- **keras** == 2.13.1
- **tensorflow** == 2.13.0
- **matplotlib** == 3.7.2

You can install these dependencies using:

```bash
pip install numpy==1.23.5 pandas==2.0.3 scikit-learn==1.3.0 keras==2.13.1 tensorflow==2.13.0 matplotlib==3.7.2

```
###  Estimate cell cycle pseudotime
```bash
run mESCs_Quartz.py

```
