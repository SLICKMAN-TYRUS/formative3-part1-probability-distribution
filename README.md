# Formative 3 - Part 1: Probability Distributions

## Bivariate Normal Distribution Implementation

Student: Ajak Chol 
Course: ALU Mathematics For Machine Learning 
Date: 25 February 2026

## Overview

This repository contains my implementation of Part 1 for the group assignment. I implemented the Bivariate Normal Distribution probability density function from scratch without using statistical libraries like scipy.stats.

## Dataset

- Name: Iris Dataset (Fisher, 1936)
- Source: UCI Machine Learning Repository via scikit-learn
- Variables Used: Petal Length (cm) and Petal Width (cm)
- Number of Samples: 150 iris flowers
- Correlation Coefficient: 0.96 (strong positive correlation)

## Implementation Details

I wrote the bivariate normal PDF formula manually using only numpy for basic array operations. The mathematical formula implemented is:

f(x,y) = [1/(2*pi*sigma_x*sigma_y*sqrt(1-rho^2))] * exp(-Q/[2(1-rho^2)])

where Q = [(x-mu_x)/sigma_x]^2 + [(y-mu_y)/sigma_y]^2 - 2*rho*[(x-mu_x)/sigma_x][(y-mu_y)/sigma_y]

Where:
- mu_x, mu_y are the means of the two variables
- sigma_x, sigma_y are the standard deviations
- rho is the correlation coefficient

No statistical libraries were used for PDF calculation. Only numpy was used for basic mathematical operations.

## Files in Repository

- notebooks/part1_bivariate_normal.ipynb - Main Jupyter notebook with complete analysis
- src/bivariate_normal.py - Modular Python implementation of BVN PDF
- src/__init__.py - Package initialization file
- figures/ - Directory for generated plots (contour and 3D surface)
- requirements.txt - Python dependencies
- README.md - This file

## How to Run

1. Install required packages:
   pip install -r requirements.txt

2. Navigate to notebooks directory:
   cd notebooks

3. Start Jupyter:
   jupyter notebook

4. Open part1_bivariate_normal.ipynb and run all cells

## Results

- Parameters calculated from data:
  - mu_x (mean petal length) = 3.759
  - mu_y (mean petal width) = 1.199
  - sigma_x (std petal length) = 1.764
  - sigma_y (std petal width) = 0.763
  - rho (correlation) = 0.962

- PDF at mean point: 0.83 (maximum value)

- Visualizations created:
  - Contour plot showing elliptical probability density contours
  - 3D surface plot showing the characteristic bell-shaped distribution

- Verification:
  - Peak of PDF occurs at the mean values
  - Volume under surface approximately 1.0 (valid PDF)
  - Strong correlation creates diagonal ridge in 3D plot

## References

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179-188.