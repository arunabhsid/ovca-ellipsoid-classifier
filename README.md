# ovca-ellipsoid-classifier

Quadratic (ellipsoid) classifier separating ovarian cancer patients from controls using FAM/HEX marker assays.

## Overview

This repository contains the analysis code for an ovarian cancer classification study. The data are marker measurements from six two-channel assays, each with a FAM and a HEX fluorescence concentration, for ovarian cancer patients and healthy controls. For every assay we derived a sum and a ratio of the two channels on the log scale, and used these to fit a classifier that separates patients from controls. The analysis evaluates each assay on its own, identifies the assay combinations that separate the two groups without error in this dataset, and reports a single classifier based on all six assays.

## Method

The classifier encloses the patient samples within a convex region and keeps the controls outside it. For a second-order polynomial kernel that region is an ellipsoid, which we obtain by solving a semidefinite program. Because a missed cancer case matters more than a false alarm, the program includes a parameter $\mu$ between 0 and 1 that shifts the fit toward higher sensitivity ($\mu$ near 1) or higher specificity ($\mu$ near 0) when the two groups cannot be separated perfectly.

Solving the program returns an ellipsoid of the form

$$
(s - s_0)^T A (s - s_0) = 1,
$$

with a center $s_0$ and a matrix $A$, which is then used both to classify samples and to draw the decision boundary. The full semidefinite-program formulation is given in `OC.Rmd`.

## Repository contents

- `scripts/OC.Rmd`: the full analysis: data preparation, the ellipsoid fits, the figures, the assay-combination search, and the final classifier.
- `scripts/OC_ellipsoid.py`: sets up and solves the semidefinite program that produces the ellipsoid classifier. This is the method used throughout the analysis.
- `scripts/OC_linear_prog.py`: a linear-programming variant of the separation problem, kept for reference.

The Python files are sourced into the R session through reticulate. The R Markdown reads from `raw_data/` and writes to `results/` and `plots/`, so it is run from inside `scripts/` with those folders alongside it.

## Data

The individual-level patient and control data are not included, as they cannot be shared for data protection reasons. The code documents the analysis rather than reproducing it end to end, so it will not run as-is without the underlying data.

## Dependencies

R, with the packages openxlsx, tidyr, dplyr, ggplot2, forcats, reticulate, ggforce, matlib, gridExtra, zplyr, GGally, ggpubr, and cowplot.

Python 3, with numpy, pandas, scipy, and cvxopt. The semidefinite program is solved through cvxopt's DSDP interface (`solver = "dsdp"`), so cvxopt needs to be built with DSDP support.

Add the R and Python version numbers and the package versions.

## Status

Manuscript submitted.
