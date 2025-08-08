## Overview
Current ECG signal analysis in large-scale health datasets primarily relies on summary statistics, such as wavelet energy measures, to assess the relationship between heart signals and disease. This project examines the individual detailed wavelet coefficients in an effort to uncover new predictive biomarkers and potentially improve disease risk prediction performance.

We also explore the reconstruction of ECG waveforms from reduced-dimensional representations, allowing interpretable recovery of signal morphology from compressed data. In parallel, our objective is to estimate the heritability of the principal components derived from wavelets using genome-wide association studies (GWAS), which may reveal genetic influences on different ECG features.

## Data Pipeline
We had two primary sources of data:
  * UK Biobank: ECG signal files for 48,842 individuals
  * Demographic data, genetic principal components, biomarkers and disease phenotypes (3,511 columns)

We followed the following data processing steps:
  1. Edit pipeline to extract detailed energy coefficients from raw waveform data\
     **Data Shape**: 48,842 × 60,733
  2. Merge data: demographic data, genetic PCs, and 132 binary disease phenotype IDs. Relevant phenotype IDs were identified from previous linear regressions run on energy measures\
     **Data Shape**: 48,842 × 5,139
  3. Run (updated LDSC)
  4. Wavelet Decomposition
  5. GWAS
  6. Run LDSC for heritability and genetic correlation


