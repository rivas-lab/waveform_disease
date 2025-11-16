## Overview
Current ECG signal analysis in large-scale health datasets primarily relies on summary statistics, such as wavelet energy measures, to assess the relationship between heart signals and disease. This project examines the individual detailed wavelet coefficients in an effort to uncover new predictive biomarkers and potentially improve disease risk prediction performance.

We also explore the reconstruction of ECG waveforms from reduced-dimensional representations, allowing interpretable recovery of signal morphology from compressed data. In parallel, our objective is to estimate the heritability and genetic correlation of the energy features derived from wavelets using genome-wide association studies (GWAS), which may reveal genetic influences on different ECG features.
We also explore the reconstruction of ECG waveforms from reduced-dimensional representations, allowing interpretable recovery of signal morphology from compressed data. In parallel, our objective is to estimate the heritability and genetic correlation of the energy features derived from wavelets using genome-wide association studies (GWAS), which may reveal genetic influences on different ECG features.

## Data Pipeline


We had two primary sources of data:

- **UK Biobank:** ECG signal files for 47,052 individuals white british only
- **Demographic data:** Genetic principal components, biomarkers, and disease phenotypes

**Data Processing Steps:**

1. **Extract Energy Coefficients**
   - Extract energy coefficients from raw waveform coefficient data.

2. **Wavelet Decomposition**
   - Use the script `ecg_energy.py` (utilizing the PyWavelets library) to decompose ECG signals per lead using the Daubechies 6 (db6) wavelet at level 6.
   - Calculate energy features by summing the squares of coefficients per lead, per individual.
   - The resulting dataset: 72,716 rows × 85 columns. After mapping IDs to match the `master.phe` UK Biobank file and removing duplicates and keeping white british, we get our phenotype file wavelet_dedup_new.phe: 47,052 rows × 86 columns.

3. **GWAS Analysis**
   - For each energy feature phenotype, perform Genome-Wide Association Studies (GWAS) using PLINK2.
   - Adjust for covariates (age, sex, principal components), apply quantile normalization, and output results for chromosomes 1-22.

   Example PLINK2 command (replace placeholders as needed):

   ```bash
   ./plink2 --chr 1-22 \
     --covar /oak/stanford/groups/mrivas/ukbb24983/phenotypedata/master_phe/master.phe \
     --covar-name age,sex,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 \
     --covar-variance-standardize \
     --glm qt-residualize hide-covar omit-ref \
     --keep /oak/stanford/groups/mrivas/ukbb24983/sqc/population_stratification_w24983_20211020/ukb24983_white_british.phe \
     --out [INSERT OUTPUT DIRECTORY HERE] \
     --pfile /oak/stanford/groups/mrivas/ukbb24983/array-combined/pgen/ukb24983_cal_hla_cnv.p \
     --pheno [INCLUDE PHENOTYPE FILE] \
     --pheno-name [INCLUDE PHENOTYPE NAME] \
     --pheno-quantile-normalize \
     --threads 20 \
     --vif 100000
   ```
   For our analyses, we use the phenotype file `wavelet_dedup_new.phe`.

4. **LDSC Regression (Heritability & Genetic Correlation)**
   - Use the [LDSC GitHub repository](https://github.com/bulik/ldsc) to run SNP-based heritability and genetic correlation analyses via LDSC regression.
   - The code and environment have been updated for compatibility with Python 3.8 and modern dependencies.

5. **Munge GWAS Files**
   - Prepare GWAS summary statistics for LDSC using `munge_all.sh`, modifying input paths as needed.

6. **Run Heritability Analysis**
   - Use `ldsc_all_h2.sh` to compute heritability between each pair of munged GWAS files.

7. **Run Genetic Correlation Analysis**
   - Use `ldsc_all_rg.sh` to compute genetic correlation between the munged GWAS files and external reference files (e.g., munged FinnGen I9 phenotype files).
  
8. **Finemapping**
   - Using Susie Inf  

## References

- Bulik-Sullivan, B.K., Loh, P.R., Finucane, H.K., Ripke, S., Yang, J., Patterson, N., Daly, M.J., Price, A.L., Neale, B.M., and the Schizophrenia Working Group of the Psychiatric Genomics Consortium. LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. Nature Genetics. 2015; 47(3): 291–295.

- UK Biobank: https://www.ukbiobank.ac.uk

- PyWavelets Documentation: https://pywavelets.readthedocs.io
