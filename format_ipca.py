import pandas as pd

# Load the PCA output
df = pd.read_csv("/scratch/groups/mrivas/mrivasfinal/db6_coeff_ipca_95.tsv", sep="\t")

# Create FID and IID columns from 24983_id
df['#FID'] = df['24983_id']
df['IID'] = df['24983_id']

# Keep only #FID, IID, and the first 633 PCs
pcs = ["PC" + str(i) for i in range(1, 634)]
formatted_df = df[['#FID', 'IID'] + pcs]

# Save in PLINK-compatible phenotype format
formatted_df.to_csv("/scratch/groups/mrivas/mrivasfinal/ipca633.phe", sep='\t', index=False)

print("Saved formatted PCA file: ipca633.phe")

