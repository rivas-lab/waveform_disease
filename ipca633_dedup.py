import pandas as pd

df = pd.read_csv("/scratch/groups/mrivas/mrivasfinal/ipca633.phe", sep='\t')
df = df.drop_duplicates(subset=["#FID", "IID"])
df.to_csv("/scratch/groups/mrivas/mrivasfinal/ipca633_dedup.phe", sep='\t', index=False)
