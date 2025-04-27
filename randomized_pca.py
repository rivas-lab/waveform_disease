import pandas as pd
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

file_path = 'db6_coeff.tsv'
chunksize = 100
random_state = 42

# First pass for mean and std calculation
mean = None
var = None
n_samples = 0
lead_cols = None

for chunk_data in tqdm(pd.read_csv(file_path, sep='\t', chunksize=chunksize)):
    lead_cols = [col for col in chunk_data.columns if 'lead' in col]

    chunk = chunk_data[lead_cols].apply(pd.to_numeric, errors='coerce')

    n = chunk.shape[0]
    chunk_mean = np.mean(chunk, axis=0)
    chunk_var = np.var(chunk, axis=0)

    if mean is None:
        mean = chunk_mean
        M2 = chunk_var * n
    else:
        delta = chunk_mean - mean
        total = n_samples + n
        mean += (delta * n) / total
        M2 += chunk_var * n + (delta ** 2) * (n * n_samples / total)

    n_samples = n_samples + n

    print("Processed chunk for Step 1")

var = M2 / n_samples
        

print("Completed Step 1")

# Second pass: compute Y = X_scaled * Omega
k = 72  # number of components to approximate
Omega = np.random.randn(len(lead_cols), k + 10) # oversampling
Y = None
ids = []

for i, chunk_data in enumerate(pd.read_csv(file_path, sep="\t", chunksize=chunksize)):
    chunk_ids = chunk_data['24983_id']
    ids.extend(chunk_ids.tolist())

    chunk = chunk_data[lead_cols].apply(pd.to_numeric, errors='coerce')
    chunk_scaled = (chunk - mean) / np.sqrt(var)
    Y_chunk = chunk_scaled @ Omega
    if Y is None:
        Y = Y_chunk
    else:
        Y = np.vstack([Y, Y_chunk])

    print(f"Processed Step 2 for chunk {i}")

    

# Orthonormalize Y
Q, _ = np.linalg.qr(Y)

print("Completed Step 2")

# Third pass: compute B = Q.T @ X_scaled
B = np.zeros((len(lead_cols), Q.shape[1]))
i0 = 0
for i, chunk_data in enumerate(pd.read_csv(file_path, sep="\t", chunksize=chunksize)):
    chunk = chunk_data[lead_cols].apply(pd.to_numeric, errors='coerce')

    chunk_scaled = (chunk - mean) / np.sqrt(var)
    i1 = i0 + len(chunk_scaled)
    B += chunk_scaled.T @ Q[i0:i1]  # shape: (features, k+10)
    i0 = i1
    
    print(f"Processed Step 3 for chunk {i}")


print("Completed Step 3")

# Final SVD on small matrix B
U_tilde, S, Vt = np.linalg.svd(B.T, full_matrices=False)
U = Q @ U_tilde

pd.DataFrame(U).to_csv("pca_U.tsv", sep='\t', index=False)
pd.DataFrame(S).to_csv("pca_S.tsv", sep='\t', index=False)
pd.DataFrame(Vt).to_csv("pca_Vt.tsv", sep='\t', index=False)

projected = U[:, :k] * S[:k]  # shape: (n_samples, k)

# Combine with IDs
df_proj = pd.DataFrame(projected, columns=[f'PC{i+1}' for i in range(k)])
df_proj['24983_id'] = ids

# Reorder columns and add #FID, IID
df_proj['#FID'] = df_proj['24983_id']
df_proj['IID'] = df_proj['24983_id']
df_proj = df_proj[['#FID', 'IID'] + [f'PC{i+1}' for i in range(k)]]

# Save
df_proj.to_csv("pca_projections_with_id.tsv", sep='\t', index=False)