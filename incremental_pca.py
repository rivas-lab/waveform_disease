import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import numpy as np
import time

# Paths
input_file = "/scratch/groups/mrivas/mrivasfinal/db6_coeff.tsv"
output_file = "/scratch/groups/mrivas/mrivasfinal/db6_coeff_ipca.tsv"

# Settings
chunksize = 1000
n_components = 100
batch_size = chunksize  # Usually match chunksize

# Start timer
total_start = time.time()

print("Reading first chunk to initialize scaler and PCA...")
reader = pd.read_csv(input_file, sep='\t', chunksize=chunksize)

first_chunk = next(reader)
ids = [first_chunk['24983_id']]
X_first = first_chunk.drop(columns=['24983_id'])

# Scale
print("Fitting scaler on first chunk...")
scaler = StandardScaler()
scaler.fit(X_first)
X_scaled_first = scaler.transform(X_first)

# Initialize Incremental PCA
print("Initializing IncrementalPCA...")
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

print("Fitting IPCA on first chunk...")
ipca.partial_fit(X_scaled_first)

# Store transformed chunks
print("Transforming first chunk...")
transformed_chunks = [ipca.transform(X_scaled_first)]

# Process remaining chunks
chunk_num = 1
for chunk in reader:
    chunk_num += 1
    print("Processing chunk {}...".format(chunk_num))
    ids.append(chunk['24983_id'])
    X = chunk.drop(columns=['24983_id'])

    print("Scaling chunk {}...".format(chunk_num))
    X_scaled = scaler.transform(X)

    print("Fitting IPCA on chunk {}...".format(chunk_num))
    ipca.partial_fit(X_scaled)

    print("Transforming chunk {}...".format(chunk_num))
    transformed_chunks.append(ipca.transform(X_scaled))

print("All chunks processed. Concatenating results...")

# Combine results
X_pca = np.vstack(transformed_chunks)
all_ids = pd.concat(ids, axis=0).reset_index(drop=True)

print("Saving PCA-transformed data to {}...".format(output_file))
pca_columns = ["PC{}".format(i+1) for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_columns)
pca_df['24983_id'] = all_ids
pca_df.to_csv(output_file, sep='\t', index=False)

# Finish
total_end = time.time()
print("Incremental PCA complete: reduced to {} components.".format(X_pca.shape[1]))
print("Total time: {:.2f} minutes".format((total_end - total_start) / 60.0))
