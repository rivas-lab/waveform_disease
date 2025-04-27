import os
import numpy as np
import pandas as pd
import pywt
from pathlib import Path
import gc  # For garbage collection

# Paths
ecg_dir = "/scratch/groups/mrivas/jackos/ecgs/ukb/"
mapping_file = "/scratch/groups/mrivas/jackos/ukb22282_24983_mapping.tsv"
output_dir = "/scratch/groups/mrivas/mrivasfinal/"
output_file = os.path.join(output_dir, "db6_coeff.tsv")
temp_dir = os.path.join(output_dir, "temp_chunks")

# Create temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Load the mapping file
print("Loading mapping file...")
mapping_df = pd.read_csv(mapping_file, sep='\t', header=None, names=['old_id', 'new_id'])
id_mapping = dict(zip(mapping_df['old_id'].astype(str), mapping_df['new_id'].astype(str)))

# Wavelet parameters
wavelet = 'db6'
level = 6

# Get list of ECG files and corresponding new IDs
print("Finding valid ECG files...")
ecg_files = list(Path(ecg_dir).glob("*.npy"))
valid_files = []
new_ids = []

for ecg_file in ecg_files:
    old_id = ecg_file.name.split('_20205')[0]
    new_id = id_mapping.get(old_id, None)
    if new_id:
        valid_files.append(ecg_file)
        new_ids.append(new_id)

# Free up memory
del ecg_files, mapping_df, id_mapping
gc.collect()

# Update n_individuals to reflect valid files only
n_individuals = len(valid_files)
print(f"Total valid individuals: {n_individuals}")

# Define coefficient sizes
coeff_sizes = [88, 88, 166, 322, 634, 1258, 2505]
total_coeffs_per_lead = sum(coeff_sizes)
print(f"Total coefficients per lead: {total_coeffs_per_lead}")

# Create column names
print("Creating column names...")
feature_names = []
for lead_idx in range(12):
    for level_idx, size in enumerate(coeff_sizes):
        for coeff_idx in range(size):
            feature_names.append(f"lead{lead_idx+1}_level{level_idx}_coeff{coeff_idx+1}")

columns = ['24983_id'] + feature_names
print(f"Number of column names: {len(columns)}")

# Process in smaller, more manageable chunks
chunk_size = 1000  # Reduced chunk size
n_chunks = (n_individuals + chunk_size - 1) // chunk_size
temp_files = []

print(f"Processing {n_individuals} individuals in {n_chunks} chunks of size {chunk_size}...")

for chunk_idx in range(n_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, n_individuals)
    chunk_files = valid_files[start_idx:end_idx]
    chunk_ids = new_ids[start_idx:end_idx]
    
    print(f"Chunk {chunk_idx + 1}/{n_chunks}: {start_idx} to {end_idx}, {len(chunk_files)} files")
    
    if not chunk_files:
        print(f"Skipping empty chunk {chunk_idx + 1}")
        continue
    
    # Temporary file for this chunk
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_idx}.tsv")
    temp_files.append(temp_file)
    
    # Process this chunk
    try:
        # Load chunk into memory
        ecg_data = np.stack([np.load(f) for f in chunk_files], axis=0)
        chunk_rows = []
        
        # Process each individual in the chunk
        for i, individual_id in enumerate(chunk_ids):
            row = [individual_id]
            for lead_idx in range(12):
                lead_data = ecg_data[i, lead_idx, :]
                coeffs = pywt.wavedec(lead_data, wavelet, level=level)
                for level_coeffs in coeffs:
                    row.extend(level_coeffs.tolist())
            
            chunk_rows.append(row)
        
        # Create DataFrame for this chunk and save to temp file
        chunk_df = pd.DataFrame(chunk_rows, columns=columns)
        chunk_df.to_csv(temp_file, sep='\t', index=False)
        
        # Free memory
        del ecg_data, chunk_rows, chunk_df
        gc.collect()
        
        print(f"Saved chunk {chunk_idx + 1} to {temp_file}")
    
    except Exception as e:
        print(f"Error processing chunk {chunk_idx + 1}: {e}")
        continue

# Merge all temp files into the final output file
print(f"Merging {len(temp_files)} temporary files...")

# Write header to the final output file
with open(output_file, 'w') as f:
    f.write('\t'.join(columns) + '\n')

# Append each temp file to the final output file
for i, temp_file in enumerate(temp_files):
    print(f"Appending file {i+1}/{len(temp_files)}: {temp_file}")
    try:
        # Read the temp file without header and append to output
        with open(temp_file, 'r') as f:
            # Skip header line
            next(f)
            # Read and append the rest
            with open(output_file, 'a') as out:
                for line in f:
                    out.write(line)
        
        # Optional: Remove the temp file after merging
        os.remove(temp_file)
    except Exception as e:
        print(f"Error appending temp file {temp_file}: {e}")

print(f"Saved joint wavelet features to {output_file}")