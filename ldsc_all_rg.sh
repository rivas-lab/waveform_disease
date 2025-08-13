
#!/bin/bash

# Config paths (à adapter)
Finn_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/FinnGen_Freeze12/munged"
new_gwas="/oak/stanford/groups/mrivas/salma_backup/LDSC/new_gwas_munged"
output_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/rg_results_new_vs_finn"
log_dir="${output_dir}/logs_finn_rg"
mkdir -p "$output_dir" "$log_dir"

LDSC_SCRIPT="/oak/stanford/groups/mrivas/salma_backup/LDSC/ldsc.py"
PYTHON="python3"

REF_LD_CHR="./eur_w_ld_chr/"
W_LD_CHR="./eur_w_ld_chr/"

batch_size=10  # Nombre de fichiers new_files par job, ajuste selon besoin

# Récupération des fichiers
old_files=(${Finn_dir}/*.sumstats.gz)
new_files=(${new_gwas}/*.sumstats.gz)

echo "Found ${#old_files[@]} old files and ${#new_files[@]} new files"
echo "Batch size: $batch_size"

for ((i=0; i<${#old_files[@]}; i++)); do
  file1="${old_files[$i]}"
  base1=$(basename "$file1" .sumstats.gz)

  total_new=${#new_files[@]}

  for ((start=0; start<total_new; start+=batch_size)); do
    end=$((start + batch_size))
    if (( end > total_new )); then
      end=$total_new
    fi

    job_name="ldsc_rg_batch_${base1}_${start}_to_$((end-1))"
    out_name="${job_name}"

    script_file="slurm-${job_name}.sh"

    # Création du script SLURM
    cat <<EOF > "$script_file"
#!/bin/bash
#SBATCH --partition=mrivas
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_dir}/${out_name}.out
#SBATCH --error=${log_dir}/${out_name}.err
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate ldsc

cd /oak/stanford/groups/mrivas/salma_backup/LDSC
EOF

    # Ajout des commandes ldsc --rg pour le batch
    for ((j=start; j<end; j++)); do
      file2="${new_files[$j]}"
      base2=$(basename "$file2" .sumstats.gz)
      pair_out_name="${base1}_vs_${base2}"

      echo "${PYTHON} ${LDSC_SCRIPT} --rg ${file1},${file2} --ref-ld-chr ${REF_LD_CHR} --w-ld-chr ${W_LD_CHR} --out ${output_dir}/${pair_out_name}" >> "$script_file"
    done

    # Soumission du job SLURM
    sbatch "$script_file"
    echo "Submitted batch job ${job_name} for ${base1} new_files ${start} to $((end-1))"
  done
done

