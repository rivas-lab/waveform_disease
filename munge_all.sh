#!/bin/bash
#SBATCH --partition=mrivas
#SBATCH --job-name=munge_all
#SBATCH --output=/oak/stanford/groups/mrivas/salma_backup/LDSC/I9_phenos_munged/logs/munge_all_finR12.out
#SBATCH --error=/oak/stanford/groups/mrivas/salma_backup/LDSC/I9_phenos_munged/logs/munge_all_finR12.err

# Top-level driver: submit one job per file
input_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/I9_phenos"
output_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/I9_phenos"
log_dir="${output_dir}/logs"
whm3_path="/oak/stanford/groups/mrivas/salma_backup/LDSC/w_hm3.snplist"
# Point directly at your conda env’s python and the munging script
PYTHON=~/miniconda3/envs/ldsc/bin/python3
MUNGE_SCRIPT=/oak/stanford/groups/mrivas/salma_backup/LDSC/munge_sumstats.py

mkdir -p "$output_dir" "$log_dir"

for f in "${input_dir}"/*.gz; do
  filename=$(basename "$f")
  base="${filename%.gz}"
  cleaned="${output_dir}/${base}_input_cleaned.txt"
  out_prefix="${output_dir}/${base}_cleaned"

  sbatch <<EOF
#!/bin/bash
#SBATCH --partition=mrivas
#SBATCH --job-name=munge_${base}
#SBATCH --output=${log_dir}/${base}.out
#SBATCH --error=${log_dir}/${base}.err

# Activate your env (optional since we call the python binary directly)
# module load anaconda3
source activate ldsc

# Go where the data & script live
cd ${output_dir}

# 1) Drop rows with missing values in columns 12–15
awk -F'\t' 'NR==1 || (\$12 != "" && \$13 != "" && \$14 != "" && \$15 != "")' "${f}" > "${cleaned}"

# 2) Run the munging
${PYTHON} ${MUNGE_SCRIPT} \\
  --sumstats         "${cleaned}" \\
  --out              "${out_prefix}" \\
  --merge-alleles    "${whm3_path}" \\
  --snp              ID \\
  --N-col            OBS_CT \\
  --a1               A1 \\
  --a2               OMITTED \\
  --p                P \\
  --signed-sumstats  BETA,0
EOF

  echo "Submitted job for ${base}"
done

