#!/bin/bash

# Set paths

annot_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/new_gwas_munged"
output_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/new_gwas_h2_results"
log_dir="/oak/stanford/groups/mrivas/salma_backup/LDSC/new_gwas_h2_results/logs_ldsc_h2"
mkdir -p "$output_dir" "$log_dir"

# LDSC script path
LDSC_SCRIPT="/oak/stanford/groups/mrivas/salma_backup/LDSC/ldsc.py"
PYTHON="python3"

# LD score files
REF_LD_CHR="./eur_w_ld_chr/"
W_LD_CHR="./eur_w_ld_chr/"

# Get all .sumstats.gz files in annot
files=(${annot_dir}/*.sumstats.gz)

# Loop over each file for univariate h2 estimation
for ((i=0; i<${#files[@]}; i++)); do
   file1="${files[$i]}"
   base1=$(basename "$file1" .sumstats.gz)
   out_name="${base1}_h2"

   sbatch <<EOF
#!/bin/bash
#SBATCH --partition=mrivas
#SBATCH --job-name=ldsc_h2_${base1}
#SBATCH --output=${log_dir}/${out_name}.out
#SBATCH --error=${log_dir}/${out_name}.err
#SBATCH --time=02:00:00

source activate ldsc
cd /oak/stanford/groups/mrivas/salma_backup/LDSC

${PYTHON} ${LDSC_SCRIPT} \\
  --h2 ${file1} \\
  --ref-ld-chr ${REF_LD_CHR} \\
  --w-ld-chr ${W_LD_CHR} \\
  --out ${output_dir}/${out_name}
EOF

   echo "Submitted h2 job for ${base1}"

done

