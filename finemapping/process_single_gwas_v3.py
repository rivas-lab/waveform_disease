import sys
import pandas as pd
import numpy as np
from gentropy.method.susie_inf import SUSIE_inf
import subprocess
import os

# Paths absolus
PLINK2_BIN = "/oak/stanford/groups/mrivas/salma_backup/software_tools/plink2"
PFILE = "/oak/stanford/groups/mrivas/ukbb24983/array-combined/pgen/ukb24983_cal_hla_cnv.p"
WB_KEEP = "/oak/stanford/groups/mrivas/ukbb24983/sqc/population_stratification_w24983_20211020/ukb24983_white_british.phe"

def process_gwas(gwas_file, pval_threshold=5e-8):
    print(f"Processing {gwas_file} with p-value threshold {pval_threshold}")

    # Créer dossiers de sortie
    os.makedirs("logs", exist_ok=True)
    os.makedirs("sig_files", exist_ok=True)
    os.makedirs("clump_files", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(gwas_file):
        print(f"Erreur: {gwas_file} n'existe pas")
        return {}

    gwas_basename = os.path.basename(gwas_file)
    if not gwas_basename.startswith("lead"):
        print(f"{gwas_file}: Skipped, does not start with 'lead'")
        return {}

    # Étape 1: Charge summary stats
    try:
        df = pd.read_csv(gwas_file, sep="\t", header=0)
        print("Colonnes disponibles:", df.columns.tolist())
        required_cols = ['#CHROM', 'POS', 'ID', 'BETA', 'SE', 'P']
        if not all(col in df.columns for col in required_cols):
            print(f"Erreur: Colonnes manquantes. Attendues: {required_cols}, Trouvées: {df.columns.tolist()}")
            return {}
        print("Aperçu des données:")
        print(df.head())
        df = df.dropna(subset=['BETA', 'SE', 'ID', 'P'])
        print(f"Nombre de lignes après dropna: {len(df)}")
        n = df.get('OBS_CT', 24983).median()  # Sample size (gardé original)
        print(f"Taille échantillon estimée: {n}")

        # Débogage p-valeurs
        print("10 plus petites p-valeurs:", df['P'].nsmallest(10).tolist())
        print(f"Nombre de P < 5e-8: {len(df[df['P'] < 5e-8])}")
        print(f"Nombre de P < 1e-6: {len(df[df['P'] < 1e-6])}")
        print(f"Nombre de P < 1e-5: {len(df[df['P'] < 1e-5])}")
    except Exception as e:
        print(f"Erreur lecture {gwas_file}: {e}")
        return {}

    # Étape 2: Identifie loci sig
    sig_df = df[df['P'] < pval_threshold]
    print(f"Nombre de loci significatifs (P < {pval_threshold}): {len(sig_df)}")
    if sig_df.empty:
        print(f"{gwas_file}: No significant loci at threshold {pval_threshold}")
        return {}

    # Vérifie fichiers PGEN
    for ext in ['.pvar', '.psam', '.pgen']:
        if not os.path.exists(PFILE + ext):
            print(f"Erreur: Fichier PGEN manquant {PFILE + ext}")
            return {}

    # Fichier temporaire pour SNPs significatifs
    sig_file = f"sig_files/sig_{gwas_basename}.txt"
    try:
        sig_df[['#CHROM', 'POS', 'ID', 'P']].to_csv(sig_file, sep="\t", index=False)
    except KeyError:
        print(f"Erreur: Colonnes '#CHROM', 'POS', 'ID', 'P' non trouvées dans {gwas_file}")
        return {}

    # Clump pour loci indépendants
    clump_out = f"clump_files/clump_{gwas_basename}"
    clump_cmd = [
        PLINK2_BIN, "--pfile", PFILE, "--keep", WB_KEEP,
        "--clump", sig_file, "--clump-p1", str(pval_threshold), "--clump-r2", "0.1", "--clump-kb", "500",
        "--out", clump_out
    ]
    try:
        print(f"Exécution commande PLINK2: {' '.join(clump_cmd)}")
        subprocess.run(clump_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur PLINK2 clump {gwas_file}: {e}")
        return {}

    # Charge clumped leads
    clump_file = f"{clump_out}.clumps"
    if not os.path.exists(clump_file):
        clump_file = f"{clump_out}.clumped"
        if not os.path.exists(clump_file):
            print(f"{gwas_file}: No clumped file generated (.clumps or .clumped)")
            return {}

    # Vérifie contenu du fichier clump
    clump_df = pd.read_csv(clump_file, sep=r"\s+")
    print(f"Colonnes dans {clump_file}: {clump_df.columns.tolist()}")
    if clump_df.empty:
        print(f"{gwas_file}: Fichier de clumps vide")
        return {}

    # Gère les variations de noms de colonnes
    chrom_col = 'CHR' if 'CHR' in clump_df.columns else '#CHROM' if '#CHROM' in clump_df.columns else None
    pos_col = 'BP' if 'BP' in clump_df.columns else 'POS' if 'POS' in clump_df.columns else None
    snp_col = 'SNP' if 'SNP' in clump_df.columns else 'ID' if 'ID' in clump_df.columns else None

    if not all([chrom_col, pos_col, snp_col]):
        print(f"Erreur: Colonnes requises (CHR/#CHROM, BP/POS, SNP/ID) manquantes dans {clump_file}")
        return {}

    results = {}
    for _, lead in clump_df.iterrows():
        chrom = lead[chrom_col]
        pos = lead[pos_col]
        lead_id = lead[snp_col]
        start, end = pos - 500000, pos + 500000

        # Extrait région complète
        region_df = df[(df['#CHROM'] == chrom) & (df['POS'].between(start, end))]
        if len(region_df) < 10:
            print(f"{gwas_file} {lead_id}: Région trop petite, skipped")
            continue

        # Filtre SNPs significatifs dans la région
        sig_region_df = region_df[region_df['P'] < pval_threshold]
        if len(sig_region_df) < 10:
            print(f"{gwas_file} {lead_id}: Moins de 10 SNPs significatifs dans région, skipped")
            continue

        # Crée fichier temporaire pour SNPs significatifs (IDs seulement pour --extract)
        sig_snps_file = f"sig_files/sig_region_{gwas_basename}_{lead_id}.txt"
        sig_region_df[['ID']].to_csv(sig_snps_file, sep="\t", index=False, header=False)

        # Étape 3: Calcule LD in-sample (seulement sur SNPs significatifs)
        ld_out = f"results/ld_{gwas_basename}_{lead_id}"
        ld_cmd = [
            PLINK2_BIN, "--pfile", PFILE, "--keep", WB_KEEP,
            "--chr", str(chrom), "--from-bp", str(start), "--to-bp", str(end),
            "--extract", sig_snps_file,  # LD sur SNPs significatifs seulement
            "--r-unphased", "square", "--maf", "0.001",
            "--force-intersect",  # ← FIX: Permet combinaison de filters
            "--out", ld_out
        ]
        try:
            print(f"Exécution commande PLINK2 LD: {' '.join(ld_cmd)}")
            subprocess.run(ld_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur PLINK2 LD {gwas_file} {lead_id}: {e}")
            continue

        # Charge LD
        ld_file = f"{ld_out}.unphased.vcor1"
        if os.path.exists(ld_file):
            ld = np.loadtxt(ld_file, skiprows=0)
            ld_snps_file = f"{ld_out}.unphased.vcor1.vars"
            if os.path.exists(ld_snps_file):
                with open(ld_snps_file, 'r') as f:
                    ld_snps = [line.strip() for line in f]
            else:
                ld_snps = sig_region_df['ID'].values  # Fallback
        else:
            ld_file = f"{ld_out}.ld"
            if os.path.exists(ld_file):
                ld_df = pd.read_table(ld_file)
                ld = ld_df.pivot(index='SNP_A', columns='SNP_B', values='R').fillna(1e-4).values
                ld_snps = ld_df['SNP_A'].unique()
            else:
                print(f"Erreur: Fichier LD manquant ({ld_out}.unphased.vcor1 ou {ld_out}.ld)")
                continue

        # Remplace NaN/inf dans LD par epsilon (1e-4)
        epsilon = 1e-4
        ld = np.nan_to_num(ld, nan=epsilon, posinf=epsilon, neginf=epsilon)
        print(f"{gwas_file} {lead_id}: LD corrigé pour NaN/inf avec epsilon={epsilon}")
        print(f"LD shape: {ld.shape}")

        # Aligne region_df EXACTEMENT à l'ordre de ld_snps (fix désalignement)
        try:
            aligned_df = sig_region_df.set_index('ID').reindex(ld_snps).dropna(subset=['BETA', 'SE'])
        except KeyError as e:
            print(f"{gwas_file} {lead_id}: Erreur alignement (IDs manquants): {e}")
            continue

        if len(aligned_df) < 10:
            print(f"{gwas_file} {lead_id}: Moins de 10 SNPs après alignement, skipped")
            continue

        # Compute z et snps dans l'ordre LD
        z = (aligned_df['BETA'] / aligned_df['SE']).values
        snps = aligned_df.index.values

        # Vérification alignement (debug optionnel)
        print(f"{gwas_file} {lead_id}: {len(z)} SNPs alignés (doit matcher LD shape {ld.shape[0]})")
        if len(z) != ld.shape[0]:
            print(f"Erreur: z ({len(z)}) != LD ({ld.shape[0]})")
            continue

        # Filtrer z-scores extrêmes (après alignement)
        z_mask = np.abs(z) < 30
        if not np.any(z_mask):
            print(f"{gwas_file} {lead_id}: Tous les z-scores sont extrêmes (|z| >= 30), skipped")
            continue
        z = z[z_mask]
        snps = snps[z_mask]
        aligned_df = aligned_df.iloc[z_mask]  # Pour debug si besoin
        ld = ld[z_mask, :][:, z_mask]  # Sous-matrice LD alignée

        # Vérifie NaN/inf dans z (après filtre)
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            print(f"{gwas_file} {lead_id}: z contient NaN ou inf")
            print(f"z: {z}")
            print(f"BETA: {aligned_df['BETA'].values}")
            print(f"SE: {aligned_df['SE'].values}")
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"z corrigé: {z}")

        # Optionnel: Régularise LD pour stabilité SuSiE (évite PIP=0)
        ld += np.eye(ld.shape[0]) * 1e-6

        # Sauvegarde z et LD pour débogage
        np.savetxt(f"results/z_{gwas_basename}_{lead_id}.txt", z, fmt="%.6f")
        np.savetxt(f"results/ld_{gwas_basename}_{lead_id}.txt", ld, fmt="%.6f")
        pd.DataFrame({'SNP': snps}).to_csv(f"results/snps_{gwas_basename}_{lead_id}.csv", index=False)
        pd.DataFrame({'SNP': ld_snps}).to_csv(f"results/ld_snps_{gwas_basename}_{lead_id}.csv", index=False)

        # Étape 4: Run SuSiE-inf
        try:
            susie_result = SUSIE_inf.susie_inf(
                z=z, LD=ld, n=n, L=5, method='moments',
                est_ssq=True, est_sigmasq=True, est_tausq=True,
                maxiter=500, PIP_tol=0.001
            )
            print(f"SuSiE-inf completed for {lead_id}")
        except Exception as e:
            print(f"Erreur SuSiE-inf {gwas_file} {lead_id}: {e}")
            continue

        # Vérifie PIP
        pip_matrix = susie_result['PIP']           # matrice p x L
        pip = 1 - np.prod(1 - pip_matrix, axis=1)  # formule marginale correcte
        pip = np.clip(pip, 0, 1)                   # sécurité numérique
        print(f"{gwas_file} {lead_id}: PIP sum = {np.sum(pip)}")
        if np.sum(pip) < 0.9:
            print(f"{gwas_file} {lead_id}: Somme des PIP trop faible (< 0.9), credible set non calculé")
            pd.DataFrame({'SNP': snps, 'PIP': pip, 'P': aligned_df['P'].values}).to_csv(f"results/pip_{gwas_basename}_{lead_id}.csv", index=False)
            continue

        # Credible Sets
        try:
            cs_list = SUSIE_inf.cred_inf(
                PIP=susie_result['PIP'], n=n, coverage=0.9, purity=0.5, LD=ld
            )
        except Exception as e:
            print(f"Erreur cred_inf {gwas_file} {lead_id}: {e}")
            pd.DataFrame({'SNP': snps, 'PIP': pip, 'P': aligned_df['P'].values}).to_csv(f"results/pip_{gwas_basename}_{lead_id}.csv", index=False)
            continue

        # Stocke et output
        locus_key = f"{gwas_basename}_{lead_id}"
        results[locus_key] = {
            'PIP': pip,
            'CS': cs_list,
            'SNPs': snps
        }

        pd.DataFrame({'SNP': snps, 'PIP': pip, 'P': aligned_df['P'].values}).to_csv(f"results/pip_{locus_key}.csv", index=False)
        with open(f"results/cs_{locus_key}.txt", 'w') as f:
            for i, cs in enumerate(cs_list):
                if cs:
                    cs_snps = snps[cs]
                    f.write(f"CS {i+1}: {cs_snps}, size={len(cs)}\n")

        print(f"{locus_key} processed")

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_single_gwas.py <full_path_to_gwas_file> [pval_threshold]")
        sys.exit(1)
    gwas_file = sys.argv[1]
    pval_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-8
    process_gwas(gwas_file, pval_threshold)
