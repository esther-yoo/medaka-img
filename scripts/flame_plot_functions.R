# function to read in the GWAS result from flexlmm for a specific chromosome (https://github.com/birneylab/flexlmm/blob/main/docs/output.md#gwas)
# select the columns "chr", "pos", "ref", "alt", "pval", and "beta" from all the files of that specific chromosome and bind them together
read_gwas <- function(pheno_name){
  basepath <- sprintf("gwas/%s", pheno_name)
  files <- list.files(basepath, pattern = chr_pattern, full.names = TRUE) # all the file directories of the gwas results per phenotype
  df <- lapply(files, fread, select = c("chr", "pos", "ref", "alt", "pval", "beta")) |> rbindlist() # bind row-wise all the gwas results for the phenotype, selecting for these specific columns
  df[, pheno := pheno_name] # the `pheno` column is filled with the pheno_name argument supplied to the function
  df[, n := 1:.N] # for each SNP position within a phenotype, assign a unique number from 1 - N (N being the number of SNP positions) 
  return(df)
}

# read in all the gwas results from flexlmm corresponding to chr_name
phenotypes <- list.files("gwas")
# phenotypes <- phenotypes[phenotypes != "mean"]
chr_pattern <- sprintf("chr_%s.*.tsv.gwas.gz", chr_name)
df <- lapply(phenotypes, read_gwas) |> rbindlist()

# function to read in the permutations result for a specific chromosome (https://github.com/birneylab/flexlmm/blob/main/docs/output.md#permutations)
# select the smallest of the "min_p"s from the 10 permutations, and multiply by the stringency factor
read_p_thr <- function(pheno_name){
  df_p_min <- sprintf("permutations/%s.min_p_dist.rds", pheno_name) |> readRDS()
  p_thr <- min(df_p_min[["min_p"]]) * stringency_factor
  data.table(pheno = pheno_name, p_thr = p_thr)
}

# generate a table that has all the minimum p-values*stringency_factor from all the permutations per phenotype
stringency_factor <- 0.1
df_thr <- lapply(phenotypes, read_p_thr) |> rbindlist()

### Not sure what all these are for...
# (from pgenlibr vignette) NewPvar loads variant IDs and allele codes from a .pvar or .bim file
# Saul also said that the genotypes are all the same here, so the phenotype chosen doesn't matter -> not entirely sure I understand this yet
pvar <- pgenlibr::NewPvar(sprintf("genotypes/chr_%s_abdominal.pvar.zst", chr_name))
# (from pgenlibr vignette) NewPgen opens a .pgen or PLINK 1 .bed file
pgen <- pgenlibr::NewPgen(sprintf("genotypes/chr_%s_abdominal.pgen", chr_name))

psam <- fread(sprintf("genotypes/chr_%s_abdominal.psam", chr_name))
sample_ids <- psam[["#IID"]]

# read in the pheno, covar tables that were used as input to flexlmm
pheno <- fread("/hps/nobackup/birney/users/esther/verta_phenotypes_plink2_format_06022024.txt")[match(sample_ids, `#IID`)]
covar <- fread("/hps/nobackup/birney/users/esther/covar_20032024.tsv")[match(sample_ids, `#IID`)]

# function for getting linkage disequilibrium -> specifically, the LD r^2
get_ld <- function(i, df){
  # Buf returns a numeric buffer that Read() or ReadHardcalls() can load to
  gt <- pgenlibr::Buf(pgen)
  gt2 <- pgenlibr::Buf(pgen) # these have length 2128, which is the same as the number of (unique, filtered) samples

  # ReadHardcalls(pgen, buf, variant_num) loads the variant_num-th variant, then fills buf with 0, 1, 2, NA values indicating the number of copies of the first ALT allele each sample has
  # Therefore, for each sample, indicate whether it has 0, 1, 2, or NA of the variant allele
  # This line loads the top snp (indicated by i) to gt
  pgenlibr::ReadHardcalls(pgen, gt, i) # variant_num i is the number in column `n` of df

  inner_routine <- function(i2){
    pgenlibr::ReadHardcalls(pgen, gt2, i2) # gt2 is a temporary vector that stores the variant information of the SNP that is being compared against the top SNP
    suppressWarnings(cor(gt, gt2)^2) # calculate correlation between gt (top SNP) and gt2 (every other SNP) -> correlation is based on the variants of each sample, NOT pvalue, etc
  }

  # the phenotype selected here doesn't matter... for some reason?
  # doesn't matter because it's all merged back to df at the end of this function
  # thus, the ld r^2 is the same for each phenotype (so the colour of the points is the same for each function)
  # the shape of the flame plots is given by the pvalues (which are different)
  # return(inner_routine(n))

  df_ld <- df[, .(chr, pos, ref, alt, ld_r2 = sapply(n, inner_routine))] # create new column, ld_r2, for the correlations against the top snp

  df_ld[
    ,ld_classes := cut(
      ld_r2,
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1),
      labels = c(
        "0 to 0.2",
        "0.2 to 0.4",
        "0.4 to 0.6",
        "0.6 to 0.8",
        "0.8 to 1"
      )
    )
  ]
  # merge df_ld back with other columns of df on c("chr", "pos", "ref", "alt")
  # this is ok since the presence/absence of variants is by sample, and has nothing to do with the phenotype
  df_ld <- df_ld[df, on = c("chr", "pos", "ref", "alt")]
  return(df_ld)
}

# set colour scale for the flame plot
ld_color_scale <- scale_color_manual(
  values = c(
    "0.8 to 1"="red",
    "0.6 to 0.8"="orange",
    "0.4 to 0.6"="green",
    "0.2 to 0.4"="cornflowerblue",
    "0 to 0.2"="darkblue"
  )
)

get_ld_pheno <- function(new_pheno) {
  top_snp_pheno <- df[pheno == new_pheno][order(pval)[1]]
  df_ld_pheno <- get_ld(top_snp_pheno[,n], df[pheno == new_pheno])
  return(df_ld_pheno)
}