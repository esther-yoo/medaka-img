# Transform data frames for joining
# Use with variance-64-vector-vertebrae-count.R

library(stringr)

covar <- read.delim("/hps/nobackup/birney/users/esther/covar_20032024.tsv",
                    header = TRUE,
                    sep = '\t')

# covar_rownames <- covar[,c(1,2)]
# covar_rownames$id <- str_extract(covar_rownames$X.IID, "(?<=Loosli_lane1).*?(?=_AA)")
# covar_rownames$plateid <- as.numeric(str_extract(covar_rownames$id, "(?<=Plate)\\d+(?=F2)"))
# covar_rownames$endid <- str_extract(covar_rownames$id, "(?<=VC)[A-Za-z]\\d+")
# covar_rownames$cross_id_new <- str_replace_all(covar_rownames$cross_id, "_", " ")
# covar_rownames$cross_id_new <- str_replace_all(covar_rownames$cross_id_new, "(\\d+-\\d+) (Male|Female) x (\\d+-\\d+) (Male|Female)", "\\2 \\1 F14 x \\4 \\3 F14")
# covar_rownames$plateid_new <- as.character(sprintf("%02d", covar_rownames$plateid))
# covar_rownames$endid_new <- paste0(substr(covar_rownames$endid, 1, 1), "0", substr(covar_rownames$endid, 2, nchar(covar_rownames$endid)))
# 
# covar_rownames$final <- paste0("PLATE ", covar_rownames$plateid, " F2 VC_ ", covar_rownames$cross_id_new, "_", covar_rownames$endid)
# 
# join_mapping <- covar_rownames[,c(8,1)]

# transform vertebrae_cross_link data frame for joining
vertebrae_cross_link <- read.delim("/nfs/research/birney/users/esther/medaka-img/vertrabrate_cross_link.txt",
                                   header = TRUE,
                                   sep = "\t")
vertebrae_cross_link <- vertebrae_cross_link[!duplicated(vertebrae_cross_link),]
vertebrae_cross_link$`#IID` <- paste0(vertebrae_cross_link$seq_ids, "_", vertebrae_cross_link$seq_ids)
vertebrae_cross_link$plateid <- as.numeric(str_extract(vertebrae_cross_link$plates, "(?<=plate)\\d+"))
vertebrae_cross_link$well_char <- as.character(str_extract(vertebrae_cross_link$wells, "[A-Za-z]+"))
vertebrae_cross_link$well_num <- as.numeric(str_extract(vertebrae_cross_link$wells, "\\d+"))
vertebrae_cross_link$crossid_new <- str_replace_all(str_replace_all(str_replace_all(vertebrae_cross_link$crosses, "_", " "), "(\\d+-\\d+) (Male|Female) x (\\d+-\\d+) (Male|Female)", "\\2 \\1 x \\4 \\3"), " ", "")

# transform master data frame for joining
# master is from variance-64-vector-vertebrae-count.R
master <- master[!grepl("bright field", master$name), ]
master$plateid <- as.numeric(str_extract(master$name, "(?<=PLATE )\\d+(?= F2)"))
master$well_char <- as.character(str_extract(str_extract(master$name, "([^_]+)$"), "[A-Za-z]+"))
master$well_num <- as.numeric(str_extract(str_extract(master$name, "([^_]+)$"), "\\d+"))
master$crossid <- str_replace_all(str_replace_all(str_extract(master$name, "(?<=VC_).*(?=F14_)"), " ", ""), "F\\d+", "")

# join master and vertebrae_cross_link
joined <- merge(vertebrae_cross_link, master, 
                by.x = c("plateid", "well_char", "well_num", "crossid_new"),
                by.y = c("plateid", "well_char", "well_num", "crossid"))

final_pheno <- merge(joined, covar,
                by.x = '#IID',
                by.y = 'X.IID')
rownames(final_pheno) <- final_pheno$`#IID`
final_pheno <- final_pheno[, grepl("PC", names(joined))] # Replace what's in the quotes with "V" if not using PCs
final_pheno_withrows <- cbind('#IID' = rownames(final_pheno), final_pheno)

# Just the latent vectors
write.table(final_pheno_withrows,
            "/hps/nobackup/birney/users/esther/64feature_phenotypes_08052024.txt", 
            quote = FALSE, 
            sep = '\t', 
            row.names = TRUE)

# Principal components of the latent vectors
write.table(final_pheno_withrows,
            "/hps/nobackup/birney/users/esther/64feature_pca_phenotypes_03062024.txt", 
            quote = FALSE, 
            sep = '\t', 
            row.names = FALSE)

# joined_df <- joined[, '' || grepl("^V", names(joined))]
# 
# # master is from variance-64-vector-vertebrae-count.R
# joined <- merge(join_mapping, master, by.x = 'final', by.y = 'name', all.x = TRUE)
# 
# na_rows <- joined[is.na(joined$abdominal), ]
# 
# # covar_rownames$final <- ifelse(covar_rownames$final %in% na_rows$final, sub("VC_ ", "VC_", covar_rownames$final), covar_rownames$final) # then go back to line 13
# covar_rownames$final <- ifelse(covar_rownames$final %in% na_rows$final, paste0("PLATE ", covar_rownames$plateid, " F2 VC_ ", covar_rownames$cross_id_new, "_", covar_rownames$endid_new), covar_rownames$final)

