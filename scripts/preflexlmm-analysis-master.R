# Script to analyze the huggingface_mambavision-t-1k feature space (run on 1112 medaka images)
options(expressions=10000)

library(ggfortify)

setwd("/nfs/research/birney/users/esther/medaka-img/src_files")

pheno <- read.delim("/nfs/research/birney/users/esther/medaka-img/src_files/train_set_2024-10-03.csv",
                    header = TRUE,
                    sep = ",",
                    check.names = FALSE)
# data <- read.csv("/nfs/research/birney/users/esther/medaka-img/src_files/feature_matrix_train_set_2024-10-03_MambaVision-T-1K.csv",
#                  header = TRUE,
#                  sep = ",",
#                  row.names = 1)
data <- read.csv("/nfs/research/birney/users/esther/medaka-img/features/convnet-ae-pytorch-medaka/amber-sweep-1-epoch500-feature_matrix.csv",
                 header = TRUE,
                 sep = ",",
                 row.names = 1)

master <- merge(pheno[, c("abdominal", "caudal", "img_name")], data, by.x = 'img_name', by.y = "row.names")

# Perform a Spearman correlation test between the 640 latent variables (Bonazzola et al., 2024)
# Empty matrix to store results
cor_results <- matrix(NA, ncol(data), ncol(data))
p_values <- matrix(NA, ncol(data), ncol(data))

for(i in 1:ncol(data)) {
  for(j in i:ncol(data)) {
    test <- cor.test(data[,i], data[,j])
    cor_results[i,j] <- test$estimate
    p_values[i,j] <- test$p.value
  }
}

logical_mat <- cor_results > 0.95
indices <- which(logical_mat, arr.ind = TRUE) # RESULT: all cor_results > 0.95 are along the diagonal
all(indices[,1] == indices[,2])

# PCA
data.pca <- prcomp(data, scale = FALSE)
ggplot(data.pca, aes(x = PC1, y = PC2)) + geom_point()
ggplot(as.table(summary(data.pca)[["importance"]]["Proportion of Variance", c(1:20)]), aes(x = Var1, y = Freq)) + 
  geom_bar(stat='identity') +
  scale_y_continuous(limits = c(0, NA)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Principal component") +
  ylab("Proportion of Variance") 

# Linear/aov model
linear_model_abdominal_str <- as.formula(paste0('abdominal ~ ', paste(colnames(master)[4:length(colnames(master))], collapse = ' + ')))
linear_model_abdominal <- lm(linear_model_abdominal_str, data = master)
linear_model_abdominal_summary <- as.data.frame(summary(linear_model_abdominal)['coefficients'])
linear_model_abdominal_summary[linear_model_abdominal_summary[,"coefficients.Pr...t.."] < 0.001, ]

linear_model_caudal_str <- as.formula(paste0('caudal ~ ', paste(colnames(master)[4:length(colnames(master))], collapse = ' + ')))
linear_model_caudal <- lm(linear_model_caudal_str, data = master)
linear_model_caudal_summary <- as.data.frame(summary(linear_model_caudal)['coefficients'])
linear_model_caudal_summary[linear_model_caudal_summary[,"coefficients.Pr...t.."] < 0.001, ]

aov_model_abdominal <- aov(linear_model_abdominal_str, data = master)
aov_model_abdominal_summary <- as.data.frame(summary(aov_model_abdominal)[[1]])
aov_model_abdominal_summary[aov_model_abdominal_summary[,"Pr(>F)"] < 0.001, ]

aov_model_caudal <- aov(linear_model_caudal_str, data = master)
aov_model_caudal_summary <- as.data.frame(summary(aov_model_caudal)[[1]])
aov_model_caudal_summary[aov_model_caudal_summary[,"Pr(>F)"] < 0.001, ]

master <- merge(pheno[, c("sample_name", "abdominal", "caudal", "img_name")], data, by.x = 'img_name', by.y = "row.names")
rownames(master) <- master$sample_name
master <- master[,!names(master) %in% c("sample_name")]

write.csv(master,
          "/nfs/research/birney/users/esther/medaka-img/features/convnet-ae-pytorch-medaka/data_matrix_amber-sweep-1-epoch500-feature_matrix.csv",
          quote = FALSE,
          row.names = TRUE)

# Run inverse normalization, then linear/aov model
`invnorm` = function(x) {
  res = rank(x); res = qnorm(res/(length(res)+0.5))
  return(res)
}
data_norm <- apply(data, 2, invnorm)

rownames(master_norm)
master_norm <- merge(pheno[, c("sample_name", "abdominal", "caudal", "img_name")], data_norm, by.x = 'img_name', by.y = "row.names")
rownames(master_norm) <- master_norm$sample_name
master_norm$sample_name <- NULL

linear_model_abdominal_str <- as.formula(paste0('abdominal ~ ', paste(colnames(master_norm)[4:length(colnames(master_norm))], collapse = ' + ')))
linear_model_abdominal_norm <- lm(linear_model_abdominal_str, data = master_norm)
linear_model_abdominal_summary_norm <- as.data.frame(summary(linear_model_abdominal_norm)['coefficients'])
linear_model_abdominal_summary_norm[linear_model_abdominal_summary_norm[,"coefficients.Pr...t.."] < 0.001, ]

linear_model_caudal_str <- as.formula(paste0('caudal ~ ', paste(colnames(master_norm)[4:length(colnames(master_norm))], collapse = ' + ')))
linear_model_caudal_norm <- lm(linear_model_caudal_str, data = master_norm)
linear_model_caudal_summary_norm <- as.data.frame(summary(linear_model_caudal_norm)['coefficients'])
linear_model_caudal_summary_norm[linear_model_caudal_summary_norm[,"coefficients.Pr...t.."] < 0.001, ]

aov_model_abdominal_norm <- aov(linear_model_abdominal_str, data = master_norm)
aov_model_abdominal_summary_norm <- as.data.frame(summary(aov_model_abdominal_norm)[[1]])
aov_model_abdominal_summary_norm[aov_model_abdominal_summary_norm[,"Pr(>F)"] < 0.001, ]

aov_model_caudal_norm <- aov(linear_model_caudal_str, data = master_norm)
aov_model_caudal_summary_norm <- as.data.frame(summary(aov_model_caudal_norm)[[1]])
aov_model_caudal_summary_norm[aov_model_caudal_summary_norm[,"Pr(>F)"] < 0.001, ]

# write.csv(master_norm,
#           "/nfs/research/birney/users/esther/medaka-img/src_files/data_norm_matrix_train_set_2024-10-03_MambaVision-T-1K.csv",
#           quote = FALSE,
#           row.names = TRUE)
write.csv(master_norm,
          "/nfs/research/birney/users/esther/medaka-img/src_files/data_norm_matrix_amber-sweep-1-epoch500-feature_matrix.csv",
          quote = FALSE,
          row.names = TRUE)

# Run PCA (on inverse normalized data), then linear/aov model
data_norm_pca <- prcomp(data_norm, scale = FALSE) # qnorm already sets sd = 1?

master_norm_pca <- merge(pheno[, c("sample_name", "abdominal", "caudal", "img_name")], data_norm_pca$x, by.x = 'img_name', by.y = "row.names")
rownames(master_norm_pca) <- master_norm_pca$sample_name
master_norm_pca$sample_name <- NULL

linear_model_abdominal_str <- as.formula(paste0('abdominal ~ ', paste(colnames(master_norm_pca)[4:length(colnames(master_norm_pca))], collapse = ' + ')))
linear_model_abdominal_norm_pca <- lm(linear_model_abdominal_str, data = master_norm_pca)
linear_model_abdominal_summary_norm_pca <- as.data.frame(summary(linear_model_abdominal_norm_pca)['coefficients'])
linear_model_abdominal_summary_norm_pca[linear_model_abdominal_summary_norm_pca[,"coefficients.Pr...t.."] < 0.001, ]

linear_model_caudal_str <- as.formula(paste0('caudal ~ ', paste(colnames(master_norm_pca)[4:length(colnames(master_norm_pca))], collapse = ' + ')))
linear_model_caudal_norm_pca <- lm(linear_model_caudal_str, data = master_norm_pca)
linear_model_caudal_summary_norm_pca <- as.data.frame(summary(linear_model_caudal_norm_pca)['coefficients'])
linear_model_caudal_summary_norm_pca[linear_model_caudal_summary_norm_pca[,"coefficients.Pr...t.."] < 0.001, ]

aov_model_abdominal_norm_pca <- aov(linear_model_abdominal_str, data = master_norm_pca)
aov_model_abdominal_summary_norm_pca <- as.data.frame(summary(aov_model_abdominal_norm_pca)[[1]])
aov_model_abdominal_summary_norm_pca[aov_model_abdominal_summary_norm_pca[,"Pr(>F)"] < 0.001, ]

aov_model_caudal_norm_pca <- aov(linear_model_caudal_str, data = master_norm_pca)
aov_model_caudal_summary_norm_pca <- as.data.frame(summary(aov_model_caudal_norm_pca)[[1]])
aov_model_caudal_summary_norm_pca[aov_model_caudal_summary_norm_pca[,"Pr(>F)"] < 0.001, ]

write.csv(master_norm_pca,
          "/nfs/research/birney/users/esther/medaka-img/src_files/pca_matrix_amber-sweep-1-epoch500-feature_matrix.csv",
          quote = FALSE,
          row.names = TRUE)


### Read all generated dataframes back in and output in correct format for flexlmm
master <- read.csv("/nfs/research/birney/users/esther/medaka-img/features/convnet-ae-pytorch-medaka/data_matrix_amber-sweep-1-epoch500-feature_matrix.csv",
                   header = TRUE)
names(master)[1] <- "#IID"
master <- master[,!names(master) %in% c("img_name")]
write.table(master,
            file = "/nfs/research/birney/users/esther/medaka-img/scripts/flexlmm_params/convnet-ae-pytorch-medaka/data/data_matrix_amber-sweep-1-epoch500-feature_matrix.txt",
            quote = FALSE,
            sep = "\t",
            row.names = FALSE,
            col.names = TRUE)

master_norm <- read.csv("/nfs/research/birney/users/esther/medaka-img/features/convnet-ae-pytorch-medaka/data_norm_matrix_amber-sweep-1-epoch500-feature_matrix.csv",
                        header = TRUE)
names(master_norm)[1] <- "#IID"
master_norm <- master_norm[,!names(master_norm) %in% c("img_name")]
write.table(master_norm,
            file = "/nfs/research/birney/users/esther/medaka-img/scripts/flexlmm_params/convnet-ae-pytorch-medaka/data/data_norm_matrix_amber-sweep-1-epoch500-feature_matrix.txt",
            quote = FALSE,
            sep = "\t",
            row.names = FALSE,
            col.names = TRUE)

master_norm_pca <- read.csv("/nfs/research/birney/users/esther/medaka-img/features/convnet-ae-pytorch-medaka/pca_matrix_amber-sweep-1-epoch500-feature_matrix.csv",
                            header = TRUE)
names(master_norm_pca)[1] <- "#IID"
master_norm_pca <- master_norm_pca[,!names(master_norm_pca) %in% c("img_name")]
write.table(master_norm_pca,
            file = "/nfs/research/birney/users/esther/medaka-img/scripts/flexlmm_params/convnet-ae-pytorch-medaka/data/pca_matrix_amber-sweep-1-epoch500-feature_matrix.txt",
            quote = FALSE,
            sep = "\t",
            row.names = FALSE,
            col.names = TRUE)
