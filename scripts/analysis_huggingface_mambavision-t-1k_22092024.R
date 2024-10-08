# Script to analyze the huggingface_mambavision-t-1k_22092024 feature space (run on 2353 medaka images)
options(expressions=10000)

library(ggfortify)

setwd("/nfs/research/birney/users/esther/medaka-img/features/huggingface_mambavision-t-1k_22092024/")

pheno <- read.delim("/nfs/research/birney/users/esther/medaka-img/verta_df.txt",
                    header = TRUE,
                    sep = "\t",
                    check.names = FALSE)
pheno$name <- sub("\\.tif$", "", pheno$name)

samplenames <- read.csv('obs.csv', header=FALSE)[,1]
data <- as.data.frame(read.csv('X.csv', header=FALSE))
rownames(data) <- samplenames

master <- merge(pheno, data, by.x = 'name', by.y = "row.names")

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

logical_mat <- cor_results > 0.95 & cor_results < 1.000000000
indices <- which(logical_mat, arr.ind = TRUE) # RESULT: all cor_results > 0.95 are along the diagonal

# PCA
data.pca <- prcomp(data, scale = FALSE)
ggplot(data.pca, aes(x = PC1, y = PC2)) + geom_point()
ggplot(as.table(summary(data.pca)[["importance"]]["Proportion of Variance", c(1:5)]), aes(x = Var1, y = Freq)) + 
  geom_bar(stat='identity') +
  scale_y_continuous(limits = c(0, NA)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Principal component") +
  ylab("Proportion of Variance") 

# Linear/aov model
data <- merge(pheno, data, by.x = 'name', by.y = "row.names")

linear_model_abdominal_str <- as.formula(paste0('abdominal ~ ', paste(colnames(data)[4:length(colnames(data))], collapse = ' + ')))
linear_model_abdominal <- lm(linear_model_abdominal_str, data = data)
sum(summary(linear_model_abdominal)['coefficients'][["coefficients"]][,'Pr(>|t|)'] < 0.001)

linear_model_caudal_str <- as.formula(paste0('caudal ~ ', paste(colnames(data)[4:length(colnames(data))], collapse = ' + ')))
linear_model_caudal <- lm(linear_model_caudal_str, data = data)
sum(summary(linear_model_caudal)['coefficients'][["coefficients"]][,'Pr(>|t|)'] < 0.001)

aov_model_abdominal <- aov(linear_model_abdominal_str, data = data)
sum(summary(aov_model_abdominal)[[1]]['Pr(>F)'] < 0.001, na.rm = TRUE)

aov_model_caudal <- aov(linear_model_caudal_str, data = data)
sum(summary(aov_model_caudal)[[1]]['Pr(>F)'] < 0.001, na.rm = TRUE)


