# Calculate variance explained for the elements of the latent vector
library(ggfortify)

setwd("/nfs/research/birney/users/esther/medaka-img/features/pythae_ResNet_AE_CIFAR_flipped_resized_input/")

cam_colours <- readRDS("~/resources/cam_colours.rds")

pheno <- read.delim("/nfs/research/birney/users/esther/medaka-img/verta_df.txt",
                    header = TRUE,
                    sep = "\t",
                    check.names = FALSE)
pheno$name <- sub("\\.tif$", "", pheno$name)

samplenames <- read.csv('obs.csv', header=FALSE)[,1]
data <- as.data.frame(read.csv('X.csv', header=FALSE))
rownames(data) <- samplenames

# outliers <- c("PLATE 25 F2 VC_ Male 11-2 F15 x Female 72-1 F14_C10 bright field", "PLATE 13 F2 VC_ Female 95-1 F14 x Male 72-1 F14_C5")
# data <- data[!(row.names(data) %in% outliers), , drop = FALSE]


### Try PCA before ANOVA
data.pca <- prcomp(data, scale = FALSE) # TODO: decide what to set scale as?
# scale: a logical value indicating whether the variables should be scaled to have unit variance before the analysis takes place

# Plot PCA
ggplot(data.pca, aes(x = PC1, y = PC2)) + geom_point()

# Plot scree plot
ggplot(as.table(summary(data.pca)[["importance"]]["Proportion of Variance", c(1:20)]), aes(x = Var1, y = Freq)) + 
  geom_bar(stat='identity') +
  scale_y_continuous(limits = c(0, NA)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Principal component") +
  ylab("Proportion of Variance") 

master <- merge(pheno, data.pca$x, by.x = 'name', by.y = "row.names")

ggplot(master, aes(x = PC1, y = PC2)) + geom_point(aes(colour = abdominal))

linear_model_abdominal_str <- as.formula(paste0('abdominal ~ ', paste(colnames(master)[4:67], collapse = ' + ')))
linear_model_abdominal <- lm(linear_model_abdominal_str, data = master)

linear_model_caudal_str <- as.formula(paste0('caudal ~ ', paste(colnames(master)[4:67], collapse = ' + ')))
linear_model_caudal <- lm(linear_model_caudal_str, data = master)

saveRDS(linear_model_abdominal, "/nfs/research/birney/users/esther/medaka-img/features/linear_model_abdominal.rds")
saveRDS(linear_model_caudal, "/nfs/research/birney/users/esther/medaka-img/features/linear_model_caudal.rds")

aov_model_abdominal <- aov(linear_model_abdominal_str, data = master)
aov_model_caudal <- aov(linear_model_caudal_str, data = master)

saveRDS(aov_model_abdominal, "/nfs/research/birney/users/esther/medaka-img/features/aov_model_abdominal.rds")
saveRDS(aov_model_caudal, "/nfs/research/birney/users/esther/medaka-img/features/aov_model_caudal.rds")

data_withrows <- cbind('#IID' = rownames(data), data)
write.table(data_withrows, "/hps/nobackup/birney/users/esther/64feature_phenotypes_07052024.txt", quote = FALSE, sep = '\t', row.names = TRUE)


###########################################################################################################

# Using the new final_pheno_withrows, which only contains a subset of the samples from the above^ master
# final_pheno_withrows is from transform-dataframes.R

master <- merge(merge(pheno, joined[, c('#IID', 'name')],
                      by.x = 'name',
                      by.y = 'name'),
                final_pheno_withrows,
                by.x = '#IID',
                by.y = '#IID')

ggplot(master, aes(x = PC1, y = PC2)) + 
  geom_point(aes(colour = as.factor(abdominal)))
  # scale_fill_manual(values = cam_colours[c(1,2,3,4,5)])
# cut(abdominal, c(10,11, 12, 13, 14, 15))

linear_model_abdominal_str <- as.formula(paste0('abdominal ~ ', paste(colnames(master)[4:67], collapse = ' + ')))
linear_model_abdominal <- lm(linear_model_abdominal_str, data = master)

linear_model_caudal_str <- as.formula(paste0('caudal ~ ', paste(colnames(master)[4:67], collapse = ' + ')))
linear_model_caudal <- lm(linear_model_caudal_str, data = master)

aov_model_abdominal <- aov(linear_model_abdominal_str, data = master)
aov_model_caudal <- aov(linear_model_caudal_str, data = master)

saveRDS(linear_model_abdominal, "/nfs/research/birney/users/esther/medaka-img/models/2024-06-02/linear_model_abdominal.rds")
saveRDS(linear_model_caudal, "/nfs/research/birney/users/esther/medaka-img/models/2024-06-02/linear_model_caudal.rds")
saveRDS(aov_model_abdominal, "/nfs/research/birney/users/esther/medaka-img/models/2024-06-02/aov_model_abdominal.rds")
saveRDS(aov_model_caudal, "/nfs/research/birney/users/esther/medaka-img/models/2024-06-02/aov_model_caudal.rds")
