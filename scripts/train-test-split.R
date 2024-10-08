# Script to split the medaka image data into training vs zero-shot test set
# Total number of samples: 2128 (2353 images, 2255 samples with valid phenotype data, 2128 samples with valid phenotype data & > 0.1 coverage)
#   We want at least n = 1000 in the zero-shot test
#   ~45% in the zero-shot test makes:
#     - 1058 in the zero-shot test
#     - 1303 in the training & cross-validation

library(dplyr)
library(ggplot2)
library(caret)

### Read in inputs
# # (Tom's original, but with columns renamed) covariates file -> total 2255 samples
# covariates <- read.delim("/hps/nobackup/birney/users/esther/covar_renamed_cols.tsv", header = TRUE, sep = "\t", row.names = "X.IID")
# 
# # Phenotypes file (low coverage samples removed, but the 2 outliers found during fine-mapping not removed -> total 2128 samples)
# # Has the same number of rows as Tom's: /hps/nobackup/birney/users/tomas/nf_lauchdir/flexlmm/verta_phenotypes_plink2_format_filtered_0.1_cov.txt
# phenotypes <- read.delim("/hps/nobackup/birney/users/esther/verta_phenotypes_plink2_format_filtered_0.1_cov_new.txt", header = TRUE, sep = "\t", row.names = "X.IID")
# 
# 
# # left join on phenotypes to get the correct total number of samples to be split into train/test
# all_samples <- merge(x = phenotypes, y = covariates, by.x = 0, by.y = 0, all.x = TRUE)

# Final file mapping image names to sample_names
all_samples <- read.csv("/nfs/research/birney/users/esther/medaka-img/src_files/all_image_sample_name_metadata_map.csv",
                        header = TRUE,
                        sep = ",")

n_samples = nrow(all_samples) # n = 2122 as of 03/10/2024
n_test = 1000 # Set number of testing samples to be 1000 to be sufficient for GWAS
n_train = n_samples - n_test # Set number of testing samples to be the rest of the dataset -> for n_samples = 2353 and n_train = 1058, n_test = 1295

cat(paste0("Total number of samples: ", n_samples, "\n Training set: ", n_train, "\n Test set: ", n_test))


### Split using caret library
set.seed(1)
index <- createDataPartition(all_samples$crosses,
                             times = 1,
                             p = n_test/n_samples, # p specifies the percentage to be delegated per class; may not fit exactly with n_train, n_test
                             list = FALSE)

test <- all_samples[index,]
train <- all_samples[-index,]

train$split <- "train"
test$split <- "test"

all_samples_split <- rbind(train, test)

### Visusalize the split
prop.table(table(all_samples_split$crosses))


# proportion of classes in each split
all_samples_counts <- all_samples_split %>%
  group_by(crosses, split) %>%
  summarise(n = n())

all_samples_counts$split <- factor(all_samples_counts$split, levels = c("train", "test")) # makes the training set appear before the test set on plots

# plot1 <- ggplot(all_samples_counts, aes(fill=split, x=cross_id, y=n)) + 
#   geom_bar(position="dodge", stat="identity") +
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
#   facet_wrap(~split)
# plot1

# Barplot of train-test split, facet wrapped by class
plot2 <- ggplot(all_samples_counts, aes(x = split, y = n, fill = split)) +
  geom_bar(stat = "identity") +
  xlab("") +
  geom_text(aes(label=n), vjust=2, size=3.5) +
  facet_wrap(~crosses) +
  labs(title = paste0("Train (", nrow(train), ") / Test (", nrow(test), ") split of medaka image+genotype data per cross"), 
                      y = "Number of samples", fill = "Data split")
plot2

# Save the train/test split and associated plot
outputdir <- "/nfs/research/birney/users/esther/medaka-img/src_files/"

savefile = paste0(outputdir, "train_test_split_", Sys.Date())
ggsave(file=paste0(savefile, ".png"), plot=plot2, width=unit(12, "inches"), height=unit(12, "inches"))
ggsave(file=paste0(savefile, ".svg"), plot=plot2, width=unit(12, "inches"), height=unit(12, "inches"))

write.table(train, paste0(outputdir, "train_set_", Sys.Date(), ".csv"),
            quote = FALSE,
            sep = ",",
            row.names = FALSE,
            col.names = TRUE)

write.table(test, paste0(outputdir, "test_set_", Sys.Date(), ".csv"),
            quote = FALSE,
            sep = ",",
            row.names = FALSE,
            col.names = TRUE)
