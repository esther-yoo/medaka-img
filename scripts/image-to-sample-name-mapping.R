# Mapping image names to sample names
library(readxl)
library(dplyr)
library(stringr)
library(purrr)

####################################################################
# Example image name:
# PLATE 1 F2 VC_Female 95-1 F14 x Male 72-1 F14_A01.tif

# Example sample name:
# AAAVWMLHV_Pool1-4F2_VC_22s004612-1-1_Loosli_lane1Plate1F2VCA1
####################################################################

### Needed inputs
# VC_ARC_F2_15_04-2021.xlsx                            Contains the image names and their associated phenotypes (abdominal, caudal count), but not the sample names -> 8984 rows
# verta_phenotypes_plink2_format_filtered_0.1_cov.txt  Contains the sample names and their associated phenotypes (abdominal, caudal count + others) but not the image names
#                                                      This is the resultant file after the filtering out of low-coverage samples -> 2128 rows
# vertrabrate_cross_link.txt                           Contains the sample names and identifying columns that could be used to map them to their image names


### Read in inputs
# Get all the sheet names of the VC_ARC_F2_15_04-2021.xlsx file 
sheet_names <- grep(pattern = "Plate",
                    x = excel_sheets("/nfs/research/birney/users/esther/medaka-img/VC_ARC_F2_15_04-2021.xlsx"),
                    value = TRUE)

# Read in file with mapping between seq_id's (ie. sample names) and columns with image name identifiers
vertebrae_img_name_mapping <- read.delim("/nfs/research/birney/projects/indigene/vertebrae-f2/gwas/sample_links/vertrabrate_cross_link.txt", # 2258 rows
                                         header = TRUE,
                                         sep = "\t")
vertebrae_img_name_mapping <- distinct(vertebrae_img_name_mapping) # 2255 rows after dropping duplicates

# Phenotypes file (low coverage samples removed, but the 2 outliers found during fine-mapping not removed -> total 2128 samples)
# Has the same number of rows as Tom's: /hps/nobackup/birney/users/tomas/nf_lauchdir/flexlmm/verta_phenotypes_plink2_format_filtered_0.1_cov.txt
phenotypes <- read.delim("/hps/nobackup/birney/users/esther/verta_phenotypes_plink2_format_filtered_0.1_cov_new.txt", header = TRUE, sep = "\t", row.names = "X.IID")

### Read in all the sheets of VC_ARC_F2_15_04-2021.xlsx as-is (has the image name and associated vertebrae counts)
image_name_vertebrae_count <- lapply(sheet_names, function(sheet) {
  print(sheet)
  
  df <- data.frame(lapply(read_excel("/nfs/research/birney/users/esther/medaka-img/VC_ARC_F2_15_04-2021.xlsx", sheet = sheet), as.character))
  
  if (sheet == "Plate 9"){ # Sheet 9 is missing the headers on abdominal, caudal for some reason
    colnames(df) <- c("Name", "Abdominal", "Caudal", "Total", "Comments")
  }
  
  return(df)
  # colnames(df) <- tolower(colnames(df))
  # df <- df[, c("name", "abdominal", "caudal")]
}) %>% bind_rows()

# > sum(is.na(image_name_vertebrae_count$Abdominal))
# [1] 6822
# > sum(is.na(image_name_vertebrae_count$Caudal))
# [1] 6824


### Clean up the dataframe
# Coalesce the two abdominal and caudal columns (one all lowercase, one capitalized) to one column
image_name_vertebrae_count$Abdominal <- coalesce(image_name_vertebrae_count$Abdominal, image_name_vertebrae_count$abdominal)
image_name_vertebrae_count$Caudal <- coalesce(image_name_vertebrae_count$Caudal, image_name_vertebrae_count$caudal)

# > sum(is.na(image_name_vertebrae_count$Abdominal))
# [1] 6726 -> leaves 2258 rows with non-missing Abdominal counts
# > sum(is.na(image_name_vertebrae_count$Caudal))
# [1] 6728 -> leaves 2256 rows with non-missing Caudal counts

image_name_vertebrae_count_clean <- image_name_vertebrae_count[, c("Name", "Abdominal", "Caudal")] # 8984 rows
colnames(image_name_vertebrae_count_clean) <- c("img_name", "abdominal", "caudal")
image_name_vertebrae_count_clean <- distinct(image_name_vertebrae_count_clean) # 4614 rows after removing duplicates 
                                                                               # -> but there are still duplicates among image names, one with NA in the caudal and abdominal columns,
                                                                               # and the other with the actual caudal and abdominal count values

### Remove rows where either the abdominal or caudal rows are NA
image_name_vertebrae_count_clean <- image_name_vertebrae_count_clean[complete.cases(image_name_vertebrae_count_clean$abdominal, image_name_vertebrae_count_clean$caudal), ] # 2256 rows

# image_name_vertebrae_count_clean <- na.omit(image_name_vertebrae_count_clean) # 2160 rows after removing NA entries
# image_name_vertebrae_count_clean <- image_name_vertebrae_count_clean[!is.na(image_name_vertebrae_count_clean$img_name), ] # 2258 rows after removing entries where the img_name is NA

### Parse the image name to extract the plate, cross, and well
image_name_vertebrae_count_clean$plate_id <- as.numeric(str_extract(image_name_vertebrae_count_clean$img_name, "(?<=PLATE )\\d+(?= F2)"))
image_name_vertebrae_count_clean$plates <- paste0("plate", as.character(image_name_vertebrae_count_clean$plate_id))

# The well_id is anything after "_" and before the file extension (to account for non-tiff files? there's one png file (row 1799))
image_name_vertebrae_count_clean$well_id <- str_extract(image_name_vertebrae_count_clean$img_name, "(?<=_)\\w+(?=\\.[^.]+$)")
# Handle the image with "bright field" in its name separately
image_name_vertebrae_count_clean[2244,]$well_id <- str_extract(image_name_vertebrae_count_clean[2244,]$img_name, "(?<=_)\\w+(?= bright field\\.tif)")
# Get rid of the unncessary leading 0 in well id's, eg. "D08" to "D8"
image_name_vertebrae_count_clean$wells <- sub("0([0-9])", "\\1", image_name_vertebrae_count_clean$well_id)

### Merge the cleaned image names + phenotypes file (image_name_vertebrae_count_clean) with the sample names + image identifiers file (vertebrae_img_name_mapping)
# image_name_vertebrae_count_clean has 2256 rows
# vertebrae_img_name_mapping has 2255 rows
# -> image_name_vertebrae_count_clean has 1 more row because of the "bright field" image that has duplicated plate, well id as another name
# Remove "bright field" image from the dataframe manually
image_name_vertebrae_count_clean <- image_name_vertebrae_count_clean[image_name_vertebrae_count_clean$img_name != "PLATE 25 F2 VC_ Male 11-2 F15 x Female 72-1 F14_C10 bright field.tif", ]
# Now, checking for duplication among the plate, well ids gives FALSE
# > any(duplicated(image_name_vertebrae_count_clean[c("plates", "wells")]))

# Should give a resulting dataframe with 2255 rows
image_name_sample_name_phenotypes <- merge(image_name_vertebrae_count_clean, vertebrae_img_name_mapping, by = c("plates", "wells"), all.x = TRUE)
image_name_sample_name_phenotypes$sample_name <- paste0(image_name_sample_name_phenotypes$seq_ids, "_", image_name_sample_name_phenotypes$seq_ids)

### Merge with Tom's phenotypes file that has low-coverage samples filtered out -> the final dataframe should have 2128 samples
# nrow(phenotypes) -> 2128
# nrow(image_name_sample_name_phenotypes) -> 2255
image_name_sample_name_phenotypes <- image_name_sample_name_phenotypes[, c("sample_name", "img_name", "crosses", "abdominal", "caudal", "plates", "wells")]
phenotypes <- phenotypes[, c("abdominal", "caudal")]
phenotypes$sample_name <- rownames(phenotypes)
phenotypes <- phenotypes[, c("sample_name", "abdominal", "caudal")]

final_mapping <- merge(phenotypes, image_name_sample_name_phenotypes, by = c("sample_name", "abdominal", "caudal"))

# Manually remove/fix file records with weird metadata/file names
final_mapping <- final_mapping[!grepl("PLATE 1 F2 VC_Female 14-2 F14 x Male 129-1 F14_H8.tif", final_mapping$img_name),]
final_mapping <- final_mapping[!grepl("PLATE 1 F2 VC_Female 14-2 F14 x Male 129-1 F14_H9.tif", final_mapping$img_name),]
final_mapping <- final_mapping[!grepl("PLATE 1 F2 VC_Female 14-2 F14 x Male 129-1 F14_H10.tif", final_mapping$img_name),]
final_mapping <- final_mapping[!grepl("PLATE 1 F2 VC_Female 14-2 F14 x Male 129-1 F14_H11.tif", final_mapping$img_name),]
final_mapping <- final_mapping[!grepl("PLATE 1 F2 VC_Female 14-2 F14 x Male 129-1 F14_H12.tif", final_mapping$img_name),]
final_mapping <- final_mapping[!grepl("PLATE 1 F2 VC_Female 95-1 F14 x Male 129-1 F14_G12.tif", final_mapping$img_name),]
final_mapping <- final_mapping[!grepl("PLATE 20 F2 VC_ Male 95-1 F14 x Female 33-1 F14_F1.tif", final_mapping$img_name),]
final_mapping$img_name[final_mapping$img_name == "PLATE 22 F2 VC_ Male 14-2 F14 x Female 10-1 F14_A4.tif"] <-  "PLATE 22F2 VC_ Male 14-2 F14 x Female 10-1 F14_A4.tif"

# nrow(final_mapping)
# -> 2121

write.table(final_mapping, "/nfs/research/birney/users/esther/medaka-img/src_files/all_image_sample_name_metadata_map.csv",
          quote = FALSE,
          sep = ",",
          row.names = FALSE,
          col.names = TRUE)


