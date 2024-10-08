library("tiff")
library(raster)

img <- readTIFF("/nfs/research/birney/users/esther/medaka-img/votj_aligned_images_align_any/PLATE 1 F2 VC_Female 95-1 F14 x Male 72-1 F14_A11.tif-aligned.tif")
grid::grid.raster(img[,,1])

# Get pixel values
View(img@.Data[1:dim(img), 1:dim(img), 1])

img_matrix = as.data.frame(img[,,1]) # transform image to a dataframe and remove columns that have all the same value (0 variance; cannot PCA)
img_matrix_cropped = img_matrix[, sapply(img_matrix, function(x) length(unique(x)) > 1)]

pca <- prcomp(as.data.frame(img_matrix_cropped, scale. = TRUE))
pca <- prcomp(as.data.frame(img[,,1]), scale. = FALSE)

fish.pc <- pca$x
fish$pc1 <- fish.pc[,1]
fish$pc2 <- fish.pc[,2]
fish$pc3 <- fish.pc[,3]

plot(pca$x[,1], col = cm.colors(15), axes = FALSE)
