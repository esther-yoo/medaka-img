### transformed_images
# import os
# import torch
# from natsort import natsorted
# from pl_bolts.models.autoencoders import AE

# pretrained = 'cifar10-resnet18'
# model = AE(input_height=256, enc_type='resnet18', latent_dim=5).from_pretrained(pretrained)
# encoder = model.encoder
# encoder.eval

# feature_matrix = []
# counter = 0

# for img_name in natsorted(os.listdir(path = '../transformed_images/tensor/')):
#     counter += 1
#     img = torch.load('../transformed_images/tensor/' + img_name)
#     features = encoder(img.unsqueeze(0))

#     feature_matrix.append(features.detach())
#     if counter % 10 == 0:
#         print(counter)


# torch.save(feature_matrix, ('../features/' + pretrained + '_' + 'feature_matrix.pt'))


### flipped_transformed_images
import os
import torch
from natsort import natsorted
from pl_bolts.models.autoencoders import AE

pretrained = 'cifar10-resnet18'
model = AE(input_height=256, enc_type='resnet18', latent_dim=5).from_pretrained(pretrained)
encoder = model.encoder
encoder.eval

feature_matrix = []
counter = 0

for img_name in natsorted(os.listdir(path = '../flipped_transformed_images/tensor/')):
    counter += 1
    img = torch.load('../flipped_transformed_images/tensor/' + img_name)
    features = encoder(img.unsqueeze(0))

    feature_matrix.append(features.detach())
    if counter % 10 == 0:
        print(counter)


torch.save(feature_matrix, ('../features/' + pretrained + '_' + 'flipped_feature_matrix.pt'))