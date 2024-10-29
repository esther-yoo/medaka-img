import torch.nn as nn
import torch

class AutoEncoderConv(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoderConv, self).__init__()
        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim[0], 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.embedding = nn.Linear(1024 * 14 * 14, latent_dim)

        # decoder
        self.deconv1 = nn.Sequential(nn.Linear(latent_dim, 1024 * 28 * 28))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv5 = nn.Sequential(
                nn.ConvTranspose2d(128, input_dim[0], 5, 1, padding=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.embedding(x.reshape(x.shape[0], -1))
        x = self.deconv1(x)
        x = x.view(-1, 1024, 28, 28)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dim0, dim1, dim2, dim3, dim4):
        super(AutoEncoder, self).__init__()
        # encoder
        self.fc1 = nn.Linear(input_dim, dim0)
        self.fc2 = nn.Linear(dim0, dim1)
        self.fc3 = nn.Linear(dim1, dim2)
        self.fc4 = nn.Linear(dim2, dim3)
        self.fc5 = nn.Linear(dim3, dim4)
        # decoder
        self.fc6 = nn.Linear(dim4, dim3)
        self.fc7 = nn.Linear(dim3, dim2)
        self.fc8 = nn.Linear(dim2, dim1)
        self.fc9 = nn.Linear(dim1, dim0)
        self.fc10 = nn.Linear(dim0, input_dim)

    def encoder(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

    def decoder(self, x):
        x = torch.sigmoid(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        x = torch.sigmoid(self.fc8(x))
        x = torch.sigmoid(self.fc9(x))
        x = torch.sigmoid(self.fc10(x))
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoderRelu(nn.Module):
    def __init__(self, input_dim, dim0, dim1, dim2, dim3, dim4):
        super(AutoEncoderRelu, self).__init__()
        # encoder
        self.fc1 = nn.Linear(input_dim, dim0)
        self.fc2 = nn.Linear(dim0, dim1)
        self.fc3 = nn.Linear(dim1, dim2)
        self.fc4 = nn.Linear(dim2, dim3)
        self.fc5 = nn.Linear(dim3, dim4)
        # decoder
        self.fc6 = nn.Linear(dim4, dim3)
        self.fc7 = nn.Linear(dim3, dim2)
        self.fc8 = nn.Linear(dim2, dim1)
        self.fc9 = nn.Linear(dim1, dim0)
        self.fc10 = nn.Linear(dim0, input_dim)

    def encoder(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x

    def decoder(self, x):
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x