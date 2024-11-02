import torch.nn as nn
import torch

class AutoEncoderVGGNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoderVGGNet, self).__init__()
        # encoder
        # vggnet = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=False)
        # vggnet.features = make_encoder_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], batch_norm=False)
        vggnet = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

        self.encoder_features = nn.Sequential(*list(vggnet.features.children()))

        self.encoder_adaptive_pool = vggnet.avgpool

        self.encoder_linear = nn.Sequential(*list(vggnet.classifier.children())[:-1])

        self.embedding = nn.Linear(4096, latent_dim)

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512 * 7 * 7)
        )

        self.decoder_adaptive_unpool = nn.ConvTranspose2d(512 * 7 * 7, 512, kernel_size=7, stride=1, padding=0)

        self.decoder_features =  nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (36) MaxPool2d
            
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (27) MaxPool2d
            
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (18) MaxPool2d

            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (9) MaxPool2d
            
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder_features(x)
        x = self.encoder_adaptive_pool(x)
        x = self.encoder_linear(x.view(x.size(0), -1))
        x = self.embedding(x)
        x = self.decoder_linear(x)
        x = self.decoder_adaptive_unpool(x.view(x.size(0), 512 * 7 * 7, 1, 1))
        x = self.decoder_features(x)
        return x

class AutoEncoderResNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoderResNet, self).__init__()
        # encoder
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=4, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        )
        self.residual = nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32)
        )
        self.embedding = nn.Linear(128 * 14 * 14, latent_dim)

        # decoder
        self.delayer1 = nn.Linear(latent_dim, 128 * 14 * 14)
        self.delayer2  = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.deresidual = nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            ResBlock(in_channels=128, out_channels=32)
        )
        self.delayer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=5, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )
        self.delayer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1, output_padding=1)
        )
        self.delayer5 = nn.Sequential(
                nn.ConvTranspose2d(64, input_dim[0], 4, 2, padding=1),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.residual(x)
        x = self.embedding(x.reshape(x.shape[0], -1))
        x = self.delayer1(x)
        x = self.delayer2(x.reshape(x.shape[0], 128, 14, 14))
        x = self.deresidual(x)
        x = self.delayer3(x)
        x = self.delayer4(x)
        x = self.delayer5(x)
        return x

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


class AutoEncoderSigmoid(nn.Module):
    def __init__(self, input_dim, dim0, dim1, dim2, dim3, dim4):
        super(AutoEncoderSigmoid, self).__init__()
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)