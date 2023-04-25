import torch
from torch import nn


class Img2PcdModel(nn.Module):
    """
    A neural network of single image to 3D.
    """

    def __init__(self, device):
        super(Img2PcdModel, self).__init__()
        # TODO: Design your network layers.
        # Example:
        #     self.linear = nn.Linear(3 * 256 * 256, 1024 * 3)
        #     self.act = nn.Sigmoid()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=2, padding=2)
        
        self.fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048 * 2, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=256*3)
        self.fc5 = nn.Linear(in_features=32*32*3, out_features=768*3)
        self.fc6 = nn.Linear(in_features=256*3, out_features=1024*3)
    
        
        # Decoder
        self.decov1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.decov2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.decov3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.decov4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
        self.decov5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)

        self.dcov1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dcov2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dcov3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dcov4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dcov5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dcov6 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(3)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.device = device
        self.to(device)

    def forward(self, x):  # shape = (B, 3, 256, 256)
        # TODO: Design your network computation process.
        # Example:
        #     batch_size = x.shape[0]
        #     x = self.linear(x)
        #     x = self.act(x)
        #     x = x.reshape(batch_size, 1024, 3)
        #     return x
        batch_size = x.shape[0]
        # encoder
        x = self.conv1(x)
        x = self.bn16(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn16(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn16(x)
        x = self.relu(x)
        x0 = x
        x = self.conv3(x)
        x = self.bn32(x)
        x = self.relu(x)

        # 32 
        x = self.conv4(x)
        x = self.bn32(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn32(x)
        x = self.relu(x)
        x1 = x
        x = self.conv5(x)
        x = self.bn64(x)
        x = self.relu(x)
        
        # 64
        x = self.conv6(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn64(x)
        x = self.relu(x)
        x2 = x
        x = self.conv7(x)
        x = self.bn128(x)
        x = self.relu(x)
        
        # 128
        x = self.conv8(x)
        x = self.bn128(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn128(x)
        x = self.relu(x)
        x3 = x
        x = self.conv9(x)
        x = self.bn256(x)
        x = self.relu(x)

        # 256
        x = self.conv10(x)
        x = self.bn256(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn256(x)
        x = self.relu(x)
        x4 = x
        x = self.conv11(x)
        x = self.bn512(x)
        x = self.relu(x)

        # 512
        x = self.conv12(x)
        x = self.bn512(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.bn512(x)
        x = self.relu(x)
        x5 = x
        x = self.conv13(x)
        x = self.bn512(x)

        # 256
        x_t = x.view(batch_size, -1)
        x_addition = self.relu(self.fc1(x_t))
        x5 = self.dcov1(x5)
        x5 = self.bn256(x5)
        x = self.decov1(x)
        x = self.bn256(x)
        x = self.relu(torch.add(x, x5))
        x = self.conv10(x)
        x = self.bn256(x)
        x = self.relu(x)
        x5 = x
        x = self.decov2(x)
        x = self.bn128(x)

        # 128
        x4 = self.dcov2(x4)
        x4 = self.bn128(x4)
        x = self.relu(torch.add(x, x4))
        x = self.conv8(x)
        x = self.bn128(x)
        x = self.relu(x)
        x4 = x
        x = self.decov3(x)
        x = self.bn64(x)

        # 64
        x3 = self.dcov3(x3)
        x3 = self.bn64(x3)
        x = self.relu(torch.add(x, x3))
        x = self.conv6(x)
        x = self.bn64(x)
        x = self.relu(x)
        x3 = x
        x = self.decov4(x)
        x = self.bn32(x)

        # 32
        x2 = self.dcov4(x2)
        x2 = self.bn32(x2)
        x = self.relu(torch.add(x, x2))
        x = self.conv4(x)
        x = self.bn32(x)
        x = self.relu(x)
        x2 = x
        x = self.decov5(x)
        x = self.bn16(x)

        # 16
        x1 = self.dcov5(x1)
        x1 = self.bn16(x1)
        x = self.relu(torch.add(x, x1))
        x = self.conv2(x)
        x = self.bn16(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn32(x)

        # 32
        x2 = self.conv4(x2)
        x2 = self.bn32(x2)
        x = self.relu(torch.add(x, x2))
        x = self.conv4(x)
        x = self.bn32(x)
        x = self.relu(x)
        x2 = x
        x = self.conv5(x)
        x = self.bn64(x)

        # 64
        x3 = self.conv6(x3)
        x3 = self.bn64(x3)
        x = self.relu(torch.add(x, x3))
        x = self.conv6(x)
        x = self.bn64(x)
        x = self.relu(x)
        x3 = x
        x = self.conv7(x)
        x = self.bn128(x)

        # 128
        x4 = self.conv8(x4)
        x4 = self.bn128(x4)
        x = self.relu(torch.add(x, x4))
        x = self.conv8(x)
        x = self.bn128(x)
        x = self.relu(x)
        x4 = x
        x = self.conv9(x)
        x = self.bn256(x)

        # 256
        x5 = self.conv10(x5)
        x5 = self.bn256(x5)
        x = self.relu(torch.add(x, x5))
        x = self.conv10(x)
        x = self.bn256(x)
        x = self.relu(x)
        x5 = x
        x = self.conv11(x)
        x = self.bn512(x)
        x = self.relu(x)

        x_addition = self.fc2(x_addition)
        x_t = x.view(batch_size, -1)
        x_addition = self.relu(torch.cat([x_addition, self.fc1(x_t)], dim=1))

        x = self.decov1(x)
        x = self.bn256(x)

        # 256
        x5 = self.conv10(x5)
        x5 = self.bn256(x5)
        x = self.relu(torch.add(x, x5))
        x = self.conv10(x)
        x = self.bn256(x)
        x = self.relu(x)
        x5 = x
        x = self.decov2(x)
        x = self.bn128(x)

        # 128
        x4 = self.conv8(x4)
        x4 = self.bn128(x4)
        x = self.relu(torch.add(x, x4))
        x = self.conv8(x)
        x = self.bn128(x)
        x = self.relu(x)
        x4 = x
        x = self.decov3(x)
        x = self.bn64(x)

        # 64
        x3 = self.conv6(x3)
        x3= self.bn64(x3)
        x = self.relu(torch.add(x, x3))
        x = self.conv6(x)
        x = self.bn64(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn64(x)
        x = self.relu(x)

        x_addition = self.fc3(x_addition)
        x_addition = self.relu(x_addition)
        x_addition =self.fc4(x_addition)
        x_addition = x_addition.reshape(batch_size, 256, 3)

        x = self.dcov6(x)
        x = self.bn3(x)
        x = x.reshape(batch_size, 32*32*3)
        x = self.fc5(x)
        x = x.reshape(batch_size, 32*24, 3)
        x = torch.cat([x_addition, x], dim=1)
        x = x.reshape(batch_size, 1024, 3)
        
        return x
