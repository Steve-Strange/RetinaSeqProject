import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super().__init__()
        # Encoder
        self.e1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.e4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.b = ConvBlock(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d4 = ConvBlock(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        s1 = self.e1(x)
        p1 = self.pool1(s1)
        s2 = self.e2(p1)
        p2 = self.pool2(s2)
        s3 = self.e3(p2)
        p3 = self.pool3(s3)
        s4 = self.e4(p3)
        p4 = self.pool4(s4)
        
        b = self.b(p4)
        
        d1 = self.d1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.d2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.d3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.d4(torch.cat([self.up4(d3), s1], dim=1))
        
        return self.out(d4)