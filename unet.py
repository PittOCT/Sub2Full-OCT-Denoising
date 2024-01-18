import torch
import torch.nn as nn
from torchsummary import summary


class UNet(nn.Module):
    """Use the same U-Net architecture as in Noise2Noise (Lehtinen et al., 2018)."""

    def __init__(self, in_channels=1, out_channels=1, feature_maps=48):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.feature_maps, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.feature_maps, self.feature_maps, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(self.feature_maps, self.feature_maps, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(self.feature_maps, self.feature_maps, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(self.feature_maps*2, self.feature_maps*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.feature_maps*2, self.feature_maps*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(self.feature_maps*3, self.feature_maps*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.feature_maps*2, self.feature_maps*2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(self.feature_maps*2 + self.in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, self.out_channels, 3, stride=1, padding=1),
            nn.Identity())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Kaiming Initialization (He et al., 2015)."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data, a=0.01, nonlinearity='leaky_relu')
                m.bias.data.zero_()

    def forward(self, x):
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self._block6(concat1)

        return out


if __name__ == '__main__':
    # Test the network
    model = UNet().to('cuda:0')
    # input_size should be C*H*W
    summary(model, input_size=(1, 384, 384), batch_size=2)