import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


class MaskedBatchNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """
        Masked Batch Normalization layer for 3D inputs.
        Args:
            num_features (int): Number of features in the input tensor
            eps (float): Small constant to prevent division by zero
            momentum (float): The value used for the running_mean and running_var computation
            affine (bool): If True, this module has learnable affine parameters
        """
        super(MaskedBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.gamma = nn.Parameter(torch.ones(num_features)) if affine else None
        self.beta = nn.Parameter(torch.zeros(num_features)) if affine else None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Masked Batch Normalization layer
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Normalized tensor
        """
        mask = (x[:, :11, :, :].sum(dim=1, keepdim=True) != 0).float()

        # Compute the masked mean and variance
        masked_mean = (x * mask).sum(dim=(0, 2, 3)) / (
            mask.sum(dim=(0, 2, 3)) + self.eps
        )
        masked_var = ((x - masked_mean.view(1, -1, 1, 1)) ** 2 * mask).sum(
            dim=(0, 2, 3)
        ) / (mask.sum(dim=(0, 2, 3)) + self.eps)
        if self.training:
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * masked_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * masked_var
            )
        else:
            masked_mean = self.running_mean
            masked_var = self.running_var

        x = (x - masked_mean.view(1, -1, 1, 1)) / torch.sqrt(
            masked_var.view(1, -1, 1, 1) + self.eps
        )

        if self.affine:
            x = x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block for the UNet architecture.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the decoder block.
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.scse = scSE(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor"""
        x = self.conv(x)
        x = self.relu(x)
        x = self.scse(x)
        return x


class scSE(nn.Module):
    """
    Spatial and Channel Squeeze and Excitation block."""

    def __init__(self, in_channels: int) -> None:
        """
        Initialize the scSE block.
        Args:
            in_channels (int): Number of input channels"""
        super().__init__()
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid()
        )
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the scSE block.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor"""
        sse_out = x * self.spatial_se(x)
        cse_out = x * self.channel_se(x)
        return torch.max(sse_out, cse_out)


class RSPRUNetPlusPlus(nn.Module):
    """
    RSPRUNet++ model.
    """

    def __init__(self, num_classes: int = 2, input_channels: int = 24) -> None:
        """
        Initialize the RSPRUNet++ model.
        Args:
            num_classes (int): Number of classes
            input_channels (int): Number of input channels
        """
        super(RSPRUNetPlusPlus, self).__init__()

        torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
        model_version = "resnest101"
        resnest = torch.hub.load(
            "zhanghang1989/ResNeSt", model_version, pretrained=True
        )
        print(model_version)
        self.initial_conv = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        with torch.no_grad():
            resnest_first_conv_weight = resnest.conv1[0].weight
            self.initial_conv.weight[:, :3] = resnest_first_conv_weight
        nn.init.kaiming_normal_(self.initial_conv.weight[:, 3:])

        # Encoder
        self.encoder0 = nn.Sequential(
            self.initial_conv,
            resnest.conv1[1],
            resnest.conv1[2],
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        )
        self.encoder1 = nn.Sequential(resnest.maxpool, resnest.layer1)
        self.encoder2 = resnest.layer2
        self.encoder3 = resnest.layer3
        self.encoder4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Additional convolutions to reduce channel count
        self.conv1x1_1 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(2048, 512, kernel_size=1)

        # Decoder
        l0, l1, l2, l3, l4 = 128, 64, 128, 256, 512

        self.decoder0_1 = DecoderBlock(l0 + l1, l0)
        self.decoder0_2 = DecoderBlock(2 * l0 + l1, l0)
        self.decoder0_3 = DecoderBlock(3 * l0 + l1, l0)
        self.decoder0_4 = DecoderBlock(4 * l0 + l1, l0)

        self.decoder1_1 = DecoderBlock(l1 + l2, l1)
        self.decoder1_2 = DecoderBlock(2 * l1 + l2, l1)
        self.decoder1_3 = DecoderBlock(3 * l1 + l2, l1)

        self.decoder2_1 = DecoderBlock(l2 + l3, l2)
        self.decoder2_2 = DecoderBlock(2 * l2 + l3, l2)

        self.decoder3_1 = DecoderBlock(l3 + l4, l3)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Final 1x1 convolutions
        self.final1 = nn.Sequential(nn.Conv2d(l0, num_classes, kernel_size=1))
        self.final2 = nn.Sequential(nn.Conv2d(l0, num_classes, kernel_size=1))
        self.final3 = nn.Sequential(nn.Conv2d(l0, num_classes, kernel_size=1))
        self.final4 = nn.Sequential(nn.Conv2d(l0, num_classes, kernel_size=1))

        self.combiner_conv = nn.Conv2d(
            num_classes * 4, num_classes, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RSPRUNet++ model.
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Output tensor
        """
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e1 = self.conv1x1_1(e1)
        e2 = self.conv1x1_2(e2)
        e3 = self.conv1x1_3(e3)
        e4 = self.conv1x1_4(e4)

        d3_1 = self.decoder3_1(torch.cat([e3, e4], dim=1))
        d2_1 = self.decoder2_1(
            torch.cat(
                [
                    e2,
                    F.interpolate(
                        e3, size=e2.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        d2_2 = self.decoder2_2(
            torch.cat(
                [
                    d2_1,
                    F.interpolate(
                        d3_1, size=d2_1.shape[2:], mode="bilinear", align_corners=True
                    ),
                    e2,
                ],
                dim=1,
            )
        )

        d1_1 = self.decoder1_1(
            torch.cat(
                [
                    e1,
                    F.interpolate(
                        e2, size=e1.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        d1_2 = self.decoder1_2(
            torch.cat(
                [
                    d1_1,
                    F.interpolate(
                        d2_1, size=d1_1.shape[2:], mode="bilinear", align_corners=True
                    ),
                    e1,
                ],
                dim=1,
            )
        )
        d1_3 = self.decoder1_3(
            torch.cat(
                [
                    d1_2,
                    F.interpolate(
                        d2_2, size=d1_2.shape[2:], mode="bilinear", align_corners=True
                    ),
                    d1_1,
                    e1,
                ],
                dim=1,
            )
        )

        d0_1 = self.decoder0_1(
            torch.cat(
                [
                    e0,
                    F.interpolate(
                        e1, size=e0.shape[2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
        )
        d0_2 = self.decoder0_2(
            torch.cat(
                [
                    d0_1,
                    F.interpolate(
                        d1_1, size=d0_1.shape[2:], mode="bilinear", align_corners=True
                    ),
                    e0,
                ],
                dim=1,
            )
        )
        d0_3 = self.decoder0_3(
            torch.cat(
                [
                    d0_2,
                    F.interpolate(
                        d1_2, size=d0_2.shape[2:], mode="bilinear", align_corners=True
                    ),
                    d0_1,
                    e0,
                ],
                dim=1,
            )
        )
        d0_4 = self.decoder0_4(
            torch.cat(
                [
                    d0_3,
                    F.interpolate(
                        d1_3, size=d0_3.shape[2:], mode="bilinear", align_corners=True
                    ),
                    d0_2,
                    d0_1,
                    e0,
                ],
                dim=1,
            )
        )

        # Final 1x1 convolutions
        output1 = self.final1(d0_1)
        output2 = self.final2(d0_2)
        output3 = self.final3(d0_3)
        output4 = self.final4(d0_4)

        # Upsample outputs to input size
        output1 = nn.functional.interpolate(
            output1, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        output2 = nn.functional.interpolate(
            output2, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        output3 = nn.functional.interpolate(
            output3, size=x.shape[2:], mode="bilinear", align_corners=True
        )
        output4 = nn.functional.interpolate(
            output4, size=x.shape[2:], mode="bilinear", align_corners=True
        )

        combined_output = torch.cat([output1, output2, output3, output4], dim=1)
        final_output = self.combiner_conv(combined_output)
        # final_output = (output1 + output2 + output3 + output4) / 4
        return final_output


# Example usage
if __name__ == "__main__":
    model = RSPRUNetPlusPlus(num_classes=3, input_channels=24)
    input_tensor = torch.randn(
        2, 24, 256, 256
    )  # Batch size 1, 24 channels, 256x256 image
    outputs = model(input_tensor)
    for i, output in enumerate(outputs, 1):
        print(f"Output {i} shape:", output.shape)
