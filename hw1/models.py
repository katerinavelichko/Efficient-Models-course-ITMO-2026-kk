import torch
from torch import nn


def conv(c_in: int, c_out: int, k: int, stride: int = 1) -> list[nn.Module]:
    return [
        nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=k // 2, bias=False),
        nn.ReLU(inplace=True),
    ]


class Net(nn.Module):

    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.features = nn.Sequential(
            # инпут: B x 3 x S x S

            # Conv 7x7, stride=2
            # B x 3 x S x S -> B x 32 x S/2 x S/2
            *conv(3, 32, 7, stride=2),

            # MaxPool 3x3, stride=2
            # B x 32 x S/2 x S/2 -> B x 32 x S/4 x S/4
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Conv 5x5, stride=1
            # остается S/4
            # B x 32 x S/4 x S/4 -> B x 64 x S/4 x S/4
            *conv(32, 64, 5),

            # Conv 3x3, stride=2
            # S/4 -> S/8
            *conv(64, 128, 3, stride=2),

            # Conv 1x1 меняет только число каналов
            # S/8
            *conv(128, 256, 1),

            # Conv 3x3, stride=2
            # S/8 -> S/16
            *conv(256, 256, 3, stride=2),

            # Conv 1x1
            # B x 512 x S/16 x S/16
            *conv(256, 512, 1),
        )

        self.head = nn.Sequential(
            # B x 512 x S/16 x S/16 -> B x 512 x 1 x 1
            nn.AdaptiveAvgPool2d(1),

            # B x 512 x 1 x 1 -> B x 512
            nn.Flatten(),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            # аутпут: 100 классов
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

def build_model(device: str = "cuda") -> Net:
    torch.manual_seed(0)
    return Net().to(device).eval()
