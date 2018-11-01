from torch.nn import *
from .library import *

model = Sequential(
    Reshape(32, 1, 6),
    ConvTranspose2d(32, 64, 3, stride=2),
    ReLU(),
    ConvTranspose2d(64, 64, 3, stride=2, output_padding=(1, 0)),
    ReLU(),
    ConvTranspose2d(64, 32, 3, stride=2, output_padding=(1, 0)),
    ReLU(),
    ConvTranspose2d(32, 16, 3, stride=2, output_padding=(0, 1)),
    ReLU(),
    ConvTranspose2d(16, 3, 3, stride=2)
)
