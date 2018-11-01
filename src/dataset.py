import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt


def display_image(arr):
    image = arr[:3, :, :].detach().cpu().numpy() * 255
    image = np.clip(image, 0, 255).astype('uint8')
    arr = np.transpose(image, [1, 2, 0])
    plt.imshow(arr)
    plt.savefig('temp_plot.png')


class NumberColorsDataset(torch.utils.data.Dataset):
    def __init__(self, example_tensor):
        super().__init__()
        self.numbers = np.load("data/numbers.npy")
        self.example_tensor = example_tensor

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        number = idx % 1000

        digits = np.zeros(3, dtype='i')
        digits[2] = number % 10
        number //= 10
        digits[1] = number % 10
        number //= 10
        digits[0] = number % 10

        color = idx // 1000
        image = self.example_tensor.new_tensor(
            np.concatenate([
                self.numbers[digits[0]],
                self.numbers[digits[1]],
                self.numbers[digits[2]]], axis=1)).float()

        w, h = image.size()
        result = image.new_zeros(3, w, h)
        result[color] = image

        color = self.example_tensor.new_tensor(color, dtype=torch.long)

        return result, color
