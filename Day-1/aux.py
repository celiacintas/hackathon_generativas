import matplotlib.pyplot as plt
from torch._utils import _accumulate
from torch import randperm, utils
from torchvision import datasets, transforms
import numpy as np


## Auxiliary functions for the ipynb

def create_plot_window(vis, xlabel, ylabel, title):
    return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

def show_samples(images, groundtruth):
    """
    Show examples of ISIC 2019 along with
    condition classification.
    Input: paths, labels
    Output: grid images
    """

    f, axarr = plt.subplots(3, 4)
    f.suptitle("Ejemplos de Vasijas")
    curr_row = 0
    for index, name in enumerate(images):
        # print(name.stem)
        a = plt.imread(name)
        # find the column by taking the current index modulo 3
        col = index % 3
        # plot on relevant subplot
        axarr[col, curr_row].imshow(a)
        axarr[col, curr_row].text(
            5,
            5,
            str(groundtruth[index]),
            bbox={"facecolor": "white"},
        )
        if col == 2:
            curr_row += 1

    f.tight_layout()
    return f


class Subset(utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
