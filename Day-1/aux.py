import matplotlib.pyplot as plt

## Auxiliary functions for the ipynb


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