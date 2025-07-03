"""
Display learning rates, from learning rate finder,
show the labels nearby
"""

import matplotlib as plt
import numpy as np
import itertools


def format_lrs(lrs) -> None:
    descriptions = {
        "steep": "learning rate when the slope is the steepest, \nloss is decreasing most rapidly",
        "slide": "learning rate following an interval slide rule, \nthis could be when loss is decreasing just before the loss starts to increase rapidly",
        "minimum": "1/10th of the minimum point for the loss function, \nlargest sensible value to use for learning rate, since minimum is to high",
        "valley": "learning rate from the longest valley, \nthis could be before the loss starts to increase rapidly",
    }

    for key, value in sorted(lrs._asdict().items(), key=lambda item: item[1]):
        print("metric: %s,  learning_rate:  %5.6f" % (key, value))
        print(descriptions[key])
        print()


"""
    Workaround to plot the confussion matrix with vocabulary
"""


def plot_confusion_matrix_vocab(
    self,
    normalize: bool = False,  # Whether to normalize occurrences
    title: str = "Confusion matrix",  # Title of plot
    cmap: str = "Blues",  # Colormap from matplotlib
    norm_dec: int = 2,  # Decimal places for normalized occurrences
    plot_txt: bool = True,  # Display occurrence in matrix
    **kwargs,
) -> None:
    "Plot the confusion matrix, with `title` and using `cmap`."
    # This function is mainly copied from the sklearn docs
    cm = self.confusion_matrix()
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(self.vocab))
    plt.xticks(tick_marks, self.vocab, rotation=90)
    plt.yticks(tick_marks, self.vocab, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            coeff = f"{cm[i, j]:.{norm_dec}f}" if normalize else f"{cm[i, j]}"
            # change the location in here based on mapping
            plt.text(
                j,
                i,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax = fig.gca()
    ax.set_ylim(len(self.vocab) - 0.5, -0.5)

    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.grid(False)
