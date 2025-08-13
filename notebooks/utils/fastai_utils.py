import itertools
import pickle
import types

import matplotlib.pyplot as plt
import numpy as np

from fastai.interpret import ClassificationInterpretation  # type: ignore

__all__ = ["format_lrs", "store_fastai_classification_recordings"]

"""
Display learning rates, from learning rate finder,
show the labels nearby
"""


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
    Store fastai recordings data into mlflow.
    Since building interpretation clears plot, this is done in correct order
"""


def store_fastai_classification_recordings(learn, mlfclient, run, vocabulary=None):
    _store_fastai_train_recordings(learn, mlfclient, run)
    interp = ClassificationInterpretation.from_learner(learn)
    _store_fastai_interpretation(interp, mlfclient, run, vocabulary)
    return interp


"""
    Workaround to plot the confussion matrix with vocabulary.
"""


def _plot_confusion_matrix_vocab(
    self,
    vocab_sorted: list,
    normalize: bool = False,  # Whether to normalize occurrences
    title: str = "Confusion matrix",  # Title of plot
    cmap: str = "Blues",  # Colormap from matplotlib
    norm_dec: int = 2,  # Decimal places for normalized occurrences
    plot_txt: bool = True,  # Display occurrence in matrix
    **kwargs,
):
    "Plot the confusion matrix, with `title` and using `cmap`."
    # This function is mainly copied from the sklearn docs
    cm = self.confusion_matrix()
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(**kwargs)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(vocab_sorted))
    plt.xticks(tick_marks, vocab_sorted, rotation=90)
    plt.yticks(tick_marks, vocab_sorted, rotation=0)

    if plot_txt:
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            mapping_i = vocab_sorted.index(self.vocab[i])
            mapping_j = vocab_sorted.index(self.vocab[j])

            coeff = f"{cm[i, j]:.{norm_dec}f}" if normalize else f"{cm[i, j]}"
            plt.text(
                mapping_j,
                mapping_i,
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

    return ax


"""
    Add methods to interpretation object.
    If you plan to serialize the interpretation object, 
    do this only after serializing object, 
    because loading with pickle won't work,
    if object was saved with this method.
"""


def _add_methods_to_interpetation(interp):
    interp.plot_confusion_matrix_vocab = types.MethodType(
        _plot_confusion_matrix_vocab, interp
    )


"""
    Remove added custom methods from interpretation object
"""


def _remove_methods_from_interpetation(interp):
    delattr(interp, "plot_confusion_matrix_vocab")


"""
    Store inside the mlflow for the selected run,
    learner object, as well as information available from run.
"""


def _store_fastai_train_recordings(learn, mlfclient, run):
    # save loss recording plot
    loss_plot_path = "loss_plot.png"
    learn.recorder.plot_loss(show_epochs=True).figure.savefig(loss_plot_path)
    mlfclient.log_artifact(
        run.info.run_id, local_path=loss_plot_path, artifact_path="figures"
    )

    # save learning rate and momentum comparison
    learn.recorder.plot_sched()
    image_plot_sched = "scheduled_lr_mom.png"
    plt.gcf().savefig(image_plot_sched)

    mlfclient.log_artifact(
        run.info.run_id, local_path=image_plot_sched, artifact_path="figures"
    )


"""
    Store fastai interpretation object.
    This should be called after saving train recordings, 
    since creating interpretation object clears learn.recorder data.  
    Several other functions, like learn.predict() or learn.show_results() will also reset the recorder.
"""


def _store_fastai_interpretation(interp, mlfclient, run, vocabulary=None):
    # save interpretation object
    interp_path = "interpretation.pkl"
    with open(interp_path, "wb") as f:
        pickle.dump(interp, f)
    # Log the interpretation object as an artifact to the current MLflow run
    mlfclient.log_artifact(
        run.info.run_id, local_path=interp_path, artifact_path="interpetation"
    )

    # save confussion matrix image
    if vocabulary is not None:
        # this needs to be performed after saving pickle object,
        # so that you can serialize it.
        _add_methods_to_interpetation(interp)
        interp.plot_confusion_matrix_vocab(
            vocabulary, figsize=(12, 12), dpi=60
        ).figure.savefig("confussion_matrix")
        _remove_methods_from_interpetation(interp)
    else:
        interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
        # default method does not return a confussion matrix
        plt.gcf().savefig("confussion_matrix.png")

    mlfclient.log_artifact(
        run.info.run_id, local_path="confussion_matrix.png", artifact_path="figures"
    )
