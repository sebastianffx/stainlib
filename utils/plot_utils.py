import numpy as np
import pylab as plt
from PIL import Image

def _plot_imagegrid(images, nrows_ncols=None, figsize=(10,10), show=True, titles=None, ylabels=None, wspace=0.01, hspace=0,
                   save_path=None):
    if isinstance(images, np.ndarray):
        images = np.squeeze(images)

    if nrows_ncols is None:
        ncols = int(np.ceil(np.sqrt(len(images))))
        nrows = int(np.floor(np.sqrt(len(images))))
        nrows_ncols = (nrows, ncols)

    fig, axes = plt.subplots(nrows=nrows_ncols[0], ncols=nrows_ncols[1], figsize=figsize,
                             gridspec_kw={'wspace': wspace, 'hspace': hspace})

    axes = list(axes.flat)
    for i, _ in enumerate(images):
        ax = axes[i]
        im = ax.imshow(np.squeeze(images[i]))
        # ax.set_axis_off()
        ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        ax.set_yticks([], [])
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        if ylabels is not None and i < len(ylabels) and ylabels[i] is not None:
            ax.set_ylabel(ylabels[i], fontsize=10)

    # fig.subplots_adjust(right=0.85)
    # #[left, bottom, width, height
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()