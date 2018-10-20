import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import # This import registers the 3D projection, but is otherwise unused.


def create_error_figures(error_df,error_df_tr,x_array,name,lim):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    ax = axes.flatten()
    for col in error_df.iloc[:, 1:6].columns.values:
        axnr = error_df.columns.get_loc(col) - 1
        ax[axnr].set_title(col)
        ax[axnr].scatter(x_array, error_df[col], label="val")
        ax[axnr].scatter(x_array, error_df_tr[col], label="train")
        ax[axnr].legend()
        if lim == 1:
            ax[axnr].set_xlim([0, 50])
        ax[axnr].set_xlabel(name)
    fig.tight_layout()
    # plt.show()
    return fig

def scatter_colors_2d(title,x,y,labelnrs,cmap = "hsv"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(title)
    ax.scatter(x, y, c=labelnrs, cmap = cmap,alpha=0.3)
    plt.show()

def scatter_colors_2d_labels(title,x,y,labelnrs, labelnames,cmap="hsv"):
    fig, ax = plt.subplots(figsize = (10,10))
    ax.scatter(x, y, c = labelnrs, cmap = cmap, alpha = 0.3)
    fig.suptitle(title)
    for i, txt in enumerate(labelnames):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

def scatter_colors_3d(title,x,y,z,labelnrs, cmap="hsv"):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title)
    ax.scatter(x, y, z, c=labelnrs, cmap = cmap, alpha=0.3)
    plt.show()
