"""
contourf.
"""
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable


def show(w_1, w_1_title, w_2, w_2_title):
    """
    Show the weight matrices of two layer feed-forward neural nets.
    :param w_1: the weight matrix W2 of first layer.
    :param w_1_title: the title of the weight matrix W of first layer.
    :param w_2: the weight matrix W2 of last layer.
    :param w_2_title: the title of the weight matrix U of last layer.
    :return: None.
    """
    # show w_1 matrix of first layer.
    # fig, (axes_w_1, axes_w_2) = plt.subplots(2, 1, sharey=True)
    axes_w_1 = plt.subplot(2, 1, 1)
    im_1 = axes_w_1.imshow(w_1)
    divider = make_axes_locatable(axes_w_1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)  # "left", "right", "top", "bottom"
    # plt.colorbar(im_1, cax=cax)
    cax_1 = divider.append_axes("bottom", size="5%", pad=0.35)
    plt.colorbar(im_1, orientation="horizontal", cax=cax_1)
    axes_w_1.set_xlabel("$x_{1}$")
    axes_w_1.set_ylabel("$h_{1}$")
    axes_w_1.set_xticks([])
    axes_w_1.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_1[0]), len(w_1))
    w_1_title += matrix_size
    axes_w_1.set_title(w_1_title)


    # show w_2 matrix of last layer.
    axes_w_2 = plt.subplot(2, 1, 2)
    im_2 = axes_w_2.imshow(w_2)
    divider = make_axes_locatable(axes_w_2)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im_2, cax=cax)
    cax_2 = divider.append_axes("top", size="5%", pad=0.35)
    plt.colorbar(im_2, orientation="horizontal", cax=cax_2)
    axes_w_2.set_xlabel("$h_{1}$")
    axes_w_2.set_ylabel("$h_{2}$")
    axes_w_2.set_xticks([])
    axes_w_2.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_2[0]), len(w_2))
    w_1_title += matrix_size
    axes_w_2.set_title(w_1_title)

    # plt.tight_layout()

    # show the six matrices.
    plt.show()


if __name__ == "__main__":
    w = np.random.random((200, 300))
    w_1 = np.random.random((200, 300))
    w_2 = np.random.random((200, 250))
    w_1_title = "$First\ layer\ of\ MLP: $\n $h^{(1)} = \\tanh(W^{(1)}x+b^{(1)})$\n $W^{(1)}$"
    w_2_title = "$Last\ layer\ of\ MLP: $\n $h^{(2)} = \sigma(W^{(2)}h^{(1)}+b^{(2)})$\n $W^{(2)}$"
    show(w_1, w_1_title, w_2, w_2_title)
