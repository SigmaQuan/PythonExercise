"""
contourf.
"""
import matplotlib.pyplot as plt
import numpy as np



def show(w, w_title):
    """
    Show a weight matrix.
    :param w: the weight matrix.
    :param w_title: the title of the weight matrix
    :return: None.
    """
    # show w_z matrix of update gate.
    axes_w = plt.gca()
    plt.imshow(w)
    plt.colorbar()
    plt.xlabel("$w_{1}$")
    plt.ylabel("$w_{2}$")
    axes_w.set_xticks([])
    axes_w.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w[0]), len(w))
    w_title += matrix_size
    plt.title(w_title)

    # show the matrix.
    plt.show()

if __name__ == "__main__":
    w = np.random.random((200, 300))
    w = w * 100
    title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$\n $W^{(z)}$"
    show(w, title)
