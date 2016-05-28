"""
contourf.
"""
import matplotlib.pyplot as plt
import numpy as np


def show(w_z, w_z_title, u_z, u_z_title, w_r, w_r_title, u_r, u_r_title,
         w_h, w_h_title, u_h, u_h_title):
    """
    Show the weight matrices of GRU.
    :param w_z: the weight matrix W of update gate.
    :param w_z_title: the title of the weight matrix W of update gate.
    :param u_z: the weight matrix U of update gate.
    :param u_z_title: the title of the weight matrix U of update gate.
    :param w_r: the weight matrix W of reset gate.
    :param w_r_title: the title of the weight matrix W of reset gate.
    :param u_r: the weight matrix U of reset gate.
    :param u_r_title: the title of the weight matrix U of reset gate.
    :param w_h: the weight matrix W of hidden cell.
    :param w_h_title: the title of the weight matrix W of hidden.
    :param u_h: the weight matrix U of hidden cell.
    :param u_h_title: the title of the weight matrix U of hidden.
    :return: None.
    """
    # show w_z matrix of update gate.
    axes_w_z = plt.subplot(2, 3, 1)
    plt.imshow(w_z)
    plt.colorbar()
    plt.xlabel("$x_{t}$")
    plt.ylabel("$z_{t}$")
    axes_w_z.set_xticks([])
    axes_w_z.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_z[0]), len(w_z))
    w_z_title += matrix_size
    plt.title(w_z_title)

    # show w_r matrix of reset gate.
    axes_w_r = plt.subplot(2, 3, 2)
    plt.imshow(w_r)
    plt.colorbar()
    plt.xlabel("$x_{t}$")
    plt.ylabel("$r_{t}$")
    axes_w_r.set_xticks([])
    axes_w_r.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_r[0]), len(w_r))
    w_r_title += matrix_size
    plt.title(w_r_title)

    # show w_h matrix of hidden cell.
    axes_w_h = plt.subplot(2, 3, 3)
    plt.imshow(w_h)
    plt.colorbar()
    plt.xlabel("$x_{t}$")
    plt.ylabel("$h_{t}$")
    axes_w_h.set_xticks([])
    axes_w_h.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_h[0]), len(w_h))
    w_h_title += matrix_size
    plt.title(w_h_title)

    # show u_z matrix of update gate.
    axes_u_z = plt.subplot(2, 3, 4)
    plt.imshow(u_z)
    plt.colorbar()
    plt.xlabel("$h_{t}$")
    plt.ylabel("$z_{t}$")
    axes_u_z.set_xticks([])
    axes_u_z.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_z[0]), len(u_z))
    u_z_title += matrix_size
    plt.title(u_z_title)

    # show u_r matrix of reset gate.
    axes_u_r = plt.subplot(2, 3, 5)
    plt.imshow(u_r)
    plt.colorbar()
    plt.xlabel("$h_{t}$")
    plt.ylabel("$r_{t}$")
    axes_u_r.set_xticks([])
    axes_u_r.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_r[0]), len(u_r))
    u_r_title += matrix_size
    plt.title(u_r_title)

    # show u_h matrix of hidden cell.
    axes_u_h = plt.subplot(2, 3, 6)
    plt.imshow(u_h)
    plt.colorbar()
    plt.xlabel("$h_{t}$")
    plt.ylabel("$h_{t}$")
    axes_u_h.set_xticks([])
    axes_u_h.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_h[0]), len(u_h))
    u_h_title += matrix_size
    plt.title(u_h_title)

    # show the six matrices.
    plt.show()


if __name__ == "__main__":
    w = np.random.random((200, 300))
    w_z = np.random.random((200, 300))
    u_z = np.random.random((200, 250))
    w_r = w
    u_r = w
    w_h = w
    u_h = w
    w_z_title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$\n $W^{(z)}$"
    u_z_title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$\n $U^{(z)}$"
    w_r_title = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+U^{(r)}h_{t-1}+b^{(r)})$\n $W^{(r)}$"
    u_r_title = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+U^{(r)}h_{t-1}+b^{(r)})$\n $U^{(r)}$"
    w_h_title = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+U(r_{t}\odot h_{t-1})+b^{(h)})$\n $W$"
    u_h_title = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+U(r_{t}\odot h_{t-1})+b^{(h)})$\n $U$"
    show(w_z, w_z_title, u_z, u_z_title, w_r, w_r_title, u_r, u_r_title,
         w_h, w_h_title, u_h, u_h_title)
