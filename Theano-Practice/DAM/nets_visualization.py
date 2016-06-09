"""
Networks visualization
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages


def show_weight(w, w_title, image_file):
    """
    Show a weight matrix.
    :param w: the weight matrix.
    :param w_title: the title of the weight matrix
    :param image_file: save image to file.
    :return: None.
    """
    # set figure size
    plt.figure(figsize=(24, 12))
    plt.ion()

    # show w_z matrix of update gate.
    # axes_w = plt.gca()
    # plt.imshow(w)
    # plt.colorbar(orientation="horizontal")
    # plt.xlabel("$w_{1}$")
    # plt.ylabel("$w_{2}$")
    # axes_w.set_xticks([])
    # axes_w.set_yticks([])
    # matrix_size = "$:\ %d \\times\ %d$" % (len(w[0]), len(w))
    # w_title += matrix_size
    # plt.title(w_title)
    axes_w = plt.gca()
    im_w = axes_w.imshow(w)
    divider_w = make_axes_locatable(axes_w)
    cax_w = divider_w.append_axes("bottom", size="5%", pad=0.3)
    plt.colorbar(im_w, orientation="horizontal", cax=cax_w)
    axes_w.set_xlabel("$h_{t}$")
    axes_w.set_ylabel("$h_{t}$")
    axes_w.set_xticks([])
    axes_w.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w[0]), len(w))
    w_title += matrix_size
    axes_w.set_title(w_title)

    # show the matrix.
    plt.show()

    # save image
    pp = PdfPages(image_file)
    plt.savefig(pp, format='pdf')
    pp.close()

    # close plot GUI
    plt.close()


def show_gru_weight(w_z, w_z_title, u_z, u_z_title, w_r, w_r_title, u_r, u_r_title,
         w_h, w_h_title, u_h, u_h_title, image_file):
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
    :param image_file: save image to file.
    :return: None.
    """
    # set figure size
    plt.figure(figsize=(24, 12))
    plt.ion()

    # show w_z matrix of update gate.
    axes_w_z = plt.subplot(2, 3, 1)
    im_w_z = axes_w_z.imshow(w_z)
    divider_w_z = make_axes_locatable(axes_w_z)
    # "left", "right", "top", "bottom"
    cax_w_z = divider_w_z.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_w_z, cax=cax_w_z)
    axes_w_z.set_xlabel("$x_{t}$")
    axes_w_z.set_ylabel("$z_{t}$")
    axes_w_z.set_xticks([])
    axes_w_z.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_z[0]), len(w_z))
    w_z_title += matrix_size
    axes_w_z.set_title(w_z_title)

    # show w_r matrix of reset gate.
    # axes_w_r = plt.subplot(2, 3, 2)
    # plt.imshow(w_r)
    # plt.colorbar()
    # plt.xlabel("$x_{t}$")
    # plt.ylabel("$r_{t}$")
    # axes_w_r.set_xticks([])
    # axes_w_r.set_yticks([])
    # matrix_size = "$:\ %d \\times\ %d$" % (len(w_r[0]), len(w_r))
    # w_r_title += matrix_size
    # plt.title(w_r_title)
    axes_w_r = plt.subplot(2, 3, 2)
    im_w_r = axes_w_r.imshow(w_r)
    divider_w_r = make_axes_locatable(axes_w_r)
    cax_w_r = divider_w_r.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_w_r, cax=cax_w_r)
    axes_w_r.set_xlabel("$x_{t}$")
    axes_w_r.set_ylabel("$r_{t}$")
    axes_w_r.set_xticks([])
    axes_w_r.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_r[0]), len(w_r))
    w_r_title += matrix_size
    axes_w_r.set_title(w_r_title)

    # show w_h matrix of hidden cell.
    # axes_w_h = plt.subplot(2, 3, 3)
    # plt.imshow(w_h)
    # plt.colorbar()
    # plt.xlabel("$x_{t}$")
    # plt.ylabel("$h_{t}$")
    # axes_w_h.set_xticks([])
    # axes_w_h.set_yticks([])
    # matrix_size = "$:\ %d \\times\ %d$" % (len(w_h[0]), len(w_h))
    # w_h_title += matrix_size
    # plt.title(w_h_title)
    axes_w_h = plt.subplot(2, 3, 3)
    im_w_h = axes_w_h.imshow(w_h)
    divider_w_h = make_axes_locatable(axes_w_h)
    cax_w_h = divider_w_h.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_w_h, cax=cax_w_h)
    axes_w_h.set_xlabel("$x_{t}$")
    axes_w_h.set_ylabel("$h_{t}$")
    axes_w_h.set_xticks([])
    axes_w_h.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_h[0]), len(w_h))
    w_h_title += matrix_size
    axes_w_h.set_title(w_h_title)

    # show u_z matrix of update gate.
    # axes_u_z = plt.subplot(2, 3, 4)
    # plt.imshow(u_z)
    # plt.colorbar()
    # plt.xlabel("$h_{t}$")
    # plt.ylabel("$z_{t}$")
    # axes_u_z.set_xticks([])
    # axes_u_z.set_yticks([])
    # matrix_size = "$:\ %d \\times\ %d$" % (len(u_z[0]), len(u_z))
    # u_z_title += matrix_size
    # plt.title(u_z_title)
    axes_u_z = plt.subplot(2, 3, 4)
    im_u_z = axes_u_z.imshow(u_z)
    divider_u_z = make_axes_locatable(axes_u_z)
    cax_u_z = divider_u_z.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_u_z, cax=cax_u_z)
    axes_u_z.set_xlabel("$h_{t}$")
    axes_u_z.set_ylabel("$z_{t}$")
    axes_u_z.set_xticks([])
    axes_u_z.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_z[0]), len(u_z))
    u_z_title += matrix_size
    axes_u_z.set_title(u_z_title)

    # show u_r matrix of reset gate.
    # axes_u_r = plt.subplot(2, 3, 5)
    # plt.imshow(u_r)
    # plt.colorbar()
    # plt.xlabel("$h_{t}$")
    # plt.ylabel("$r_{t}$")
    # axes_u_r.set_xticks([])
    # axes_u_r.set_yticks([])
    # matrix_size = "$:\ %d \\times\ %d$" % (len(u_r[0]), len(u_r))
    # u_r_title += matrix_size
    # plt.title(u_r_title)
    axes_u_r = plt.subplot(2, 3, 5)
    im_u_r = axes_u_r.imshow(u_r)
    divider_u_r = make_axes_locatable(axes_u_r)
    cax_u_r = divider_u_r.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_u_r, cax=cax_u_r)
    axes_u_r.set_xlabel("$h_{t}$")
    axes_u_r.set_ylabel("$r_{t}$")
    axes_u_r.set_xticks([])
    axes_u_r.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_r[0]), len(u_r))
    u_r_title += matrix_size
    axes_u_r.set_title(u_r_title)

    # show u_h matrix of hidden cell.
    # axes_u_h = plt.subplot(2, 3, 6)
    # plt.imshow(u_h)
    # plt.colorbar()
    # plt.xlabel("$h_{t}$")
    # plt.ylabel("$h_{t}$")
    # axes_u_h.set_xticks([])
    # axes_u_h.set_yticks([])
    # matrix_size = "$:\ %d \\times\ %d$" % (len(u_h[0]), len(u_h))
    # u_h_title += matrix_size
    # plt.title(u_h_title)
    axes_u_h = plt.subplot(2, 3, 6)
    im_u_h = axes_u_h.imshow(u_h)
    divider_u_h = make_axes_locatable(axes_u_h)
    cax_u_h = divider_u_h.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im_u_h, cax=cax_u_h)
    axes_u_h.set_xlabel("$h_{t}$")
    axes_u_h.set_ylabel("$h_{t}$")
    axes_u_h.set_xticks([])
    axes_u_h.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(u_h[0]), len(u_h))
    u_h_title += matrix_size
    axes_u_h.set_title(u_h_title)

    # adjust margin
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.4)
    # plot_margin = 0.25
    # x0, x1, y0, y1 = axes_w_z.axis()
    # axes_w_z.axis((x0 - plot_margin,
    #           x1 + plot_margin,
    #           y0 - plot_margin,
    #           y1 + plot_margin))

    # show the six matrices.
    plt.show()

    # save image
    pp = PdfPages(image_file)
    plt.savefig(pp, format='pdf')
    pp.close()

    # close plot GUI
    plt.close()



def show_attention_weight(w_1, w_1_title, w_2, w_2_title, image_file):
    """
    Show the weight matrices of two layer feed-forward neural nets.
    :param w_1: the weight matrix W2 of first layer.
    :param w_1_title: the title of the weight matrix W of first layer.
    :param w_2: the weight matrix W2 of last layer.
    :param w_2_title: the title of the weight matrix U of last layer.
    :return: None.
    """
    # set figure size
    plt.figure(figsize=(24, 12))
    plt.ion()

    # show w_1 matrix of first layer.
    # fig, (axes_w_1, axes_w_2) = plt.subplots(2, 1, sharey=True)
    axes_w_1 = plt.subplot(2, 1, 1)
    im_1 = axes_w_1.imshow(w_1)
    divider = make_axes_locatable(axes_w_1)
    # "left", "right", "top", "bottom"
    # cax = divider.append_axes("right", size="5%", pad=0.05)
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
    cax_2 = divider.append_axes("bottom", size="5%", pad=0.35)
    plt.colorbar(im_2, orientation="horizontal", cax=cax_2)
    axes_w_2.set_xlabel("$h_{1}$")
    axes_w_2.set_ylabel("$h_{2}$")
    axes_w_2.set_xticks([])
    axes_w_2.set_yticks([])
    matrix_size = "$:\ %d \\times\ %d$" % (len(w_2[0]), len(w_2))
    w_2_title += matrix_size
    axes_w_2.set_title(w_2_title)

    # plt.tight_layout()

    # show the six matrices.
    plt.show()

    # save image
    pp = PdfPages(image_file)
    plt.savefig(pp, format='pdf')
    pp.close()

    # close plot GUI
    plt.close()

