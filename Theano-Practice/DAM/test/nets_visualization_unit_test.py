from nets_visualization import show_weight, show_attention_weight, show_gru_weight
import numpy as np


def unit_test_attention_weight_matrix_visualization():
    w = np.random.random((30, 200))
    w_1 = np.random.random((30, 200))
    w_2 = np.random.random((30, 200))
    w_1_title = "$First\ layer\ of\ MLP: $\n $h^{(1)} =" \
                " \\tanh(W^{(1)}x+b^{(1)})$\n $W^{(1)}$"
    w_2_title = "$Last\ layer\ of\ MLP: $\n $h^{(2)} = " \
                "\sigma(W^{(2)}h^{(1)}+b^{(2)})$\n $W^{(2)}$"
    image_file = "attention.pdf"
    show_attention_weight(w_1, w_1_title, w_2, w_2_title, image_file)


def unit_test_gru_weight_matrix_visualization():
    # w = np.random.random((200, 300))
    # w_z = np.random.random((200, 300))
    # u_z = np.random.random((200, 250))
    w = np.random.random((300, 300))
    w_z = np.random.random((300, 300))
    u_z = np.random.random((300, 300))
    w_r = w
    u_r = w
    w_h = w
    u_h = w
    w_z_title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+" \
                "U^{(z)}h_{t-1}+b^{(z)})$\n $W^{(z)}$"
    u_z_title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+" \
                "U^{(z)}h_{t-1}+b^{(z)})$\n $U^{(z)}$"
    w_r_title = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+" \
                "U^{(r)}h_{t-1}+b^{(r)})$\n $W^{(r)}$"
    u_r_title = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+" \
                "U^{(r)}h_{t-1}+b^{(r)})$\n $U^{(r)}$"
    w_h_title = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+" \
                "U(r_{t}\odot h_{t-1})+b^{(h)})$\n $W$"
    u_h_title = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+" \
                "U(r_{t}\odot h_{t-1})+b^{(h)})$\n $U$"
    image_file = "gru_weight.pdf"
    show_gru_weight(
        w_z, w_z_title, u_z, u_z_title,
        w_r, w_r_title, u_r, u_r_title,
        w_h, w_h_title, u_h, u_h_title, image_file)


def unit_test_weight_matrix_visualization():
    w = np.random.random((200, 300))
    w = w * 100
    title = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+" \
            "U^{(z)}h_{t-1}+b^{(z)})$\n $W^{(z)}$"
    image_file = "matrix.pdf"
    show_weight(w, title, image_file)


if __name__ == "__main__":
    unit_test_weight_matrix_visualization()
    unit_test_gru_weight_matrix_visualization()
    unit_test_attention_weight_matrix_visualization()