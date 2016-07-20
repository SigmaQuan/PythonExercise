import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages


def dynamic_show_1():
    fig = plt.figure()
    plt.axis([0, 10, 0, 1])
    plt.ion()

    train_acc = []
    test_acc = []
    epochs = []

    for epoch in range(10):
        training_acc = np.random.randn()
        testing_acc = np.random.randn()

        train_acc.append(training_acc)
        test_acc.append(testing_acc)

        epochs.append(epoch)
        plt.gca().cla()
        plt.plot(epochs, train_acc, 'r.-', label="Train")
        plt.plot(epochs, test_acc, 'g.-', label="Test")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc=4)
        network_name = 'task_%s.epoch_%03d.' \
                       'train_%06.3f.test_%06.3f' % (5, 10, 100, 99)
        plt.title(network_name)
        plt.draw()
        plt.grid(True)
        plt.pause(0.05)

    imageName = '%s/task_%s.pdf' % ('experiment', 1)
    pp = PdfPages(imageName)
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.close()


def dynamic_show_2():
    fig = plt.figure()
    plt.axis([0, 10, 0, 1])
    plt.ion()

    train_error = []
    test_error = []
    epochs = []

    for epoch in range(50):
        train_error.append(np.random.random())
        test_error.append(np.random.random())
        epochs.append(epoch)

        plt.gca().cla()
        if epoch % 10 == 0:
            plt.axis([0, (int(epoch/10)+1)*10, 0, 1])
        plt.plot(epochs, train_error, 'r--', label="Training error")
        plt.plot(epochs, test_error, 'g.-', label="Testing error")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102),
                   loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.draw()
        plt.grid(True)
        plt.pause(0.05)

    while True:
        plt.pause(0.05)
