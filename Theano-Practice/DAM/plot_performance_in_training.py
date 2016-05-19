import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


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
    network_name = 'task_%s.epoch_%03d.train_%06.3f.test_%06.3f' % (5, 10, 100, 99)
    plt.title(network_name)
    plt.draw()
    plt.grid(True)
    plt.pause(0.05)

imageName = '%s/task_%s.pdf' % ('experiment', 1)
pp = PdfPages(imageName)
plt.savefig(pp, format='pdf')
pp.close()
plt.close()
