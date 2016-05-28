import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.draw()
    plt.grid(True)
    plt.pause(0.05)

while True:
    plt.pause(0.05)
