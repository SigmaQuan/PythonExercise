"""
contourf.
"""
import matplotlib.pyplot as plt
import numpy as np

name = "$W^{(z)}$ in $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$"
w = np.random.random((200, 300))

w = w * 100
axes1 = plt.subplot(3, 2, 1)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
plt.title(name)

plt.subplot(3, 2, 2)
plt.imshow(w)
plt.colorbar()
plt.subplot(3, 2, 3)
plt.imshow(w)
plt.colorbar()
plt.subplot(3, 2, 4)
plt.imshow(w)
plt.colorbar()
plt.subplot(3, 2, 5)
plt.imshow(w)
plt.colorbar()
plt.subplot(3, 2, 6)
plt.imshow(w)
plt.colorbar()

# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)

# plt.colorbar()
plt.show()
