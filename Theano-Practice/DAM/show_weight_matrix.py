"""
contourf.
"""
import matplotlib.pyplot as plt
import numpy as np

w = np.random.random((200, 300))

w = w * 100
axes1 = plt.subplot(2, 3, 1)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
name11 = "$Update\ gate: $\n $z_{t} = \sigma(W^{(z)}x_{t}+U^{(z)}h_{t-1}+b^{(z)})$\n $W^{(z)}$"
plt.title(name11)


axes1 = plt.subplot(2, 3, 2)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
name12 = "$Reset\ gate: $\n $r_{t} = \sigma(W^{(r)}x_{t}+U^{(r)}h_{t-1}+b^{(r)})$\n $W^{(r)}$"
plt.title(name12)


axes1 = plt.subplot(2, 3, 3)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
name13 = "$Hidden: $\n $\\tilde{h}_{t} = \\tanh(Wx_{t}+U(r_{t}\odot h_{t-1})+b^{(h)})$\n $W$"
plt.title(name13)


axes1 = plt.subplot(2, 3, 4)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
name21 = "$U^{(z)}$"
plt.title(name21)


axes1 = plt.subplot(2, 3, 5)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
name22 = "$U^{(r)}$"
plt.title(name22)


axes1 = plt.subplot(2, 3, 6)
plt.imshow(w)
plt.colorbar()
plt.xlabel('%d' % len(w[0]))
plt.ylabel('%d' % len(w))
axes1.set_xticks([])
axes1.set_yticks([])
name23 = "$U$"
plt.title(name23)

# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)

# plt.colorbar()
plt.show()
