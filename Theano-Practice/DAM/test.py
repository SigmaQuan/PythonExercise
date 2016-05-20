import util
import numpy as np

w1 = util.uniform_param(std=100, shape=(100, 200))


print(w1)

w = np.random.random((200, 300))

print(w1.get_value())

print(w)

# w = [(x, y) for x, y in w1]

# for kk, vv in w1.items():
#     w[kk].setvalue(vv)

print(w)
