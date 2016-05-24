# Something weird happens when you run this code. Find something that is
# not quite right. Figure out which compilation modes make the problem
# more obvious. Explain why what is happening is happening.
import numpy as np
from theano import function
from theano import tensor as T
x = T.vector()
y = T.vector()
z = T.zeros_like(y)
a = x + z
f = function([x, y], a)
# print(np.zeros((2,), dtype=x.dtype))
# output = f(np.zeros((1,), dtype=x.dtype), np.zeros((2,), dtype=y.dtype))
output = f(np.zeros((2,), dtype=x.dtype), np.zeros((2,), dtype=y.dtype))
print(output)
# The weird thing is that if you think about how the function call is
# implemented, the two arguments have different shapes, and so should
# the resulting values of x and z. The line adding x and z should therefore
# result in a ValueError. However, when run in the default mode it does not.
# The reason is that the optimizations realize that z is always 0, so adding
# z to x has no effect. The optimizations thus remove the addition of z.
# However, this causes the function to fail to raise an error for bad values
# of x and y. To use fewer optimizations and see the bug, you can use
# THEANO_FLAGS="mode=FAST_COMPILE". DEBUG_MODE will also catch the bug.
