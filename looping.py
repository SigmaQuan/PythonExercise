# Guide
#   The scan functions provides the basic functionality needed to do
#   loops in Theano.scan comes with many whistles and bells, which will
#   introduce by way of examples.

# Simple loop with accumulation: computing A^k
# result = 1
# for i in range(k):
#     result = result * A

# There are three things here that we need to handle:
#   1. the initial value assigned to result,
#   2. the accumulation of results in result, and
#   3. the unchanging variable A.
# Unchanging variables are passed to scan as non_sequences. Initialization
# occurs in outputs_info, and the accumulation happens automatically.
#
# # The equivalent Theano code would be:
# import theano
# import theano.tensor as T
#
# k = T.iscalar("k")
# A = T.vector("A")
#
# # Symbolic description of the result.
# result, updates = theano.scan(
#     fn=lambda prior_result, A: prior_result * A,
#     outputs_info=T.ones_like(A),
#     non_sequences=A,
#     n_steps=k)
#
# # We only care about A**k, but scan has provided us with A**1 through A**k.
# # Discard the values that we don't care about. Scan is smart enough to
# # notice this and not waste memory saving them.
# final_reult = result[-1]
#
# # Compiled function that returns A**k
# power = theano.function(inputs=[A, k], outputs=final_reult, updates=updates)
#
# # Test function
# print(power(range(10),2))
# print(power(range(10),4))
# # Using gpu device 1: GeForce GTX TITAN (CNMeM is enabled with initial
# # size: 85.0% of memory, CuDNN 4007)
# # [  0.   1.   4.   9.  16.  25.  36.  49.  64.  81.]
# # [  0.00000000e+00   1.00000000e+00   1.60000000e+01   8.10000000e+01
# #    2.56000000e+02   6.25000000e+02   1.29600000e+03   2.40100000e+03
# #    4.09600000e+03   6.56100000e+03]

# Let us go through the example line by line. What we did is first to
# construct a function (using a lambda expression) that given prior_result
# and A returns prior_result * A. The order of parameters is fixed by
# scan: the output of the prior call to fn (or the initial value,
# initially) is the first parameter, followed by all non-sequences.

# Next we initialize the output as a tensor with shape and dtype as A,
# filled with ones. We give A to scan as a non sequence parameter and
# specify the number of steps k to iterate over out lambda expression.

# Scan returns a tuple containing our result (result) and a dictionary of
# updates (empty in this case). Note that the result is not a matrix, but
# a 3D tensor containing the value of A**k for each step. We want the last
# value (after k steps) so we compile a function to return just that. Note
# that there is an optimization, that at compile time will detect that you
# are using just the last value of the result and ensure that scan not
# store all the intermediate values that are used. So do not worry if A and
# k are large.


# Iterating over the first dimension of a tensor: Calculating a polynomial

# In addition to looping a fixed number of times, scan can iterate over the
# leading dimension of tensors (similar to Python's  for x in a_list).

# The tensor(s) to be looped over should be provided to scan the sequence
# keyword argument.
#
# # Here is an example that builds a symbolic calculation of a polynomial
# # from a list of its coefficients:
# import theano
# import theano.tensor as T
# import numpy
#
# coefficients = theano.tensor.vector("coefficients")
# x = T.scalar("x")
#
# max_coefficients_supported = 10000
#
# # Generate the components of the polynomial
# components, updates = theano.scan(
#     fn=lambda coefficient, power, free_variable:
#     coefficient * (free_variable ** power),
#     outputs_info=None,
#     sequences=[coefficients, theano.tensor.arange(
#         max_coefficients_supported)],
#     non_sequences=x)
#
# # Sum them up
# polynomial = components.sum()
#
# # Compile a function
# calculate_polynomial = theano.function(
#     inputs=[coefficients, x], outputs=polynomial)
#
# # Test
# test_coefficients = numpy.asarray([1, 0, 2], dtype=numpy.float32)
# test_value = 3
# print(calculate_polynomial(test_coefficients, test_value))
# print(1.0*(3**0)+0.0*(3**1)+2.0*(3**2))
# # Using gpu device 1: GeForce GTX TITAN (CNMeM is enabled with initial
# # size: 85.0% of memory, CuDNN 4007)
# # 19.0
# # 19.0

# There are a few things to note here.
#   First, we calculate the polynomial by first generating each of the
# coefficients, and then summing them at the end. (we could also have
# accumulated them along the way, and then taken the last one, which would
# have been more memory-efficient, but this is an example.)
#   **Second, there is no accumulation of results, we can set ouputs_info to
# None. This indicates to scan that it doesn't need to pass the prior
# result to fn. The general order of function parameters to fn is:
#    sequences (if any), prior result(s) (if need), non-sequences (if any)
#   **Third, there's handy trick used to simulate python's enumerate: simply
# include theano.tensor.arange to the sequences.
#   **Fourth, given multiple sequences of uneven lengths, scan will
# truncate to the shortest of them. This makes it safe to pass a very long
# arange, which we need to do for generality, since arange must have its
# length specified at creation time.


# Simple accumulation into a scalar, ditching lambda
#
# # Although this example would seem almost self-explanatory, it stresses a
# # pitfall to be careful of: the initial output state that is supplied,
# # that is ouputs_info, must be of a shape similar to that of the output
# # variable generated at each iteration and moreover, it must not involve
# # an implicit downcast of the latter.
# import numpy as np
# import theano
# import theano.tensor as T
#
# up_to = T.iscalar("up_to")
#
# # Define a named function, rather than using lambda
# def accumulate_by_adding(arange_val, sum_to_date):
#     return sum_to_date + arange_val
# seq = T.arange(up_to)
#
# # An unauthorized implicit donwcast from the dtype of 'seq', to that of
# # 'T.as_tensor_variable(0)' which is of dtype 'int8' by default would occur
# # if this instruction were to be used instead of the next one:
# # outputs_info = T.as_tensor_variable(0)
#
# outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
# scan_result, scan_updates = theano.scan(
#     fn=accumulate_by_adding,
#     outputs_info=outputs_info,
#     sequences=seq
# )
#
# triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)
#
# # test
# some_num = 15
# print(triangular_sequence(some_num))
# print([n*(n+1)//2 for n in range(some_num)])
# # Using gpu device 1: GeForce GTX TITAN (CNMeM is enabled with initial
# # size: 85.0% of memory, CuDNN 4007)
# # [  0   1   3   6  10  15  21  28  36  45  55  66  78  91 105]
# # [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105]

# Another simple example
# Unlike some of the prior examples, this one is hard to reproduce except
# by using scan.
#
# # This takes a sequence of array indices, and values to place there, and a
# # "model" output array (whose shape and dtype will be mimicked), and
# # produces a sequence of arrays with the shape and dtype of the model, with
# # all values set to zero except at the provided array indices.
# import numpy as np
# import theano
# import theano.tensor as T
# location = T.imatrix("location")
# values = T.vector("values")
# output_model = T.matrix("output_model")
#
#
# def set_value_at_position(a_location, a_value, output_model):
#     zeros = T.zeros_like(output_model)
#     zeros_subtensor = zeros[a_location[0], a_location[1]]
#     return T.set_subtensor(zeros_subtensor, a_value)
#
# result, updates = theano.scan(
#     fn=set_value_at_position,
#     sequences=[location, values],
#     non_sequences=output_model
# )
#
# assign_values_at_positions = theano.function(inputs=[location, values, output_model], outputs=result)
#
# # Test
# test_locations = np.asarray([[1, 1], [2, 3]], dtype=np.int32)
# test_value = np.asarray([42, 50], dtype=np.float32)
# test_output_model = np.zeros((5, 5), dtype=np.float32)
# print(assign_values_at_positions(test_locations, test_value, test_output_model))
# # Using gpu device 1: GeForce GTX TITAN (CNMeM is enabled with initial
# # size: 85.0% of memory, CuDNN 4007)
# # [[[  0.   0.   0.   0.   0.]
# #   [  0.  42.   0.   0.   0.]
# #   [  0.   0.   0.   0.   0.]
# #   [  0.   0.   0.   0.   0.]
# #   [  0.   0.   0.   0.   0.]]
# #
# #  [[  0.   0.   0.   0.   0.]
# #   [  0.   0.   0.   0.   0.]
# #   [  0.   0.   0.  50.   0.]
# #   [  0.   0.   0.   0.   0.]
# #   [  0.   0.   0.   0.   0.]]]

# This demonstrates that you can introduce new Theano variable into a scan
# function.

# Using shared variables - Gibbs sampling

# Another useful feature of scan, is that it can handle shared variables.
# For example, if we want to implement a Gibbs chain of length 10 we would
# do the following:
import theano
from theano import tensor as T
# We assume that "W_values" contains the initial values of you weight
# matrix.
W = theano.shared(W_values)

bvis = theano.shared(bvis_values)
bhid = theano.shared(bhid_values)

trng = T.shared_randomstreams.RandomStreams(1234)


def OneStep(vsample):
    hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
    hsample = trng.binomia(size=hmean.shape, n=1, p=hmean)
    vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
    return trng.binomial(size=vsample.shape, n=1, p=vmean,
                         dtype=theano.config.floatX)

sample = theano.tensor.vector()

values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)

gibbs10 = theano.function([sample], values[-1], updates=updates)
