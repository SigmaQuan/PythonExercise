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
# assign_values_at_positions = theano.function(
#   inputs=[location, values, output_model], outputs=result)
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
#
# # Another useful feature of scan, is that it can handle shared variables.
# # For example, if we want to implement a Gibbs chain of length 10 we would
# # do the following:
# import numpy as np
# import theano
# from theano import tensor as T
# # We assume that "W_values" contains the initial values of you weight
# # matrix.
# W_values = np.matrix('1 2; 3 4')
# print W_values
# W = theano.shared(W_values)
# print W
#
# bvis_values = 1
# bhid_values = 1
# bvis = theano.shared(bvis_values)
# bhid = theano.shared(bhid_values)
#
# trng = T.shared_randomstreams.RandomStreams(1234)
#
#
# def OneStep(vsample):
#     hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
#     hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
#     vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
#     return trng.binomial(size=vsample.shape, n=1, p=vmean,
#                          dtype=theano.config.floatX)
#
# sample = theano.tensor.vector()
#
# values, updates = theano.scan(OneStep, outputs_info=sample, n_steps=10)
#
# gibbs10 = theano.function([sample], values[-1], updates=updates)
# print gibbs10
#
# # The first, and probably most crucial observation is that the updates
# # dictionary becomes important in this case. It links a shared varible with
# # its updated value after k steps. In this case it tells how the random
# # streams get updated after 10 iterations. If you do not pass this update
# # dictionary to your function, you will always get the same 10 sets of
# # random numbers. You can even use the updates dictionary afterwards. Look
# # at this example:
# a = theano.shared(1)
# values, updates = theano.scan(lambda: {a: a+1}, n_steps=10)
#
# # In this case the lambda expression does not require and input parameters
# # and returns an update dictionary with tells how a should be updated after
# # each step of scan. If we write:
# b = a + 1
# c = updates[a] + 1
# f = theano.function([], [b, c], updates=updates)
#
# print b
# print c
# print a.get_value()
# # *****
# # We will see that because b does not use the updated version of a, it will
# # be 2, c will be 12, while a.value is 11. If we call the function again, b
# # will become 12, c will be 22 and a.value 21. If we do not pass the
# # updates dictionary to the function, then a.value will always be 2 and c
# # will always be 12.

#
# # The second observation is that if we use shared variables (W, bvis, bhid)
# # but we do not iterate over them (i.e. scan does not really need to know
# # anything in particular about them, just that they are used inside the
# # function applied at each step) you do not need to pass them as arguments.
# # Scan Op calling any earlier (external) Op over and over. This results in
# # a simpler computational graph. However, passing them to the scan function
# # is a good practice, as it avoids Scan Op calling any earlier (external)
# # Op over and over. This results in a simpler computational graph, which
# # speeds up the optimization and the execution. *****To pass the shared
# # variables to Scan you need to put them in a list and give it to the
# # non_sequences argument. Here is the Gibbs sampling code updated:
# import numpy as np
# import theano
# from theano import tensor as T
# W_values = np.matrix('1 2; 3 4')
# print W_values
# W = theano.shared(W_values)
# print W
#
# bvis_values = 1
# bhid_valuds = 1
# bvis = theano.shared(bvis_values)
# bhid = theano.shared(bhid_valuds)
#
# trng = T.shared_randomstreams.RandomStreams(1234)
#
#
# # OneStep, with explicit use fo the shared variables (W, bvis, bhid)
# def OneStep(vsample, W, bvis, bhid):
#     hmean = T.nnet.sigmoid(theano.dot(vsample, W) + bhid)
#     hsample = trng.binomial(size=hmean.shape, n=1, p=hmean)
#     vmean = T.nnet.sigmoid(theano.dot(hsample, W.T) + bvis)
#     return trng.binomial(size=vsample.shape, n=1, p=vmean,
#                          dtype=theano.config.floatX)
#
# sample = theano.tensor.vector()
#
# # The new scan, with the shared variables passed as non_sequences
# values, updates = theano.scan(
#     fn=OneStep,
#     outputs_info=sample,
#     non_sequences=[W, bvis, bhid],
#     n_steps=10
# )
#
# gibbs10 = theano.function([sample], values[-1], updates=updates)
# # sample_value = theano.tensor.vector('1,2')
# # print(gibbs10(sample_value))
#
# # Using shared variables - the strict flag
#
# # As we just saw, passing the shared variables to scan many result in a
# # simpler computational graph, which speeds up the optimization and the
# # execution. A good way to remember to pass every shared variable used
# # during scan is to use the strict flag. When set to true, scan assumes
# # that all the necessary shared variables in fn are passed as a part of
# # non_sequences. This has to be ensured by the user. Otherwise, it will
# # result in an error.
#
# # Using the previous Gibbs sampling example:
# # The new scan, using strict=True
# values, updates =  theano.scan(
#     fn=OneStep,
#     outputs_info=sample,
#     non_sequences=[W, bvis, bhid],
#     strict=True
# )
#
# # If you omit to pass W, bvis or bhid, as a non_sequence, it will result in
# # an error.
#
#
# # Multiple outputs, several taps values - Recurrent Neural Network wth scan
#
# # The example above showed simple uses of scan. However, scan also supports
# # referring not only to the prior result and the current sequence value,
# # but also looking back more than one step.
# #
# # This is needed, for example, to implement a RNN using scan. Assume that
# # our RNN is defined as follows:
# # x_{n} = \tanh(
# #   Wx_{n-1}+
# #   W^{in}_{1}u_{n}+
# #   W^{in}_{2}u_{n-4}+
# #   W^{feedback}y_{n-1}
# # )
# # y_{n} = W^{out}x_{n-3}
# # Note that this network is far from a classical recurrent network and
# # might be useless. The reason  we defined as such is to better illustrate
# # the features of scan.
# #
# # In this case we have a sequence over which we need to iterate u, and two
# # outputs x and y. To implement this with scan we first construct a
# # function that computes one iteration step:
# def oneStep(u_tm4, u_t, x_tm3, x_tm1,
#             y_tm1, W, W_in_1, W_in_2, W_feedback, W_out):
#     x_t = T.tanh(
#         theano.dot(x_tm1, W) +
#         theano.dot(u_t, W_in_1) +
#         theano.dot(u_tm4, W_in_2) +
#         theano.dot(y_tm1, W_feedback)
#     )
#     y_t = theano.dot(x_tm3, W_out)
#     return [x_t, y_t]
# # As naming convention for the variables we used a _tmb to mean a at t-b
# # and a_tpb to be a at b+b. Note the order in which the parameter are
# # given, and in which the result is returned. Try to respect
# # chronological order among the taps (time slices of sequences or outputs)
# # used. For scan is crucial only for the variables representing the
# # different time taps to be in the same order as the one in which these
# # taps should respect an order, but also variables, since this is how scan
# # figures out what should be represented by what. Given that we have all
# # the Theano avriables needed we construct our RNN as follows:
# W = T.matrix()
# W_in_1 = T.matrix()
# W_in_2 = T.matrix()
# W_feedback = T.matrix()
# W_out = T.matrix()
#
# u = T.matrix()  # it is a sequence of vectors
# # initial state of x has to be a matrix, since it has to cover x[-3]
# x0 = T.matrix()
# # y0 is just a vector since scan has only to provide y[-1]
# y0 = T.vector()
#
# ([x_vals, y_vals], updates) = theano.scan(
#     fn=oneStep,
#     sequences=dict(input=u, taps=[-4, -0]),
#     outputs_info=[dict(initial=x0, taps=[-3,-1]), y0],
#     non_sequences=[W, W_in_1, W_in_2, W_feedback, W_out],
#     strict=True
# )  # for second input y, scan adds -1 in output_taps by default
#
# # Now x_vals and y_vals are symbolic variables pointing to the sequence of
# # x and y values generated by iterating over u. The sequence_taps,
# # output_taps give to scan information about what slices are exactly
# # needed. Note that if we want to use x[t-k] we do not need to also have
# # x[t-(k-1)] x[t-(k-2)], ..., but when applying the compiled function, the
# # numpy array given to represent this sequence should be large enough to
# # cover this values. Assume that we compile the above function, and we give
# # as u the array uvals = [0,,1,2,3,4,5,6,7,8]. By abusing notations, scan
# # will consider uval[0] as u[-4], and will start scaning from uvals[4]
# # towards the end.


# Conditional ending of scan

# Scan can also be used as a repeat-until block. In such a case scan will
# stop when either the maximal number of iteration is reached, or the
# provided condition evaluates to True.
#
# For an example, we will compute all powers of two smaller then some
# provided value max_value.
import theano
import theano.tensor as T


def power_of_2(previous_power, max_value):
    return previous_power*2, theano.scan_module.until(
            previous_power*2 > max_value)

max_value = T.scalar()
values, _ = theano.scan(
    power_of_2,
    outputs_info=T.constant(1.),
    non_sequences=max_value,
    n_steps=1024
)

f = theano.function([max_value], values)

print(f(45))
# [  2.   4.   8.  16.  32.  64.]

# As you can see, in order to terminate on condition, the only thing
# required is that the inner function power_of_2 to return also the
# condition wrapped in the class theano.scan_module.until. The condition
# has to be exptesssed in terms of the arguments of the inner function (in
# this case previous_power and max_value).
#

# As a rule, scan always, expects the condition to be the last thing
# returned by the inner function, otherwise an error will be raised.
#
# Optimizing Scan's performance
#
# This section covers some ways to improve performance of a Theano function
# using Scan.
#
# Minimizing Scan usage
#
# Scan makes it possible to define simple and compact graphs that can do
# same work as much larger and more complicated graphs. However, it comes
# with a significant overhead. As such, when performance is the objective,
# a good rule of thumb is to perform as much of the computation as
# possible outside of Scan. This may have the effect of increasing memory
# usage but can also reduce the overhead introduces by using Scan.
#
# Explicitly passing inputs of the inner function to scan
#
# It is possible, inside of Scan, to use variables previously defined
# outside of Scan without explicitly passing them as inputs to the Scan.
# However, it is often more efficient to explicitly pass them as
# non-sequence inputs instead. Section Using shared variables-Gibbs
# sampling provides an explanation for this and section Using shared
# variables-the strict flag describes the strict flag, a tool that Scan
# provides to help ensure that the inputs to the function inside Scan have
# all been provided as explicit inputs to the scan() function.
#
#
# Deativating garbage collecting in Scan
#
# Deactivating the garbage collection for Scan can allow it to reuse memory
# between executions instead of always having to allocate new memory
# between iterations of the same execution but frees the momory after the
# last iteration.
#
# There are two ways to achieve this, using the Theano flag
#   config.scan.allow_gc and setting it to False, or using the argument
#   allow_gc of the function theano.scan() and set it to False (when a
# value is not provided for this argument, the value of the flag
# config.scan.allow_gc is used).
#
#
# Graph optimizations
#
# This one is simple but still worth pointing out. Theano is able to auto-
# matically recognize and optimize many computation patterns. However,
# there are patterns that Theano doesn't optimize because doing so would
# change the user interface (such as merging shared variable together into
# a single one, for instance). Additionaly, Theano doesn't catch every case
# that it could optimize and so it remains useful for performance that the
# user defines and efficient graph in the first place. This is also the
# case, and sometimes even more so, for the graph inside of Scan. This is
# because it will be executed many times for every execution of the Theano
# function that contain it.
#
# The LSTM tutorial on Deeplearning.net provides an example of an
# optimization that Theano cannot perform. Instead of performing many
# matrix multiplications between matrix x_{t} and each of the shared
# matrix W_{i}, W_{c}, W_{f} and W_{o}, the matrices W_{*} are merged into
# a single shared matrix W and the graph performs a single larger matrix
# multiplication between W and x_{t}. The resulting matrix is then sliced
# to obtain the results of the small individual matrix multiplications
# would have produced. This optimization replaces several small and
# inefficient matrix multiplications by a single larger one and thus
# improves performance at the cost of a potentially higher memory usage.
