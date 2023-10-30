# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)
import math

import numpy as np
from numba import njit, prange


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
def generate_kernels(input_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0

    for i in range(num_kernels):
        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings


# @njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64)")
# def generate_kernels_v2(input_length, num_kernels):
#     candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
#     length = np.random.choice(candidate_lengths, 1)
#     lengths = np.repeat(length, num_kernels)
#
#     weights = np.zeros(lengths.sum(), dtype=np.float64)
#     biases = np.zeros(num_kernels, dtype=np.float64)
#
#     dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (length - 1)))
#     dilations = np.repeat(np.int32(dilation), num_kernels)
#
#     padding = ((length - 1) * dilations[0]) // 2 if np.random.randint(2) == 1 else 0
#     paddings = np.repeat(padding, num_kernels)
#
#     a1 = 0
#
#     for i in range(num_kernels):
#         _length = lengths[i]
#
#         _weights = np.random.normal(0, 1, _length)
#
#         b1 = a1 + _length
#         weights[a1:b1] = _weights - _weights.mean()
#
#         biases[i] = np.random.uniform(-1, 1)
#
#         # dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
#         # dilation = np.int32(dilation)
#         # dilations[i] = dilation
#         #
#         # padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
#         # paddings[i] = padding
#
#         a1 = b1
#
#     return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1
    return _max


#    return _ppv / output_length, _max

@njit(fastmath=True)
def ts_features(arr):
    total = 0.0
    count = 0

    min_val = arr[0]
    max_val = arr[0]

    for item in arr:
        total += item
        count += 1
        if item < min_val:
            min_val = item
        if item > max_val:
            max_val = item

    avg = total / count

    variance = 0.0
    for item in arr:
        variance += (item - avg) ** 2
    variance = variance / count

    # Calculate the standard deviation
    std_deviation = variance ** 0.5
    return avg, std_deviation, min_val, max_val

# removing here ppv and max-pooling
@njit(fastmath=True)
def apply_kernel_v2(X, weights, length, bias, dilation, padding):
    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    features = []
    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if index > -1 and index < input_length:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation
        features.append(_sum)
        # if _sum > _max:
        #     _max = _sum
        #
        # if _sum > 0:
        #     _ppv += 1

    avg, std, min, max = ts_features(features)

    return avg, std, min, max



def smoothed_average(features, window):
    smoothed_features = []
    for i in range(0, len(features), window):
        smoothed_features.append(np.mean(features[i:(i + window)]))
    return smoothed_features


@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel=True,
      fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float64)  # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0  # for weights
        a2 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
                apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X


@njit("float64[:,:](float64[:,:],int32,Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))", parallel=True,
      fastmath=True)
def apply_kernels_v2(X, num_features, kernels):
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * num_features), dtype=np.float64)  # 2 features per kernel

    # smoothed_X = []
    for i in prange(num_examples):

        a1 = 0  # for weights
        a2 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + num_features

            _X[i, a2:b2] = apply_kernel_v2(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X
