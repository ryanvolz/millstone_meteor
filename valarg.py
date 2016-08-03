#-----------------------------------------------------------------------------
# Copyright (c) 2014, Ryan Volz
# All rights reserved.
#
# Distributed under the terms of the BSD 3-Clause ("BSD New") license.
#
# The full license is in the LICENSE file, distributed with this software.
#-----------------------------------------------------------------------------

# things I wish were in numpy
import numpy as np

__all__ = ['valargcmp', 'valargmax', 'valargmin']

def valargcmp(arr, axis=None, fun=np.argmax):
    args = fun(arr, axis)
    if axis is None:
        vals = arr.ravel()[args]
    else:
        idx = list(np.ix_(*[xrange(k) for k in args.shape]))
        idx.insert(axis, args)
        vals = arr[idx].squeeze()

    return vals, args

def valargmax(arr, axis=None):
    return valargcmp(arr, axis, np.argmax)

def valargmin(arr, axis=None):
    return valargcmp(arr, axis, np.argmin)
