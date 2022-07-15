import numpy as np
from numba import guvectorize
from scipy.integrate import simps

@guvectorize(["void(float32[:,:], float32[:])",
              "void(float64[:,:], float64[:])"],
             "(n, m),(n)", nopython=True, cache=True)
def rectangle(wf_in, sum_out):
    """
    Calculate the integral of a waveform by summing 
    its value at each sample.
    """
    n, m = wf_in.shape
    sum_out[:] = np.sum(wf_in[:,:], 1)
    
@guvectorize(["void(float32[:,:], int32, int32, float32[:])",
              "void(float64[:,:], int64, int64, float64[:])"],
             "(n, m),(),(),(n)", forceobj=True)
def simpson(wf_in, lower, upper, int_out):
    """
    Calculate the integral of a waveform using Simpson's rule
    """
    n, m = wf_in.shape
    int_out[:] = simps(wf_in[:,lower:upper])