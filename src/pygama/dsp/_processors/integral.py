import numpy as np
from numba import guvectorize
from scipy.integrate import simps

@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int64, int64, float64[:])"],
             "(n)->()", nopython=True, cache=True)
def rectangle(wf_in, lower, upper, sum_out):
    """
    Calculate the integral of a waveform by summing 
    the waveform value at each sample.
    """
    sum_out[0] = np.nan
    sum_out[0] = np.sum(wf_in[lower:upper], 1)
    
@guvectorize(["void(float32[:], int32, int32, float32[:])",
              "void(float64[:], int64, int64, float64[:])"],
             "(n),(),()->()", forceobj=True)
def simpson(wf_in, lower, upper, int_out):
    """
    Calculate the integral of a waveform using Simpson's rule
    """
    int_out[0] = np.nan
    int_out[0] = simps(wf_in[lower:upper])