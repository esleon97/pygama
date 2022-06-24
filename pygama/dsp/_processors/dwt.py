import numpy as np
from numba import guvectorize
from pywt import wavedec

def dwt(level, wavelet):
    """
    Apply a discrete wavelet transform to the waveform and return only 
    the approximate coefficients. Note that it is composed of a
    factory function that is called using the init_args argument and that
    the function the waveforms are passed to using args.

    Initialization Parameters
    -------------------------
    level   : int
              The length of the filter to be convolved
    wavelet : float
              The wavelet type for convolution ('haar', 'db', ...)

    Parameters
    ----------
    w_in : array-like
           The input waveform
    w_out: array-like
           The approximate coefficients 

    Processing Chain Example
    ------------------------
    "dwt":{
        "function": "dwt",
        "module": "pygama.dsp.processors",
        "args": ["wf_blsub", "dwt"],
        "unit": "ADC",
        "prereqs": ["wf_blsub"],
        "init_args": ["3", "haar"]
        }
    """
        
    @guvectorize(["void(float32[:], float32[:])",
                  "void(float64[:], float64[:])"],
                 "(n),(m)", forceobj=True)
    def dwt_out(wf_in, wf_out):
        
        wf_out[:] = np.nan
        coeffs = wavedec(wf_in, wavelet, level=level) #always rounds up number of samples 
        wf_out[:] = coeffs[0][:]
        
    return dwt_out