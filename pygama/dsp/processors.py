"""
Contains a list of dsp processors used by the legend experiment, implemented
using numba's guvectorize to implement numpy's ufunc interface. In other words,
all of the functions are void functions whose outputs are given as parameters.
The ufunc interface provides additional information about the function
signatures that enables broadcasting the arrays and SIMD processing. Thanks to
the ufunc interface, they can also be called to return a numpy array, but if
 this is done, memory will be allocated for this array, slowing things down.
"""

# I think there's a way to do this recursively, but I'll figure it out later...
from ._processors.mean_stdev import mean_stdev
<<<<<<< HEAD
from ._processors.pole_zero import pole_zero
=======
from ._processors.pole_zero import pole_zero, double_pole_zero
>>>>>>> Add files via upload
from ._processors.trap_filter import trap_filter
from ._processors.current import avg_current
from ._processors.asym_trap_filter import asymTrapFilter
from ._processors.fixed_time_pickoff import fixed_time_pickoff
from ._processors.trap_norm import trap_norm
from ._processors.trap_pickoff import trap_pickoff
from ._processors.time_point_frac import time_point_frac
from ._processors.time_point_thresh import time_point_thresh
<<<<<<< HEAD
from ._processors.linear_fit import linear_fit
from ._processors.zac_filter import zac_filter
=======
from ._processors.time_point_frac import time_point_frac
>>>>>>> Update remote
from ._processors.param_lookup import param_lookup
from ._processors.cusp_filter import cusp_filter
from ._processors.fftw import dft, inv_dft, psd
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from ._processors.integral import sum_wf
from ._processors.presum import presum
=======
from ._processors.presum import presum
from ._processors.integral import sum_wf
>>>>>>> Update processors.py
=======
from ._processors.presum import presum
from ._processors.integral import sum_wf
>>>>>>> Fixed calibration function. Added processors to .json config file
=======
from ._processors.linear_slope_fit import linear_slope_fit
from ._processors.log_check import log_check
from ._processors.min_max import min_max
from ._processors.presum import presum
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> Add files via upload
=======
from ._processors.integral import sum_wf
>>>>>>> Modified analysis files
=======
from ._processors.integral import sum_wf
from ._processors.windower import windower
from ._processors.bl_subtract import bl_subtract
from ._processors.convolutions import cusp_filter, zac_filter, t0_filter
from ._processors.trap_filters import trap_filter, trap_norm, asym_trap_filter, trap_pickoff
from ._processors.moving_windows import moving_window_left, moving_window_right, moving_window_multi, avg_current
from ._processors.soft_pileup_corr import soft_pileup_corr, soft_pileup_corr_bl
from ._processors.optimize import optimize_1pz, optimize_2pz
from ._processors.saturation import saturation
<<<<<<< HEAD
>>>>>>> modified analysis files
=======
from ._processors.gaussian_filter1d import gaussian_filter1d
from ._processors.get_multi_local_extrema import get_multi_local_extrema
from ._processors.multi_t_filter import multi_t_filter, remove_duplicates
from ._processors.multi_a_filter import multi_a_filter
from ._processors.Wiener_filter import Wiener_filter
from ._processors.pulse_injector import inject_sig_pulse,inject_exp_pulse
from ._processors.dwt import dwt 
>>>>>>> Added dwt processor
