import numpy as np
import rkl
import math
import xarray as xr
from time_utils import datetime_to_float
import pandas as pd

# interferance and clutter removal

# frequency bank of matched filters for a single pulse
# mf_rx = 2D array (delay, frequency)
def freq_dft(array1, array2):
    y = rkl.delay_multiply.delaymult_like_arg2(array2.values, array1.values/np.sum(array1.values), R=1)
    z = np.fft.fft(y)
    global f
    f = np.fft.fftfreq(array1.shape[0], 1e-6)
    
    array = np.arange(array2.delay.values[0]-(len(array1)-1), array2.delay.values[0])
    delay_array = np.append(array, array2.delay.values)

    mf_rx = xr.DataArray(z, coords=dict(t=array2.t.values, delay=('delay', delay_array, {'label': 'Delay (samples)'}), frequency=('frequency', f)), dims=('delay', 'frequency',), name='mf_rx')
    global fs
    fs = array2.sample_rate
    return mf_rx

# meteor signal detection for a single pulse
# need to include range in output
def is_there_a_meteor(data, snr_val, snr_idx, thres, fmin, fmax):
    meteor_list = []
    if snr_val >= thres:
        if fmin < f[snr_idx[1]] < fmax:
            # returns object's time, range, frequency at max SNR
            signal_range = (3e8*data.delay.values[snr_idx[0]])/(2*fs)
    	    t = datetime_to_float(data.t.values)
            info = (t, signal_range, f[snr_idx[1]])
            meteor_list.extend(info)
    return meteor_list 

# work-in progress 
def summary(events):
    d = {}
    idx = []
    initial_t = []
    duration = []
    initial_r = []
    range_rate = []
    for i in range(0, len(events)):
        initial_t.append(events[i]['t'][0])
        t = (events[i]['t'][events[i]['t'].shape[0] - 1] - events[i]['t'][0])
        duration.append(t)
        initial_r.append(events[i]['r'][0])
        rrate = (events[i]['r'][events[i]['r'].shape[0] - 1] - events[i]['r'][0])/t
        range_rate.append(rrate)
        idx.append(i + 1)

    d['initial t'] = initial_t
    d['duration'] = duration
    d['initial r'] = initial_r
    d['range rate'] = range_rate
    cluster_summary = pd.DataFrame(d, index=[list(idx)])
    return cluster_summary


