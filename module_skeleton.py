import numpy as np
import rkl
import math
import xarray as xr
from time_utils import datetime_to_float, datetime_from_float
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
    	    t = datetime_to_float(data.t.values)
            signal_range = (3e8*data.delay.values[snr_idx[0]])/(2*fs)
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
    snr_mean = []
    snr_var = []
    snr_peak = []
    range_rates = []
    lstsq = []
    for i in range(0, len(events)):
        t0 = datetime_from_float(events[i]['t'][0], 'ms')
        initial_t.append(t0)
        t = events[i]['t'][events[i]['t'].shape[0] - 1] - events[i]['t'][0]
        duration.append(t)
        initial_r.append(events[i]['r'][0])
        range_rate.append((events[i]['r'][0] - events[i]['r'][events[i]['r'].shape[0] - 1])/t)
        idx.append(i + 1)
        snr_mean.append(np.mean(events[i]['snr']))
        snr_var.append(np.var(events[i]['snr']))
        snr_peak.append(np.max(events[i]['snr']))
        rr2 = []
        def rr(data, start):
            if start < (data[i]['t'].shape[0] - 1):
                rr1 = (data[i]['r'][start] - data[i]['r'][start + 1])
                dt1 = np.abs((data[i]['t'][start] - data[i]['t'][start + 1]))
                y = rr1/dt1
                rr2.append(y)
                rr(data, start + 1)
        rr(events, 0)
        range_rates.append(rr2)
        A = np.vstack([events[i]['t'] - events[i]['t'][0], np.ones(events[i]['t'].shape[0])]).T
        m, c = np.linalg.lstsq(A, events[i]['r'])[0]
        lstsq.append(m)
    d['initial t'] = initial_t
    d['duration'] = duration
    d['initial r'] = initial_r
    d['range rate'] = range_rate
    d['snr mean'] = snr_mean
    d['snr var'] = snr_var
    d['snr peak'] = snr_peak
    d['range rates'] = range_rates
    d['lstsq'] = lstsq 
    cluster_summary = pd.DataFrame(d, index=[list(idx)])
    return cluster_summary


