import numpy as np
import math
import xarray as xr
from scipy.constants import c

import rkl
from time_utils import datetime_to_float, datetime_from_float
from valarg import valargmax

def matched_filter(tx, rx, rmin_km, rmax_km):
    """Frequency bank of matched filters for a single pulse.

    Filter rx with tx, keeping data only from ranges rmin_km to rmax_km.

    Output mf_rx = 2D array (delay, frequency).

    """
    fs = rx.sample_rate
    fc = rx.center_frequencies

    # convert range bounds to bounds on the delay in number of samples
    delay_min = int(np.floor((2*fs*rmin_km*1000)/c))
    delay_max = int(np.floor((2*fs*rmax_km*1000)/c)) + 1 # +1 so non-inclusive

    # indexes into rx array that correspond to desired delay window
    rx_start = max(delay_min - rx.delay.values[0], 0)
    # want extra len(tx)-1 samples past delay_max so full correlation can be done
    rx_stop = min(delay_max + (tx.shape[0] - 1) - rx.delay.values[0], rx.shape[0])

    rx_corr = rx.values[rx_start:rx_stop]
    # normalize tx data so that noise level is the same after matched filtering
    tx_normalized = tx.values/np.linalg.norm(tx.values)

    # perform matched filter calculation as (delay and multiply) + (FFT)
    y = rkl.delay_multiply.delaymult_like_arg2(rx_corr, tx_normalized, R=1)
    z = np.fft.fft(y)

    # calculate coordinates for matched filtered data
    delays = np.arange(-(tx.shape[0] - 1), rx_corr.shape[0]) + rx.delay.values[rx_start]
    freqs = np.fft.fftfreq(tx.shape[0], 1/fs)

    # subset matched filtered data to desired delay bounds
    mfrx_start = max(delay_min - delays[0], 0)
    mfrx_stop = min(delay_max - delays[0], delays.shape[0])
    delays = delays[mfrx_start:mfrx_stop]
    z_subset = z[mfrx_start:mfrx_stop, :]

    # calculate derived coordinates
    ranges = delays*c/(2*fs)
    vels = -freqs*c/(2*fc) # positive frequency shift is negative range rate

    mf_rx = xr.DataArray(
        z_subset,
        coords=dict(
            t=rx.t.values,
            delay=('delay', delays, {'label': 'Delay (samples)'}),
            range=('delay', ranges, {'label': 'Range (m)'}),
            frequency=('frequency', freqs, {'label': 'Doppler frequency shift (Hz)'}),
            range_rate=('frequency', vels, {'label': 'Range rate (m/s)'}),
        ),
        dims=('delay', 'frequency',),
        name='mf_rx',
        attrs=rx.attrs,
    )
    return mf_rx

def detect_meteors(mf_rx, snr_thresh, vmin_kps, vmax_kps):
    """Meteor signal detection for a single pulse.

    Returns a list of detected meteor points in time-range-velocity space.

    """
    snr_vals = (mf_rx.values.real**2 + mf_rx.values.imag**2)/mf_rx.noise_power

    # for now, only detect based on highest SNR point in delay-frequency space
    snr, snr_idx = valargmax(snr_vals)
    delay_idx, freq_idx = np.unravel_index(snr_idx, snr_vals.shape)
    v = mf_rx.range_rate.values[freq_idx]

    # detection conditions based on SNR threshold and velocity window
    # v is negative and m/s, vmin and vmax are positive and km/s
    if snr >= snr_thresh and -vmax_kps <= v/1e3 <= -vmin_kps:
        meteor = dict(
            t=datetime_to_float(mf_rx.t.values),
            r=mf_rx.range.values[delay_idx],
            v=v,
            snr=snr,
        )
        return [meteor]
    return []

def summarize_meteor(events):
    """Calculates some statistics on a head echo cluster and summarizes it."""
    cols = ['start_t', 'end_t', 'duration', 'snr_mean', 'snr_var', 'snr_peak',
            'rcs_mean', 'rcs_peak', 'rcs_var', 'pulse_num',
            'range', 'range_var', 'range_rate', 'range_rate_var']
    if events is None:
        return cols

    N = len(events.index)
    t = events.t.values
    r = events.r.values
    v = events.v.values
    snr = events.snr.values
    rcs = events.rcs.values

    d = {}
    d['start_t'] = datetime_from_float(t[0], 'ms')
    d['end_t'] = datetime_from_float(t[-1], 'ms')
    d['duration'] = t[-1] - t[0]
    d['snr_mean'] = np.mean(snr)
    d['snr_var'] = np.var(snr)
    d['snr_peak'] = np.max(snr)
    d['rcs_mean'] = np.mean(rcs)
    d['rcs_var'] = np.var(rcs)
    d['rcs_peak'] = np.max(rcs)
    d['pulse_num'] = events.pulse_num.values[0]
    A1 = np.append(np.ones(N), np.zeros(N))
    A2 = np.append(t - t[0], np.ones(N))
    A = np.vstack([A1, A2]).T
    x, residuals, rank, s = np.linalg.lstsq(A, np.append(r, v))
    r0, v0 = x
    d['range'] = r0
    d['range_var'] = np.var(r)
    d['range_rate'] = v0
    d['range_rate_var'] = np.var(v)

    return d
