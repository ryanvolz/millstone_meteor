import numpy as np
from scipy.constants import c
import xarray as xr

import rkl
from time_utils import datetime_from_float, datetime_to_float
from valarg import valargmax


def matched_filter(tx, rx, rmin_km=None, rmax_km=None):
    """Frequency bank of matched filters for a single pulse.

    Filter rx with tx, keeping data only from ranges rmin_km to rmax_km.

    Output mf_rx = 2D array (delay, frequency).

    """
    fs = rx.samples_per_second
    fc = rx.center_frequencies

    # convert range bounds to bounds on the delay in number of samples
    if rmin_km is None:
        delay_min = rx.delay.values[0]
    else:
        delay_min = int(np.floor((2 * fs * rmin_km * 1000) / c))
    if rmax_km is None:
        delay_max = rx.delay.values[-1] + 1
    else:
        delay_max = (
            int(np.floor((2 * fs * rmax_km * 1000) / c)) + 1
        )  # +1 so non-inclusive

    # indexes into rx array that correspond to desired delay window
    rx_start = max(delay_min - rx.delay.values[0], 0)
    # want extra len(tx)-1 samples past delay_max so full correlation can be done
    rx_stop = min(delay_max + (tx.shape[0] - 1) - rx.delay.values[0], rx.shape[0])

    rx_corr = rx.values[rx_start:rx_stop]
    # normalize tx data so that noise level is the same after matched filtering
    # when the noise is white
    tx_normalized = tx.values / np.linalg.norm(tx.values)

    # perform matched filter calculation as (delay and multiply) + (FFT)
    y = rkl.delay_multiply.delaymult_like_arg2(rx_corr, tx_normalized, R=1)
    z = np.fft.fft(y)

    # calculate coordinates for matched filtered data
    delays = np.arange(-(tx.shape[0] - 1), rx_corr.shape[0]) + rx.delay.values[rx_start]
    freqs = np.fft.fftfreq(tx.shape[0], 1 / fs)

    # subset matched filtered data to desired delay bounds
    mfrx_start = max(delay_min - delays[0], 0)
    mfrx_stop = min(delay_max - delays[0], delays.shape[0])
    delays = delays[mfrx_start:mfrx_stop]
    z_subset = z[mfrx_start:mfrx_stop, :]

    # calculate derived coordinates
    ranges = delays * (c / (2 * fs))
    vels = -freqs * (c / (2 * fc))  # positive frequency shift is negative range rate

    mf_rx = xr.DataArray(
        z_subset,
        coords=dict(
            t=rx.t.values,
            delay=("delay", delays, {"label": "Delay (samples)"}),
            range=("delay", ranges.astype(np.float_), {"label": "Range (m)"}),
            frequency=(
                "frequency",
                freqs.astype(np.float_),
                {"label": "Doppler frequency shift (Hz)"},
            ),
            range_rate=(
                "frequency",
                vels.astype(np.float_),
                {"label": "Range rate (m/s)"},
            ),
            noise_power=rx.noise_power.data,
        ),
        dims=("delay", "frequency"),
        name="mf_rx",
        attrs=dict(
            sample_rate_denominator=rx.sample_rate_denominator,
            sample_rate_numerator=rx.sample_rate_numerator,
            samples_per_second=rx.samples_per_second,
            center_frequencies=rx.center_frequencies,
        ),
    )
    return mf_rx


def detect_meteors(mf_rx, snr_thresh, vmin_kps, vmax_kps):
    """Meteor signal detection for a single pulse.

    Returns a list of detected meteor points in time-range-velocity space.

    """
    snr_vals = (
        mf_rx.values.real ** 2 + mf_rx.values.imag ** 2
    ) / mf_rx.noise_power.values

    vmax_kps_detect = vmax_kps * 4 / 3
    vmin_kps_detect = vmin_kps * 4 / 3
    v_valid = (-mf_rx.range_rate.values <= vmax_kps_detect * 1e3) & (
        -mf_rx.range_rate.values >= vmin_kps_detect * 1e3
    )
    freq_idx_map = np.arange(v_valid.shape[0])[v_valid]
    # for now, only detect based on highest snr point in delay-frequency space
    snr_sub = snr_vals[:, v_valid]

    # for now, only detect based on highest SNR point in delay-frequency space
    # within the detection Doppler shift range specified
    snr, snr_idx = valargmax(snr_sub)
    delay_idx, freq_idx = np.unravel_index(snr_idx, snr_sub.shape)
    freq_idx = freq_idx_map[freq_idx]

    # detection conditions based on SNR threshold and velocity window
    # v is negative and m/s, vmin and vmax are positive and km/s
    v = mf_rx.range_rate.values[freq_idx]
    if snr >= snr_thresh and -vmax_kps <= v / 1e3 <= -vmin_kps:
        meteor = dict(
            t=datetime_to_float(mf_rx.t.values),
            r=mf_rx.range.values[delay_idx],
            v=v,
            snr=snr,
        )
        return [meteor], freq_idx
    else:
        return [], 0


def summarize_meteor(events, rmeas_var=1, vmeas_var=1, debug=False):
    """Calculate some statistics on a head echo cluster and summarizes it."""
    cols = [
        "start_t",
        "end_t",
        "duration",
        "pulse_num",
        "n_detections",
        "range",
        "range_var",
        "range_rate",
        "range_rate_var",
        "snr_mean",
        "snr_peak",
        "snr_var",
        "rcs_mean",
        "rcs_peak",
        "rcs_var",
    ]
    if events is None:
        return cols

    N = len(events.index)
    t = events.t.values
    r = events.r.values
    v = events.v.values
    snr = events.snr.values
    rcs = events.rcs.values

    d = {}
    d["start_t"] = datetime_from_float(t[0], "ms")
    d["end_t"] = datetime_from_float(t[-1], "ms")
    d["duration"] = t[-1] - t[0]
    d["n_detections"] = N
    d["pulse_num"] = events.pulse_num.values[0]
    d["snr_mean"] = np.mean(snr)
    d["snr_var"] = np.var(snr)
    d["snr_peak"] = np.max(snr)
    d["rcs_mean"] = np.mean(rcs)
    d["rcs_var"] = np.var(rcs)
    d["rcs_peak"] = np.max(rcs)

    A1 = np.concatenate((np.ones(N), np.zeros(N)))
    A2 = np.concatenate((t - t[0], np.ones(N)))
    A = np.stack((A1, A2), axis=1)
    y = np.concatenate((r, v))
    w = np.concatenate(
        (np.ones_like(r) / np.sqrt(rmeas_var), np.ones_like(v) / np.sqrt(vmeas_var))
    )
    x, residuals, rank, s = np.linalg.lstsq(A * w[:, np.newaxis], y * w, rcond=None)
    yfit = np.matmul(A, x)
    rfit = yfit[: len(r)]
    vfit = yfit[len(r) :]
    r0, v0 = x
    d["range"] = r0
    d["range_var"] = np.var(r - rfit)
    d["range_rate"] = v0
    d["range_rate_var"] = np.var(v - vfit)

    if debug:
        yhat = np.dot(A, x)
        import matplotlib
        import matplotlib.pyplot as plt

        formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fig, axarr = plt.subplots(2, 1)
        axarr[0].plot(t - t[0], r / 1e3, ".b", label="Data")
        axarr[0].plot(t - t[0], yhat[:N] / 1e3, "or", label="Fit")
        axarr[0].yaxis.set_major_formatter(formatter)
        axarr[0].legend()
        axarr[0].set_ylabel("Range (km)")
        axarr[1].plot(t - t[0], v / 1e3, ".b", label="Data")
        axarr[1].plot(t - t[0], yhat[N:] / 1e3, "or", label="Fit")
        axarr[1].yaxis.set_major_formatter(formatter)
        axarr[1].legend()
        axarr[1].set_ylabel("Range Rate (km/s)")
        axarr[1].set_xlabel("Time (s)")
        fig.tight_layout()
        plt.show()

    return d
