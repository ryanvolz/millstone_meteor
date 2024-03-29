#!/usr/bin/env python

"""A script to run on Digital RF data to detect and summarize meteors."""
import ast
import collections
import csv
import datetime
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import six
import xarray as xr

import digital_rf as drf
import meteor_processing as mp
from clustering import Clustering

import warnings
import h5py

warnings.simplefilter("ignore", category=h5py.h5py_warnings.H5pyDeprecationWarning)

noise_pwr_rv = stats.chi2(2)
# med_pwr_est_factor = noise_pwr_rv.mean() / noise_pwr_rv.median()
# particular to way outliers are removed and then median is taken for noise estimate
med_pwr_est_factor = noise_pwr_rv.mean() / noise_pwr_rv.ppf(
    0.5 * noise_pwr_rv.cdf(3 * noise_pwr_rv.median())
)


def interval_range(start, stop, step=1):
    iterable = iter(range(start, stop, step))
    a = b = six.next(iterable)
    for b in iterable:
        yield (a, b)
        a = b
    yield (b, stop)


def pulse_generator(ido, tmm_hdf5, s0, s1, ds=None, offsets=None):
    fs = ido.get_samples_per_second()
    if ds is None:
        # needs to be long enough to get at least one noise metadata sample
        ds = int(5 * fs)

    if offsets is None:
        offsets = {}
    codes = {}
    rasters = {}
    with h5py.File(tmm_hdf5, "r") as tmm_file:
        for code_acronym, code_group in tmm_file["codes"].items():
            code_list = code_group["code_list"][()]
            bauds = (code_list["I"] + 1j * code_list["Q"]).astype(np.complex64)
            code_idx = (
                np.arange(0, code_group.attrs["pulse_length"], 1e9 / fs)
                / code_group.attrs["baud_length"]
            ).astype(np.int_)
            sampled_codes = bauds[:, code_idx]

            raster_dict = {
                r.decode(): (a, b) for (r, a, b) in code_group["raster_list"]
            }

            base_id_code = code_group.attrs["base_id_code"]
            sweep_count = code_group.attrs["sweep_count"]
            for sweep_k in range(sweep_count):
                sweepid = base_id_code + sweep_k
                codes[sweepid] = sampled_codes[sweep_k, :]
                rasters[sweepid] = raster_dict

    for ss, se in interval_range(s0, s1, ds):
        idmd = ido.read(ss, se, "sweepid")
        for ks, sweepid in idmd.items():
            code = codes[sweepid]
            raster_table = rasters[sweepid]

            full_rasters = tuple(
                int(np.round(s * fs / 1e9)) for s in raster_table["full"]
            )

            # adjust ks to get to start of pulse by subtracting index 0 of full raster
            idx_shift = full_rasters[0]
            # most modes seem to have an offset of about 14 us from what the
            # raster says it should be
            offset = offsets.get(sweepid, int(np.round(14e-6 * fs)))

            pulse_sample_idx = ks - idx_shift + offset

            if pulse_sample_idx + full_rasters[1] <= se:
                # only yield a pulse with full raster inside the desired sample window
                yield pulse_sample_idx, sweepid, raster_table, code


def noise_generator(no, s0, s1, ds=None, columns=None):
    fs = no.get_samples_per_second()
    if ds is None:
        # needs to be long enough to get at least one noise metadata sample
        ds = int(5 * fs)

    for ss, se in interval_range(s0, s1, ds):
        noisemd = xr.Dataset.from_dataframe(
            pd.DataFrame.from_dict(no.read(ss - ds, se, columns), orient="index")
        )

        yield noisemd, ss, se


def data_generator(
    rfo, ido, no, tmm_hdf5, s0, s1, rxch, txch=None, offsets=None, debug=False
):
    ch_list = [rxch]
    if txch is not None:
        ch_list.append(txch)
    ch_attr_list = [rfo.get_properties(ch) for ch in ch_list]
    rx_attrs = ch_attr_list[0]

    for ch, attrs in zip(ch_list, ch_attr_list):
        mdo = rfo.get_digital_metadata(ch)
        md_bounds = mdo.get_bounds()
        md_dict = mdo.read(md_bounds[0], md_bounds[1])
        md_idx = sorted(md_dict.keys())
        # get index into md_idx so that md_idx[md_loc] is less than
        # or equal to sstart and md_idx[md_loc] is greater than sstart
        # (this gives the metadata value assuming a forward fill)
        md_loc = np.searchsorted(md_idx, s0, side="right") - 1
        md = md_dict[md_idx[md_loc]]
        attrs.update(md)

    rxfs = ch_attr_list[0]["samples_per_second"]

    if no is not None:
        nch = rxch.split("-")[0]
        ntattr = "temp_{0}_median".format(nch)
        npattr = "{0}_off_100_kHz_median".format(nch)

        noise_gen = noise_generator(no, s0, s1, columns=[ntattr, npattr])
        noisemd, nss, nse = six.next(noise_gen)
    else:
        noise_ests = collections.OrderedDict()

    for s, sweepid, raster, code in pulse_generator(
        ido, tmm_hdf5, s0, s1, offsets=offsets
    ):
        t = np.round(s * (1e9 / rxfs)).astype("datetime64[ns]")

        if no is not None:
            if s >= nse:
                noisemd, nss, nse = six.next(noise_gen)

            noise_attrs = noisemd.sel(index=s, method="ffill").data_vars
            noise = dict(
                noise_temp=noise_attrs[ntattr].item(),
                noise_power=noise_attrs[npattr].item()
                * med_pwr_est_factor
                * float(rxfs / 100e3),
            )
        else:
            noise_raster = tuple(
                int(np.round(ks * rxfs / 1e9)) for ks in raster["noise"]
            )
            try:
                noise_data = rfo.read_vector_1d(
                    s + noise_raster[0], noise_raster[1] - noise_raster[0], rxch
                )
            except IOError:
                noise_data = np.zeros(
                    noise_raster[1] - noise_raster[0],
                    dtype=np.complex64,
                )
            noise_pwr = noise_data.real ** 2 + noise_data.imag ** 2
            noise_pwr_med = np.median(noise_pwr)
            # spike removal
            noise_data[noise_pwr > 3 * noise_pwr_med] = np.nan
            # decimate to 100 kHz
            dec = int(rxfs // 100e3)
            noise_dec = np.nan_to_num(
                np.nanmean(
                    noise_data[: (len(noise_data) // dec * dec)].reshape((-1, dec)),
                    axis=1,
                )
            )
            # estimate power from median
            noise_est = (
                med_pwr_est_factor
                * np.median(noise_dec.real ** 2 + noise_dec.imag ** 2)
                * float(rxfs / 100e3)
            )
            # FUTURE WORK: would prefer not having to average noise over pulses here
            noise_ests[s] = noise_est
            for samp in list(noise_ests.keys()):
                if (s - samp) / rxfs > 1.0:
                    noise_ests.pop(samp)
                else:
                    break
            noise = dict(noise_power=np.median(list(noise_ests.values())))

        rx_raster = (
            int(np.round(raster["blank"][1] * rxfs / 1e9)),
            int(np.round(raster["signal"][1] * rxfs / 1e9)),
        )
        try:
            rx_data = rfo.read_vector_1d(
                s + rx_raster[0], rx_raster[1] - rx_raster[0], rxch
            )
        except IOError:
            rx_data = np.zeros(rx_raster[1] - rx_raster[0], dtype=np.complex64)

        rx_attrs.update(noise)
        # FUTURE WORK: would prefer to use a Dataset here with tx, rx, noise as arrays
        rx = xr.DataArray(
            rx_data,
            coords=dict(
                t=t,
                delay=(
                    "delay",
                    np.arange(
                        rx_raster[0],
                        rx_raster[1],
                        dtype=np.result_type(
                            np.min_scalar_type(rx_raster[0]),
                            np.min_scalar_type(rx_raster[1]),
                        ),
                    ),
                    {"label": "Delay (samples)"},
                ),
            ),
            dims=("delay",),
            name="rx",
            attrs=rx_attrs,
        )

        if txch is not None:
            tx_attrs = ch_attr_list[1]
            txfs = tx_attrs["samples_per_second"]

            tx_raster = (
                int(np.round(raster["tx"][0] * txfs / 1e9)),
                int(np.round(raster["tx"][1] * txfs / 1e9)),
            )
            try:
                tx_data = rfo.read_vector_1d(
                    int(np.round(s * txfs / rxfs)) + tx_raster[0],
                    tx_raster[1] - tx_raster[0],
                    txch,
                )
            except IOError:
                tx_data = np.zeros(tx_raster[1] - tx_raster[0], dtype=np.complex64)
                tx_data[0] = 1

            tx_attrs.update(sweepid=sweepid)
            tx = xr.DataArray(
                tx_data,
                coords=dict(
                    t=t,
                    delay=(
                        "delay",
                        np.arange(
                            tx_raster[0],
                            tx_raster[1],
                            dtype=np.result_type(
                                np.min_scalar_type(tx_raster[0]),
                                np.min_scalar_type(tx_raster[1]),
                            ),
                        ),
                        {"label": "Delay (samples)"},
                    ),
                ),
                dims=("delay",),
                name="tx",
                attrs=tx_attrs,
            )
        else:
            tx = xr.DataArray(
                code,
                coords=dict(
                    t=t,
                    delay=(
                        "delay",
                        np.arange(
                            0,
                            len(code),
                            dtype=np.min_scalar_type(len(code)),
                        ),
                        {"label": "Delay (samples)"},
                    ),
                ),
                dims=("delay",),
                name="tx",
                attrs=dict(sweepid=sweepid),
            )

        if debug:
            import matplotlib.pyplot as plt

            fig, axarr = plt.subplots(2, 1)
            np.abs(tx).plot.line(ax=axarr[0])
            np.abs(rx).plot.line(ax=axarr[1])
            fig.tight_layout()
            plt.show()
            # turn off debug plotting for subsequent loops
            debug = False

        yield s, tx, rx


def detect_meteors(
    rf_dir,
    output_dir,
    id_dir,
    tmm_hdf5,
    noise_dir=None,
    t0=None,
    t1=None,
    rxch="zenith-l",
    txch=None,
    snr_thresh=1,
    pointing_el=90,
    amin_km=70,
    amax_km=140,
    vmin_kps=-72,
    vmax_kps=72,
    eps=0.5,
    min_samples=5,
    tscale=0.001,
    rscale=150,
    vscale=3787,
    offsets=None,
    resume=True,
    save_mf_bank=False,
    debug=False,
):
    """Function to detect and summarize meteor head echoes.

    Arguments
    ---------

    rf_dir : string or list
        RF data directory or directories.

    output_dir : string
        Meteor data output directory.

    id_dir : string
        ID code metadata directory.

    tmm_hdf5 : string
        Path to timing mode HDF5 file.

    noise_dir : string
        RX noise metadata directory.

    t0 : float, optional
        Start time, seconds since epoch. If None, start at beginning of data.

    t1 : float, optional
        End time, seconds since epoch. If None, end at end of data.

    rxch : string, optional
        Receiver channel to process.

    txch : string, optional
        Transmitter channel. If None, use the ideal code.

    """
    # set up reader objects for data and metadata
    rfo = drf.DigitalRFReader(rf_dir)
    ido = drf.DigitalMetadataReader(id_dir)
    if noise_dir is not None:
        no = drf.DigitalMetadataReader(noise_dir)
    else:
        no = None

    rxfs = rfo.get_properties(rxch)["samples_per_second"]

    mf_output_dir = pathlib.Path(output_dir) / f"mf_{rxch}"

    if not save_mf_bank:
        mf_output_basename = "mf"
    else:
        mf_output_basename = "mfbank"

    # if resuming, get start time from already processed data
    if t0 is None and resume:
        mf_file_paths = sorted(
            mf_output_dir.glob("{0}@*.h5".format(mf_output_basename))
        )
        for mf_path in reversed(mf_file_paths):
            try:
                last_mf_ds = xr.open_dataset(mf_path, engine="h5netcdf")
            except Exception:
                continue
            t0 = np.round(pd.Timestamp(last_mf_ds.t.values[-1]).timestamp())
            break

    # infer time window to process based on bounds of data and metadata
    if t0 is None or t1 is None:
        bounds = []
        bounds.append(rfo.get_bounds(rxch))
        if txch is not None:
            bounds.append(rfo.get_bounds(txch))
        bounds.append(ido.get_bounds())
        if no is not None:
            bounds.append(no.get_bounds())
        bounds = np.asarray(bounds)

        ss = np.max(bounds[:, 0])
        se = np.min(bounds[:, 1])

    if t0 is None:
        s0 = ss
    else:
        s0 = int(np.uint64(t0 * rxfs))

    if t1 is None:
        s1 = se
    else:
        s1 = int(np.uint64(t1 * rxfs))

    # convert altitudes to ranges
    rng2alt = np.cos(np.pi / 2 - pointing_el * np.pi / 180)
    rmin_km = amin_km / rng2alt
    rmax_km = amax_km / rng2alt

    # ensure output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initalize generator that steps through data pulse by pulse
    pulse_data = data_generator(
        rfo, ido, no, tmm_hdf5, s0, s1, rxch, txch, offsets=offsets, debug=debug
    )

    # initialize clustering object for grouping detections
    clustering = Clustering(eps, min_samples, tscale, rscale, vscale)

    # initialize CSV file for saving meteor clusters
    timestr = str(datetime.datetime.utcnow().replace(microsecond=0))
    timestr = timestr.replace(":", "-").replace(" ", "_")
    csvpath = os.path.join(
        output_dir, "cluster_summaries_{ch}_v{t}.txt".format(ch=rxch, t=timestr)
    )
    csvfile = open(csvpath, "w", 1)  # 1 => use line buffering
    cols = mp.summarize_meteor(None)
    csvwriter = csv.DictWriter(csvfile, cols)
    csvwriter.writeheader()

    # initialize storage and settings for saving Doppler-shifted matched filter results
    save_rx = []
    mf_output_dir.mkdir(parents=True, exist_ok=True)
    curwrite_t = (
        np.round(s0 * (1e9 / rxfs)).astype("datetime64[ns]").astype("datetime64[s]")
    )
    tdelta = np.timedelta64(10, "s")
    nextwrite_t = curwrite_t + tdelta

    # loop that steps through data one pulse at a time
    for k, (s, tx, rx) in enumerate(pulse_data):
        # marching periods as status update
        if (k % 100) == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        t = np.round(s * (1e9 / rxfs)).astype("datetime64[ns]")

        # matched filter
        mf_rx = mp.matched_filter(tx, rx, rmin_km, rmax_km)
        mf_rx["alt"] = mf_rx.range * rng2alt

        # meteor signal detection
        meteors, freq_idx = mp.detect_meteors(mf_rx, snr_thresh, vmin_kps, vmax_kps)
        mf_rx_best = mf_rx.isel(frequency=freq_idx)

        if not save_mf_bank:
            # summary frequency computations, power-weighted frequency moments
            mf_rx_pwr = mf_rx.real ** 2 + mf_rx.imag ** 2
            mf_rx_max_pwr_freq = mf_rx_pwr.idxmax(dim="frequency").reset_coords(
                "range_rate", drop=True
            )
            mf_rx_max_pwr_freq.attrs[
                "label"
            ] = "Doppler frequency of maximum power (Hz)"
            mf_rx_freq_moment_0 = mf_rx_pwr.sum(dim="frequency")
            mf_rx_freq_moment_0.attrs["label"] = "Total power across frequency"
            mf_rx_freq_moment_1 = (mf_rx_pwr * mf_rx_pwr.coords["frequency"]).sum(
                dim="frequency"
            ) / mf_rx_freq_moment_0
            mf_rx_freq_moment_1.attrs["label"] = "Mean power-weighted frequency (Hz)"
            mf_rx_freq_moment_2 = (
                mf_rx_pwr * (mf_rx_pwr.coords["frequency"] - mf_rx_freq_moment_1) ** 2
            ).sum(dim="frequency") / mf_rx_freq_moment_0
            mf_rx_freq_moment_2.attrs[
                "label"
            ] = "Variance of power-weighted frequency (Hz)"

            # combine matched filtered data and summary computations into single dataset
            mf_ds = xr.Dataset(
                data_vars=dict(
                    mf_rx=mf_rx_best,
                    freq_mean=mf_rx_freq_moment_1,
                    freq_var=mf_rx_freq_moment_2,
                    freq_max_pwr=mf_rx_max_pwr_freq,
                ),
                attrs=mf_rx.attrs,
            )
        else:
            mf_ds = mf_rx.to_dataset()

        # clustering of detections into single meteor head echoes
        for meteor in meteors:
            sys.stdout.write("*")
            sys.stdout.flush()
            new_clusters = clustering.addnext(pulse_num=k, **meteor)
            for c in new_clusters:
                sys.stdout.write("{0}".format(c.cluster.values[0]))
                # summarize head echo and save to a data file
                cluster_summary = mp.summarize_meteor(
                    c, rmeas_var=rscale ** 2, vmeas_var=vscale ** 2, debug=debug
                )
                csvwriter.writerow(cluster_summary)

        # save result of doppler-shifted matched filter
        if t >= nextwrite_t:
            # join and write previous data
            save_data = xr.concat(save_rx, "t")

            timestr = np.datetime_as_string(curwrite_t)
            timestr = timestr.replace(":", "-").replace(" ", "_")
            fname = "{basename}@{t}.h5".format(basename=mf_output_basename, t=timestr)
            filepath = mf_output_dir / fname

            save_data.to_netcdf(
                filepath,
                mode="w",
                format="NETCDF4",
                engine="h5netcdf",
                invalid_netcdf=True,
            )
            save_data.close()
            del save_data

            save_rx.clear()
            save_rx.append(mf_ds)
            curwrite_t = nextwrite_t
            nextwrite_t = curwrite_t + tdelta
        else:
            save_rx.append(mf_ds)

    # join and write previous data
    save_data = xr.concat(save_rx, "t")

    timestr = np.datetime_as_string(curwrite_t, unit="s")
    timestr = timestr.replace(":", "-").replace(" ", "_")
    fname = "{basename}@{t}.h5".format(basename=mf_output_basename, t=timestr)
    filepath = mf_output_dir / fname

    save_data.to_netcdf(
        filepath, mode="w", format="NETCDF4", engine="h5netcdf", invalid_netcdf=True
    )
    save_data.close()
    del save_data

    # tell clustering object that data is exhausted and to return any final clusters
    new_clusters = clustering.finish()
    for c in new_clusters:
        # summarize head echo and save to a data file
        cluster_summary = mp.summarize_meteor(c)
        csvwriter.writerow(cluster_summary)

    csvfile.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("rf_dir", nargs="+", help="RF Data directory or directories.")
    parser.add_argument(
        "-i",
        "--id_dir",
        required=True,
        help="ID code metadata directory.",
    )
    parser.add_argument(
        "-n",
        "--noise_dir",
        default=None,
        help="RX noise metadata directory.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Meteor data output directory.",
    )
    parser.add_argument(
        "--tmm",
        dest="tmm_hdf5",
        default="/tmp/tmm.hdf5",
        help="Path to timing mode HDF5 file. (default %(default)s)",
    )
    parser.add_argument(
        "-0",
        "--t0",
        type=float,
        default=None,
        help="Start time, seconds since epoch. (default: beginning of data)",
    )
    parser.add_argument(
        "-1",
        "--t1",
        type=float,
        default=None,
        help="End time, seconds since epoch. (default: end of data)",
    )
    parser.add_argument(
        "-c",
        "--rxch",
        default="zenith-l",
        help="Receiver channel to process. (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--txch", default=None, help="Transmitter channel. (default: %(default)s)"
    )
    parser.add_argument(
        "--snr_thresh",
        type=float,
        default=1,
        help="SNR detection threshold. (default: %(default)s)",
    )
    parser.add_argument(
        "--pointing_el",
        type=float,
        default=90,
        help=(
            "Antenna pointing elevation (from horizon) in Degrees."
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--amin",
        type=float,
        default=70,
        help=(
            "Lower boundary of altitude window for detection (km)."
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--amax",
        type=float,
        default=140,
        help=(
            "Upper boundary of altitude window for detection (km)."
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=7,
        help="Lower limit of range rate for detection (km/s). (default: %(default)s)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=72,
        help="Upper limit of range rate for detection (km/s). (default: %(default)s)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=20,
        help="Size of neighborhood for purposes of clustering. (default: %(default)s)",
    )
    parser.add_argument(
        "--min_samples",
        type=float,
        default=4,
        help=(
            "Minimum number of points in neighborhood that constitutes a cluster."
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tscale",
        type=float,
        default=0.002,
        help="Distance scale of time units (s). (default: %(default)s)",
    )
    parser.add_argument(
        "--rscale",
        type=float,
        default=150,
        help="Distance scale of range units (m). (default: %(default)s)",
    )
    parser.add_argument(
        "--vscale",
        type=float,
        default=3787,
        help="Distance scale of range rate units (m/s). (default: %(default)s)",
    )
    parser.add_argument(
        "--offsets",
        action="append",
        default=None,
        metavar="{SWEEPID}={OFFSET}",
        help="Mode sample offset sweepid=offset pairs. (default: %(default)s)",
    )
    parser.add_argument(
        "--ignore_existing",
        dest="resume",
        default=True,
        action="store_false",
        help=(
            "Ignore existing matched filtered data and do not resume processing."
            " (default: False)"
        ),
    )
    parser.add_argument(
        "--save_mf_bank",
        dest="save_mf_bank",
        default=False,
        action="store_true",
        help=(
            "Save the full results from the bank of matched filters, instead of the"
            " per-pulse highest SNR match. (default: False)"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Turn on debugging (activates interactive plots, etc.).",
    )

    a = parser.parse_args()

    # convert offset strings to a dictionary
    if a.offsets is not None:
        offset_dict = {}
        for arg in a.offsets:
            k, v = arg.split("=")
            try:
                k = ast.literal_eval(k)
            except ValueError:
                pass
            try:
                v = ast.literal_eval(v)
            except ValueError:
                pass
            offset_dict[k] = v
        a.offsets = offset_dict

    detect_meteors(
        rf_dir=a.rf_dir,
        output_dir=a.output_dir,
        id_dir=a.id_dir,
        tmm_hdf5=a.tmm_hdf5,
        noise_dir=a.noise_dir,
        t0=a.t0,
        t1=a.t1,
        rxch=a.rxch,
        txch=a.txch,
        snr_thresh=a.snr_thresh,
        pointing_el=a.pointing_el,
        amin_km=a.amin,
        amax_km=a.amax,
        vmin_kps=a.vmin,
        vmax_kps=a.vmax,
        eps=a.eps,
        min_samples=a.min_samples,
        tscale=a.tscale,
        rscale=a.rscale,
        vscale=a.vscale,
        offsets=a.offsets,
        resume=a.resume,
        save_mf_bank=a.save_mf_bank,
        debug=a.debug,
    )
