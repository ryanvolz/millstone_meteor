#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc

import plotting


def plot_mf_snr(mf_ds):
    # use an average noise value for SNR for plotting so pulse-to-pulse variations
    # in noise are visible if they exist
    avg_noise_pwr = np.median(mf_ds.noise_power.values)
    snr = (mf_ds.mf_rx.values.real**2 + mf_ds.mf_rx.values.imag**2) / avg_noise_pwr

    width, height, dpi = plotting.size_dpi_nointerp(snr.shape[0], snr.shape[1], 8, 4.5)
    fig, ax = plt.subplots(figsize=(width + 1.5, height + 1.5), dpi=dpi)
    plotting.make_axes_fixed(ax, width, height)

    plotting.rtiplot(
        10 * np.log10(snr),
        mf_ds.t.values,
        mf_ds.alt.values / 1e3,
        ax=ax,
        xlabel="Time of pulse",
        ylabel="Altitude (km)",
        clabel="SNR (db)",
        title="Doppler-matched filtered data",
        xistime=True,
        exact_ticks=False,
        vmin=-5,
        vmax=30,
    )
    fig.tight_layout()

    return fig, dpi


def plot_mf_hsv(mf_ds):
    # use an average noise value for SNR for plotting so pulse-to-pulse variations
    # in noise are visible if they exist
    avg_noise_pwr = np.median(mf_ds.noise_power.values)
    snr = (mf_ds.mf_rx.values.real**2 + mf_ds.mf_rx.values.imag**2) / avg_noise_pwr
    snr_db = 10 * np.log10(snr)
    snr_normalization = plt.Normalize(vmin=-5, vmax=30, clip=True)
    value = snr_normalization(snr_db)

    freq_max = mf_ds.attrs["samples_per_second"] / 2
    saturation = plt.Normalize(vmin=-freq_max, vmax=freq_max, clip=True)(
        np.sqrt(mf_ds.freq_var.values)
    )
    hue = plt.Normalize(vmin=-freq_max, vmax=freq_max)(mf_ds.freq_mean.values)

    width, height, dpi = plotting.size_dpi_nointerp(snr.shape[0], snr.shape[1], 8, 4.5)
    fig, ax = plt.subplots(figsize=(width + 1.5, height + 1.5), dpi=dpi)
    plotting.make_axes_fixed(ax, width, height)

    plotting.rtiplot(
        mpl.colors.hsv_to_rgb(np.stack((hue, saturation, value), axis=-1)),
        mf_ds.t.values,
        mf_ds.alt.values / 1e3,
        ax=ax,
        xlabel="Time of pulse",
        ylabel="Altitude (km)",
        clabel="SNR (db)",
        title="Doppler-matched filtered data",
        xistime=True,
        exact_ticks=False,
        norm=snr_normalization,
        cmap="binary_r",
    )
    fig.tight_layout()

    return fig, dpi


def plot_mf_doppler(mf_ds, estimator="max_pwr", range_rate=False, freq_max=None):
    if estimator == "max_pwr":
        doppler_freqs = mf_ds.freq_max_pwr.values
    elif estimator == "mean":
        doppler_freqs = mf_ds.freq_mean.values
    else:
        raise ValueError(f"estimator {estimator} must be one of 'max_pwr' or 'mean'")
    if freq_max is None:
        freq_max = mf_ds.attrs["samples_per_second"] / 2
    if range_rate:
        rr = -doppler_freqs * (spc.c / (2 * mf_ds.attrs["center_frequencies"]))
        val = rr / 1e3
        val_max = freq_max * (spc.c / (2 * mf_ds.attrs["center_frequencies"])) / 1e3
        clabel = "Range rate (km/s)"
    else:
        val = doppler_freqs / 1e3
        val_max = freq_max / 1e3
        clabel = "Doppler frequency (kHz)"

    width, height, dpi = plotting.size_dpi_nointerp(val.shape[0], val.shape[1], 8, 4.5)
    fig, ax = plt.subplots(figsize=(width + 1.5, height + 1.5), dpi=dpi)
    plotting.make_axes_fixed(ax, width, height)

    plotting.rtiplot(
        val,
        mf_ds.t.values,
        mf_ds.alt.values / 1e3,
        ax=ax,
        xlabel="Time of pulse",
        ylabel="Altitude (km)",
        clabel=clabel,
        title=f"Estimated Doppler frequency shift (estimator = {estimator})",
        xistime=True,
        exact_ticks=False,
        vmin=-val_max,
        vmax=val_max,
        cmap="coolwarm",
    )
    fig.tight_layout()

    return fig, dpi


if __name__ == "__main__":
    import argparse
    import os
    import pathlib

    import xarray as xr

    plt.switch_backend("Agg")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datasets",
        type=pathlib.Path,
        nargs="*",
        help="Matched filtered data mf@*.h5 dataset files.",
    )
    parser.add_argument("--savedir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing plots.",
    )
    parser.add_argument(
        "--nosnr",
        dest="snr",
        action="store_false",
        default=True,
        help="Do not create an SNR plot (default: False).",
    )
    parser.add_argument(
        "--hsv",
        action="store_true",
        help="Create HSV color plot using Doppler frequency mean and stddev and SNR.",
    )
    parser.add_argument(
        "--freq_max_pwr",
        action="store_true",
        help="Create plot of Doppler frequency estimated by filter bank maximum power match.",
    )
    parser.add_argument(
        "--freq_mean",
        action="store_true",
        help="Create plot of Doppler frequency estimated by filter bank mean.",
    )
    parser.add_argument(
        "--rr_max_pwr",
        action="store_true",
        help="Create plot of Doppler range rate estimated by filter bank maximum power match.",
    )
    parser.add_argument(
        "--rr_mean",
        action="store_true",
        help="Create plot of Doppler range rate estimated by filter bank mean.",
    )
    parser.add_argument(
        "--max_freq",
        default=100e3,
        type=float,
        help="Frequency value to use for vmin/vmax of freq/range rate plots.",
    )

    args = parser.parse_args()

    if args.savedir is None:
        commonpath = pathlib.Path(os.path.commonpath(args.datasets))
        if len(args.datasets) == 1 or commonpath == args.datasets[0].parent:
            args.savedir = args.datasets[0].parent / "figures"
        else:
            args.savedir = pathlib.Path("figures")
    args.savedir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(args.datasets):
        mf_ds = xr.open_dataset(filepath, engine="h5netcdf")

        savename_template = "{id}_" + filepath.stem + ".png"

        if args.snr:
            savepath = args.savedir / savename_template.format(id="snr")
            if not savepath.exists() or args.overwrite:
                fig, dpi = plot_mf_snr(mf_ds)
                fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
        if args.hsv:
            savepath = args.savedir / savename_template.format(id="hsv")
            if not savepath.exists() or args.overwrite:
                fig, dpi = plot_mf_hsv(mf_ds)
                fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
        if args.freq_max_pwr:
            savepath = args.savedir / savename_template.format(id="freq_max_pwr")
            if not savepath.exists() or args.overwrite:
                fig, dpi = plot_mf_doppler(
                    mf_ds, estimator="max_pwr", range_rate=False, freq_max=args.max_freq
                )
                fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
        if args.freq_mean:
            savepath = args.savedir / savename_template.format(id="freq_mean")
            if not savepath.exists() or args.overwrite:
                fig, dpi = plot_mf_doppler(
                    mf_ds, estimator="mean", range_rate=False, freq_max=args.max_freq
                )
                fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
        if args.rr_max_pwr:
            savepath = args.savedir / savename_template.format(id="rr_max_pwr")
            if not savepath.exists() or args.overwrite:
                fig, dpi = plot_mf_doppler(
                    mf_ds, estimator="max_pwr", range_rate=True, freq_max=args.max_freq
                )
                fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
        if args.rr_mean:
            savepath = args.savedir / savename_template.format(id="rr_mean")
            if not savepath.exists() or args.overwrite:
                fig, dpi = plot_mf_doppler(
                    mf_ds, estimator="mean", range_rate=True, freq_max=args.max_freq
                )
                fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)
