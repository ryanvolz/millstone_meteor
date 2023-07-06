#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plotting


def plot_mf(mf_ds):
    # use an average noise value for SNR for plotting so pulse-to-pulse variations
    # in noise are visible if they exist
    avg_noise_pwr = np.median(mf_ds.noise_power.values)
    snr = (mf_ds.mf_rx.values.real ** 2 + mf_ds.mf_rx.values.imag ** 2) / avg_noise_pwr

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
    snr = (mf_ds.mf_rx.values.real ** 2 + mf_ds.mf_rx.values.imag ** 2) / avg_noise_pwr
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
        "--hsv",
        action="store_true",
        help="Create HSV color plot using Doppler frequency mean and stddev and SNR.",
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
        savepath = args.savedir / (filepath.stem + ".png")
        if not args.overwrite and savepath.exists():
            continue

        mf_ds = xr.open_dataset(filepath, engine="h5netcdf")
        if args.hsv:
            fig, dpi = plot_mf_hsv(mf_ds)
        else:
            fig, dpi = plot_mf(mf_ds)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
