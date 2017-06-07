#!/usr/bin/env python

"""A script to run on Digital RF data to detect and summarize meteors."""
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import scipy.stats
import os
import sys
import csv

import digital_rf_hdf5 as drf
import digital_metadata as dmd
import TimingModeManager
from clustering import Clustering
import meteor_processing as mp

noise_pwr_rv = sp.stats.chi2(2)
#med_pwr_est_factor = noise_pwr_rv.mean()/noise_pwr_rv.median()
med_pwr_est_factor = noise_pwr_rv.mean()/noise_pwr_rv.ppf(
                         0.5*noise_pwr_rv.cdf(3*noise_pwr_rv.median()))

def interval_range(start, stop, step=1):
    iterable = iter(xrange(start, stop, step))
    a = b = iterable.next()
    for b in iterable:
        yield (a, b)
        a = b
    yield (b, stop)

def pulse_generator(ido, no, tmm, s0, s1, nch, ds=None):
    fs = ido.get_samples_per_second()
    if ds is None:
        # needs to be long enough to get at least one noise metadata sample
        ds = int(5*fs)

    rasters = {}
    ntattr = 'temp_{0}_median'.format(nch)
    npattr = '{0}_off_100_kHz_median'.format(nch)

    for ss, se in interval_range(s0, s1, ds):
        noisemd = xr.Dataset.from_dataframe(
            pd.DataFrame.from_dict(no.read(ss-ds, se, [ntattr, npattr]), orient='index')
        )

        idmd = ido.read(ss, se, 'sweepid')
        for sp, sweepid in idmd.iteritems():
            # skip all but uncoded pulses for now
            #if sweepid != 300:
                #continue
            try:
                raster_table = rasters[sweepid]
            except KeyError:
                sweep = tmm.getTimingSweepByID(sweepid)
                code = sweep.getCodeObject()
                raster_table = code.getRasterTable()
                rasters[sweepid] = raster_table

            offset = -int(np.round(raster_table['full'][0]*fs/1e9))

            noise_attrs = noisemd.sel(index=sp, method='ffill').data_vars
            noise_dict = dict(
                noise_temp=noise_attrs[ntattr].item(),
                noise_power=noise_attrs[npattr].item()*med_pwr_est_factor*fs/100e3,
            )

            yield sp+offset, sweepid, raster_table, noise_dict

def data_generator(rfo, ido, no, tmm, s0, s1, rxch, txch):
    rx_attrs = {k: v.value for k, v, in rfo.get_metadata(rxch).iteritems()}
    tx_attrs = {k: v.value for k, v, in rfo.get_metadata(txch).iteritems()}

    rxfs = rx_attrs['sample_rate']
    txfs = tx_attrs['sample_rate']

    nch = rxch.split('-')[0]

    for s, sweepid, raster, noise in pulse_generator(ido, no, tmm, s0, s1, nch):
        t = np.round(s * (1e9/rxfs)).astype('datetime64[ns]')

        tx_raster = (int(np.round(raster['tx'][0]*txfs/1e9)),
                     int(np.round(raster['tx'][1]*txfs/1e9)))
        tx_data = rfo.read_vector_c81d(int(np.round(s*txfs/rxfs)) + tx_raster[0],
                                       tx_raster[1] - tx_raster[0], txch)

        rx_raster = (int(np.round(raster['blank'][1]*rxfs/1e9)),
                     int(np.round(raster['signal'][1]*rxfs/1e9)))
        rx_data = rfo.read_vector_c81d(s + rx_raster[0],
                                       rx_raster[1] - rx_raster[0], rxch)

        tx_attrs.update(sweepid=sweepid)
        tx = xr.DataArray(
            tx_data,
            coords=dict(t=t,
                        delay=('delay', np.arange(tx_raster[0], tx_raster[1]),
                               {'label': 'Delay (samples)'}),
                   ),
            dims=('delay',),
            name='tx',
            attrs=tx_attrs,
        )

        rx_attrs.update(noise)
        rx = xr.DataArray(
            rx_data,
            coords=dict(t=t,
                        delay=('delay', np.arange(rx_raster[0], rx_raster[1]),
                               {'label': 'Delay (samples)'}),
                   ),
            dims=('delay',),
            name='rx',
            attrs=rx_attrs,
        )

        yield tx, rx

def detect_meteors(rf_dir, id_dir, noise_dir, output_dir,
        t0=None, t1=None, rxch='zenith-l', txch='tx-h',
        snr_thresh=1, rmin_km=70, rmax_km=140, vmin_kps=7, vmax_kps=72,
        eps=0.5, min_samples=5, tscale=1, rscale=1, vscale=1,
        debug=False,
    ):
    """Function to detect and summarize meteor head echoes.


    Arguments
    ---------

    rf_dir : string or list
        RF data directory or directories.

    id_dir : string
        ID code metadata directory.

    noise_dir : string
        RX noise metadata directory.

    output_dir : string
        Meteor data output directory.

    t0 : float, optional
        Start time, seconds since epoch. If None, start at beginning of data.

    t1 : float, optional
        End time, seconds since epoch. If None, end at end of data.

    rxch : string, optional
        Receiver channel to process.

    txch : string, optional
        Transmitter channel.

    """
    # set up reader objects for data and metadata
    rfo = drf.read_hdf5(rf_dir)
    ido = dmd.read_digital_metadata(id_dir)
    no = dmd.read_digital_metadata(noise_dir)

    # infer time window to process based on bounds of data and metadata
    if t0 is None or t1 is None:
        bounds = []
        bounds.append(rfo.get_bounds(rxch))
        bounds.append(rfo.get_bounds(txch))
        bounds.append(ido.get_bounds())
        bounds.append(no.get_bounds())
        bounds = np.asarray(bounds)

        ss = np.max(bounds[:, 0])
        se = np.min(bounds[:, 1])

        fs = rfo.get_metadata(rxch)['sample_rate'].value

        if t0 is None:
            s0 = ss
        else:
            s0 = int(np.round(t0*fs))

        if t1 is None:
            s1 = se
        else:
            s1 = int(np.round(t1*fs))

    # load pulse/coding information
    tmm = TimingModeManager.TimingModeManager()
    if os.path.exists('/tmp/tmm.hdf5'):
        tmm.loadFromHdf5('/tmp/tmm.hdf5', skip_lowlevel=True)
    else:
        tmm.loadFromHdf5(skip_lowlevel=True)

    # initalize generator that steps through data pulse by pulse
    pulse_data = data_generator(rfo, ido, no, tmm, s0, s1, rxch, txch)

    # initialize clustering object for grouping detections
    clustering = Clustering(eps, min_samples, tscale, rscale, vscale)

    # initialize CSV file for saving meteor clusters
    csvpath = os.path.join(output_dir, 'cluster_summaries.txt')
    csvfile = open(csvpath, "wb", 1) # 1 => use line buffering
    cols = mp.summarize_meteor(None)
    csvwriter = csv.DictWriter(csvfile, cols)
    csvwriter.writeheader()

    # loop that steps through data one pulse at a time
    for k, (tx, rx) in enumerate(pulse_data):
        # marching periods as status update
        if (k % 100) == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

        # matched filter
        mf_rx = mp.matched_filter(tx, rx, rmin_km, rmax_km)

        # meteor signal detection
        meteors = mp.detect_meteors(mf_rx, snr_thresh, vmin_kps, vmax_kps)

        # clustering of detections into single meteor head echoes
        for meteor in meteors:
            sys.stdout.write('*')
            sys.stdout.flush()
            new_clusters = clustering.addnext(pulse_num=k, **meteor)
            for c in new_clusters:
                sys.stdout.write('{0}'.format(c.cluster.values[0]))
                # summarize head echo and save to a data file
                cluster_summary = mp.summarize_meteor(c, debug=debug)
                csvwriter.writerow(cluster_summary)

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

    parser.add_argument(
        'rf_dir', nargs='+',
        help='RF Data directory or directories.',
    )
    parser.add_argument(
        '-i', '--id_dir', required=True,
        help='ID code metadata directory.',
    )
    parser.add_argument(
        '-n', '--noise_dir', required=True,
        help='RX noise metadata directory.',
    )
    parser.add_argument(
        '-o', '--output_dir', required=True,
        help='Meteor data output directory.',
    )
    parser.add_argument(
        '-0', '--t0', type=float, default=None,
        help='Start time, seconds since epoch. (default: beginning of data)',
    )
    parser.add_argument(
        '-1', '--t1', type=float, default=None,
        help='End time, seconds since epoch. (default: end of data)',
    )
    parser.add_argument(
        '-c', '--rxch', default='zenith-l',
        help='Receiver channel to process. (default: %(default)s)',
    )
    parser.add_argument(
        '-t', '--txch', default='tx-h',
        help='Transmitter channel. (default: %(default)s)',
    )
    parser.add_argument(
        '--snr_thresh', type=float, default=1,
        help='SNR detection threshold. (default: %(default)s)',
    )
    parser.add_argument(
        '--rmin', type=float, default=70,
        help='Lower boundary of range window for detection (km). (default: %(default)s)',
    )
    parser.add_argument(
        '--rmax', type=float, default=140,
        help='Upper boundary of range window for detection (km). (default: %(default)s)',
    )
    parser.add_argument(
        '--vmin', type=float, default=7,
        help='Lower limit of range rate for detection (km/s). (default: %(default)s)',
    )
    parser.add_argument(
        '--vmax', type=float, default=72,
        help='Upper limit of range rate for detection (km/s). (default: %(default)s)',
    )
    parser.add_argument(
        '--eps', type=float, default=20,
        help='Size of neighborhood for purposes of clustering. (default: %(default)s)',
    )
    parser.add_argument(
        '--min_samples', type=float, default=4,
        help='Minimum number of points in neighborhood that constitutes a cluster. (default: %(default)s)',
    )
    parser.add_argument(
        '--tscale', type=float, default=0.002,
        help='Distance scale of time units (s). (default: %(default)s)',
    )
    parser.add_argument(
        '--rscale', type=float, default=150,
        help='Distance scale of range units (m). (default: %(default)s)',
    )
    parser.add_argument(
        '--vscale', type=float, default=3787,
        help='Distance scale of range rate units (m/s). (default: %(default)s)',
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Turn on debugging (activates interactive plots, etc.).',
    )

    a = parser.parse_args()

    detect_meteors(
        a.rf_dir, a.id_dir, a.noise_dir, a.output_dir,
        a.t0, a.t1, a.rxch, a.txch,
        snr_thresh=a.snr_thresh, rmin_km=a.rmin, rmax_km=a.rmax,
        vmin_kps=a.vmin, vmax_kps=a.vmax, eps=a.eps, min_samples=a.min_samples,
        tscale=a.tscale, rscale=a.rscale, vscale=a.vscale, debug=a.debug,
    )

