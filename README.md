Scripts for processing Millstone Hill radar data to detect and analyze meteor head echoes.

# Installation
First, get a copy of this repository by either downloading it or using `git clone`:

```
git clone https://github.com/ryanvolz/millstone_meteor
```

For installating dependencies and managing environments, I recommend [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html). `micromamba` is a self-contained executable that can be used in place of `conda` to create environments and install packages. You can simply download the [latest version for your platform](https://github.com/mamba-org/micromamba-releases/releases/) and run directly it with the typical `conda` commands: `./micromamba activate ENV`. But the gentlest approach is to follow the [installation instructions provided by prefix.dev](https://prefix.dev/docs/mamba/introduction#installation).

Once you have micromamba/mamba/conda, create and activate an environment for running the `millstone_meteor` code:

```
micromamba create -n meteor
micromamba activate meteor
micromamba config prepend channels conda-forge
micromamba config set channel_priority strict
```

Then from within the environment you just created, install the following dependencies:

```
micromamba install dask digital_rf h5netcdf ipython matplotlib numba numpy pandas scipy xarray zarr
```

The `millstone_meteor` code should now be able to run from within the activated conda environment.


# Running matched filtering and meteor detection
The processing is run by either by importing the `detect_meteors` module and running the `detect_meteors(...)` function, or by running `detect_meteors.py` as a terminal script. Command line arguments for the script can be viewed by running:

```
python detect_meteors.py --help
```

An example command is as follows:

```
python detect_meteors.py --snr_thresh=20 --pointing_el 13.3 --amin 70 --amax 120 --vmin=-72 --vmax 72 --eps 20 --min_samples 8 --tscale 0.0041 --rscale 150 --vscale 262.1 -c misa-l -t misa-l --offsets "10000001=15" -i /data/2022-08-12_meteor/id_metadata -o /data/2022-08-12_meteor/output --tmm /data/tmm.hdf5 /data/2022-08-12_meteor
```
