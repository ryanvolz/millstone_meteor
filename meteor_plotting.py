import matplotlib.pyplot as plt
import numpy as np

from plotting import rtiplot
from time_utils import datetime_to_float


# plotting windows
def plot(mf, r, t, x, n, header):
    plt.figure()
    rtiplot(
        10 * np.log10(mf),
        t,
        r / 1e3,
        title=header,
        ylabel="Range (km)",
        clabel="SNR (dB)",
        vmin=-10,
        vmax=10,
    )
    plt.tight_layout()
    plt.savefig(n + str(x))


def hist_hits(data):
    times = []
    for i in range(0, len(data)):
        event_time = datetime_to_float(data[i]["initial t"][0])
        times.append(event_time)
    plt.hist(times)
    plt.xlabel("Time (Min)")
    plt.ylabel("Num of Head Echoes")
    plt.title("Number of Hits per Minute")
    plt.grid()
    plt.savefig("hist_hits")


def hist_v(data):
    v_avg = []
    for i in range(0, len(data)):
        v = data["rate rate"][i] / 1e3
        v_avg.append(v)
    plt.hist(v_avg, bins=np.arange(-70, 10, 5))
    plt.xlabel("Range Rate (km/s)")
    plt.ylabel("Num of Head Echoes")
    plt.title("Rate Rate Distribution")
    plt.grid()
    plt.savefig("hist_v")


def hist_r(data):
    r_avg = []
    for i in range(0, len(data)):
        r = data["range"][i] / 1e3
        r_avg.append(r)
    plt.hist(r_avg, bins=np.arange(70, 145, 2))
    plt.xlabel("Range (km)")
    plt.ylabel("Num of Head Echoes")
    plt.title("Range Distribution")
    plt.grid()
    plt.savefig("hist_r")


def hist_snr(data):
    snr_avg = []
    for i in range(0, len(data)):
        snr = 10 * np.log10(data["snr mean"][i])
        snr_avg.append(snr)
    plt.hist(snr_avg, bins=np.arange(0, 70, 2))
    plt.xlabel("SNR (dB)")
    plt.ylabel("Num of Head Echoes")
    plt.title("SNR Distribution")
    plt.grid()
    plt.savefig("hist_snr")
