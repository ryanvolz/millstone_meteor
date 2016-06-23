import numpy as np
import rkl
import math
import xarray as xr
from collections import namedtuple

# interferance and clutter removal

# determines the index of the strongest signal
def strongest_signal(signal):
    f = np.max(np.abs(signal), axis=0)
    return np.argmax(f)

# frequency bank of matched filters for a single pulse
def freq_dft(array1, array2):
    y = rkl.delay_multiply.delaymult_like_arg2(array2.values/array2.noise_power, array1.values, R=1)
    z = np.fft.fft(y)
    maxfreqidx = strongest_signal(z)
    f = np.fft.fftfreq(array1.shape[0], 1e-6)
 
    array = np.arange(array2.delay.values[0]-len(array1), array2.delay.values[0])
    delay_array = np.append(array, array2.delay.values)

    data = []

    for q in range(0, len(z[1])):
         data.append(f[q])

    freq_array = np.array(np.asarray(data))
    snr_vals = (np.abs(array2.values)**2)/array2.noise_power
    Pulse_Info = namedtuple('Pulse_Info', 'delay frequencies data_range snr time')
    drange = (delay_array*3e8/array2.sample_rate*2)/1e3
    event = Pulse_Info(delay=delay_array, frequencies=freq_array, data_range=drange, snr=snr_vals, time=array2.t)

    return event

# meteor signal detection for a single pulse
# need to include range in output
def is_there_a_meteor(named_tuple, thres, fmin, fmax):
    list_of_meteors = []
    gen = (val for val in list(named_tuple[3]) if val >= thres)
    for val in gen:
        val_ind = named_tuple[3].index(val) 
        if fmin < f[val_ind] < fmax:
            # returns object's time, snr, frequency
            meteor_list = [named_tuple[4], f[val_ind], val]
            return meteor_list 

# clustering class then runs

# speed of object = (meteor_list[3]*3e8/(440e6*2))/1e3

"""
# summarize the results of the previous functions
def summary(dict_of_meteors):
    df_meteors = pd.DataFrame(dict_of_meteors, index=['time', 'range', 'velocity', 'max snr', 'frequency'])
    return df_meteors
"""
    
    
        
        
    

   
            
       

        
