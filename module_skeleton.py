import numpy as np
import rkl
import math

# interferance and clutter removal

# determines the index of the strongest signal
def strongest_signal(signal):
    f = np.max(np.abs(signal), axis=0)
    return np.argmax(f)

# frequency bank of matched filters for a single pulse
def freq_dft(array1, array2):
    y = rkl.delay_multiply.delaymult_like_arg2(array2, array1, R=1)
    z = np.fft.fft(y)
    maxfreqidx = strongest_signal(z)
    f = np.fft.fftfreq(array1.shape[0], 1e-6) 
    new_array1 = np.array(y[0], dtype=complex)
    data = []
    for q in range(0, len(z[1])):
         data.append(f[q])
    new_array2 = np.array(np.asarray(data))
    saved_data = z[:, maxfreqidx]
    new_array3 = np.array((new_array1, new_array2))
    return new_array3, saved_data

"""
# meteor signal detection for a single pulse
# need the signal data for snr threshold
def meteor_present(array, thres, fmin, fmax):
    if np.max(20*np.log10(np.abs(saved_data))) >= thres:
        if fmin < array[1] < fmax:
            return meteor_list = [..., ..., array[1]]


# event clustering function
# ts, rs, fs, threshold have not been defined...
# currently identifies which meteor detections(list) are below the threshold
def cluster(list_of_data):
    dist_list = []
    for i in range(0, len(list_of_data)-1):
        y = np.asarray(list_of_data[i])
        t1 = y[0]
        r1 = y[1]
        f1 = y[2]
        y2 = np.asarray(list_of_data[i+1])
        t2 = y2[0]
        r2 = y2[1]
        f2 = y2[2]
        dist = math.sqrt((t1 - t2)**2/ts + (r1 - r2)**2/rs + (f1 - f2)**2/fs)
        if dist < ...:
            dist_list.append(list_of_data[i])  

# summarize the results of the previous functions
def summary(list_of_meteors): """

   
            
       

        
