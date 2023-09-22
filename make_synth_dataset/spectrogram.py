from gwpy.timeseries import timeseries
from gwpy.signal.window import recommended_overlap
import scipy.signal as scisig
import numpy as np

def make_spectrogram(data, Tc=256, To=2, Q=(16, 16)):
    """

        takes TimeSeries data from gwpy package and retruns an array of 2 second spectrograms


        input:
        - data : gwpy.timeseries.Timeseries
        - Tc : duration of the input chunk
        - To : overlap between chunks
        - Q : the range of Q used in the Q-transform

        output:
        - tuple of an array of spectrograms, SNR_offset, SNR_norm, times,
        - spectrograms is a Nx256x256 array where N is the number of spectrograms
        - the outputed spectrogram is normalized to have values between 0 to 1.
        - SNR_offset is the spectrogram minimum before normalization, numpy array of Nx1
        - SNR_norm is the spectrogram maximum before normalization, numpy arry of Nx1

    """
    if np.isnan(data.value).any():
        print('Nan in chunk')
        return

    data = data - data.mean()
    data.highpass(frequency=20, filtfilt=True)

    Nc = len(data)
    Tc = Nc * data.dt.value
    window = scisig.tukey(M=Nc, alpha=1.0 * To / Tc, sym=True)
    data = data * window

    window = 'hann'
    fftlength = timeseries._fft_length_default(data.dt)
    overlap = None
    method = "median"  # supposed to be mean?
    if fftlength == data.duration.value:
        method = "median"
        overlap = 0
    else:
        overlap = recommended_overlap(window) * fftlength

    ASD = data.asd(fftlength, overlap, window=window, method=method)

    with np.errstate(all='raise'):
        whitened_data = data.whiten(asd=ASD, fduration=2,
                                    highpass=None)

    frq = int(1 / whitened_data.dt.value)

    # desired spectrogram duration and resolution
    duration = 2
    tres = duration / 256
    fres = 256

    # Q transform sections
    # returns the first spectorgram only
    original_size = 4
    cropped = whitened_data[0:original_size * frq]
    qt = cropped.q_transform(frange=(10, 2048), qrange=Q, whiten=False, tres=tres, fres=fres, logf=True)
    qt = qt[fres // 2:-fres // 2]
    SNR_offset = qt.min()
    qt += np.abs(qt.min())

    SNR_norm = qt.max()
    specs = (qt / qt.max()).value
    times = qt.t0.value

    return specs, SNR_offset, SNR_norm, times