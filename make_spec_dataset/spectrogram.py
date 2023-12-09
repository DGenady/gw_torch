import scipy.signal as scisig
import numpy as np
from config.conf import conf
data_conf = conf['data_config']

def make_spectrogram(data, To=2):
    if np.isnan(data.value).any():
        print('Nan in chunk')
        return

    data = data - data.mean()
    data.highpass(frequency=data_conf['highpass_freq'], filtfilt=True)

    Nc = len(data)
    Tc = Nc * data.dt.value
    window = scisig.tukey(M=Nc, alpha=1.0 * To / Tc, sym=True)
    data = data * window

    duration = data_conf['spec_duration']
    tres = duration / 256
    fres = 256

    qt = data.q_transform(tres=tres, fres=fres, **data_conf['q_transform'])
    qt = qt[int(To / duration / qt.dt.value):-int(To / duration / qt.dt.value)]

    two_sec_slice = int(2 / tres)
    num_of_slices = qt.shape[0] // two_sec_slice

    qts = np.empty((num_of_slices, 256, 256))
    times = np.empty(num_of_slices)
    for i in range(num_of_slices):
        qts[i] = qt[i * two_sec_slice:(i + 1) * two_sec_slice].value
        times[i] = qt[i * two_sec_slice:(i + 1) * two_sec_slice].t0.value
        qts[i] = qts[i] + np.abs(qts[i].min())
        qts[i] = qts[i] / qts[i].max()

    return qts, times
