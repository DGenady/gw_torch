import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries as cbcTS
from scipy.signal import square
from scipy.signal import chirp, iirdesign, filtfilt
import functools

def projection_decorator(sig_func):
    @functools.wraps(sig_func)
    def inner(*args, **kwargs):
        detector = kwargs.pop('detector')
        dt = kwargs.pop('dt')
        declination = np.random.uniform(-np.pi / 2, np.pi / 2)
        right_ascension = np.random.uniform(0, np.pi)
        polarization = np.random.uniform(0, np.pi)
        hp, hc, params = sig_func(*args, **kwargs)
        hp = cbcTS(hp, delta_t=dt)
        hc = cbcTS(hc, delta_t=dt)

        det = Detector(detector)  # H1 or L1 or V1
        # Choose randomly sky location, and polarization phase for the merger
        # NOTE: Right ascension and polarization phase runs from 0 to 2pi
        #       Declination runs from pi/2. to -pi/2 with the poles at pi/2. and -pi/2.

        params.update({
            "detector": detector,
            "declination": declination,
            "right_ascension": right_ascension,
            "polarization": polarization,
        })
        signal = det.project_wave(hp, hc, right_ascension, declination, polarization)
        signal_series = TimeSeries(signal, dt=dt)
        return signal_series, params

    return inner


def gw_signal(dt, detector, **kwargs):
    mass_1 = np.random.uniform(10, 30)
    mass_2 = np.random.uniform(10, 30)
    declination = np.random.uniform(-np.pi / 2, np.pi / 2)
    right_ascension = np.random.uniform(0, np.pi)
    polarization = np.random.uniform(0, np.pi)
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                             mass1=mass_1,
                             mass2=mass_2,
                             delta_t=dt,
                             f_lower=30)
    det = Detector(detector)  # H1 or L1 or V1
    # Choose randomly sky location, and polarization phase for the merger
    # NOTE: Right ascension and polarization phase runs from 0 to 2pi
    #       Declination runs from pi/2. to -pi/2 with the poles at pi/2. and -pi/2.

    params = {
        "detector": detector,
        "mass_1": mass_1,
        "mass_2": mass_2,
        "declination": declination,
        "right_ascension": right_ascension,
        "polarization": polarization,
    }
    signal = det.project_wave(hp, hc, right_ascension, declination, polarization)
    signal_series = TimeSeries(signal, dt=dt)
    return signal_series, params


def square_signal(dt, w=5, **kwargs):
    signal_duration = np.random.uniform(0.1, 0.2)  # seconds
    t = np.linspace(0, 1, int(signal_duration / dt), endpoint=False)
    square_signal = square(2 * np.pi * w * t) * 1e-19  # roughly the aplitude scale of the noise
    square_signal = TimeSeries(square_signal, dt=dt)
    params = {
        "w": w,
        "signal_duration": signal_duration
    }

    return square_signal, params


def gaussian_env(times, t_inj, tau):
    """Generate gaussian envelope.
    """
    env = np.exp(-(times - t_inj) ** 2 / tau ** 2)
    return env


@projection_decorator
def sine_gaussian(times, t_inj):
    """Generate sine-gaussian waveform.
    """
    f0 = np.random.uniform(100.0, 1000.0)
    Q = np.random.uniform(30.0, 200.0)
    tau = Q / (np.sqrt(2) * np.pi * f0)
    params = {
        "f0": f0,
        "Q": Q,
        "tau": tau
    }
    env = gaussian_env(times, t_inj, tau)
    Hp = env * np.sin(2 * np.pi * f0 * (times - t_inj))
    Hc = env * np.cos(2 * np.pi * f0 * (times - t_inj))
    return Hp, Hc, params


@projection_decorator
def gaussian(times, t_inj, tau):
    """Generate gaussian waveform.
    """
    Hp = gaussian_env(times, t_inj, tau)
    Hc = np.zeros(Hp.shape)
    return Hp, Hc


@projection_decorator
def ringdown(times, t_inj):
    """Generate ringdown waveform.
    """
    f0 = np.random.uniform(100.0, 1000.0)
    tau = np.random.uniform(0.01, 0.2)
    params = {
        "f0": f0,
        "tau": tau,
    }
    env = np.exp(-(times - t_inj) / tau) * np.heaviside(times - t_inj, 0)
    env[np.isnan(env)] = 0
    Hp = env * np.sin(2 * np.pi * f0 * (times - t_inj))
    Hc = env * np.cos(2 * np.pi * f0 * (times - t_inj))
    return Hp, Hc, params


@projection_decorator
def chirp_gaussian(times, t_inj, method='linear'):
    """Generate chirp-gaussian waveform.
    """
    f0 =  np.random.uniform(150.0, 1000.0)
    Q = np.random.uniform(15.0, 500.0)
    A = np.random.uniform(1e-21, 4e-21)
    t1 = Q / (np.sqrt(2) * np.pi * f0)
    f1 = f0 / 2
    tau = Q / (np.sqrt(2) * np.pi * f0)
    params = {
        "f0": f0,
        "Q": Q,
        "t1": t1,
        "f1": f1,
        "tau": tau
    }
    env = gaussian_env(times, t_inj, tau)
    Hp = env * chirp(times - t_inj, f0, t1, f1, method=method)
    Hc = env * chirp(times - t_inj, f0, t1, f1, method=method, phi=90)
    return Hp, Hc, params


@projection_decorator
def chirp_gaussian_inc(times, t_inj, method='linear'):
    """Generate chirp-gaussian waveform.
    """
    f0 = np.random.uniform(30.0, 1000.0)
    Q = np.random.uniform(3.0, 200.0)
    f1 = f0 * 20.0
    t1 = Q / (np.sqrt(2) * np.pi * f0)
    params = {
        "f0": f0,
        "Q": Q,
        "f1": f1,
        "t1": t1
    }
    env = gaussian_env(times, t_inj, t1)
    Hp = env * chirp(times - t_inj + np.sqrt(2) * t1, f0, t1, f1, method=method)
    Hc = env * chirp(times - t_inj + np.sqrt(2) * t1, f0, t1, f1, method=method, phi=90)
    return Hp, Hc, params


@projection_decorator
def double_chirp_gaussian(times, t_inj, method='linear'):
    """Generate sine-gaussian-chirp waveform.
    """
    f0 = np.random.uniform(30.0, 1000.0)
    Q = np.random.uniform(3.0, 200.0)
    f1 = np.random.uniform(600.0, 1400.0)
    a = np.random.uniform(0.2, 0.6)
    f2 = np.random.uniform(60.0, 84.0)
    tau = Q / (np.sqrt(2) * np.pi * f0)
    t1 = tau

    params = {
        "f0": f0,
        "Q": Q,
        "f1": f1,
        "f2": f2,
        "a": a,
        "tau": tau,
        "t1": t1
    }
    env = gaussian_env(times, t_inj, tau)
    Hp = env * (chirp(times - t_inj + tau, f0, t1, f1, method=method) +
                a * chirp(times - t_inj + tau, f0, t1, f2, method=method))
    Hc = env * (chirp(times - t_inj + tau, f0, t1, f1, method=method, phi=90) +
                a * chirp(times - t_inj + tau, f0, t1, f2, method=method, phi=90))
    return Hp, Hc, params


@projection_decorator
def white_noise(times, t_inj, f_low, f_high, tau, fs=4096 * 4):
    """Generate white noise waveform
    """
    env = gaussian_env(times, t_inj, tau)
    sig_p = randn(len(times))
    sig_c = randn(len(times))
    b, a = iirdesign(wp=(f_low, f_high), ws=(f_low * 2 / 3., min(f_high * 1.5, fs / 2.)), gpass=2, gstop=30, fs=fs)
    Hp = env * filtfilt(b, a, sig_p)
    Hc = env * filtfilt(b, a, sig_c)
    return Hp, Hc
