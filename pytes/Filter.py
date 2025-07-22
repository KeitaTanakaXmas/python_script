import numpy as np
from scipy.signal import lfilter

def median_filter(arr, sigma):
    """
    Noise filter using Median and Median Absolute Deviation for 1-dimentional array
    """
    
    if sigma is None:
        return np.ones(arr.size, dtype='b1')
    
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
                
    # Tiny cheeting for mad = 0 case
    if mad == 0:
        absl = np.abs(arr - med)
        if len(absl[absl > 0]) > 0:
            mad = (absl[absl > 0])[0]
        else:
            mad = np.std(arr) / 1.4826
                
    return (arr >= med - mad*1.4826*sigma) & (arr <= med + mad*1.4826*sigma)

def reduction(data, sigma=3, **kwargs):
    """
    Do data reduction with sum, max and min for pulse/noise using median filter (or manual min/max)
    
    Parameters (and their default values):
        data:   array of pulse/noise data (NxM or N array-like)
        sigma:  sigmas allowed for median filter
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (mask)
        mask:   boolean array for indexing filtered data
    """
    
    data = np.asarray(data)
    
    if "min" in kwargs:
        min_mask = (data.min(axis=1) > kwargs["min"][0]) & (data.min(axis=1) < kwargs["min"][1])
    else:
        min_mask = median_filter(data.min(axis=1), sigma)

    if "max" in kwargs:
        max_mask = (data.max(axis=1) > kwargs["max"][0]) & (data.max(axis=1) < kwargs["max"][1])
    else:
        max_mask = median_filter(data.max(axis=1), sigma)

    if "sum" in kwargs:
        sum_mask = (data.sum(axis=1) > kwargs["sum"][0]) & (data.sum(axis=1) < kwargs["sum"][1])
    else:
        sum_mask = median_filter(data.sum(axis=1), sigma)

    return min_mask & max_mask & sum_mask

def ntrigger(pulse, noise, threshold=20, sigma=3, smooth=10, avg_pulse=None, **kwargs):
    """
    Number of trigger
    
    Parameters (and their default values):
        pulse:      array of pulse data (NxM)
        noise:      array of noise data (NxM or N array-like)
        threshold:  sigmas of noise to use as threshold (Default: 20)
        sigma:      sigmas allowed for median filter, or None to disable noise filtering (Default: 3)    
        smooth:     number of boxcar taps to smooth pulse (Default: 10)
        avg_pulse:  averaged pulse to use for offset subtraction (Default: None)

    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum

    Return (mask)
        mask:   array of boolean
    """

    # if given data is not numpy array, convert them
    pulse = np.asarray(pulse).copy()
    noise = np.asarray(noise).copy()
    
    # Subtract offset from pulse
    p = pulse - offset(pulse, max_shift=0, avg_pulse=avg_pulse)[:, np.newaxis]
    
    # Smooth pulse
    p = lfilter([smooth**-1]*smooth, 1, p)
    
    # Data reduction for noise
    if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
        noise = noise[reduction(noise, sigma, **kwargs)]

    thre = np.std(noise) * threshold
    
    # Trigger
    ntrigger = np.sum((p >= thre) & np.roll(p < thre, 1), axis=-1)
    ntrigger += np.sum((p <= -thre) & np.roll(p > -thre, 1), axis=-1)
    
    return ntrigger

def average_pulse(pulse, sigma=3, r=0.2, rr=0.1, max_shift=None, return_shift=False, **kwargs):
    """
    Generate an averaged pulse
    
    Parameters (and their default values):
        pulse:          array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:          sigmas allowed for median filter, or None to disable filtering (Default: 3)
        r:              amount in ratio of removal in total for data reduction (Default: 0.2)
        rr:             amount in ratio of removal for each step for data reduction (Default: 0.1)
        max_shift:      maximum allowed shifts to calculate maximum cross correlation (Default: None = length / 2)
        return_shift:   return array of shifts if True (Default: False)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (averaged_pulse)
        averaged_pulse:     averaged pulse
        shift:              shifted values (only if return_shift is True)
    """
    
    # if given data is not numpy array, convert them
    pulse = np.asarray(pulse).copy()
    
    s = []
    
    # Calculate averaged pulse
    if pulse.ndim == 2:
        # Data reduction
        if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
            pulse = pulse[reduction(pulse, sigma, **kwargs)]

        plen = len(pulse)

        while (len(pulse) > (plen*(1.0-r))):
            avg = np.average(pulse, axis=0)
            pulse = pulse[((pulse - avg)**2).sum(axis=-1).argsort() < (len(pulse) - plen*rr - 1)]

        # Align pulses to the first pulse
        if max_shift is None or max_shift > 0:
            max_shift = pulse.shape[-1]/2 if max_shift is None else max_shift
            if len(pulse) > 1:
                s.append(0)
                for i in range(1, len(pulse)):
                    _s = cross_correlate(pulse[0], pulse[i], max_shift=max_shift)[1]
                    pulse[i] = np.roll(pulse[i], _s)
                    s.append(_s)

        avg_pulse = np.average(pulse, axis=0)
    
    elif pulse.ndim == 1:
        # Only one pulse data. No need to average
        avg_pulse = pulse
    
    else:
        raise ValueError("object too deep for desired array")

    if return_shift:
        return avg_pulse, s
    else:
        return avg_pulse

def power(data):
    """
    Calculate power spectrum
    
    Parameter:
        data:   pulse/noise data (NxM or N array-like)
    
    Return (power)
        power:  calculated power spectrum
    """
    
    data = np.asarray(data)
    
    # Real DFT
    ps = np.abs(np.fft.rfft(data) / data.shape[-1])**2
    
    if data.shape[-1] % 2:
        # Odd
        ps[...,1:] *= 2
    else:
        # Even
        ps[...,1:-1] *= 2
    
    return ps

def average_noise(noise, sigma=3, r=0.2, rr=0.1, **kwargs):
    """
    Calculate averaged noise power
    
    Parameters (and their default values):
        noise:      array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        r:          amount in ratio of removal in total for data reduction (Default: 0.2)
        rr:         amount in ratio of removal for each step for data reduction (Default: 0.1)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (averaged_pulse)
        power_noise:    calculated averaged noise power in V^2
    """

    # Convert to numpy array
    noise = np.asarray(noise)

    if sigma is not None or 'min' in kwargs or 'max' in kwargs or 'sum' in kwargs:
        noise = noise[reduction(noise, sigma, **kwargs)]
        
        nlen = len(noise)

        while (len(noise) > (nlen*(1.0-r))):
            avg = np.average(power(noise), axis=0)
            noise = noise[((power(noise) - avg)**2).sum(axis=-1).argsort() < (len(noise) - nlen*rr - 1)]

    return np.average(power(noise), axis=0)

def generate_template(pulse, noise, cutoff=None, lpfc=None, hpfc=None, nulldc=False, **kwargs):
    """
    Generate a template of optimal filter

    Parameters (and their default values):
        pulse:  array of pulse data, will be averaged if dimension is 2
        noise:  array of noise data, will be averaged if dimension is 2
        cutoff: low-pass cut-off bin number for pulse spectrum (Default: None)
                (**note** This option is for backward compatibility only. Will be removed.)
        lpfc:   low-pass cut-off bin number for pulse spectrum (Default: None)
        hpfc:   high-pass cut-off bin number for pulse spectrum (Default: None)
        nulldc: nullify dc bin of template (Default: False)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (template)
        template:   generated template
        sn:         calculated signal-to-noise ratio
    """
    
    # Average pulse
    if pulse.ndim == 2:
        avg_pulse = average_pulse(pulse, **kwargs)
    else:
        avg_pulse = pulse

    # Real-DFT
    fourier = np.fft.rfft(avg_pulse)
    
    # Apply low-pass/high-pass filter
    m = len(avg_pulse)
    n = len(fourier)
    
    if lpfc is None and cutoff is not None:
        lpfc = cutoff
    
    if lpfc is not None and 0 < lpfc < n:
        h = np.blackman(m)*np.sinc(np.float(lpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        fourier *= np.abs(np.fft.rfft(h))

    # Apply high-pass filter
    if hpfc is not None and 0 < hpfc < n:
        h = np.blackman(m)*np.sinc(np.float(hpfc)/n*(np.arange(m)-(m-1)*0.5))
        h /= h.sum()
        fourier *= (1 - np.abs(np.fft.rfft(h)))
    
    # Null DC bin?
    if nulldc:
        fourier[0] = 0
    
    # Calculate averaged noise power
    if noise.ndim == 2:
        pow_noise = average_noise(noise, **kwargs)
    else:
        pow_noise = noise
    
    # Calculate S/N ratio
    sn = np.sqrt(power(np.fft.irfft(fourier, len(avg_pulse)))/pow_noise)
    
    # Generate template (inverse Real-DFT)
    template = np.fft.irfft(fourier / pow_noise, len(avg_pulse))
    
    # Normalize template
    norm = (avg_pulse.max() - avg_pulse.min()) / ((template * avg_pulse).sum() / len(avg_pulse))
    
    return template * norm, sn

def cross_correlate(data1, data2, max_shift=None, method='interp'):
    """
    Calculate a cross correlation for a given set of data.
    
    Parameters (and their default values):
        data1:      pulse/noise data (array-like)
        data2:      pulse/noise data (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        method:     interp - perform interpolation for obtained pha and find a maximum
                             (only works if max_shift is given)
                    integ  - integrate for obtained pha
                    none   - take the maximum from obtained pha
                    (Default: interp)
    
    Return (max_cor, shift)
        max_cor:    calculated max cross correlation
        shift:      required shift to maximize cross correlation
        phase:      calculated phase
    """

    # Sanity check
    if len(data1) != len(data2):
        raise ValueError("data length does not match")

    # if given data set is not numpy array, convert them
    data1 = np.asarray(data1).astype(dtype='float64')
    data2 = np.asarray(data2).astype(dtype='float64')
    
    # Calculate cross correlation
    if max_shift == 0:
        return np.correlate(data1, data2, 'valid')[0] / len(data1), 0, 0
    
    # Needs shift
    if max_shift is None:
        max_shift = len(data1) / 2
    else:
        # max_shift should be less than half data length
        max_shift = min(max_shift, len(data1) / 2)
    
    # Calculate cross correlation
    cor = np.correlate(data1, np.concatenate((data2[-max_shift:], data2, data2[:max_shift])), 'valid')
    ind = cor.argmax()

    if method == 'interp' and 0 < ind < len(cor) - 1:
        return (cor[ind] - (cor[ind-1] - cor[ind+1])**2 / (8 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))) / len(data1), ind - max_shift, (cor[ind-1] - cor[ind+1]) / (2 * (cor[ind-1] - 2 * cor[ind] + cor[ind+1]))
    elif method == 'integ':
        return sum(cor), 0, 0
    elif method in ('none', 'interp'):
        # Unable to interpolate, and just return the maximum
        return cor[ind] / len(data1), ind - max_shift, 0
    else:
        raise ValueError("Unsupported method")
    
def optimal_filter(pulse, template, max_shift=None, method='interp'):
    """
    Perform an optimal filtering for pulse using template
    
    Parameters (and their default values):
        pulse:      pulses (NxM array-like)
        template:   template (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        method:     interp - perform interpolation for obtained pha and find a maximum
                             (only works if max_shift is given)
                    integ  - integrate for obtained pha
                    none   - take the maximum from obtained pha
                    (Default: interp)
    
    Return (pha, lagphase)
        pha:        pha array
        phase:      phase array
    """
    
    return np.apply_along_axis(lambda p: cross_correlate(template, p, max_shift=max_shift, method=method), 1, pulse)[...,(0,2)].T

def offset(pulse, bins=None, sigma=3, max_shift=None, avg_pulse=None):
    """
    Calculate an offset (DC level) of pulses
    
    Parameters (and their default values):
        pulse:      pulses (N or NxM array-like)
        bins:       tuple of (start, end) for bins used for calculating an offset
                    (Default: None = automatic determination)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
        avg_pulse:  if given, use this for averaged pulse (Default: None)

    Return (offset)
        offset: calculated offset level
    """
    
    pulse = np.asarray(pulse)
    
    if bins is None:
        if avg_pulse is None:
            avg_pulse = average_pulse(pulse, sigma=sigma, max_shift=max_shift)
        i = np.correlate(avg_pulse, [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1]).argmax() - 16
        if i < 1:
            raise ValueError("Pre-trigger is too short")
        return pulse[..., :i].mean(axis=-1)
    else:
        return pulse[..., bins[0]:bins[1]].mean(axis=-1)