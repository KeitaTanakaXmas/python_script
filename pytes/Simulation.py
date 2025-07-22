import numpy as np
from scipy.stats import norm, cauchy
from pytes import Analysis, Constants

def pulse(pha=1.0, tr=2e-6, tf=100e-6, points=1024, t=2e-3, duty=0.5, talign=0.0):
    """
    Generate Dummy Pulse
    
    Parameters (and their default values):
        pha:    pulse height (Default: 1.0)
        tr:     rise time constant (Default: 2 us)
        tf:     fall time constant (Default: 100 us)
        points: data length (scalar or 2-dim tuple/list) (Default: 1024)
        t:      sample time (Default: 2 ms)
        duty:   pulse raise time ratio to t (Default: 0.5)
                should take from 0.0 to 1.0
        talign: time offset ratio to dt (=t/points) (Default: 0)
                should take from -0.5 to 0.5
    
    Return (pulse)
        pulse:  pulse data (array-like)
    """
    
    points = np.asarray(points)
    
    # Data counts
    c = 1 if points.ndim == 0 else np.ones(points[0])[np.newaxis].T
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Time steps
    ts = np.linspace(-t*duty, t*(1-duty), l)*c + t/l*np.asarray(talign)[np.newaxis].T
    
    # Pulse model function
    def M(t):
        with np.errstate(over='ignore'):
            return np.nan_to_num((1-np.exp(-t/tr)) * np.exp(-t/tf)) * (t>0)
    
    # Normalize coeff (= max M)
    norm = M(tr*np.log(tf/tr+1))
    
    return np.asarray(pha)[np.newaxis].T*M(ts)/norm

def white(sigma=3e-6, mean=0.0, points=1024, t=2e-3, etf=False, L=20, tf=100e-6):
    """
    Generate Gaussian White Noise
    
    Parameters (and their default values):
        sigma:  noise level in V/srHz (or gaussian sigma) (Default: 3 uV/srHz)
        mean:   DC offset of noise signal (Default: 0 V)
        t:      sample time (Default: 2 ms)
        points: data length (scalar or 2-dim tuple/list) (Default: 1024)
        etf:    simulate noise suppression by ETF (Default: False)
        L:      ETF loopgain of TES (Default: 20)
        tf:     fall time constant (Default: 100 us)
    """
    
    points = np.asarray(points)
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Time resolution of Nyquist frequency
    dt = t / l * 2

    n = (mean + sigma*norm.rvs(size=points)) / np.sqrt(dt)
    
    if etf:
        tau0 = tf*(L+1)
        fl = l/2 + 1
        w = 2*np.pi*np.arange(fl)*t**-1
        n = np.fft.irfft(np.fft.rfft(n)*(1./(L+1))**2*(1+(w*tau0)**2)/(1+(w*tf)**2), int(l))
    
    return n

def phonon(sigma=10e-6, mean=0.0, tr=2e-6, tf=100e-6, points=1024, t=2e-3):
    """
    Generate Phonon Noise
    
    Parameters (and their default values):
        sigma:  noise level in V/srHz at low frequency (Default: 10 uV/srHz)
        mean:   DC offset of noise signal (Default: 0 V)
        tr:     rise time constant (Default: 2 us)
        tf:     fall time constant (Default: 100 us)
        points: data length (scalar or 2-dim tuple/list) (Default: 1024)
        t:      sample time (Default: 2 ms)
    """

    points = np.asarray(points)
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Generate white noise
    w = white(sigma, mean, points, t)
    
    # Generate dummy pulse and generate filter
    dp = pulse(tr=tr, tf=tf, points=l, t=t)
    h = np.abs(np.fft.rfft(dp/dp.sum()))
    
    # Generate phonon noise
    return (mean + np.fft.irfft(np.fft.rfft(w)*h, int(l)))

def random(pdf, N, min, max):
    """
    Generate random values in given distribution using the rejection method
    
    Parameters:
        pdf:    distribution function
        N:      desired number of random values
        min:    minimum of random values
        max:    maximum of random values
    
    Return (values)
        values: generated random values
    """
    
    valid = np.array([])
    
    maxp = np.max(pdf(np.linspace(min, max, 1e6)))
    
    while len(valid) < N:
        r = np.random.uniform(min, max, N)
        p = np.random.uniform(0, maxp, N)
        
        valid = np.concatenate((valid, r[p < pdf(r)]))
    
    return valid[:N]

def simulate(N, width, pha=1.0, noise=3e-6, pnoise=10e-7, L=20, sps=1e6, t=2e-3, tr=2e-6, tf=100e-6, duty=0.1, atom="Mn", ratio=0.9, talign=True, kascale=1.0, kbscale=1.0):
    """
    Generate pulse (Ka and Kb) and noise
    
    Parameters (and their default values):
        N:          desired number of pulses/noises
        width:      width (FWHM) of gaussian (voigt) profile
        pha:        pulse height of Ka line in V (Default: 1 V)
        noise:      white noise level in V/srHz (Default: 3 uV/srHz)
        pnoise:     phonon noise level in V/srHz (Default: 10 uV/srHz)
        L:          ETF loopgain of TES (Default: 20)
        sps:        sampling per second (Default: 1 Msps)
        t:          sampling time (Default: 2 ms)
        tr:         rise time constant (Default: 2 us)
        tf:         fall time constant (Default: 100 us)
        duty:       trigger position (Default: 0.1)
        atom:       atom to simulate (Default: Mn)
        ratio:      Ka ratio (Default: 0.9)
        taling:     vary pulse alignment (Default: True)
        kascale:    Ka line PHA scaling (Default: 1.0)
        kbscale:    Kb line PHA scaling (Default: 1.0)
    
    Return (pulse, noise):
        pulse:  simulated pulse data (NxM array-like)
        noise:  simulated noise data (NxM array-like)
    """
    
    # Simulate Ka and Kb Lines
    e = np.concatenate((line(int(N*ratio), width, atom+"Ka"),
                        line(int(N-N*ratio), width, atom+"Kb")))
    
    # Simulate non-linearity
    p = np.polyfit([0, Constants.LE[atom+"Ka"], Constants.LE[atom+"Kb"]], [0, Constants.LE[atom+"Ka"]*kascale, Constants.LE[atom+"Kb"]*kbscale], 2)
    e = np.polyval(p, e)
    
    # Convert energy to PHA
    _pha = e / Constants.LE[atom+"Ka"] * pha
    
    # Vary talign?
    if talign:
        ta = (np.random.uniform(size=N) - 0.5)
    else:
        ta = 0
    
    # Generate pulses and noises
    points = (N, int(sps*t))
    p = pulse(_pha, tr=tr, tf=tf, points=points, t=t, duty=duty, talign=ta) + \
                white(sigma=noise, points=points, t=t, etf=True, L=L, tf=tf) + phonon(sigma=pnoise, tr=tr, tf=tf, points=points, t=t)
    
    n = white(sigma=noise, points=points, t=t, etf=True, L=L, tf=tf) + phonon(sigma=pnoise, tr=tr, tf=tf, points=points, t=t)

    return p, n

def line(N, width, line="MnKa"):
    """
    Simulate Ka/Kb Line
    
    Parameters (and their default values):
        N:      desired number of pulse
        width:  width (FWHM) of gaussian (voigt) profile
        line:   line to simulate (Default: MnKa)
    
    Return (e):
        e:      simulated data
    """

    if width == 0:
        # Simulate by Cauchy (Lorentzian)
        fs = np.asarray(Constants.FS[line])
        renorm = fs.T[2].sum()**-1
        e = np.concatenate([ cauchy.rvs(loc=f[0], scale=Analysis.fwhm2gamma(f[1]), size=int(f[2]*renorm*N)) for f in fs ])
        if N - len(e) > 0:
            e = np.concatenate((e, cauchy.rvs(loc=f[0], scale=Analysis.fwhm2gamma(f[1]), size=int(N-len(e)))))
        
    else:
        # Simulate by Voigt
        pdf = lambda E: Analysis.line_model(E, 0, width, line=line)
        Ec = np.array(Constants.FS[line])[:,0]
        _Emin = np.min(Ec) - width*50
        _Emax = np.max(Ec) + width*50
        e = random(pdf, N, _Emin, _Emax)
    
    return e[:N]