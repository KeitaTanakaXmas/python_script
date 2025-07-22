import numpy as np
import math
import datetime
import warnings
from scipy.special import wofz
from scipy.stats import cauchy, norm
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from scipy.odr import odrpack, models
from .Filter import median_filter
from .Constants import *
from functools import reduce

def baseline(sn, E=5.9e3):
    """
    Calculate a baseline resolution dE(FWHM) for the given energy.
    
    Parameter:
        sn:     S/N ratio (array-like)
        E:      energy to calculate dE
    """
    
    return 2*np.sqrt(2*np.log(2))*E/np.sqrt((sn**2).sum(axis=-1)*2)

def ka(pha, sigma=1):
    """
    Return a k-alpha data set.
    
    Parameters (and their default values):
        pha:    pha (and optional companion) data (array-like)
        sigma:  sigmas allowed for median filter (Default: 1)

    Return (data)
        data:   k-alpha data set
    """
    
    pha = np.asarray(pha)

    # Pre-selection for more robustness (dirty hack)
    if pha.ndim > 1:
        mask = median_filter(pha[:,0], sigma=sigma+1)
    else:
        mask = median_filter(pha, sigma=sigma+1)

    pha = pha[mask]
    
    if pha.ndim > 1:
        mask = median_filter(pha[:,0], sigma=sigma)
    else:
        mask = median_filter(pha, sigma=sigma)
    
    return pha[mask]

def kb(pha, sigma=1):
    """
    Find a k-beta data set.
    
    Parameters (and their default values):
        pha:    pha (and optional companion) data (array-like)
        sigma:  sigmas allowed for median filter (Default: 1)

    Return (data)
        data:   k-beta data set
    """
    
    pha = np.asarray(pha)
    
    if pha.ndim > 1:
        _pha = pha[:,0]
    else:
        _pha = pha
    
    ka_max = ka(_pha, sigma=sigma).max()
    ka_min = ka(_pha, sigma=sigma).min()
    
    return ka(pha[(_pha > ka_max) | (_pha < ka_min)], sigma)

def offset_correction(pha, offset, sigma=1, prange=None, orange=None, method='ols', flip=False, p=None, filename=None, tex=False):
    """
    Fit a pha and an offset (DC level)
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        offset:     offset data (array-like)
        sigma:      sigmas allowed for median filter (Default: 1)
        prange:     a tuple of range for pha to fit if not None (Default: None)
        orange:     a tuple of range for offset to fit if not None (Default: None)
        method:     fitting method from ols (ordinal least squares)
                    or odr (orthogonal distance regression) (Default: ols)
        flip:       flip pha and offset when fitting if True (Default: False)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (pha, p, coef):
        pha:    offset corrected pha
        p:      fitting results to pha = offset*p[1] + p[0]
        coef:   correlation coefficient

    Note:
        - If p is given, it just uses it to perform the correction
        - If p is not given, it first makes it and uses it
    """

    # Sanity check
    if len(pha) != len(offset):
        raise ValueError("data length of pha and offset does not match")
    
    pha = np.asarray(pha)
    offset = np.asarray(offset)
    
    if p is None:
        # Reduction
        if prange is not None:
            pmask = (pha >= prange[0]) & (pha <= prange[1])
        else:
            pmask = median_filter(pha, sigma=sigma)

        if orange is not None:
            omask = (offset >= orange[0]) & (offset <= orange[1])
        else:
            omask = median_filter(offset, sigma=sigma)

        mask = pmask & omask

        # Correlation coefficient
        coef = np.corrcoef(pha[mask], offset[mask])[0,1]

        # Fitting to a*x+b
        if method.lower() == 'ols':
            if flip:
                p = np.polyfit(pha[mask], offset[mask], 1)
            else:
                p = np.polyfit(offset[mask], pha[mask], 1)
        elif method.lower() == 'odr':
            m = models.polynomial(1)
            if flip:
                data = odrpack.Data(pha[mask], offset[mask])
            else:
                data = odrpack.Data(offset[mask], pha[mask])
            odr = odrpack.ODR(data, m, maxit=100)
            fit = odr.run()
            p = fit.beta[::-1]
        else:
            raise ValueError('Unknown method: %s' % method)
    
        if flip:
            p = np.array([p[0]**-1, -p[1]/p[0]])
    
        # Plot if needed
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['text.usetex'] = str(tex)
            from matplotlib import pyplot as plt

            fig = plt.figure()

            ax = plt.subplot(211)
            plt.plot(offset[mask], pha[mask], ',', c='k')
            x_min, x_max = plt.xlim()
            x = np.linspace(x_min, x_max)
            label = '$\mathrm{PHA}=\mathrm{Offset}\\times%.2f+%.2f$' % tuple(p)
            plt.plot(x, np.polyval(p, x), 'r-', label=label)
            plt.xlabel('Offset$\quad$(a.u.)')
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.legend(frameon=False)
            
            ax = plt.subplot(212)
            plt.plot(offset[mask], pha[mask] / np.polyval(p, offset[mask]), ',', c='k')
            plt.axhline(1, color='r', ls='--')
            plt.xlim(x_min, x_max)
            plt.xlabel('Offset$\quad$(a.u.)')
            plt.ylabel('PHA$\quad$(a.u.)')
            
            plt.tight_layout()
            plt.savefig(filename)
    else:
        coef = None
    
    return pha / np.polyval(p, offset), p, coef

def phase_correction(phase, pha, sigma=1, deg=10, p=None, filename=None, tex=False):
    """
    Perform a phase correction

    Parameters (and their default values):
        phase:      phase (array-like)
        pha:        pha data (array-like)
        sigma:      sigmas for median filter (Default: 1)
        deg:        degree of the fitting polynomial (Default: 10)
        p:          polynomial coefficients to use for correction (Default: None)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (pha, func):
        pha:        drift corrected pha
        p:          polynomial coefficients used for correction

    Note:
        - If p is given, it just uses it to perform the correction
        - If p is not given, it first makes it and uses it
    """

    if p is None:
        # Zip pha and phase
        data = np.vstack((pha, phase)).T

        # Correction using K-alpha
        ka_pha, ka_phase = ka(data, sigma=sigma).T

        ka_phase_w = np.concatenate((ka_phase - 1, ka_phase, ka_phase + 1))
        ka_pha_w = np.concatenate((ka_pha, ka_pha, ka_pha))
        
        mask = (ka_phase_w > -0.75) & (ka_phase_w < 0.75)

        p = np.polyfit(ka_phase_w[mask], ka_pha_w[mask], deg)

        # Plot if needed
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['text.usetex'] = str(tex)
            from matplotlib import pyplot as plt
        
            plt.figure()
            
            plt.subplot(211)
            plt.plot(ka_phase, ka_pha, ',', c='k')
            plt.xlim(-0.5, 0.5)
            x = np.linspace(-0.5, 0.5)
            plt.plot(x, np.polyval(p, x), 'r-')
            plt.ylabel('PHA$\quad$(a.u.)')
            
            plt.subplot(212)
            plt.plot(ka_phase, ka_pha/np.polyval(p, ka_phase), ',', c='k')
            plt.axhline(1, color='r', ls='--')
            plt.xlim(-0.5, 0.5)
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.xlabel('Phase$\quad$(a.u.)')
            
            plt.tight_layout()
            plt.savefig(filename)
    
    return pha / np.polyval(p, phase), p

def drift_correction(ts, pha, atom='Mn', sigma=1, method="fitting", tail=False, ignorekb=False, npoints=300, nstep=100, func=None, filename=None, tex=False):
    """
    Perform a drift correction

    Parameters (and their default values):
        ts:         timestamp (array-like)
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas for median filter (Default: 1)
        method:     fitting or median to detect lines (Default: fitting)
        tail:       enable low energy tail (Default: False)
        ignorekb:   do not use Kb (Default: False)
        npoints:    number of points to use for line fittings (Default: 300)
        nstep:      number of internval points for line fittings (Default: 100)
        func:       function to use for correction (Default: None)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (pha, func):
        pha:        drift corrected pha
        func:       function used for correction

    Note:
        - If func is given, it just uses it to perform the correction
        - If func is not given, it first makes it and uses it
    """
    
    if method.lower() == "fitting":
        return _drift_correction_by_fitting(ts, pha, atom=atom, sigma=sigma, tail=tail, ignorekb=ignorekb, npoints=npoints, nstep=nstep, func=func, filename=filename, tex=tex)
    elif method.lower() == "median":
        return _drift_correction_by_median(ts, pha, atom=atom, sigma=sigma, tail=tail, ignorekb=ignorekb, npoints=npoints, nstep=nstep, func=func, filename=filename, tex=tex)
    else:
        raise ValueError("Unsupported method: %s" % method)
    
def _drift_correction_by_fitting(ts, pha, atom='Mn', sigma=1, tail=False, ignorekb=False, npoints=300, nstep=100, func=None, filename=None, tex=False):
    """
    Perform a drift correction by fitting

    Parameters (and their default values):
        ts:         timestamp (array-like)
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas for median filter (Default: 1)
        tail:       enable low energy tail (Default: False)
        ignorekb:   do not use Kb (Default: False)
        npoints:    number of points to use for line fittings (Default: 300)
        nstep:      number of internval points for line fittings (Default: 100)
        func:       function to use for correction (Default: None)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (pha, func):
        pha:        drift corrected pha
        func:       function used for correction

    Note:
        - If func is given, it just uses it to perform the correction
        - If func is not given, it first makes it and uses it
    """
    
    if func is None:
        # Linearity correction
        lc_p, lc_pr, lc_r = _linearity_correction_by_fitting(pha, atom=atom, sigma=sigma, tail=tail, ignorekb=ignorekb)
        
        # Zip pha and timestamp
        data = np.vstack((pha, ts)).T

        # Correction using K-alpha
        ka_pha, ka_ts = ka(data, sigma=sigma).T

        ka_pha = ka_pha[ka_ts.argsort()]
        ka_ts = ka_ts[ka_ts.argsort()]
        
        if tail:
            # Pre-fit
            x = fit(np.polyval(lc_p, ka_pha), line=atom+"Ka", shift=True, tail=tail, method="c", error=False)[0]
            freeze = (None, None, x[2], x[3])
        else:
            freeze = None
        
        # Drift fitting
        ka_ts_f, ka_pha_f = np.array([ [ t, np.polyval(lc_pr, LE[atom + "Ka"] + fit(np.polyval(lc_p, ka_pha[i*nstep:i*nstep+npoints]), line=atom+"Ka", shift=True, tail=tail, freeze=freeze, method="c", error=False)[0][0]) ] for i, t in enumerate(ka_ts[npoints/2:-npoints/2:nstep]) ]).T
        
        func = interp1d(ka_ts_f, ka_pha_f, bounds_error=False, fill_value=(ka_pha_f[0], ka_pha_f[-1]), assume_sorted=True)

        # Plot if needed
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['text.usetex'] = str(tex)
            from matplotlib import pyplot as plt
            from matplotlib.ticker import ScalarFormatter
            from matplotlib.dates import epoch2num, DateFormatter

            fig = plt.figure()

            ax = plt.subplot(211)
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            plt.plot_date(epoch2num(ka_ts), ka_pha, ',', c='k')
            t_min, t_max = plt.xlim()
            x = np.linspace(ka_ts.min(), ka_ts.max())
            plt.plot_date(epoch2num(x), func(x), 'r-')
            plt.xlim(t_min, t_max)
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.setp(ax.get_xticklabels(), visible=False)

            ax = plt.subplot(212)
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            plt.plot_date(epoch2num(ka_ts), ka_pha/func(ka_ts), ',', c='k')
            plt.axhline(1, color='r', ls='--')
            plt.xlim(t_min, t_max)
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.xlabel('Time')
            
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(filename)

    return pha / func(ts), func

def _drift_correction_by_median(ts, pha, atom='Mn', sigma=1, tail=False, ignorekb=False, npoints=300, nstep=100, func=None, filename=None, tex=False):
    """
    Perform a drift correction by median

    Parameters (and their default values):
        ts:         timestamp (array-like)
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas for median filter (Default: 1)
        tail:       enable low energy tail (Default: False)
        ignorekb:   do not use Kb (Default: False)
        npoints:    number of points to use for line fittings (Default: 300)
        nstep:      number of internval points for line fittings (Default: 100)
        func:       function to use for correction (Default: None)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (pha, func):
        pha:        drift corrected pha
        func:       function used for correction

    Note:
        - If func is given, it just uses it to perform the correction
        - If func is not given, it first makes it and uses it
    """
    
    if func is None:
        # Zip pha and timestamp
        data = np.vstack((pha, ts)).T

        # Correction using K-alpha
        ka_pha, ka_ts = ka(data, sigma=sigma).T

        ka_pha = ka_pha[ka_ts.argsort()]
        ka_ts = ka_ts[ka_ts.argsort()]
        
        # Drift fitting
        ka_ts_f, ka_pha_f = np.array([ [ t, np.median(ka_pha[i*nstep:i*nstep+npoints]) ] for i, t in enumerate(ka_ts[npoints/2:-npoints/2:nstep]) ]).T
        
        func = interp1d(ka_ts_f, ka_pha_f, bounds_error=False, fill_value=(ka_pha_f[0], ka_pha_f[-1]), assume_sorted=True)

        # Plot if needed
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['text.usetex'] = str(tex)
            from matplotlib import pyplot as plt
            from matplotlib.ticker import ScalarFormatter
            from matplotlib.dates import epoch2num, DateFormatter

            fig = plt.figure()

            ax = plt.subplot(211)
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            plt.plot_date(epoch2num(ka_ts), ka_pha, ',', c='k')
            t_min, t_max = plt.xlim()
            x = np.linspace(ka_ts.min(), ka_ts.max())
            plt.plot_date(epoch2num(x), func(x), 'r-')
            plt.xlim(t_min, t_max)
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.setp(ax.get_xticklabels(), visible=False)

            ax = plt.subplot(212)
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            plt.plot_date(epoch2num(ka_ts), ka_pha/func(ka_ts), ',', c='k')
            plt.axhline(1, color='r', ls='--')
            plt.xlim(t_min, t_max)
            plt.ylabel('PHA$\quad$(a.u.)')
            plt.xlabel('Time')
            
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(filename)

    return pha / func(ts), func

def linearity_correction(pha, atom="Mn", sigma=1, p=None, method="fitting", fittingmethod="mle", tail=False, ignorekb=False, filename=None, tex=False):
    """
    Perform a linearity correction for PHA

    Parameters (and their default values):
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas allowed for median filter (Default: 1)
        p:          polynomial coefficients to use for correction (Default: None)
        method:     fitting or median to detect lines (Default: fitting)
        fittingmethod: fitting method among c/mle/cs/ls (Default: c)
        tail:       enable low energy tail when fitting (Default: False)
        ignorekb:   do not use Kb (Default: False)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)

    Return (pha, r, p):
        pha:        corrected pha
        r:          linearity rate
        p:          polynomial coefficients used for correction

    Note:
        - If p is given, it just uses it to perform the correction
        - If p is not given, it first makes it and uses it
    """

    if p is None:
        if method.lower() == "fitting":
            p, pr, r = _linearity_correction_by_fitting(pha, atom=atom, sigma=sigma, method=fittingmethod, tail=tail, ignorekb=ignorekb, filename=filename, tex=tex)
        elif method.lower() == "median":
            p, pr, r = _linearity_correction_by_median(pha, atom=atom, sigma=sigma, ignorekb=ignorekb, filename=filename, tex=tex)
        else:
            raise ValueError("Unsupported method: %s" % method)
    else:
        r = None

    return np.polyval(p, pha), r, p

def _linearity_correction_by_fitting(pha, atom="Mn", sigma=1, method="c", tail=False, ignorekb=False, filename=None, tex=False):
    """
    Perform a linearity correction for PHA

    Parameters (and their default values):
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas allowed for median filter (Default: 1)
        method:     method for fitting among c/mle/cs/ls (Default: c)
        tail:       enable low energy tail (Default: False)
        ignorekb:   do not use Kb (Default: False)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)

    Return (p, pr, r):
        p:          polynomial coefficients used for correction
        pr:         polynomial coefficients used for reverse correction
        r:          linearity rate
    """
    
    # Sanity check
    if atom+"Ka" not in LE:
        raise ValueError("No data for %s" % atom)

    if not ignorekb and atom+"Kb" not in LE:
        raise ValueError("No data for %s" % atom)

    if ignorekb:
        le = np.asarray([0, LE[atom+"Ka"]])
        pc = np.asarray([0, np.median(ka(pha, sigma=sigma))])
    else:
        le = np.asarray([0, LE[atom+"Ka"], LE[atom+"Kb"]])
        pc = np.asarray([0, np.median(ka(pha, sigma=sigma)), np.median(kb(pha, sigma=sigma))])
    
    # Calculate polynomial coefficients
    if ignorekb:
        p = np.array([ le[1]/pc[1], 0 ])
        pr = np.array([ pc[1]/le[1], 0 ])
    else:
        p = np.array([ (pc[2]*le[1]-pc[1]*le[2])/(pc[1]**2*pc[2]-pc[1]*pc[2]**2),
                (pc[1]**2*le[2]-pc[2]**2*le[1])/(pc[1]*pc[2]*(pc[1]-pc[2])), 0 ])

        pr = np.array([ (le[2]*pc[1]-le[1]*pc[2])/(le[1]**2*le[2]-le[1]*le[2]**2),
                (le[1]**2*pc[2]-le[2]**2*pc[1])/(le[1]*le[2]*(le[1]-le[2])), 0 ])

    # Fit spectrum
    lc_pha = np.polyval(p, pha)

    dE_ka, E_ka = fit(ka(lc_pha, sigma=sigma), line=atom+"Ka", shift=False, tail=tail, method=method, error=False)[0][:2]
    
    if ignorekb:
        dE = np.array([ dE_ka ])
    else:
        dE_kb, E_kb = fit(kb(lc_pha, sigma=sigma), line=atom+"Kb", shift=False, tail=tail, method=method, error=False)[0][:2]
        dE = np.array([ dE_ka, dE_kb ])

    # Fine-tune PHA Center
    pc[1:] = np.polyval(pr, le[1:] + dE)

    # Recalculate polynomial coefficients
    if ignorekb:
        p = np.array([ le[1]/pc[1], 0 ])
        pr = np.array([ pc[1]/le[1], 0 ])
    else:
        p = np.array([ (pc[2]*le[1]-pc[1]*le[2])/(pc[1]**2*pc[2]-pc[1]*pc[2]**2),
                (pc[1]**2*le[2]-pc[2]**2*le[1])/(pc[1]*pc[2]*(pc[1]-pc[2])), 0 ])

        pr = np.array([ (le[2]*pc[1]-le[1]*pc[2])/(le[1]**2*le[2]-le[1]*le[2]**2),
                (le[1]**2*pc[2]-le[2]**2*pc[1])/(le[1]*le[2]*(le[1]-le[2])), 0 ])

    # Calculate linearity rate
    if ignorekb:
        r = None
    else:
        r = (pc[2]/pc[1])/(le[2]/le[1])

    if filename is not None:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = str(tex)
        from matplotlib import pyplot as plt

        plt.figure()
        if ignorekb:
            label = "%sKa" % (atom)
            plt.plot(le[1], pc[1], '+', c='b', label=label)
        else:
            label = "%sKa / %sKb (%.2f $\%%$)" % (atom, atom, r*100)
            plt.plot(le[1:], pc[1:], '+', c='b', label=label)
        x = np.linspace(0, np.ceil(le[-1]/1e3)*1e3)
        if ignorekb:
            label = 'PHA$=\mathrm{%.2e}E$' % pr[0]
        else:
            label = 'PHA$=\mathrm{%.2e}E^2+\mathrm{%.2e}E$' % tuple(pr[:2])
        plt.plot(x, np.polyval(pr, x), 'r-', label=label)
        ymin, ymax = plt.ylim()
        plt.ylim(0, ymax*1.2)
        plt.xlabel('Energy$\quad$(eV)')
        plt.ylabel('PHA$\quad$(a.u.)')
        plt.legend(frameon=False, numpoints=1)
        plt.tight_layout()
        plt.savefig(filename)

    return p, pr, r

def _linearity_correction_by_median(pha, atom="Mn", sigma=1, ignorekb=False, filename=None, tex=False):
    """
    Perform a linearity correction for PHA

    Parameters (and their default values):
        pha:        pha data (array-like)
        atom:       atom to use for correction (Default: Mn)
        sigma:      sigmas allowed for median filter (Default: 1)
        ignorekb:   do not use Kb (Default: False)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)

    Return (p, r):
        p:          polynomial coefficients used for correction
        pr:         polynomial coefficients used for reverse correction
        r:          linearity rate
    """

    # Sanity check
    if atom+"Ka" not in LE:
        raise ValueError("No data for %s" % atom)

    if not ignorekb and atom+"Kb" not in LE:
        raise ValueError("No data for %s" % atom)

    if ignorekb:
        le = np.asarray([0, LE[atom+"Ka"]])
        pc = np.asarray([0, np.median(ka(pha, sigma=sigma))])
    else:
        le = np.asarray([0, LE[atom+"Ka"], LE[atom+"Kb"]])
        pc = np.asarray([0, np.median(ka(pha, sigma=sigma)), np.median(kb(pha, sigma=sigma))])
    
    # Calculate polynomial coefficients
    if ignorekb:
        p = np.array([ le[1]/pc[1], 0 ])
        pr = np.array([ pc[1]/le[1], 0 ])
    else:
        p = np.array([ (pc[2]*le[1]-pc[1]*le[2])/(pc[1]**2*pc[2]-pc[1]*pc[2]**2),
                (pc[1]**2*le[2]-pc[2]**2*le[1])/(pc[1]*pc[2]*(pc[1]-pc[2])), 0 ])

        pr = np.array([ (le[2]*pc[1]-le[1]*pc[2])/(le[1]**2*le[2]-le[1]*le[2]**2),
                (le[1]**2*pc[2]-le[2]**2*pc[1])/(le[1]*le[2]*(le[1]-le[2])), 0 ])

    # Calculate linearity rate
    if ignorekb:
        r = None
    else:
        r = (pc[2]/pc[1])/(le[2]/le[1])

    if filename is not None:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = str(tex)
        from matplotlib import pyplot as plt

        plt.figure()
        if ignorekb:
            label = "%sKa" % (atom)
            plt.plot(le[1], pc[1], '+', c='b', label=label)
        else:
            label = "%sKa / %sKb (%.2f $\%%$)" % (atom, atom, r*100)
            plt.plot(le[1:], pc[1:], '+', c='b', label=label)
        x = np.linspace(0, np.ceil(le[-1]/1e3)*1e3)
        if ignorekb:
            label = 'PHA$=\mathrm{%.2e}E$' % pr[0]
        else:
            label = 'PHA$=\mathrm{%.2e}E^2+\mathrm{%.2e}E$' % tuple(pr[:2])
        plt.plot(x, np.polyval(pr, x), 'r-', label=label)
        ymin, ymax = plt.ylim()
        plt.ylim(0, ymax*1.2)
        plt.xlabel('Energy$\quad$(eV)')
        plt.ylabel('PHA$\quad$(a.u.)')
        plt.legend(frameon=False, numpoints=1)
        plt.tight_layout()
        plt.savefig(filename)

    return p, pr, r

def voigt(E, Ec, nw, gw):
    """
    Voigt profile
     
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural (lorentzian) width (FWHM)
        gw:     gaussian width (FWHM)
    
    Return (voigt)
        voigt:  Voigt profile
    """
    
    # Sanity check
    if gw == 0:
        return lorentzian(E, Ec, nw)
    
    z = (E - Ec + 1j*fwhm2gamma(nw)) / (fwhm2sigma(gw)*np.sqrt(2))

    return wofz(z).real / (fwhm2sigma(gw)*np.sqrt(2*np.pi))

def voigt_pseudo(E, Ec, nw, gw):
    """
    Pseudo-Voigt profile
     
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural (lorentzian) width (FWHM)
        gw:     gaussian width (FWHM)
    
    Return (voigt)
        voigt:  Voigt profile
    """
    
    # Sanity check
    if gw == 0:
        return lorentzian(E, Ec, nw)
    
    sigma = fwhm2sigma(gw)
    gamma = fwhm2gamma(nw)
    
    f = (sigma**5 + 2.69269*sigma**4*gamma + 2.42843*sigma**3*gamma**2 + 4.47163*sigma**2*gamma**3 + 0.07842*sigma*gamma**4 + gamma**5)**0.2
    r = 1.36603*(gamma/f) - 0.47719*(gamma/f)**2 + 0.11116*(gamma/f)**3
    
    return r*lorentzian(E, Ec, gamma2fwhm(f)) + (1-r)*gaussian(E, Ec, sigma2fwhm(f))

def voigt_with_tail(E, Ec, nw, gw, tau):
    """
    Low energy tail profile
    
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural (lorentzian) width (FWHM)
        gw:     gaussian width (FWHM)
        tau:    decay of the tail
    
    Return (voigt_with_tail)
        voigt_with_tail:  voigt profile with low energy tail
    """

    if tau == 0:
        return voigt(E, Ec, nw, gw)
    
    # Estimate energy range
    Estep = 0.05 if tau <= 0.1 else (0.1 if tau <= 0.5 else 0.25)
    Emin = np.ceil(Ec - (nw+gw) * 20 - tau * 20 - 10)
    Emax = np.floor(Ec + (nw+gw) * 20 + 10)
    _E = np.arange(Emin, Emax, Estep)

    if _E.size > 2000:
        convolve = fftconvolve
    else:
        convolve = np.convolve

    # Calculate pdf
    pdf = convolve(voigt(_E, Ec, nw, gw), np.nan_to_num(np.exp((_E-np.mean(_E))/tau)/tau)*((_E-np.mean(_E))<0), 'same')*Estep
    
    # Interpolate pdf
    func = interp1d(_E, pdf, bounds_error=False, fill_value=(0, 0), kind='linear', assume_sorted=True)

    return func(E)

def gaussian(E, Ec, width):
    """
    Gaussian profile
    
    Parameters:
        E:      energy
        Ec:     center energy
        width:  width (FWHM)
    
    Return (gauss)
        gauss:  Gaussian profile
    """

    sigma = fwhm2sigma(width)

    return norm.pdf(E, loc=Ec, scale=sigma)
    
def lorentzian(E, Ec, nw):
    """
    Lorentzian profile
    
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural width (FWHM)
    
    Return (lorentz)
        lorentz:  Lorentzian profile
    """

    gamma = fwhm2gamma(nw)

    return cauchy.pdf(E, loc=Ec, scale=gamma)

def sigma2fwhm(sigma):
    """
    Convert sigma to width (FWHM)
    
    Parameter:
        sigma:  sigma of gaussian / voigt profile
    
    Return (fwhm)
        fwhm:   width
    """
    
    return 2*sigma*np.sqrt(2*np.log(2))

def fwhm2sigma(fwhm):
    """
    Convert width (FWHM) to sigma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        sigma:  sigma of gaussian / voigt profile
    """
    
    return fwhm/(2*np.sqrt(2*np.log(2)))

def gamma2fwhm(gamma):
    """
    Convert gamma to width (FWHM)
    
    Parameter:
        gamma:  gamma of lorentzian / voigt profile
    
    Return (fwhm)
        fwhm:   width
    """
    
    return gamma*2.0

def fwhm2gamma(fwhm):
    """
    Convert width (FWHM) to gamma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        gamma:  gamma of lorentzian / voigt profile
    """
    
    return fwhm/2.0

def line_model(E, dE=0, width=0, tR=None, tT=None, line="MnKa", shift=False, tail=False, full=False):
    """
    Line model
    
    Parameters (and their default values):
        E:      energy in eV (array-like)
        dE:     shift from center energy in eV (Default: 0 eV)
        width:  FWHM of gaussian profile in eV (Default: 0 eV)
        tR:     low energy tail ratio (Default: None)
        tT:     low energy tail tau (Default: None)
        line:   line (Default: MnKa)
        shift:  treat dE as shift if True instead of scaling (Default: False)
        tail:   enable low energy tail (Default: False)
        full:   switch for return value (Default: False)
    
    Return (i) when full = False or (i, i1, i2, ...) when full = True
        i:      total intensity
        i#:     component intensities
    
    Note:
        If shift is False, adjusted center energies ec_i of fine structures
        will be
        
            ec_i = Ec_i * (1 + dE/Ec)
        
        where Ec_i is the theoretical (experimental) center energy of fine
        structures and Ec is the center energy of the overall profile, which
        is the weighted sum of each component profiles.
        
        If shift is True, ec_i will simply be
        
            ec_i = Ec_i + dE.
    """
    
    # Sanity check
    if line not in LE:
        raise ValueError("No data for %s" % line)

    if line not in FS:
        raise ValueError("No data for %s" % line)
    
    # Boundary check
    width = 0 if width < 0 else width
    
    # Center energy
    Ec = np.exp(np.log(np.asarray(FS[line])[:,(0,2)]).sum(axis=1)).sum()
    
    if shift:
        model = np.array([ p[2] * voigt(E, p[0]+dE, p[1], width) for p in FS[line] ])
    
        if tail and tR is not None and tT is not None:
            model = model * (1 - tR) + np.array([ p[2] * tR * voigt_with_tail(E, p[0]+dE, p[1], width, tT) for p in FS[line] ])
    else:
        model = np.array([ p[2] * voigt(E, p[0]*(1+(0 if Ec == 0 else dE/Ec)), p[1], width) for p in FS[line] ])

        if tail and tR is not None and tT is not None:
            model = model * (1 - tR) + np.array([ p[2] * tR * voigt_with_tail(E, p[0]*(1+(0 if Ec == 0 else dE/Ec)), p[1], width, tT) for p in FS[line] ])

    if full:
        return np.nan_to_num(np.vstack((model.sum(axis=0)[np.newaxis], model)))
    else:
        return np.nan_to_num(model.sum(axis=0))

def group_bin(n, bins, min=100):
    """
    Group PHA bins to have at least given number of minimum counts
    
    Parameters (and their default values):
        n:      counts
        bins:   bin edges
        min:    minimum counts to group (Default: 100)
    
    Return (grouped_n, grouped_bins)
        grouped_n:      grouped counts
        grouped_bins:   grouped bin edges
    """
    
    grp_n = []
    grp_bins = [bins[0]]

    n_sum = 0

    for p in zip(n, bins[1:]):
        n_sum += p[0]
        
        if n_sum >= min:
            grp_n.append(n_sum)
            grp_bins.append(p[1])
            n_sum = 0
    
    return np.asarray(grp_n), np.asarray(grp_bins)

def histogram(pha, binsize=1.0):
    """
    Create histogram
    
    Parameter:
        pha:        pha data (array-like)
        binsize:    size of bin in eV (Default: 1.0 eV)
    
    Return (n, bins)
        n:      photon count
        bins:   bin edge array
    
    Note:
        - bin size is 1eV/bin.
    """
    
    # Create histogram
    bins = np.arange(np.floor(pha.min()/binsize)*binsize, np.ceil(pha.max()/binsize)*binsize+binsize, binsize) 
    n, bins = np.histogram(pha, bins=bins)
    
    return n, bins

def fit(pha, binsize=1, min=None, line="MnKa", shift=False, tail=False, freeze=None, method='c', error=True, filename=None, tex=False):
    """
    Fitting of line spectrum
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        binsize:    size of energy bin in eV for histogram (Default: 1)
        min:        minimum counts to group bins or None for default (Default: None)
        line:       line to fit (Default: MnKa)
        shift:      treat dE as shift if True instead of scaling (Default: False)
        tail:       enable low energy tail (Default: False)
        freeze:     array-like of x0 to freeze or None to thaw (Default: None)
        method:     fitting method among c/mle/cs/ls (Default: c)
        error:      calculate error (Default: True)
        filename:   if given, make a plot and save it with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    Return (dEc, width), (dEc_error, width_error), (stat, dof) when tail is False,
        or (dEc, width, tR, tT), (dEc_error, width_error, tR_error, tT_error), (stat, dof) when tail is True

        dEc:            shift from line center
        width:          fitted gaussian width (FWHM)
        tR:             fitted low energy tail ratio
        tT:             fitted low energy tail tau
        dEc_error:      dEc error (1-sigma)
        width_error:    width error (1-sigma)
        tR_error:       tR error (1-sigma)
        tT_error:       tT error (1-sigma)
        stat:           statistics
        dof:            number of degree of freedom
    """
    
    # Sanity check
    if line not in FS:
        raise ValueError("No data for %s" % line)
    
    # Create histogram
    n, bins = histogram(pha, binsize=binsize)
    
    # Group bins
    if min is None:
        if method in ('c', 'mle'):
            min = 1
        else:
            min = 10

    gn, gbins = group_bin(n, bins, min)
    ngn = gn/np.diff(gbins)   # normalized counts in counts/eV
    ngn_sigma = np.sqrt(gn)/np.diff(gbins)
    
    bincenters = (gbins[1:]+gbins[:-1])/2

    def stat(args, bounds=None):
        # arg = (dEc, width, tR, tT)
        
        # Boundary check
        if bounds is not None:
            for x, (lower, upper) in zip(args, bounds):
                if not (upper >= x >= lower):
                    # Out of boundary
                    return np.inf
        
        # Model
        m = len(pha)*line_model(bincenters, *args, line=line, shift=shift, tail=tail)
        
        # Truncation (only for C stat and MLE)
        mask = m > 1e-25
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            if method == 'c':
                s = 2*(m[mask] - ngn[mask] + ngn[mask]*(np.log(ngn[mask]) - np.log(m[mask]))).sum()
            elif method == 'mle':
                s = (-np.log(m[mask])*ngn[mask]).sum()
            elif method == 'cs':
                s = ((ngn-m)**2/ngn_sigma**2).sum()
            elif method == 'ls':
                s = ((ngn-m)**2).sum()
            else:
                raise ValueError("Unsupported fitting method %s" % method)
        
            return s

    # Build initial params and bounds (x0 will be ignored)
    x0 = (0, np.std(pha)/2)
    bounds = ((-np.inf, np.inf), (0.01, np.inf))
    initial_simplex = ((-1, 0.5), (1, 0.5), (-1, 5))
    
    if tail:
        x0 += (0.2, 5)
        bounds += ((0, 1), (0.01, np.inf))
        initial_simplex = (
            (-1, 0.5, 0.05, 1),
            (1, 0.5, 0.05, 1),
            (-1, 5, 0.05, 1),
            (-1, 0.5, 0.5, 1),
            (-1, 0.5, 0.05, 10))

    # Parameter freezing
    if freeze is not None:
        # Backup originals
        _x0 = x0
        _bounds = bounds
        _initial_simplex = initial_simplex
        _stat = stat
        
        x0 = [ [_x] if _f is None else [] for _x, _f in zip(_x0, freeze) ]
        bounds = [ [_b] if _f is None else [] for _b, _f in zip(_bounds, freeze) ]
        
        x0 = tuple(reduce(lambda x, y: x+y, x0))
        bounds = tuple(reduce(lambda x, y: x+y, bounds))
        
        mask = [True] + [ True if _f is None else False for _f in freeze ]
        initial_simplex = np.array(initial_simplex)[mask]
        initial_simplex = initial_simplex.T[mask[1:]].T
        
        # Create wrapper function for stat
        def stat(args, bounds=None):
            # Replace args with frozen params
            args = list(args)
            _args = tuple([ args.pop(0) if _x is None else _x for _x in freeze ])
            
            if bounds is not None:
                # Rebuild bounds array
                bounds = list(bounds)
                _bounds = tuple([ bounds.pop(0) if _x is None else (-np.inf, np.inf) for _x in freeze ])
            else:
                _bounds = None
            
            return _stat(_args, _bounds)
    
    if freeze is not None and reduce(lambda x, y: x&y, [ True if _f is not None else False for _f in freeze ]):
        # All parameters are frozen
        x = x0
        s = stat(x0)
        dof = len(bincenters)
    else:
        # Minimize
        res = minimize(stat, x0=x0, method='Nelder-Mead', args=(bounds,), options={'initial_simplex': initial_simplex, 'maxiter': 500*len(x0)})
    
        if not res['success']:
            raise Exception("Fitting failed for %s" % line)

        x = res['x']
        s = res['fun']
        dof = len(bincenters) - len(x)
    
    if freeze is None:
        dE, width = x[:2]
        
        if tail:
            tR, tT = x[2:]
        else:
            tR, tT = 0, 0
    else:
        _x = list(x)
        
        dE, width = [ _x.pop(0) if _f is None else _f for _f in freeze[:2] ]
        
        if tail:
            tR, tT = [ _x.pop(0) if _f is None else _f for _f in freeze[2:] ]
        else:
            tR, tT = 0, 0

    if freeze is not None and reduce(lambda x, y: x&y, [ True if _f is not None else False for _f in freeze ]):
        # All parameters are frozen
        dE_e, width_e, tR_e, tT_e = -1, -1, -1, -1
    else:
        # Calculate Hessian matrix for standard error
        if freeze is None:
            dE_e, width_e, tR_e, tT_e = None, None, None, None
        else:
            dE_e, width_e = [ _x if _x is None else -1 for _x in freeze[:2] ]

            if tail:
                tR_e, tT_e = [ _x if _x is None else -1 for _x in freeze[2:] ]
        
        if error:
            try:
                import numdifftools as nd

                hess = nd.Hessian(stat)
                # Somehow we need x2 for the inversed hess to get the right error
                err = np.sqrt(np.diag(np.linalg.inv(hess(x))*2))

                if freeze is None:
                    dE_e, width_e = err[:2]
            
                    if tail:
                        tR_e, tT_e = err[2:]
                    else:
                        tR_e, tT_e = 0, 0
                else:
                    _err = list(err)
            
                    dE_e, width_e = [ _err.pop(0) if _x is None else -1 for _x in freeze[:2] ]
        
                    if tail:
                        tR_e, tT_e = [ _err.pop(0) if _x is None else -1 for _x in freeze[2:] ]
                    else:
                        tR_e, tT_e = 0, 0
        
                dE_e, width_e, tR_e, tT_e = np.nan_to_num([dE_e, width_e, tR_e, tT_e])

            except ImportError:
                print("Warning: Plese install numdifftools to calculate standard error.")
                
                # No error
                dE_e, width_e, tR_e, tT_e = 0, 0, 0, 0
                
            except np.linalg.LinAlgError:
                pass
    
    if filename is not None:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = str(tex)
        from matplotlib import pyplot as plt

        plt.figure(figsize=(8, 6))
        
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

        if width < 1:
            text = r'FWHM$=%.4f$' % width + (' eV' if width_e is None else ' eV (fixed)' if width_e < 0 else r'$\pm %.4f$ eV' % width_e)
        else:
            text = r'FWHM$=%.2f$' % width + (' eV' if width_e is None else ' eV (fixed)' if width_e < 0 else r'$\pm %.2f$ eV' % width_e)
        if dE < 0.1:
            text += '\n' + r'$\Delta$Ec$=%.4f$' % dE + (' eV' if dE_e is None else ' eV (fixed)' if dE_e < 0 else r'$\pm %.4f$ eV' % dE_e)
        else:
            text += '\n' + r'$\Delta$Ec$=%.2f$' % dE + (' eV' if dE_e is None else ' eV (fixed)' if dE_e < 0 else r'$\pm %.2f$ eV' % dE_e)
        if tail:
            text += '\n' + r'Low-$E$ tail:'
            text += '\n' + r'  fraction: $%.2f$' % tR + ('' if tR_e is None else ' (fixed)' if tR_e < 0 else r'$\pm %.2f$' % tR_e)
            text += '\n' + r'  decay: $%.2f$' % tT + (' eV' if tT_e is None else ' eV (fixed)' if tT_e < 0 else r'$\pm %.2f$ eV' % tT_e)
        if method == 'c':
            text += '\n' + r'c-stat = $%.1f$' % s
            text += '\n' + r'd.o.f. = $%d$' % dof
        elif method == 'cs':
            text += '\n' + r'Reduced $\chi^2$ = %.1f/%d = %.2f' % (s, dof, s/dof)
        text += '\n' + r'$%d$ counts' % len(pha)

        ax1.errorbar(bincenters, ngn, yerr=ngn_sigma, xerr=np.diff(gbins)/2, capsize=0, ecolor='k', fmt=None)

        E = np.linspace(bins.min(), bins.max(), 1000)
        m = len(pha)*line_model(E, dE, width, tR, tT, line=line, shift=shift, tail=tail, full=True)

        # Plot theoretical model
        ax1.plot(E, m[0], 'r-')

        # Plot fine structures
        if len(m) > 2:
            ax1.plot(E, m[1:].T, 'b--')

        ax1.set_ylabel(r'Normalized Counts/eV')
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax*1.1)
        ax1.text(0.02, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)
        
        ax1.ticklabel_format(axis='x', style='plain')
        xtl1 = ax1.get_xticklabels()
        plt.setp(xtl1, visible=False)
        
        # Plot residuals
        m = len(pha)*line_model(bincenters, dE, width, tR, tT, line=line, shift=shift, tail=tail)
        
        ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
        
        ax2.axhline(y=0, c='r')
        ax2.errorbar(bincenters, (ngn-m)/ngn_sigma, yerr=1, xerr=np.diff(gbins)/2, capsize=0, ecolor='k', fmt=None)
        ax2.set_xlabel(r'Energy$\quad$(eV)')
        ax2.set_ylabel(r'$\Delta/\sqrt{\lambda}$')
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(filename)
    
    if tail:
        return (dE, width, tR, tT), (dE_e, width_e, tR_e, tT_e), (s, dof)
    else:
        return (dE, width), (dE_e, width_e), (s, dof)

def fit_simul(pha, binsize=1, min=None, line="MnKa", shift=False, tail=False, freeze=None, method='c', error=True, filename=None, tex=False):
    """
    Fitting of line spectrum
    
    Parameters (and their default values):
        pha:        pha data (list of array-like)
        binsize:    size of energy bin in eV for histogram (Default: 1)
        min:        minimum counts to group bins or None for default (Default: None)
        line:       line to fit (Default: MnKa)
        shift:      treat dE as shift if True instead of scaling (Default: False)
        tail:       enable low energy tail (Default: False)
        freeze:     array-like of x0 to freeze, None to thaw, False to link (Default: None)
        method:     fitting method among c/mle/cs/ls (Default: c)
        error:      calculate error (Default: True)
        filename:   if given, make a plot and save it as a multipage PDF with the given name (Default: None)
        tex:        use tex for the plot (Default: False)
    
    None:
        * freeze could be either:
            None
            (dEc, width) or (dEc, width, tR, tT)
        * some examples for freeze
            (0, None)                   => fix dEc to 0 and fit width individually
            (None, False)               => thaw dEc and link width
            (None, None, False, False)  => link tR ant tT
            (None, None, 0.2, False)    => fix tR to 0.2 and link tT
    
    Return (dEc, width), (dEc_error, width_error), (stat, dof) when tail is False,
        or (dEc, width, tR, tT), (dEc_error, width_error, tR_error, tT_error), (stat, dof) when tail is True

        dEc:            shift from line center
        width:          fitted gaussian width (FWHM)
        tR:             fitted low energy tail ratio
        tT:             fitted low energy tail tau
        dEc_error:      dEc error (1-sigma)
        width_error:    width error (1-sigma)
        tR_error:       tR error (1-sigma)
        tT_error:       tT error (1-sigma)
        stat:           statistics
        dof:            number of degree of freedom
    """
    
    # Sanity check
    if line not in FS:
        raise ValueError("No data for %s" % line)
    
    # Create histogram
    n, bins = np.array([ histogram(_pha, binsize=binsize) for _pha in pha ]).T
    
    # Group bins
    if min is None:
        if method in ('c', 'mle'):
            min = 1
        else:
            min = 10

    gn, gbins = np.array([ group_bin(_n, _bins, min) for _n, _bins in zip(n, bins) ]).T
    ngn = [ _gn/np.diff(_gbins) for _gn, _gbins in zip(gn, gbins) ] # normalized counts in counts/eV
    ngn_sigma = [ np.sqrt(_gn)/np.diff(_gbins) for _gn, _gbins in zip(gn, gbins) ]
    
    bincenters = [ (_gbins[...,1:]+_gbins[...,:-1])/2 for _gbins in gbins ]

    # Build initial params and bounds
    if freeze is None:
        _freeze = None
    else:
        # Change False to None for individual fitting (and this needs to be taken care later)
        _freeze = tuple([ None if _f is True else _f for _f in freeze ])
    
    x0 = [ fit(_pha, binsize=binsize, min=min, line=line, shift=shift, tail=tail, freeze=_freeze, method=method, error=False, filename=None, tex=False)[0] for _pha in pha ]
    
    if tail:
        bounds = (((-np.inf, np.inf), (0.01, np.inf), (0, 1), (0.01, np.inf)),)*len(pha)
    else:
        bounds = (((-np.inf, np.inf), (0.01, np.inf)),)*len(pha)

    def stat(args, bounds=None):
        # arg = (dEc, width, tR, tT)
        
        # Rebuild full parameter list
        x0, bounds = _fit_param_rebuild(args, bounds, freeze=freeze, N=len(pha))
        
        # List for results
        s = []
        
        for i, _pha in enumerate(pha):
            # Boundary check
            if bounds is not None:
                for x, (lower, upper) in zip(x0[i], bounds[i]):
                    if not (upper >= x >= lower):
                        # Out of boundary
                        s.append(np.inf)
                        continue

            # Model
            m = len(_pha)*line_model(bincenters[i], *x0[i], line=line, shift=shift, tail=tail)
        
            # Truncation (only for C stat and MLE)
            mask = m > 1e-25
        
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
        
                if method == 'c':
                    _s = 2*(m[mask] - ngn[i][mask] + ngn[i][mask]*(np.log(ngn[i][mask]) - np.log(m[mask]))).sum()
                elif method == 'mle':
                    _s = (-np.log(m[mask])*ngn[i][mask]).sum()
                elif method == 'cs':
                    _s = ((ngn[i]-m)**2/ngn_sigma[i]**2).sum()
                elif method == 'ls':
                    _s = ((ngn[i]-m)**2).sum()
                else:
                    raise ValueError("Unsupported fitting method %s" % method)
        
                s.append(_s)
        
        return np.array(s).sum()
    
    if freeze is not None and reduce(lambda x, y: x&y, [ False if _f is None or _f is True else True for _f in freeze ]):
        # All parameters are frozen
        x = x0
        s = stat(x0)
        dof = np.array([ len(_bincenters) for _bincenters in bincenters ]).sum()
    else:
        # Flatten parameters
        _x0, _bounds = _fit_param_flatten(x0, bounds, freeze=freeze, N=len(pha))
        
        # Minimize
        res = minimize(stat, x0=_x0, method='L-BFGS-B', bounds=_bounds, options={'maxiter': 500*len(_x0)})
    
        if not res['success']:
            raise Exception("Fitting failed for %s" % line)
        
        x = res['x']
        s = res['fun']
        dof = np.array([ len(_bincenters) for _bincenters in bincenters ]).sum() - len(x)

    # Rebuild parameter arrays
    x, bounds = _fit_param_rebuild(x, freeze=freeze, N=len(pha))
    
    if freeze is not None and reduce(lambda x, y: x&y, [ False if _f is None or _f is True else True for _f in freeze ]):
        # All parameters are frozen
        err = ((-1, -1, -1, -1),)*len(pha)
    else:
        # Calculate Hessian matrix for standard error
        if error:
            try:
                import numdifftools as nd

                hess = nd.Hessian(stat)
                
                # Somehow we need x2 for the inversed hess to get the right error
                err = np.nan_to_num(np.sqrt(np.diag(np.linalg.inv(hess(res['x']))*2)))

                # Rebuild error arrays
                err = _fit_error_rebuild(err, freeze=freeze, N=len(pha))

            except ImportError:
                print("Warning: Plese install numdifftools to calculate standard error.")
                
                # No error
                err = ((0, 0, 0, 0),)*len(pha)
                
            except np.linalg.LinAlgError:
                pass
    
    if filename is not None:
        import matplotlib
        from matplotlib.backends.backend_pdf import PdfPages
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = str(tex)
        from matplotlib import pyplot as plt
        
        pdf = PdfPages(filename)

        for i, _pha in enumerate(pha):
            fig = plt.figure(figsize=(8, 6))
        
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

            text = r'FWHM$=%.2f$' % x[i][1] + (' eV' if err[i][1] is None else ' eV (fixed)' if err[i][1] < 0 else r'$\pm %.2f$ eV' % err[i][1])
            text += '\n' + r'$\Delta$Ec$=%.2f$' % x[i][0] + (' eV' if err[i][0] is None else ' eV (fixed)' if err[i][0] < 0 else r'$\pm %.2f$ eV' % err[i][0])
            if tail:
                text += '\n' + r'Low-$E$ tail:'
                text += '\n' + r'  fraction: $%.2f$' % x[i][2] + ('' if err[i][2] is None else ' (fixed)' if err[i][2] < 0 else r'$\pm %.2f$' % err[i][2])
                text += '\n' + r'  decay: $%.2f$' % x[i][3] + (' eV' if err[i][3] is None else ' eV (fixed)' if err[i][3] < 0 else r'$\pm %.2f$ eV' % err[i][3])
            if method == 'c':
                text += '\n' + r'c-stat = $%.1f$' % s
                text += '\n' + r'd.o.f. = $%d$' % dof
            elif method == 'cs':
                text += '\n' + r'Reduced $\chi^2$ = %.1f/%d = %.2f' % (s, dof, s/dof)
            text += '\n' + r'$%d$ counts' % len(_pha)
            
            ax1.errorbar(bincenters[i], ngn[i], yerr=ngn_sigma[i], xerr=np.diff(gbins[i])/2, capsize=0, ecolor='k', fmt=None)

            E = np.linspace(bins[i].min(), bins[i].max(), 1000)
            m = len(_pha)*line_model(E, *x[i], line=line, shift=shift, tail=tail, full=True)

            # Plot theoretical model
            ax1.plot(E, m[0], 'r-')

            # Plot fine structures
            if len(m) > 2:
                ax1.plot(E, m[1:].T, 'b--')

            ax1.set_ylabel(r'Normed Count$\quad$(count/eV)')
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(ymin, ymax*1.1)
            ax1.text(0.02, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)
        
            # Plot residuals
            m = len(_pha)*line_model(bincenters[i], *x[i], line=line, shift=shift, tail=tail)
        
            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
        
            ax2.axhline(y=0, c='r')
            ax2.errorbar(bincenters[i], (ngn[i]-m)/ngn_sigma[i], yerr=1, xerr=np.diff(gbins[i])/2, capsize=0, ecolor='k', fmt=None)
            ax2.set_xlabel(r'Energy$\quad$(eV)')
            ax2.set_ylabel(r'$\Delta/\sqrt{\lambda}$')
        
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.5)
            pdf.savefig()

        pdf.close()        
    
    return x, err, (s, dof)