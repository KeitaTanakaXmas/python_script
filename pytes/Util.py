import warnings
import numpy as np
import time
from datetime import datetime
from struct import unpack
from .Filter import median_filter
from . import Analysis, Filter, Constants

def savefits(data, filename, vmax=1.0, sps=1e6, bits=14, noise=False, clobber=True):
    """
    Save pulse/noise to FITS file
    """
    
    import astropy.io.fits as pf
    
    # Prepare data
    data = (np.asarray(data)/vmax*2**(bits-1)).round()
    
    # Column Name
    if noise:
    	colname = 'NoiseRec'
    else:
    	colname = 'PulseRec'
        
    # Columns
    col_t = pf.Column(name='TIME', format='1D', unit='s', array=np.zeros(data.shape[0], dtype=int))
    col_data = pf.Column(name=colname, format='%dI' % data.shape[1], unit='V', array=data)
    
    cols = pf.ColDefs([col_t, col_data])
    tbhdu = pf.BinTableHDU.from_columns(cols)
    
    # Name of extension
    exthdr = tbhdu.header
    exthdr['EXTNAME'] = ('Record', 'name of this binary table extension')
    exthdr['EXTVER'] = (1, 'extension version number')
    
    # Add more attributes
    exthdr['TSCAL2'] = (vmax/2**(bits-1), '[V/ch]')
    exthdr['TZERO2'] = (0., '[V]')
    exthdr['THSCL2'] = (sps**-1, '[s/bin] horizontal resolution of record')
    exthdr['THZER2'] = (0, '[s] horizontal offset of record')
    exthdr['THSAM2'] = (data.shape[1], 'sample number of record')
    exthdr['THUNI2'] = ('s', 'physical unit of sampling step of record')
    exthdr['TRMIN2'] = (-2**(bits-1)+1, '[channel] minimum number of each sample')
    exthdr['TRMAX2'] = (2**(bits-1)-1, '[channel] maximum number of each sample')
    exthdr['TRBIN2'] = (1, '[channel] default bin number of each sample')
    
    # More attributes
    exthdr['TSTART'] = (0, 'start time of experiment in total second')
    exthdr['TSTOP'] = (0, 'end time of experiment in total second')
    exthdr['TEND'] = (0, 'end time of experiment (obsolete)')
    exthdr['DATE'] = (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), 'file creation date (UT)')
    
    # We anyway need Primary HDU
    hdu = pf.PrimaryHDU()
    
    # Write to FITS
    thdulist = pf.HDUList([hdu, tbhdu])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thdulist.writeto(filename, clobber=clobber)

def fopen(filename):
    """
    Read FITS file
    
    Parameters
    ==========
        filename:   file number to read
    
    Returns
    =======
        t:          time array
        wave:       waveform array
    """
    
    import astropy.io.fits as pf

    # Open fits file and get pulse/noise data
    header = pf.open(filename)
    wave = header[1].data.field(1).copy()
    dt = header[1].header['THSCL2']
    t = np.arange(wave.shape[-1]) * dt
    header.close()
    
    return t, wave

def yopen(filenumber, summary=False, nf=None, tmin=None, tmax=None, raw=False):
    """
    Read Yokogawa WVF file
    
    Parameters
    ==========
        filenumber: file number to read
        summary:    to summary waves (default: False)
        nf:         sigmas for valid data using median noise filter, None to disable noise filter (default: None)
        tmin:       lower boundary of time for partial extraction, scaler or list (Default: None)
        tmax:       upper boundary of time for partial extraction, scaler or list (Default: None)
        raw:        returns raw data without scaling/offsetting if True (Default: False)
    
    Returns
    =======
        if summary is False:
            [ t1, d1, t2, d2, t3, d3, ... ]
        
        if summary is True:
            [ t1, d1, err1, t2, d2, err2, ... ]
        
        if raw is True:
            t1 is a tuple of (hres1, hofs1, vres1, vofs1)
        
        where t1 is timing for 1st ch, d1 is data for 1st ch, err1 is error (1sigma) for 1st ch, and so on.
    """
    
    # Read header (HDR)
    h = open(str(filenumber) + ".HDR")
    lines = h.readlines()
    h.close()
    
    # Parse $PublicInfo
    for line in lines:
        token = line.split()
        
        if len(token) > 0:
            # Check endian
            if token[0] == "Endian":
                endian = '>' if token[1] == "Big" else '<'
            
            # Check data format
            if token[0] == "DataFormat":
                format = token[1]
                assert format == "Block"
                
            # Check # of groups
            if token[0] == "GroupNumber":
                groups = int(token[1])
            
            # Check # of total traces
            if token[0] == "TraceTotalNumber":
                ttraces = int(token[1])
            
            # Check data offset
            if token[0] == "DataOffset":
                offset = int(token[1])
    
    # Initialize containers
    traces = [None] * groups        # Number of traces for each group
    blocks = [None] * ttraces       # Number of blocks for each trace
    bsizes = [None] * ttraces       # Block size for each trace
    vres = [None] * ttraces         # VResolution for each trace
    voffset = [None] * ttraces      # VOffset for each trace
    hres = [None] * ttraces         # HResolution for each trace
    hoffset = [None] * ttraces      # HOffset for each trace
    
    # Parse $Group
    for line in lines:
        token = line.split()

        if len(token) > 0:
            # Read current group number
            if token[0][:6] == "$Group":
                cgn = int(token[0][6:]) - 1  # Current group number (minus 1)
            
            # Check # of traces in this group
            if token[0] == "TraceNumber":
                traces[cgn] = int(token[1])
                traceofs = np.sum(traces[:cgn], dtype=int)
                        
            # Check # of Blocks
            if token[0] == "BlockNumber":
                blocks[traceofs:traceofs+traces[cgn]] = [ int(token[1]) ] * traces[cgn]
            
            # Check Block Size
            if token[0] == "BlockSize":
                bsizes[traceofs:traceofs+traces[cgn]] = [ int(s) for s in token[1:] ]
            
            # Check VResolusion
            if token[0] == "VResolution":
                vres[traceofs:traceofs+traces[cgn]] = [ float(res) for res in token[1:] ]
            
            # Check VOffset
            if token[0] == "VOffset":
                voffset[traceofs:traceofs+traces[cgn]] = [ float(ofs) for ofs in token[1:] ]
            
            # Check VDataType
            if token[0] == "VDataType":
                assert token[1] == "IS2"
            
            # Check HResolution
            if token[0] == "HResolution":
                hres[traceofs:traceofs+traces[cgn]] = [ float(res) for res in token[1:] ]
            
            # Check HOffset
            if token[0] == "HOffset":
                hoffset[traceofs:traceofs+traces[cgn]] = [ float(ofs) for ofs in token[1:] ]
        
    # Data Initialization
    time = [ np.array(list(range(bsizes[t]))) * hres[t] + hoffset[t] for t in range(ttraces) ]
    data = [ [None] * blocks[t] for t in range(ttraces) ]
    
    # Open WVF
    f = open(str(filenumber) + ".WVF", 'rb')
    f.seek(offset)
    
    # Read WVF
    if format == "Block":
        # Block format (assuming block size is the same for all the traces in Block format)
        for b in range(blocks[0]):
            for t in range(ttraces):
                if raw:
                    data[t][b] = np.array(unpack(endian + 'h'*bsizes[t], f.read(bsizes[t]*2)), dtype='int64')
                else:
                    data[t][b] = np.array(unpack(endian + 'h'*bsizes[t], f.read(bsizes[t]*2))) * vres[t] + voffset[t]
    else:
        # Trace format
        for t in range(ttraces):
            for b in range(blocks[t]):
                if raw:
                    data[t][b] = np.array(unpack(endian + 'h'*bsizes[t], f.read(bsizes[t]*2)), dtype='int64')
                else:
                    data[t][b] = np.array(unpack(endian + 'h'*bsizes[t], f.read(bsizes[t]*2))) * vres[t] + voffset[t]

    # Array conversion
    for t in range(ttraces):
        if raw:
            data[t] = np.array(data[t], dtype='int64')
        else:
            data[t] = np.array(data[t])
            
    
    # Tmin/Tmax filtering
    for t in range(ttraces):
        if type(tmin) == list or type(tmax) == list:
            if not (type(tmin) == list and type(tmax) == list and len(tmin) == len(tmax)):
                raise ValueError("tmin and tmax both have to be list and have to have the same length.")
            mask = np.add.reduce([ (time[t] >= _tmin) & (time[t] < _tmax) for (_tmax, _tmin) in zip(tmax, tmin)], dtype=bool)
        else:
            _tmin = np.min(time[t]) if tmin is None else tmin
            _tmax = np.max(time[t]) + 1 if tmax is None else tmax
            mask = (time[t] >= _tmin) & (time[t] < _tmax)
        
        data[t] = data[t][:, mask]
        time[t] = time[t][mask]
        
    f.close()
    
    if summary is False:
        # Return wave data as is
        if raw:
            return [ [ (hres[t], hoffset[t], vres[t], voffset[t]), data[t] ] for t in range(ttraces)  ]
        else:
            return [ [ time[t], data[t] ] for t in range(ttraces)  ]
    else:
        if nf is None:
            # Noise filter is off
            if raw:
                return [ [ (hres[t], hoffset[t], vres[t], voffset[t]), np.mean(data[t].astype(dtype='float64'), axis=0), np.std(data[t].astype(dtype='float64'), axis=0, ddof=1) ]
                            for t in range(ttraces) ]
            else:
                return [ [ time[t], np.mean(data[t], axis=0), np.std(data[t], axis=0, ddof=1) ]
                            for t in range(ttraces) ]
        else:
            # Noise filter is on
            if raw:
                return [ [ (hres[t], hoffset[t], vres[t], voffset[t]),
                            np.apply_along_axis(lambda a: np.mean(a[median_filter(a, nf)]), 0,  data[t].astype(dtype='float64')),
                            np.apply_along_axis(lambda a: np.std(a[median_filter(a, nf)], ddof=1), 0, data[t].astype(dtype='float64')) ]
                                for t in range(ttraces) ]
            else:
                return [ [ time[t],
                            np.apply_along_axis(lambda a: np.mean(a[median_filter(a, nf)]), 0,  data[t]),
                            np.apply_along_axis(lambda a: np.std(a[median_filter(a, nf)], ddof=1), 0, data[t]) ]
                                for t in range(ttraces) ]

def popen(filename, ch=None, raw=False, bufsize=1024*1024):
    """
    Read pls file
    
    Parameters
    ==========
        filename:   file name to read
        ch:         returns data only for the given channel if given (Default: None)
        raw:        returns raw data without scaling/offsetting if True (Default: False)
        bufsize:    file read buffer size (Default: 1 MB)
    
    Returns
    =======
        if raw is True:
            [ header, vres, vofs, hres, hofs, timestamp, num, data, edata ]
        else:
            [ header, t, timestamp, num, data ]
    """
    
    # Initialize
    header = {'COMMENT': []}
    vres = {}
    vofs = {}
    hres = {}
    hofs = {}
    ts = {}
    num  = {}
    data = {}
    edata = {}
    
    preciseTime = False
    
    # Parser
    def parser():
        """
        PLS Data Parser (generator)
        """
        
        # Initialization
        samples = -1
        extra = 0
        chunk = b''
        isHeader = True
        
        while True:
            while len(chunk) < 2:
                chunk += yield
        
            # Get the magic character
            magic = chunk[0]

            if isHeader and magic == ord(b'C'):
                # Comment
                while len(chunk) < 80:
                    chunk += yield
                header['COMMENT'].append(chunk[2:80].decode())
                chunk = chunk[80:]
            
            elif isHeader and magic == ord(b'V'):
                # Version
                while len(chunk) < 80:
                    chunk += yield
                header['VERSION'] = chunk[2:80].decode()
                chunk = chunk[80:]
            
            elif isHeader and magic == ord(b'O'):
                # Date
                while len(chunk) < 10:
                    chunk += yield
                _m, _d, _y = list(map(int, chunk[2:10].decode().split()))
                header['DATE'] = "%d/%d/%d" % (_y, _m, _d)
                chunk = chunk[10:]

            elif isHeader and magic == ord(b'S'):
                # Number of Samples
                while len(chunk) < 7:
                    chunk += yield
                header['SAMPLES'] = samples = int(chunk[2:7])
                chunk = chunk[7:]

            elif isHeader and magic == ord(b'E'):
                # Extra Bytes
                while len(chunk) < 7:
                    chunk += yield
                header['EXTRA'] = extra = int(chunk[2:7])
                chunk = chunk[7:]
                
            elif isHeader and magic == ord(b'P'):
                # Discriminator
                while len(chunk) < 78:
                    chunk += yield
                _dis = chunk[2:78].decode().split()
                if _dis[0] == '01':
                    header['ULD'] = eval(_dis[1])
                elif _dis[0] == '02':
                    header['LLD'] = eval(_dis[1])
                chunk = chunk[78:]
            
            elif isHeader and magic == ord(b'N'):
                # Normalization
                while len(chunk) < 47:
                    chunk += yield
                _ch, _hofs, _hres, _vofs, _vres = chunk[2:47].decode().split()
                _ch = int(_ch)
                vres[_ch] = eval(_vres)
                vofs[_ch] = eval(_vofs)
                hres[_ch] = eval(_hres)
                hofs[_ch] = eval(_hofs)
                chunk = chunk[47:]
            
            elif magic == ord(b'D'):
                # Data
                isHeader = False
                
                if samples < 0:
                    raise ValueError("Invalid number of samples.")
                while len(chunk) < (11 + samples*2):
                    chunk += yield
                _ch, _ts, _num = unpack('<BII', chunk[2:11])

                if ch is not None and _ch != ch:
                    pass
                else:
                    if _ch not in data:
                        data[_ch] = bytearray()
                        ts[_ch] = []
                        num[_ch] = []
                        edata[_ch] = bytearray()
            
                    data[_ch] += chunk[11:11 + samples*2]
                    ts[_ch].append(_ts)
                    num[_ch].append(_num)
                    edata[_ch] += chunk[11 + samples*2:11 + samples*2 + extra]
                
                chunk = chunk[11 + samples*2 + extra:]
                
            else:
                # Skip unknown magic
                chunk = chunk[1:]
    
    # Open pls file and read by chunks
    f = open(filename, 'rb')

    # Start parser
    p = parser()
    next(p)
    
    # Read by chunk and parse it
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            p.send(chunk)

    # Convert buffer to numpy array
    for k in ([ch] if ch is not None else list(data.keys())):
        data[k] = np.frombuffer(data[k], dtype='>i2').astype('>i4').reshape(-1, header['SAMPLES'])
        if 'EXTRA' in header and header['EXTRA'] == 4:
            edata[k] = np.frombuffer(edata[k], dtype='<u4')
            preciseTime = True
        else:
            edata[k] = np.frombuffer(edata[k], dtype='>u1').reshape(-1, data[k].shape[0])
    
    if raw:
        if ch is not None:
            return header, vres[ch], vofs[ch], hres[ch], hofs[ch], ts[ch], num[ch], data[ch], edata[ch]
        else:
            return header, vres, vofs, hres, hofs, ts, num, data, edata
    else:
        t = {}

        for k in ([ch] if ch is not None else list(data.keys())):
            # Normalize data using res/ofs
            t[k] = (np.arange(header['SAMPLES']) + hofs[k]) * hres[k]
            data[k] = (np.asarray(data[k]) + vofs[k]) * vres[k]
            
            # Precise Time
            if preciseTime:
                ts[k] = ts[k] + edata[k] * 1e-6
        
        if ch is not None:
            return header, t[ch], ts[ch], num[ch], data[ch]
        else:
            return header, t, ts, num, data

def xopen(filename, ch=None, raw=False):
    """
    Read xdf file
    
    Parameters
    ==========
        filename:   file name to read
        ch:         returns data only for the given channel if given (Default: None)
        raw:        returns raw data without scaling/offsetting if True (Default: False)
    
    Returns
    =======
        if raw is True:
            [ header, vres, vofs, hres, hofs, tick, num, data, edata ]
        else:
            [ header, t, tick, num, data, edata ]
    """
    
    # Open pls file and read by chunks
    f = open(filename, 'rb')

    # Read header
    h = f.read(1024)
    
    runnum, filenum, xdfdate, xdftime, detector, squid, comments = unpack('8s8s8s8s32s32s256s', h[:352])
    nchan, samples, sps, num, pretrig, trigsrc, trigslope, trigcpl, triglevel, trigtimeout = unpack('1s8s8s8s8s3s3s2s8s8s', h[352:409])
    chAcpl, chArange, chBcpl, chBrange, chCcpl, chCrange, chDcpl, chDrange = unpack('2s3s2s3s2s3s2s3s', h[409:429])
    acqcardid, acqprog, acqver = unpack('8s16s3s', h[429:456])
    dummy, bits = unpack('567s1s', h[456:])
    
    # Reformat
    xdfdate = time.mktime(datetime.strptime(xdfdate.strip(), '%Y%m%d').timetuple())
    nchan = int(nchan.strip())
    samples = int(samples.strip())

    # Initialize for parser
    enum = []
    tick = []
    data = []
    
    # Parser
    def parser():
        """
        XDF Data Parser (generator)
        """
        
        # Initialize
        chunk = ''
        data.append(bytearray())
        
        if nchan < 0 or samples < 0:
            raise ValueError("Invalid number of channels or samples.")
        
        while True:
            while len(chunk) < (32 + nchan*samples*2):
                chunk += yield
            _num, _tick, _trig, _dummy = unpack('<8s12s2s10s', chunk[:32])
            
            data[0] += chunk[32:32+nchan*samples*2]
            enum.append(int(_num.strip()))
            tick.append((datetime.strptime('1970/01/01 ' + _tick.strip(), '%Y/%m/%d %H:%M:%S.%f') - datetime.utcfromtimestamp(0)).total_seconds())

            chunk = chunk[32 + nchan*samples*2:]
    
    # Start parser
    p = parser()
    next(p)

    # Read by chunk and parse it
    while True:
        chunk = f.read(1024*1024)   # read 1 MB
        if not chunk:
            break
        p.send(chunk)
    
    f.close()

    # Convert buffer to numpy array
    data = np.frombuffer(data[0], dtype='>i2').reshape(-1, nchan*samples)
    
    # Convert to epoch
    tick = np.asarray(tick) + xdfdate
    
    return enum, tick, data

def lopen(filename, raw=False, bufsize=1024*1024):
    """
    Read LJH file
    
    Parameters
    ==========
        filename:   file name to read
        raw:        returns raw data without scaling/offsetting if True (Default: False)
        bufsize:    file read buffer size (Default: 1 MB)

    Returns
    =======
        if raw is True:
            [ vres, vofs, hres, hofs, timestamp, inv, data ]
        else:
            [ t, timestamp, data ]
    """
    
    # Initialization
    nbits = -1
    dsize = -1
    vres = -1
    vofs = -1
    hres = -1
    hofs = -1
    inv = False
    samples = -1
    
    # Open ljh file and read by chunks
    f = open(filename, 'rb')

    # Read header
    while True:
        l = f.readline()
        
        if l[:14] == b"#End of Header":
            break
        
        token = l.split(':')

        # Number of Bits
        if token[0] == b"Bits":
            nbits = int(token[-1])
        
        # Data size
        if token[0] == b"Digitized Word Size in Bytes":
            dsize = int(token[-1])
        
        # Horizontal resolution
        if token[0] == b"Timebase":
            hres = float(token[-1])

        # Horizontal offset (in points)
        if token[0] == b"Presamples":
            hofs = int(token[-1])
        
        # Vertical resolution
        if token[0] == b"Range":
            vres = float(token[-1])

        # Vertical offset
        if token[0] == b"Offset":
            vofs = float(token[-1])

        # Number of samples
        if token[0] == b"Total Samples":
            samples = int(token[-1])
        
        # Pulse inversion
        if token[0] == b"Inverted":
            if token[-1][:2] == b"No":
                inv = False
            else:
                inv = True

    
    # Initialize for parser
    ts = [ bytearray() ]
    data = [ bytearray() ]
    
    # Parser
    def parser():
        """
        LJH Data Parser (generator)
        """
        
        # Initialize
        chunk = ''
        
        while True:
            while len(chunk) < (16 + dsize*samples):
                chunk += yield

            ts[0] += chunk[8:16]
            data[0] += chunk[16:16+dsize*samples]

            chunk = chunk[16 + dsize*samples:]
    
    # Start parser
    p = parser()
    next(p)

    # Read by chunk and parse it
    while True:
        chunk = f.read(bufsize)
        if not chunk:
            break
        p.send(chunk)
    
    f.close()

    # Convert buffer to numpy array
    ts = np.frombuffer(ts[0], dtype='<u8') * 1e-6
    data = np.frombuffer(data[0], dtype='<u%d' % dsize).reshape(-1, samples)
    
    if raw:
        return vres, vofs, hres, hofs, ts, inv, data
    else:
        # Convert data to physical units
        t = (np.arange(samples) - hofs) * hres
        data = vofs + (data/float(2**(nbits-1)) - 1) * vres * (-1 if inv else 1)
        
        return t, ts, data
