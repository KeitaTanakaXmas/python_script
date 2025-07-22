
#!/usr/bin/env python3
"""Generate a DS9 region file (fk5) for selected detector pixels.

* Input telescope pointing in **degrees** (RA, Dec)
* Counter‑clockwise rotation angle (deg)
* List of pixel IDs to export
* Optional JSON file that maps pixel ID → [row, col] (0‑based)

If no mapping file is given, a default row‑major mapping is assumed for an
N×N grid determined by the detector and pixel sizes (defaults: 3.05′ detector,
0.1666′ pixels).

Mapping file format (JSON)
--------------------------
{
  "0": [0, 0],
  "1": [0, 1],
  "37": [2, 1],
  ...
}

Each entry gives **row, col** indices of that pixel within the detector grid.

Usage
-----
python generate_ds9_region_deg.py \
       --center 188.7366  12.3759 \
       --rotation 27 \
       --pixels 0 1 37 38 \
       --outfile selected.reg
"""

import argparse
import json
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

DET_ARCMIN_DEFAULT = 3.05
PIX_ARCMIN_DEFAULT = 0.508333


def id_to_rowcol_default(pid: int, nside: int):
    """Default row‑major mapping."""
    return divmod(pid, nside)


def rowcol_from_mapping(pid: int, mapping: dict):
    try:
        return mapping[str(pid)]
    except KeyError as e:
        raise ValueError(f'Pixel ID {pid} not found in mapping file') from e


def pixel_centre_offset(row: int, col: int, pix_arcmin: float, nside: int):
    half = (nside * pix_arcmin) / 2.0
    dx = (col + 0.5) * pix_arcmin - half   # +x → east
    dy = (row + 0.5) * pix_arcmin - half   # +y → north
    return dx, dy


def rotate(dx, dy, theta_deg):
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy


def offsets_to_sky(center: SkyCoord, dx_arcmin, dy_arcmin):
    dra  = (dx_arcmin / np.cos(center.dec.radian)) * u.arcmin
    ddec = dy_arcmin * u.arcmin
    return SkyCoord(ra=center.ra + dra.to(u.deg),
                    dec=center.dec + ddec.to(u.deg),
                    frame='fk5')


def main():
    p = argparse.ArgumentParser(description='Create DS9 region for detector pixels.')
    p.add_argument('--center', nargs=2, type=float, required=True,
                   metavar=('RA_DEG', 'DEC_DEG'),
                   help='Pointing centre in DEGREES (e.g. 188.7366 12.3759)')
    p.add_argument('--rotation', type=float, default=0.0,
                   help='Detector rotation CCW on sky (deg)')
    p.add_argument('--pixels', type=int, nargs='+', required=True,
                   help='Pixel IDs to include')
    p.add_argument('--outfile', required=True, help='Output .reg file')
    p.add_argument('--mapfile', help='Optional JSON file with id → [row,col] mapping')
    p.add_argument('--det_size', type=float, default=DET_ARCMIN_DEFAULT,
                   help='Detector side length (arcmin)')
    p.add_argument('--pixsize', type=float, default=PIX_ARCMIN_DEFAULT,
                   help='Pixel side length (arcmin)')
    args = p.parse_args()

    # prepare mapping
    if args.mapfile:
        mapping = json.loads(Path(args.mapfile).read_text())
        mapper = lambda pid: rowcol_from_mapping(pid, mapping)
        nside = None   # not used
    else:
        nside = int(round(args.det_size / args.pixsize))
        mapper = lambda pid: id_to_rowcol_default(pid, nside)

    center = SkyCoord(args.center[0]*u.deg, args.center[1]*u.deg, frame='fk5')

    lines = ['# Region file format: DS9 version 4.1',
             'global color=green dashlist=8 3 width=1',
             'fk5']

    for pid in args.pixels:
        row, col = mapper(pid)
        dx, dy = pixel_centre_offset(row, col, args.pixsize, nside if nside else max(row,col)+1)
        rdx, rdy = rotate(dx, dy, args.rotation)
        sky = offsets_to_sky(center, rdx, rdy)
        lines.append(f'box({sky.ra.deg:.6f},{sky.dec.deg:.6f},'
                     f'{args.pixsize:.4f}\',{args.pixsize:.4f}\',{args.rotation})')

    Path(args.outfile).write_text('\n'.join(lines))
    print(f'Wrote {len(args.pixels)} pixels → {args.outfile}')


if __name__ == '__main__':
    main()
