import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from astropy.coordinates import SkyCoord,Galactic
from astropy_healpix import HEALPix
from astropy import units as u
from astropy.io import fits
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import numpy as np
import aplpy
import sys
import meander
import h5py
from Basic import Plotter
from xspec_tools import FitResultOut
import glob
# FITSファイルを読み込む

class CatalogSearch:

    def __init__(self) -> None:
        self.hdul = fits.open('/Users/keitatanaka/Downloads/erass1cl_primary_v3.2.fits')
        self.hdul.info()
        data = self.hdul[1].data
        ra = data['RA']
        dec = data['DEC']
        match = data['MATCH_NAME']
        kT   = data['kT']
        PCONT   = data['PCONT']
        EXT_LIKE   = data['EXT_LIKE']
        DET_LIKE = data['DET_LIKE_0']

def main(ra_reg, dec_reg):
    hdul = fits.open('/Users/keitatanaka/Downloads/erass1cl_primary_v3.2.fits')

    # FITSファイルの中身を表示する
    hdul.info()

    # データテーブルを取得する（0番目のテーブルを使用する場合）
    data = hdul[1].data

    # カラムを指定してデータを取得する（例: 'RA'というカラムを抜き出す場合）
    ra = data['RA']
    dec = data['DEC']
    match = data['MATCH_NAME']
    kT   = data['kT']
    PCONT   = data['PCONT']
    EXT_LIKE   = data['EXT_LIKE']
    DET_LIKE = data['DET_LIKE_0']
    print(EXT_LIKE.shape)
    CR_300kpc = data['CR300kpc']
    mask = match == ''
    kT_id = np.argsort(kT)
    CR300k_id = np.argsort(CR_300kpc[mask])
    # もしくは、カラムのインデックスを指定することも可能
    # ra_column = data.field(0)  # インデックス0は'RA'カラム

    # plt.scatter(DET_LIKE[mask], PCONT[mask])
    # plt.show()
    # plt.hist(DET_LIKE[mask], bins=1000, histtype='step')
    # plt.grid(linestyle='dashed')
    # plt.show()
    # plt.hist(EXT_LIKE[mask], bins=1000, histtype='step')
    # plt.grid(linestyle='dashed')
    # plt.show()
    # plt.hist(PCONT[mask], bins=1000, histtype='step')
    # plt.grid(linestyle='dashed')
    # plt.show()
    # name = 'CR300k_id'
    # for i in range(0,1000):
    #     print(DET_LIKE[mask][CR300k_id][-i],EXT_LIKE[mask][CR300k_id][-i],PCONT[mask][CR300k_id][-i],kT[mask][CR300k_id][-i], ra[mask][CR300k_id][-i], dec[mask][CR300k_id][-i])
    radec_mask = (min(ra_reg) < ra) & (ra < max(ra_reg)) & (min(dec_reg) < dec) & (dec < max(dec_reg))
    
    ra = ra[radec_mask]
    dec = dec[radec_mask]
    match=match[radec_mask]
    
    # FITSファイルを閉じる
    hdul.close()
    return ra, dec, match

def catalog_search(z:list,kT:list):
    pass



def plot():
    data_fits_R = glob.glob("*025_Image_c010.fits")[0]
    data_fits_G = glob.glob("*026_Image_c010.fits")[0]
    data_fits_B = glob.glob("*027_Image_c010.fits")[0]
    fitss = [data_fits_R, data_fits_G, data_fits_B]
    hdul = fits.open(data_fits_R)

    RA_CEN = hdul[0].header["RA_CEN"]
    DEC_CEN = hdul[0].header["DEC_CEN"]
    RA_REG = [RA_CEN-1.8, RA_CEN+1.8]
    DEC_REG = [DEC_CEN-1.8, DEC_CEN+1.8]

    ra, dec, match = main(RA_REG, DEC_REG)
    color_min_R = 30.0
    color_max_R = 1500.0
    color_min_G = 30.0
    color_max_G = 1500.0
    color_min_B = 10.0
    color_max_B = 100.0
    colorval = "%.1f_%.1f_%.1f_%.1f_%.1f_%.1f"%(color_min_R, color_max_R, color_min_G, color_max_G, color_min_B, color_max_B)
    save_png_name = "RGB_%s"%(colorval)+'.png'
    aplpy.make_rgb_image(fitss, save_png_name,pmin_r=0.0,pmax_r=99.5)
    fig = plt.figure(figsize=(8, 8))
    f = aplpy.FITSFigure(data_fits_R, slices=[0], figure=fig, convention='wells')
    f.show_rgb(save_png_name)
    f.ticks.set_color('w')
    print(ra, dec, match)
    f.show_circles(ra, dec, 0.5, layer=False, coords_frame='world', color="white", linestyle="dashed")

    # f.show_colorscale(cmap='plasma',smooth=3)
    # f = aplpy.FITSFigure(data_fits_G, slices=[0], figure=fig, convention='wells')
    # f.show_colorscale(cmap='plasma',smooth=3)
    # f = aplpy.FITSFigure(data_fits_B, slices=[0], figure=fig, convention='wells')
    # f.show_colorscale(cmap='plasma',smooth=3)
    f.set_xaxis_coord_type("scalar")
    f.set_yaxis_coord_type("scalar")
    f.save('RGB_aplpy.pdf', dpi=300)
    plt.show()
