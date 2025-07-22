import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import numpy as np
import aplpy

def halosat_obs_out():
    file = open('halomaster.tdat.txt')
    inf_list = []
    for data in file:
      if len(data.split('|')) > 5:
        inf_list.append(data.split('|'))
    df = pd.DataFrame(inf_list)
    print(df)
    obsid =df.loc[:,0].astype('str')  ## obsID  
    ra  = df.loc[:,2].astype('float') ## ra [degree]
    dec = df.loc[:,3].astype('float') ## dec [degree]

    Cor = SkyCoord(ra*u.degree,dec*u.degree,frame='icrs')
    GalCor = Cor.galactic
    l_list = GalCor.l.deg
    b_list = GalCor.b.deg
    return list(obsid),l_list,b_list,ra,dec



def PSPC_plot(filename,vmax):
    PSPC = fits.open(filename)[0]
    fig = plt.figure(figsize=(8,8))
    f = aplpy.FITSFigure(PSPC, slices=[0], convention='wells', figure=fig)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f.show_colorscale(vmin=0, vmax=vmax, stretch='linear', cmap="viridis", aspect="equal",interpolation=None)
    f.add_colorbar()
    f.colorbar.show()
    f.add_grid()
    f.grid.set_color('white')
    f.show_regions('xis0_sel.reg')
    obsid, l_list, b_list, ra, dec = halosat_obs_out()
    print(obsid)
    f.show_circles(l_list,b_list,radius=5)
    for l,b,obs in zip(l_list,b_list,obsid):
      f.add_label(l,b,obs)
    #f.show_layer()
    plt.show()