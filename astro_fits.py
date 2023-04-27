import matplotlib 
#matplotlib.use('TkAgg')
import glob 
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LogNorm
from astropy.wcs import WCS
from astropy.io import fits

name="NGC4051"

hri = glob.glob('*ime*.fits')

F = plt.figure(figsize=(10,8))

hrifilename = hri[0]
hriname = hrifilename.replace(".fits",".png")
hrihdu = fits.open(hrifilename)[0]
hriwcs = WCS(hrihdu.header)
hridata = hrihdu.data

hrixlen, hriylen = hridata.shape
hricx = int(0.5 * hrixlen)
hricy = int(0.5 * hriylen)
hridx = int(hrixlen*0.1)
hriwcscut = hriwcs[hricx-hridx:hricx+hridx,hricy-hridx:hricy+hridx]

plt.figtext(0.45,0.93, name, size="large")
plt.figtext(0.15,0.9, "X-ray, Rosat HRI")


try:
    plt.imshow(hridata, origin='lower', norm=LogNorm())
    plt.colorbar()                  
except:
    print("ERROR, couldn't plot log-z scale")           
    plt.close()

plt.grid(color='white', ls='solid')
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')

plt.imshow(hridata[hricx-hridx:hricx+hridx,hricy-hridx:hricy+hridx], origin='lower')
plt.colorbar()      
plt.grid(color='white', ls='solid')
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')



plt.show()

