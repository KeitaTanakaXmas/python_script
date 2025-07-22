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

class SkyMap:

    def CorGal(self, ra, dec):
        Cor = SkyCoord(ra*u.degree,dec*u.degree,frame='icrs')
        GalCor = Cor.galactic
        l_list = GalCor.l.deg
        b_list = GalCor.b.deg
        return l_list,b_list

    def compute_contours(self, proportions, samples):
        r''' Plot containment contour around desired level.
        E.g 90% containment of a PDF on a healpix map

        Parameters:
        -----------
        proportions: list
            list of containment level to make contours for.
            E.g [0.68,0.9]
        samples: array
            array of values read in from healpix map
            E.g samples = hp.read_map(file)
        Returns:
        --------
        theta_list: list
            List of arrays containing theta values for desired contours
        phi_list: list
            List of arrays containing phi values for desired contours
        '''

        levels = []
        sorted_samples = list(reversed(list(sorted(samples))))
        nside = hp.pixelfunc.get_nside(samples)
        sample_points = np.array(hp.pix2ang(nside,np.arange(len(samples)))).T
        for proportion in proportions:
            level_index = (np.cumsum(sorted_samples) > proportion).tolist().index(True)
            level = (sorted_samples[level_index] + (sorted_samples[level_index+1] if level_index+1 < len(samples) else 0)) / 2.0
            levels.append(level)
        contours_by_level = meander.spherical_contours(sample_points, samples, levels)

        theta_list = []; phi_list=[]
        for contours in contours_by_level:
            for contour in contours:
                theta, phi = contour.T
                phi[phi<0] += 2.0*np.pi
                theta_list.append(theta)
                phi_list.append(phi)

        return theta_list, phi_list

    def halosat_obs_out(self):
        file = open('halomaster.tdat.txt')
        inf_list = []
        for data in file:
            if len(data.split('|')) > 5:
                inf_list.append(data.split('|'))
        df = pd.DataFrame(inf_list)
        print(df)
        obsid =df.loc[:,0].astype('str')  ## obsID  
        ra  = df.loc[:,2].astype('float')*u.deg ## ra [degree]
        dec = df.loc[:,3].astype('float')*u.deg ## dec [degree]
        exp = df.loc[:,6].astype('float')*u.deg ## dec [degree]
        ra.to(u.rad)
        Cor = SkyCoord(ra.to(u.rad),dec.to(u.rad),frame='icrs')
        GalCor = Cor.galactic
        l_list = GalCor.l.deg
        b_list = GalCor.b.deg
        return list(obsid),l_list,b_list,ra,dec,exp

    def suzaku_obs_out(self):
        file = open('SUZAKU_MASTER.txt')
        inf_list = []
        for data in file:
            if len(data.split('|')) > 5:
                inf_list.append(data.split('|'))
        df = pd.DataFrame(inf_list)[1:]
        print(df)
        target_name = df.loc[:,1].astype('str')  ## target_name  
        obsid = df.loc[:,9].astype('str')  ## obsID  
        category = df.loc[:,36].astype('str')  ## category
        explanation = df.loc[:,35].astype('str')  ## category
        ra  = df.loc[:,2].astype('float') ## ra [degree]
        dec = df.loc[:,3].astype('float') ## dec [degree]
        Cor = SkyCoord(ra*u.degree,dec*u.degree,frame='icrs',unit='deg')
        GalCor = Cor.galactic
        l_list = GalCor.l.deg
        b_list = GalCor.b.deg
        print('----')
        print(obsid)
        print(category)
        return list(obsid),l_list,b_list,ra,dec,list(target_name),category,explanation

    def search_target(self,string_list,keyword):
        filtered_indices = []
        filtered_elements = []

        for index, s in enumerate(string_list):
            if keyword in s:
                filtered_indices.append(index)
                filtered_elements.append(s)

        print("Filtered Indices:", filtered_indices)
        print("Filtered Elements:", filtered_elements)
        return filtered_elements, filtered_indices     

    def searching(self):
        obsid, l_list, b_list, ra, dec, target_name, category, explanation = self.suzaku_obs_out()
        filtered_elements, filtered_indices = self.search_target(string_list=target_name,keyword='ABELL')
        cluster_element, cluster_indices = self.search_target(string_list=explanation, keyword='cluster')
        obsid = np.array(obsid)
        l_list = np.array(l_list)
        b_list = np.array(b_list)
        target_name = np.array(target_name)
        category = np.array(category)
        explanation = np.array(explanation)

        print(target_name[cluster_indices])
        #filtered_indices = cluster_indices
        obsid = obsid[filtered_indices]
        l_list = l_list[filtered_indices]
        b_list = b_list[filtered_indices]
        target_name = target_name[filtered_indices]
        category = category[filtered_indices]
        print(obsid,target_name,l_list)
        print(explanation)
        import re

        # 数字を格納するための辞書
        number_dict = {}

        # 各文字列に対して数字を抽出し、辞書に格納
        for string,l,b in zip(target_name,l_list,b_list):
            match = re.search(r'\b(\d+)', string)
            if match:
                number = match.group(1)
                if number not in number_dict:
                    number_dict[number] = {}
                    number_dict[number]['target_name'] = [string]
                    number_dict[number]['l'] = [l]
                    number_dict[number]['b'] = [b]
                else:
                    number_dict[number]['target_name'].append(string)
                    number_dict[number]['l'].append(l)
                    number_dict[number]['b'].append(b)

        print(number_dict)

        obj_name = list(number_dict.keys())
        l_avg = [np.average(number_dict[number]['l']) for number in obj_name]
        b_avg = [np.average(number_dict[number]['b']) for number in obj_name]

        self.healplot('RASS_SXRB_R5.fits',300)
        for ob,l,b in zip(target_name,l_list,b_list):
            theta_loop = np.linspace(0,2*np.pi,50)
            l_circ = l + (17.8*np.sqrt(2)/2/60)*np.cos(theta_loop)
            b_circ = b + (17.8*np.sqrt(2)/2/60)*np.sin(theta_loop)
            print(l,b)
            hp.projplot(l_circ,b_circ,'b-',coord='G',lonlat=True,lw=0.5,color="Orange")
            #hp.projtext(l,b,ob,coord='G',lonlat=True,color='Orange',fontsize=8)

        # # テキスト
        # text = plt.text(0, 0, '', va='bottom', ha='left')

        # # カーソルが動いたときの処理
        # def on_move(event):
        #     if event.xdata is not None and event.ydata is not None:
        #         text.set_text(f'({event.xdata:.2f}, {event.ydata:.2f})')
        #         text.set_position((event.xdata, event.ydata))
        #         plt.draw()

        # # イベントを処理
        # plt.connect('motion_notify_event', on_move)

        for ob,l,b in zip(obj_name,l_avg,b_avg):
            hp.projtext(l,b,f'Abell {ob}',coord='G',lonlat=True,color='Orange')

        plt.show()

    def gal2healpix(self,obs:str,**kwargs):
        if obs == 'suzaku':
            obsid, l_list, b_list, ra, dec = self.suzaku_obs_out()
            if 'select_ID' in kwargs:
                select_ID = kwargs['select_ID']
                print(select_ID)
                obsid = np.array(obsid)
                mask = np.isin(obsid,select_ID)
                print(mask)
                obsid = obsid[mask]
                l_list = l_list[mask]
                b_list = b_list[mask]
                print('ID selected')
                print(obsid)
            for ob,l,b in zip(obsid,l_list,b_list):
                theta_loop = np.linspace(0,2*np.pi,50)
                l_circ = l + (17.8*np.sqrt(2)/2/60)*np.cos(theta_loop)
                b_circ = b + (17.8*np.sqrt(2)/2/60)*np.sin(theta_loop)
                print(l,b)
                hp.projplot(l_circ,b_circ,'b-',coord='G',lonlat=True,lw=2,color="Orange")
                hp.projtext(l,b,ob,coord='G',lonlat=True,color='Orange')
        elif obs == 'halosat':
            obsid, l_list, b_list, ra, dec, exp = self.halosat_obs_out()
            if 'select_ID' in kwargs:
                select_ID = kwargs['select_ID']
                print(select_ID)
                obsid = np.array(obsid)
                mask = np.isin(obsid,select_ID)
                print(mask)
                obsid = obsid[mask]
                l_list = l_list[mask]
                b_list = b_list[mask]
                print('ID selected')
                print(obsid)
            for obs,l,b in zip(obsid,l_list,b_list):
                theta_loop = np.linspace(0,2*np.pi,50)
                l_circ = l + 5*np.cos(theta_loop)
                b_circ = b + 5*np.sin(theta_loop)
                hp.projplot(l_circ,b_circ,'w--',coord='G',lonlat=True,lw=0.5,color="White")
                hp.projtext(l,b,obs,coord='G',lonlat=True,color='White')
        else:
            print(f'{obs} is not defined.')
            print('please select suzaku or halosat')

    def obsid2coo(self,obs:str,**kwargs):
        if obs == 'suzaku':
            obsid, l_list, b_list, ra, dec = self.suzaku_obs_out()
            if 'select_ID' in kwargs:
                select_ID = kwargs['select_ID']
                print(select_ID)
                obsid = np.array(obsid)
                mask = np.isin(obsid,select_ID)
                print(mask)
                obsid = obsid[mask]
                l_list = l_list[mask]
                b_list = b_list[mask]
                print('ID selected')
                print(obsid)
                if all(obsid == select_ID):
                    print("ID check over")
                else:
                    print("ID check not over")
                    print('Error: configuration failed', file=sys.stderr)
                    sys.exit(1)

        elif obs == 'halosat':
            obsid, l_list, b_list, ra, dec, exp = self.halosat_obs_out()
            if 'select_ID' in kwargs:
                select_ID = kwargs['select_ID']
                print(select_ID)
                obsid = np.array(obsid)
                mask = np.isin(obsid,select_ID)
                print(mask)
                obsid = obsid[mask]
                l_list = l_list[mask]
                b_list = b_list[mask]
                print('ID selected')
                print(obsid)
                if all(obsid == select_ID):
                    print("ID check over")
                else:
                    print("ID check not over")
                    print('Error: configuration failed', file=sys.stderr)
                    sys.exit(1)

        else:
            print(f'{obs} is not defined.')
            print('please select suzaku or halosat')
            obsid, l_list, b_list = None, None, None

        return obsid, l_list, b_list



    def healpix_cor(self,filename):
        heal_map = fits.open(filename)
        nside = heal_map[1].header['NSIDE']
        order = heal_map[1].header['ORDERING']
        ttype = heal_map[1].header['TTYPE1']
        tmap  = heal_map[1].data[ttype]
        tmap  = np.ravel(tmap[:])
        hp = HEALPix(nside=nside, order=order, frame=Galactic())
        # sample a 360, 180 grid in RA/Dec
        factor = 10
        ra = np.linspace(180., -180., 360*factor) * u.deg
        dec = np.linspace(90., -90., 180*factor) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)

        # set up astropy coordinate objects
        coords = SkyCoord(ra_grid.ravel(), dec_grid.ravel(), frame='galactic')

        # interpolate values
        tmap = hp.interpolate_bilinear_skycoord(coords, tmap)
        tmap = tmap.reshape((180*factor, 360*factor))
        print(heal_map[0])
        print('-------')
        print(heal_map[1])
        return heal_map,tmap

    def healplot(self,filename,vmax):
        heal_map = hp.read_map(filename)
        maps = hp.mollview(
            heal_map,
            title = 'ROSAT(Color map) + WHAM(Red)',
            coord = ['G'],
            min = 1,
            max = vmax,
            rot = [0,0],
            return_projected_map=True,
            unit = 'count',
            xsize=1600
            )
        hp.graticule(coord='G')

    def heal_contour(self,filename,level,color):
        heal_map = hp.read_map(filename)
        probs = hp.pixelfunc.ud_grade(heal_map, 64) #reduce nside to make it faster
        probs = probs/np.sum(probs)
        nside = hp.pixelfunc.get_nside(heal_map)
        levels = [level]
        theta_contour, phi_contour = self.compute_contours(levels,probs)
        hp.projplot(theta_contour[0],phi_contour[0],linewidth=1,c=color)
        for i in range(1,len(theta_contour)):
            hp.projplot(theta_contour[i],phi_contour[i],linewidth=1,c=color)


    def RASS_WHAM_cont(self,vmax,select_ID):
        self.healplot('RASS_SXRB_R4.fits',vmax)
        #self.heal_contour('lambda_WHAM_1_256.fits',level=0.77,color='Red')
        #heal_contour('RASS_SXRB_R4.fits',level=0.77,color='blue')
        #self.gal2healpix('halosat')
        #self.gal2healpix('suzaku',select_ID=select_ID)
        #hp.projplot(l_list[0],b_list[0],'bo')
        plt.show()

    def load_csv(self,file:str,word:str):
        """_summary_

        Args:
            file (str): csv file includeing fitting result. ex) "test.csv"
            word (str): specific word in csv file. ex) "EM"

        Returns:
            pandas dataframe : df[obsid:str,word:str]
            df is sorted by obsid.
        """

        df_header = pd.read_csv(file, index_col=0, dtype={0: str,1: str})
        df = df_header[["obsid",word]]
        df = df.sort_values("obsid")
        print(df["obsid"])
        return df

    def result_with_skymap(self, file:str, word:str, max=None, min=None, scale=1):
        """_summary_

        Args:
            file (str): _description_
            word (str): _description_
            max (float): max value of color map
            min (float): min value of color map
            scale: scale
        """
        P = Plotter()
        df = self.load_csv(file,word)
        target = df[word]*scale
        print(df)
        if max == None :
            MAX = np.max(target)
        else :
            MAX = max

        if min == None :
            MIN = np.min(target)
        else :
            MIN = min

        fig = plt.figure(figsize=(7,7))
        ax  = plt.subplot(111)
        ax.grid(linestyle="dashed")
        obsid, l_list, b_list= self.obsid2coo("halosat",select_ID=df["obsid"])
        color = cm.jet((target-MIN)/(MAX-MIN))
        for l,b,col in zip(l_list,b_list,color):
            c = patches.Circle(xy=(l, b), radius=5, fc=col)
            ax.add_patch(c)
        im = plt.scatter([0,0],[0,0],c=[MAX,MIN],cmap=cm.jet)
        ax.set_xlabel("Galactic Longtitude",fontsize=20)
        ax.set_ylabel("Galactic Lattitude",fontsize=20)
        ax.set_xlim(170,230)
        ax.set_ylim(-60,0)
        ax.set_aspect('equal')
        fig.colorbar(im,ax=ax,shrink=0.8)
        ax.invert_xaxis()
        print(df)
        plt.show()
        fig.savefig(f'./graph/{word}_cmap.png',dpi=300)

    def result_with_skymap_scatter(self, file:str, word:str, max=None, min=None, scale=1):
        """_summary_

        Args:
            file (str): _description_
            word (str): _description_
            max (float): max value of color map
            min (float): min value of color map
            scale: scale
        """
        P = Plotter()
        ep = self.load_csv(file,f'{word}_ep')
        em = self.load_csv(file,f'{word}_em')
        df = self.load_csv(file,word)
        minus = [em[f'{word}_em'].values*(-1)*scale]
        pulse = [ep[f'{word}_ep'].values*scale]
        err = np.vstack((minus,pulse))
        print('---------')
        print(err)
        print(ep)
        print(em)
        if max == None :
            MAX = np.max(df[word])
        else :
            MAX = max

        if min == None :
            MIN = np.min(df[word])
        else :
            MIN = min

        fig = plt.figure(figsize=(8,6))
        ax  = plt.subplot(111)
        ax.grid(linestyle="dashed")
        obsid, l_list, b_list= self.obsid2coo("halosat",select_ID=df["obsid"])
        print('-----')
        ax.errorbar(b_list,df[word]*scale,yerr=err, capsize=5, fmt='.', markersize=15, ecolor="black", elinewidth=0.5, markeredgecolor="black", color="w")
        mask = df['obsid'].isin(['004201', '036601', '004101', '036501', '004001', '011701', '013701', '033801','003901'])
        new = df[word][mask]*scale
        ax.scatter(b_list[mask],new,color="red",s=100)
        ax.set_ylabel(r"$\rm Emission  \ measure \ [10^3 \ cm^{-6} \ pc]$",fontsize=20)
        ax.set_xlabel("Galactic Lattitude",fontsize=20)
        ax.invert_xaxis()
        print(np.median(df[word]))
        print(np.average(df[word]))
        plt.show()
        fig.savefig(f'./graph/{word}_cmap_scatter.png',dpi=300)

    def result_with_skymap_bgd(self, file:str, word:str, max=None, min=None, scale=1):
        """_summary_

        Args:
            file (str): _description_
            word (str): _description_
            max (float): max value of color map
            min (float): min value of color map
            scale: scale
        """
        P = Plotter()
        ep = self.load_csv(file,f'{word}_ep')
        em = self.load_csv(file,f'{word}_em')
        df = self.load_csv(file,word)
        minus = [em[f'{word}_em'].values*(-1)*scale]
        pulse = [ep[f'{word}_ep'].values*scale]
        err = np.vstack((minus,pulse))
        rate = self.load_csv('OES_HS_inf.csv','s38_hardrate')
        print('---------')
        print(err)
        print(ep)
        print(em)
        if max == None :
            MAX = np.max(df[word])
        else :
            MAX = max

        if min == None :
            MIN = np.min(df[word])
        else :
            MIN = min

        fig = plt.figure(figsize=(8,6))
        ax  = plt.subplot(111)
        ax.grid(linestyle="dashed")
        obsid, l_list, b_list= self.obsid2coo("halosat",select_ID=df["obsid"])
        print('-----')
        #ax.errorbar(rate['s38_hardrate'],df[word],yerr=err, capsize=5, fmt='.', markersize=15, ecolor="black", elinewidth=0.5, markeredgecolor="black", color="w")
        mask = df['obsid'].isin(['004201', '036601', '004101', '036501', '004001', '011701', '013701', '033801','003901'])
        ax.scatter(rate['s38_hardrate'],df[word],color="black")
        ax.set_ylabel("s38 pow norm",fontsize=20)
        ax.set_xlabel("hard rate",fontsize=20)
        print(np.median(df[word]))
        print(np.average(df[word]))
        plt.show()
        fig.savefig(f'./graph/{word}_cmap_scatter.png',dpi=300)

    def inf_with_skymap(self,file,word):
        P = Plotter()
        F = FitResultOut('dummy.log')
        df = F.loadhdf5_inf()
        s14_data = df[f's14_{word}'].to_numpy()
        s38_data = df[f's38_{word}'].to_numpy()
        s54_data = df[f's54_{word}'].to_numpy()
        data = np.array([np.sum([i,j,k]) for i,j,k in zip(s14_data,s38_data,s54_data)])
        print('data')
        print(data)
        scale = 1
        fig = plt.figure(figsize=(7,7))
        ax  = plt.subplot(111)
        ax.grid(linestyle="dashed")
        #cm = plt.cm.get_cmap('jet')
        obsid, l_list, b_list= self.obsid2coo("halosat",select_ID=df["obsid"])
        color = cm.jet((data-np.min(data))/(np.max(data)-np.min(data)))
        for l,b,col in zip(l_list,b_list,color):
            c = patches.Circle(xy=(l, b), radius=5, fc=col)
            ax.add_patch(c)
        im = plt.scatter([0,0],[0,0],c=[np.max(data),np.min(data)],cmap=cm.jet)
        plt.axis('scaled')
        #ax.set_title(word,fontsize=20)
        ax.set_xlabel("Galactic Longtitude",fontsize=20)
        ax.set_ylabel("Galactic Lattitude",fontsize=20)
        ax.set_aspect('equal')
        fig.colorbar(im,ax=ax,shrink=0.8)
        ax.invert_xaxis()
        plt.show()
        fig.savefig(f'./graph/{word}_cmap.png',dpi=300)

    
    def ROSAT_RGB_map(self):
        PSPC = fits.open('R45_arith.fits')[0]
        print(PSPC)
        fig = plt.figure(figsize=(8,8))
        f = aplpy.FITSFigure(PSPC, slices=[0], convention='wells', figure=fig)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        f.show_colorscale(vmin=0, stretch='linear', cmap="viridis", aspect="equal",interpolation=None)
        f.add_colorbar()
        f.colorbar.show()
        f.add_grid()
        f.grid.set_color('white')
        plt.show()
        fig.savefig('ROSAT_obs.pdf',dpi=300)        
    
    def PSPC_plot(self,filename):
        PSPC = fits.open(filename)[0]
        print(PSPC)
        fig = plt.figure(figsize=(8,8))
        f = aplpy.FITSFigure(PSPC, slices=[0], convention='wells', figure=fig)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        f.show_colorscale(vmin=0, stretch='linear', cmap="viridis", aspect="equal",interpolation=None)
        f.add_colorbar()
        f.colorbar.show()
        f.add_grid()
        f.grid.set_color('white')
        obsid, l_list, b_list, ra, dec, exp_list = self.halosat_obs_out()
        print(obsid)
        #f.show_circles(l_list,b_list,radius=5,edgecolor='white')
        oblist=[]
        for l,b,obs,exp in zip(l_list,b_list,obsid,exp_list):
            if exp > 9000.0:
                if obs in ['036501','004101','036601','004201']:
                    f.add_label(l,b,obs,color='white')  
                    f.show_circles(l,b,radius=5,edgecolor='white')
                    print(exp)
                    oblist.append(obs)
                else:  
                    # f.add_label(l,b,obs,color='black')
                    # f.show_circles(l,b,radius=5,edgecolor='black')
                    pass
        print(oblist)
        plt.show()
        fig.savefig('ROSAT_halosat_obs.pdf',dpi=300)

    def PSPC_plot2(self,filename):
        PSPC = fits.open(filename)[0]
        print(PSPC)
        fig = plt.figure(figsize=(8,8))
        f = aplpy.FITSFigure(PSPC, slices=[0], convention='wells', figure=fig)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        f.show_colorscale(vmin=0, stretch='linear', cmap="viridis", aspect="equal",interpolation=None)
        f.add_colorbar()
        f.colorbar.show()
        f.add_grid()
        f.grid.set_color('white')
        obsid, l_list, b_list, ra, dec, exp_list = self.halosat_obs_out()
        print(obsid)
        #f.show_circles(l_list,b_list,radius=5,edgecolor='white')
        oblist=[]
        for l,b,obs,exp in zip(l_list,b_list,obsid,exp_list):
            if exp > 9000.0:
                if obs in ['036501','004101','036601','004201']:
                    f.add_label(l,b,obs,color='white')  
                    f.show_circles(l,b,radius=5,edgecolor='white')
                    print(exp)
                    oblist.append(obs)
                else:  
                    # f.add_label(l,b,obs,color='black')
                    # f.show_circles(l,b,radius=5,edgecolor='black')
                    pass
        print(oblist)

        obsid, l_list, b_list, ra, dec = self.suzaku_obs_out()
        print(obsid)
        #f.show_circles(l_list,b_list,radius=5,edgecolor='white')
        oblist=[]
        for l,b,obs in zip(l_list,b_list,obsid):
            f.add_label(l,b,obs,color='red')  
            f.show_circles(l,b,radius=0.3,edgecolor='red')
        plt.show()
        fig.savefig('ROSAT_halosat_obs.pdf',dpi=300)

    def HEAL_plot(self,filename,vmax):
        hdu,tmap = self.healpix_cor(filename)
        print(tmap)
        fig = plt.figure(figsize=(8,8))
        f = aplpy.FITSFigure(tmap,hdu=hdu[0].header, convention='wells', figure=fig)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        f.show_colorscale(vmin=0, vmax=vmax, stretch='linear', cmap="viridis", aspect="equal",interpolation=None)
        f.add_colorbar()
        print(tmap.shape)
        factor = 10
        ra = np.linspace(180., -180., 360*factor)*u.deg
        dec = np.linspace(90., -90., 180*factor)*u.deg
        a = f.pixel2world(45,45)
        print(a)
        #f.set_xaxis_coord_type('longitude')
        #f.set_yaxis_coord_type('latitude')
        f.colorbar.show()
        f.add_grid()
        f.grid.set_color('white')
        #f.show_contour(colors='Red')
        #f.show_regions('xis0_sel.reg')
        # obsid, l_list, b_list, ra, dec = self.halosat_obs_out()
        # print(obsid)
        # f.show_circles(l_list,b_list,radius=5)
        # for l,b,obs in zip(l_list,b_list,obsid):
        # #f.add_label(l,b,obs)
        #     f.show_layer()
        plt.show()


    def gal2cart(self,l,b):
        c_gal  = SkyCoord(l*u.degree, b*u.degree,frame='galactic')
        print(c_gal)
        c_gal.representation_type = 'cartesian'
        print(c_gal) 
        return c_gal

    def cart2gal(self,u,v,w):
        c_cart  = SkyCoord(u, v, w, unit='kpc', representation_type='cartesian', frame='galactic')
        print(c_cart)
        c_cart.representation_type = 'spherical'
        print(c_cart)
        return c_cart

    def halosat_circ(self,l,b):
        theta_loop = np.linspace(0,2*np.pi,50)
        l_circ = l + 5*np.cos(theta_loop)
        b_circ = b + 5*np.sin(theta_loop)
        return l_circ, b_circ

    def GAIA_3dplot(self,dist):
        f = fits.open('STILISM_cube.fits')[1]
        cfactor = 1442401
        cparam  = int(((400+dist)/5)*cfactor)
        print('loading xdata')
        x = np.unique(f.data['X'][cparam:cparam+cfactor])
        print(x)
        print('loading ydata')
        y = np.unique(f.data['Y'][cparam:cparam+cfactor])
        print(y)
        z = f.data['Z'][cparam:cparam+cfactor]
        print(z)
        print('loading ddata')
        d = np.reshape(f.data['dA0_dD'][cparam:cparam+cfactor],(1201,1201))
        print(d)
        d = 2.47e+21*300*d
        X, Y = np.meshgrid(x,y)
        fig, ax = plt.subplots()
        ax.pcolormesh(X,Y,d,cmap='inferno_r')
        plt.show()


    def GAIA_3dplot2(self,dist):
        f = fits.open('STILISM_cube.fits')[1]
        cfactor = 1201
        cparam  = int(((3000-dist)/5)*cfactor)
        print('loading xdata')
        x = f.data['X'][:]
        print(x)
        print('loading ydata')
        y = f.data['Y'][:]
        print(y.shape)
        z = f.data['Z'][:]
        print(z.shape)
        print('loading ddata')
        d = f.data['dA0_dD'][:]
        print(d)
        mask = ((-dist<=x) & (x<=dist)) & ((-dist<=y) & (y<=dist))
        x,y,z,d = x[mask],y[mask],z[mask],d[mask]
        print(x.shape) 
        with h5py.File('gaia_comp.hdf5',"a") as f:
            if 'result' in f:
                del f['result']
            f.create_dataset('result/x',data=x)
            f.create_dataset('result/y',data=y)
            f.create_dataset('result/z',data=z)
            f.create_dataset('result/d',data=d)


    def load_gaia_comp(self):
        with h5py.File('gaia_comp.hdf5',"r") as f:
            x, y, z, d = f['result/x'][:], f['result/y'][:], f['result/z'][:], f['result/d'][:]

        xyz = np.vstack((x,y))
        xyz = np.vstack((xyz,z))
        xyz = xyz.T
        return xyz, d

    def gaia_plot(self,dist,sigma):
        data,d = self.load_gaia_comp()
        cor = self.cart2gal(data[:,0],data[:,1],data[:,2])
        dist_list = np.array(cor.distance)
        mask = (dist-sigma<=dist_list) & (dist_list<=dist+sigma)
        l_list    = np.array(cor.l)[mask]
        b_list    = np.array(cor.b)[mask]
        dist_list = dist_list[mask]
        d         = d[mask]
        print(dist_list.shape)
        obs,l,b = self.obsid2coo('halosat',select_ID=['013701'])
        l_l,b_l = self.halosat_circ(l,b)
        obs_mask = (l-5<l_list) & (l_list<l+5) & (b-5<b_list) & (b_list<b+5)
        d_avg = np.average(d[obs_mask])*dist*2.47e+21
        plt.scatter(l_list,b_list,c=d)
        plt.plot(l_l,b_l,color='Red')
        print(d_avg)
        plt.show()        


class ImagePlot:

    def __init__(self) -> None:
        pass

    def plot(self):
        data_fits_R = "merge_E5_expcor_b8.fits"
        data_fits_G = "merge_E6_expcor_b8.fits"
        data_fits_B = "merge_E7_expcor_b8.fits"
        fitss = [data_fits_R, data_fits_G, data_fits_B]

        color_min_R = 30.0
        color_max_R = 1500.0
        color_min_G = 30.0
        color_max_G = 1500.0
        color_min_B = 10.0
        color_max_B = 100.0
        colorval = "%.1f_%.1f_%.1f_%.1f_%.1f_%.1f"%(color_min_R, color_max_R, color_min_G, color_max_G, color_min_B, color_max_B)
        save_png_name = "RGB_%s"%(colorval)+'.png'
        aplpy.make_rgb_image(fitss, save_png_name,pmin_r=0.0,pmax_r=99.5)
        fig = plt.figure(figsize=(16, 12))
        f = aplpy.FITSFigure(data_fits_R, slices=[0], figure=fig, convention='wells')
        f.show_rgb(save_png_name)
        f.ticks.set_color('w')
        # f.show_colorscale(cmap='plasma',smooth=3)
        # f = aplpy.FITSFigure(data_fits_G, slices=[0], figure=fig, convention='wells')
        # f.show_colorscale(cmap='plasma',smooth=3)
        # f = aplpy.FITSFigure(data_fits_B, slices=[0], figure=fig, convention='wells')
        # f.show_colorscale(cmap='plasma',smooth=3)
        f.save('RGB_aplpy.pdf', dpi=300)
        plt.show()