

    def savemod(self,modname):
        print(modname)
        with h5py.File(self.savefile,'a') as f:
            if modname in f.keys():
                del f[modname]
                print('model is deleted')
            f.create_group(modname)
            f.create_group(f'{modname}/fitting_result')
            f[modname].create_dataset("xs",data=self.xs)
            f[modname].create_dataset("ys",data=self.ys)
            f[modname].create_dataset("xe",data=self.xe)
            f[modname].create_dataset("ye",data=self.ye)
            f[modname].create_dataset("y",data=self.y)
            f[modname].create_dataset("yscomps",data=self.ys_comps)
            f[modname].create_dataset("xres",data=self.xres)
            f[modname].create_dataset("yres",data=self.yres)
            f[modname].create_dataset("xres_e",data=self.xres_e)
            f[modname].create_dataset("yres_e",data=self.yres_e)
            f[modname].create_dataset("statistic",data=self.statistic)
            f[modname].create_dataset("dof",data=self.dof)

            for model_number in self.fit_result.keys():
                model_components = list(self.fit_result[model_number].keys())
                for model_component in model_components:
                    for k,v in self.fit_result[model_number][model_component].items():
                        print(model_number,model_component,k,v)
                        if str(model_number) not in f[f'{modname}/fitting_result'].keys():
                            f.create_group(f'{modname}/fitting_result/{str(model_number)}/{model_component}')
                        f.create_group(f'{modname}/fitting_result/{str(model_number)}/{model_component}/{k}')
                        f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset('value',data=v['value'])
                        if f'em' in v.keys():
                            f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset("em",data=v['em'])
                            print('em =', v['em'])
                        if f'ep' in v.keys():
                            f[modname]['fitting_result'][str(model_number)][model_component][k].create_dataset("ep",data=v['ep'])   
                        print(f[modname]['fitting_result'][str(model_number)][model_component].keys())    



    def open_data_fitting(self,region='center',line='wz', rebin=1):
        for line in ['wz']:
            for region in ['center']:
                AllData.clear()
                current_dir = os.getcwd()
                identifiers = ['before', 'after']
                for e,identifier in enumerate(identifiers):  
                    os.chdir(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/{identifier}_recycle/analysis')
                    spec = f'1000_{region}_merged_b1.pi'
                    rmf  = f'1000_{region}_L_without_Lp.rmf'
                    arf  = f'1000_{region}_image_1p8_8keV_1e7.arf'
                    multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
                    self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=None,datagroup=e+1,spectrumgroup=e+1)
                os.chdir(current_dir)
                AllData.notice("2.0-17.0")
                AllData.ignore("1:**-2.0 17.0-**")
                AllData.ignore("2:**-2.0 17.0-**")
                Plot.setRebin(rebin,1000)
                Plot.device = '/xs'
                Plot('data')
                # self.set_data_range(rng=(2.0,17.0))
                # self.set_xydata()
                self.model_bvapec(line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=2,line_model='zgaus')
                self.load_apecroot(line)
                # self.bvapec_fix_some_param(fix=True)
                self.fit_error(self.error_str,False,'nxb1')
                self.nxb_fix_parameter(fix=True)
                #self.bvapec_fix_some_param(fix=False)
                self.fit_error(self.error_str,True,'nxb1')
                Xset.save(f'simultaneous_{region}_{line}_rebin_{rebin}.xcm')
                Plot('data delc')
                self.set_xydata_multi(identifiers)
                self.result_pack()
                for i in range(2,3):
                    self.result_pack_only_z(model=AllModels(i),group=i)
                self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
                #self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
                self.plotting_simult(f'simultaneous_{region}_{line}_rebin_{rebin}',line=line,bg_plot=False)


    def be_data_fitting(self,region='center',line='wz', rebin=1):
        for line in ['w']:
            for region in ['center']:
                AllData.clear()
                current_dir = os.getcwd()
                identifiers = ['before', 'after']
                for e,identifier in enumerate(identifiers):  
                    os.chdir(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/Be/{identifier}_recycle')
                    spec = f'4000_{region}_merged_b1.pi'
                    rmf  = f'4000_{region}_L_without_Lp.rmf'
                    arf  = f'4000_{region}_image_1p8_8keV_1e7.arf'
                    multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
                    self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=None,datagroup=e+1,spectrumgroup=e+1)
                os.chdir(current_dir)
                AllData.notice("2.0-17.0")
                AllData.ignore("1:**-2.0 17.0-**")
                AllData.ignore("2:**-2.0 17.0-**")
                Plot.setRebin(rebin,1000)
                Plot.device = '/xs'
                Plot('data')
                self.model_bvapec(line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False,spec_num=2)
                self.load_apecroot(line)
                self.fit_error(self.error_str,False,'nxb1')
                self.nxb_fix_parameter(fix=True)
                self.fit_error(self.error_str,True,'nxb1')
                self.set_xydata_multi(identifiers)
                self.result_pack()
                self.result_pack_only_z(model=AllModels(2),group=2)
                self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
                #self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
                self.plotting_simult(f'simultaneous_{region}_{line}_rebin_{rebin}',line=line,bg_plot=False)


    def simultaneous_fitting_adr(self,region='center',line='wz', rebin=1):
        for region in ['all','center','outer']:
            AllData.clear()
            current_dir = os.getcwd()
            identifiers = ['before', 'after', '2000', '3000', '4000']
            for e,identifier in enumerate(identifiers): 
                if identifier == 'before' or identifier == 'after': 
                    os.chdir(f'../{identifier}_recycle/analysis')
                    spec = f'1000_{region}_merged_b1.pi'
                    rmf  = f'1000_{region}_L_without_Lp.rmf'
                    arf  = f'1000_{region}_image_1p8_8keV_1e7.arf'
                    multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
                    self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=None,datagroup=e+1,spectrumgroup=e+1)
                    os.chdir(current_dir)
                else:
                    os.chdir(f'../after_recycle/analysis')
                    spec = f'{identifier}_{region}_merged_b1.pi'
                    rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
                    arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'
                    multiresp = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
                    self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp=multiresp,multiresp2=None,datagroup=e+1,spectrumgroup=e+1)
                    os.chdir(current_dir)
            AllData.notice("2.0-17.0")
            AllData.ignore("1:**-2.0 17.0-**")
            AllData.ignore("2:**-2.0 17.0-**")
            AllData.ignore("3:**-2.0 15.0-**")
            AllData.ignore("4:**-2.0 17.0-**")
            AllData.ignore("5:**-2.0 17.0-**")
            Plot.setRebin(rebin,1000)
            Plot.device = '/xs'
            for line in ['None', 'w', 'wz']:
                Plot('data')
                self.model_bvapec(line,load_nxb=True,load_FeMXS=False,sigma_fix=False,multi=False)
                self.load_apecroot(line)
                self.fit_error(self.error_str,False,'nxb1')
                self.nxb_fix_parameter(fix=True)
                self.fit_error(self.error_str,True,'nxb1')
                Plot('data delc')
                self.set_xydata_multi(identifiers)
                self.result_pack()
                for i in range(2,6):
                    self.result_pack_only_z(model=AllModels(i),group=i)
                self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
                self.plotting_simult(f'simultaneous_{region}_{line}_rebin_{rebin}',line=line,bg_plot=False,identifiers=identifiers)

    def simultaneous_some_fwdata_b(self,region='center',line='wz',optbin=True,rebin=1):
        '''
        全フィルターの同時fitting
        '''
        AllData.clear()
        identifiers = ['1000', '2000', '3000', '4000']
        col_list    = ['black', 'red', 'green', 'blue'] 
        for identifier in identifiers:
            if optbin == True:
                spec = f'{identifier}_{region}_merged_b1.pi'
            else:
                spec = f'{identifier}_{region}_b1.pi'
            rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
            arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'

            spectrum = Spectrum(spec)
            spectrum.response = rmf
            spectrum.response.arf = arf
            spectrum.multiresponse[1] = '/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf'
        self.region = region
        self.model_bvapec(line,load_nxb=True,sigma_fix=False)
        self.load_apecroot(line)
        AllData.notice("2.0-17.0")
        AllData.ignore("1:**-2.0 17.0-**")
        if '2000' in identifiers:
            AllData.ignore("2:**-2.0 15.0-**")
        if '3000' in identifiers:
            AllData.ignore("3:**-2.0 17.0-**")
        if '4000' in identifiers:
            AllData.ignore("4:**-2.0 17.0-**")
        Plot.setRebin(rebin,1000)
        Plot.device = '/xs'
        Plot('data')

        self.fit_error(self.error_str,False,'nxb1')
        self.nxb_fix_parameter(fix=True)
        self.fit_error(self.error_str,True,'nxb1')
        self.set_xydata_multi(identifiers)
        self.result_pack()
        self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
        self.plotting_multi(f'simultaneous_{region}_{line}_rebin_{rebin}',line=line,bg_plot=False)


    def simultaneous_some_fwdata(self,region='center',optbin=True,rebin=1):
        '''
        全フィルターの同時fitting
        '''
        AllData.clear()
        identifiers = ['1000', '2000', '3000', '4000']
        col_list    = ['black', 'red', 'green', 'blue'] 
        for e,identifier in enumerate(identifiers):
            if optbin == True:
                spec = f'{identifier}_{region}_merged_b1.pi'
            else:
                spec = f'{identifier}_{region}_b1.pi'
            rmf  = f'{identifier}_{region}_L_without_Lp.rmf'
            arf  = f'{identifier}_{region}_image_1p8_8keV_1e7.arf'

            self.load_spectrum(spec=spec,rmf=rmf,arf=arf,multiresp='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf/newdiag60000.rmf',datagroup=e+1,spectrumgroup=e+1)

        AllData.notice("2.0-17.0")
        AllData.ignore("1:**-2.0 17.0-**")
        if '2000' in identifiers:
            AllData.ignore("2:**-2.0 15.0-**")
        if '3000' in identifiers:
            AllData.ignore("3:**-2.0 17.0-**")
        if '4000' in identifiers:
            AllData.ignore("4:**-2.0 17.0-**")
        Plot.setRebin(rebin,1000)
        Plot.device = '/xs'
        self.region = region
        for line in ['w', 'wz']:
            self.model_bvapec(line,load_nxb=True,sigma_fix=False,multi=False,spec_num=4)
            self.load_apecroot(line)
            Plot('data')

            self.fit_error(self.error_str,False,'nxb1')
            self.nxb_fix_parameter(fix=True)
            self.fit_error(self.error_str,True,'nxb1')
            self.set_xydata_multi(identifiers)
            self.result_pack()
            for i in range(2,5):
                self.result_pack_only_z(model=AllModels(i),group=i)
            self.savemod_multi(f'simultaneous_{region}_{line}_rebin_{rebin}')
            self.plotting_simult(f'simultaneous_{region}_{line}_rebin_{rebin}',line=line,bg_plot=False,identifiers=identifiers,col_list=col_list)



## resolve_analysis.py out scripts
# 
def out_process():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["2000","3000","4000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/analysis', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                processor.process_events()


def out_process_mxs_cal():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        identifiers = ["2000","3000","4000"]
        # identifiers = ["1000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/center_outer_full_open', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_low_cnt/num100_min10/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/num100_min10', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                processor.process_events_center()

def out_process_ghf_cor():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        identifiers = ["1000","2000","3000"]
        # identifiers = ["4000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/corrected_data/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/analysis', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                processor.process_events_center_outer()

def out_process_open():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["1000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/center_outer_full_open', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_cl_data/all_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source/MXS_BrightMode', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/repro_test', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                processor.process_events()

def out_process_all():
    with open("logfile.txt", "w+") as f:
        identifiers = ["1000", '2000', '3000', '4000']
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/after_recycle/corrected_data/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/after_recycle/analysis', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                if identifier == '1000':
                    processor.process_events_open('after')
                else:
                    processor.process_events()
            processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/before_recycle/corrected_data/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/before_recycle/analysis', identifier='1000', respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
            processor.process_events_open('before')

def out_process_all2():
    with open("logfile.txt", "w+") as f:
        identifiers = ["1000", '2000', '3000', '4000']
        with redirect_stdout(f):
            # for identifier in identifiers:
            #     processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/ex_27/after_recycle', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
            #     if identifier == '1000':
            #         processor.process_events_open('after')
            #     else:
            #         processor.process_events()
            # processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/before_recycle/corrected_data/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/55Fe_gh_liner_cor/before_recycle/analysis', identifier='1000', respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
            # processor.process_events_open('before')
            processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/ex_27/before_recycle', identifier='1000', respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
            processor.process_events_open('before')

def out_process_Be(recycle='after'):
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["4000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/center_outer_full_open', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_cl_data/all_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source/MXS_BrightMode', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir=f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/Be/{recycle}_recycle', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                processor.process_events_Be(recycle)

def out_process_all_region():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source/MXS_BrightMode', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_cl_data/all_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                processor.process_events_all_region()

def out_process_55Fe():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_plot_all', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                
                # processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_cl_data/pixel_by_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                processor.process_events()

def out_pixel_by_pixel():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/OBF_ND_Be', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf')
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf')
                processor.pixel_by_pixel(time_filter=False,make_resp=False)

def out_data_rescreen_for_occulation():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_occultation', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/Abell2319/000103000/resolve/event_uf/', outfiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False,obsid='000103000')
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_uf/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_occulation/55Fe_template', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False)
                processor.rescreen_occultation()

def out_data_rescreen_for_occulation_before():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/55Fe_occultation', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True)
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/Abell2319/000103000/resolve/event_uf/', outfiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_Fe_resp', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False,obsid='000103000')
                processor.rescreen_occultation()

def out_process_occulation_pixel_by_pixel(infile='occulation_px5000_cl_max_fine.evt'):
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                # processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation/', outfiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation/pixel_by_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=infile,repro=False,obsid='000103000')
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/nominal/55Fe_occultation/10ms_thresh/max/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs/nominal/55Fe_occultation/10ms_thresh/max', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=infile,repro=True,obsid='000112000', mxs_mode=True)
                processor.pixel_by_pixel(make_resp=False)

def out_process_occulation_all_pixel(infile='occulation_nxb_px5000_cl.evt'):
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                #processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/OBF_ND_Be', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf')
                # processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe/occulation_1/all_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=infile,repro=False)
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation/', outfiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation/all_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=infile,repro=False,obsid='000103000')
                processor.process_events_for_55Fe_occulation_all_pixel()


def out_process_occulation_template(infile='occulation_px5000_cl_max_fine.evt'):
    with open("logfile.txt", "w+") as f:
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                # processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation/', outfiledir='/Volumes/SUNDISK_SSD/Abell2319/analysis/000103000_repro_occulation/center_outer', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=infile,repro=False,obsid='000103000')
                # processor.process_events_for_55Fe_occulation_center_outer()
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_occulation/55Fe_template/max/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_occulation/55Fe_template/max', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=infile,repro=False,obsid='000112000')
                processor.process_events_template()

def out_process_nxb():
        # identifiers = ["1000","2000","3000","4000"]
    identifiers = ["2000","3000","4000"]
    region = ['center','outer']
    for identifier in identifiers:
        for reg in region:
            processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir='/Volumes/SUNDISK_SSD//repro_analysis/55Fe_nasa_repro_v3/OBF_ND_Be/nxb', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False,region=reg)
            processor.make_nxb_event(evt_file=f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/OBF_ND_Be/{identifier}_{reg}.evt',region=reg)

def out_process_mxs_events(t=20,m='brt'):
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir=f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_mxs_check/{m}/{str(t)}ms', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False, input_file=f'xa000112000rsl_p0px5000_cl_{m}_mxsphase.evt.gz')
                shutil.copy(f'/Volumes/SUNDISK_SSD/PKS_XRISM/from_sawadasan/pks0745/xa000112000rsl_p0px5000_cl_{m}_mxsphase.evt.gz',f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_mxs_check/{m}/{str(t)}ms')
                os.chdir(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_mxs_check/{m}/{str(t)}ms')
                processor.process_mxs_events(mxs_phase_threshold=t*1e-3)


def out_process_mxs_events_for_cal(m='brt',t=1):
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        # identifiers = ["1000","2000","3000","4000"]
        identifiers = ["5000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/000112000/resolve/event_cl/', outfiledir=f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_cal_check/comparison_55Fe_mxs_mode/{m}', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=False, input_file=f'xa000112000rsl_p0px5000_cl_{m}_mxsphase.evt.gz')
                shutil.copy(f'/Volumes/SUNDISK_SSD/PKS_XRISM/from_sawadasan/pks0745/xa000112000rsl_p0px5000_cl_{m}_mxsphase.evt.gz',f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_mxs_check/{m}/{str(t)}ms')
                os.chdir(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_cal_check/comparison_55Fe_mxs_mode/{m}')
                processor.pixel_by_pixel_for_cal(t*1e-3)

def out_process_55Fe_fine():
    with open("logfile.txt", "w+") as f:
        identifiers = ["5000"]
        dirs = ['max', 'brt', 'brmax']
        files = ['xa000112000rsl_p0px5000_max.evt','xa000112000rsl_p0px5000_brt.evt','xa000112000rsl_p0px5000_cl_before_max.evt']
        with redirect_stdout(f):
            identifier = identifiers[0]
            for d,f in zip(dirs,files):
                os.chdir(f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source_divide_fine/analysis/{d}')
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source_divide_fine/repro_data/', outfiledir=f'/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/55Fe_nasa_repro_v3/55Fe_cl_data/for_source_divide_fine/analysis/{d}', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',input_file=f,repro=False,obsid='000112000')
                #processor.process_events_center()
                processor.process_events_outer()

def out_process_MXS_gain_ev():
    pass

def multi_mxs():
    time = [5, 10, 20]
    mode = ['brt','max']
    for m in mode:
        for t in time:
            out_process_mxs_events(t=t,m=m)
 