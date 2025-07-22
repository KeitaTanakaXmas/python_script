import os
import sys
from contextlib import redirect_stdout
import subprocess
import shutil
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

class FileOperation:
    """_summary_
    Class for file operations
    """
    def __init__(self, infiledir:str, outfiledir:str, identifier:str, respdir:str, repro=True, input_file=None, obsid='000112000'):
        """_summary_
        Constructor
        Args:
            infiledir (str): _description_
            outfiledir (str): _description_
            identifier (str): _description_
        """
        self.infiledir  = infiledir
        self.outfiledir = outfiledir
        self.identifier = identifier
        self.respdir    = respdir
        self.obsid      = obsid
        self.reprocounter = 1
        if obsid == '000112000':
            self.datadir = '/Volumes/SUNDISK_SSD/PKS_XRISM/000112000'
        else:
            self.datadir = f'/Volumes/SUNDISK_SSD/Abell2319/{obsid}'
        if input_file == None:
            self.input_file = f"xa{obsid}rsl_p0px{identifier}_cl.evt"
        else :
            self.input_file = input_file
        self.base_name  = os.path.splitext(os.path.basename(self.input_file))[0]
        self.center_arf = 'center_image_1p8_8keV_1e7.arf'
        self.outer_arf  = 'outer_image_1p8_8keV_1e7.arf'
        if repro == True :
            self.ehk      = f'{self.infiledir}/xa{obsid}.ehk'
            self.pixelgti = f'{self.infiledir}/xa{obsid}rsl_px{self.identifier}_exp.gti'
            self.hk       = f'{self.infiledir}/xa{obsid}rsl_a0.hk1'
            self.tel      = f'{self.infiledir}xa{obsid}rsl_tel.gti'
            self.gen      = f'{self.infiledir}/xa{obsid}_gen.gti'
        else :
            self.ehk      = f'{self.datadir}/auxil/xa{obsid}.ehk.gz'
            self.pixelgti = f'{self.datadir}/resolve/event_uf/xa{obsid}rsl_px{self.identifier}_exp.gti.gz'
            self.hk       = f'{self.datadir}/resolve/hk/xa{obsid}rsl_a0.hk1'
            self.tel      = f'{self.datadir}/resolve/event_uf/xa{obsid}rsl_tel.gti.gz'
            self.gen      = f'{self.datadir}/auxil/xa{obsid}_gen.gti.gz'
    
    def run_command(self, command:str):
        """_summary_
        Run shell command with subprocess.run
        Args:
            command (str): _description_
        """
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)

    def make_directory(self, directory:str, overwrite:bool=False):
        """_summary_
        Make directory if not exist.
        If directory exists and overwrite is True, remove the directory and make a new one.
        Args:
            directory (str): _description_
            overwrite (bool): _description_
        """
        if os.path.exists(directory) and overwrite:
            os.rmdir(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

class EventProcessing:
    def __init__(self, infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro_analysis/mxs', identifier='1000', respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True,input_file=None,obsid='000112000',mxs_mode=False):
        self.center_pixel           = ['00', '17', '18', '35']
        self.mxs_pixel              = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
        self.center_pixel_str       = '0,17,18,35'
        if mxs_mode == True:
            self.ALL_PIXEL          = ['00', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '33', '34', '35']
            self.all_pixel_str      = '0,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33,34,35'
            self.outer_pixel        = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '34']
            self.outer_pixel_str    = '7,8,9,10,11,13,14,15,16,19,20,21,22,23,24,25,26,33,34'
            self.all_region_file    = 'all_mxs.reg'
            self.center_region_file = 'center_det.reg'
            self.outer_region_file  = 'outer_det_mxs.reg'
        else:
            self.ALL_PIXEL          = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '34', '35']
            self.all_pixel_str      = '0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35'
            self.outer_pixel        = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '19', '20', '21', '22', '23', '24', '25', '26', '28', '29', '30', '31', '32', '33', '34']
            self.outer_pixel_str    = '1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34'
            self.all_region_file    = 'region_RSL_det_ex27.reg'
            self.center_region_file = 'center_det.reg'
            self.outer_region_file  = 'outer_det_ex27.reg'
        self.file_ops   = FileOperation(infiledir,outfiledir,identifier,respdir,repro,input_file,obsid) 
        self.repro = repro

    def reprocess_MXS(self):
        # calmethod = 'Cal-pix', 'MXS', 'Fe55'
        obs='./000112000'
        obsid = '000112000'
        command = f"""
xapipeline indir={obs} outdir=./mxs steminputs=xa{obsid} stemoutputs=DEFAULT entry_stage=1 exit_stage=2 instrument=resolve verify_input=no create_ehkmkf=no calc_pointing=yes calc_optaxis=yes calc_gtilost=no calc_adrgti=no calc_mxsgti=yes rsl_gainfile=CALDB calmethod=MXS linetocorrect=CrKa seed=650504 numevent=1000 minevent=200 extraspread=40 spangti=yes cleanup=yes clobber=yes       
"""
        self.file_ops.run_command(command)

    def reprocess_Fe(self):
        obs='/Users/keitatanaka/Dropbox/share/work/astronomy/PKS/000112000/'
        obsid = '000112000'
        command = f"""
xapipeline indir={obs} outdir={obs}rsl_repro steminputs=xa{obsid} stemoutputs=DEFAULT entry_stage=1 exit_stage=2 instrument=resolve verify_input=no create_ehkmkf=no calc_pointing=yes calc_optaxis=yes calc_gtilost=no calc_adrgti=no calc_mxsgti=no rsl_gainfile=CALDB calmethod=Fe55 linetocorrect=MnKa numevent=1000 minevent=200 extraspread=40 spangti=no clobber=yes rsl_linefitfile=CALDB        
"""
        self.file_ops.run_command(command)

    def rescreen_occultation(self):
        command = f"""
ahgtigen infile={self.file_ops.ehk} outfile=xa{self.file_ops.obsid}rsl_fe55_ehk.gti gtifile=none gtiexpr="(SAA_SXS==0).and.(ELV<-5)" mergegti=AND telescop=XRISM
"""
        self.file_ops.run_command(command)

        if self.repro == False:
            ufevt = f'{self.file_ops.infiledir}xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_uf.evt.gz'
        else:
            ufevt = f'{self.file_ops.infiledir}xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_uf.evt'
        ntegti = f'xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_nte.gti'
        command = f"""
ftmgtime "{ufevt}[GTI],{ufevt}[GTIADROFF],{ufevt}[GTIMKF],{self.file_ops.tel}[GTITEL],{self.file_ops.gen}[GTIOBS],xa{self.file_ops.obsid}rsl_fe55_ehk.gti[GTI]" {ntegti} AND
"""
        self.file_ops.run_command(command)



        if self.repro == False:
            command = f"""
ahscreen infile={self.file_ops.infiledir}xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_uf.evt.gz outfile=occulation_nxb_px{self.file_ops.identifier}_cl.evt gtifile={ntegti} expr=NONE selectfile=None label=NONE mergegti=AND
"""
        else:
            command = f"""
ahscreen infile={self.file_ops.infiledir}xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_uf.evt outfile=occulation_nxb_px{self.file_ops.identifier}_cl.evt gtifile={ntegti} expr=NONE selectfile=None label=NONE mergegti=AND
"""

        self.file_ops.run_command(command)

        self.edit_header(file=f'{self.file_ops.outfiledir}/occulation_nxb_px{self.file_ops.identifier}_cl.evt', keyword='TLMIN46', value=0) 

    def rescreen_occultation_before_analysis(self):
        command = f"""
maketime {self.file_ops.ehk} {self.file_ops.identifier}_occultation_gti.fit \"(SAA_SXS==0).and.(ELV<-5)\" anything anything TIME no extname=GTI
"""
        self.file_ops.run_command(command)  
        if self.repro == False:
            command = f"""
ahscreen infile={self.file_ops.infiledir}xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_uf.evt.gz outfile=occulation_nxb_px{self.file_ops.identifier}_cl.evt gtifile={self.file_ops.identifier}_occultation_gti.fit expr=NONE selectfile=CALDB label=PIXELALL mergegti=AND
"""
        else:
            command = f"""
ahscreen infile={self.file_ops.infiledir}xa{self.file_ops.obsid}rsl_p0px{self.file_ops.identifier}_uf.evt outfile=occulation_nxb_px{self.file_ops.identifier}_cl.evt gtifile={self.file_ops.identifier}_occultation_gti.fit expr=NONE selectfile=CALDB label=PIXELALL mergegti=AND
"""

        self.file_ops.run_command(command)

        self.edit_header(file=f'{self.file_ops.outfiledir}/occulation_nxb_px{self.file_ops.identifier}_cl.evt', keyword='TLMIN46', value=0) 

    def make_nxb_event(self,evt_file,region):
        nxb_dir = '/Volumes/SUNDISK_SSD/PKS_XRISM/nxb'
        nxb_evt = f'{nxb_dir}/merged_nxb_resolve_gtifix.evt.gz'
        nxb_ehk = f'{nxb_dir}/merged_reduced_rev3_fix2.ehk.gz'

        command = f"""
maketime {nxb_ehk} {self.file_ops.identifier}_{region}_ehkSelA.gti \"T_SAA_SXS>0 && ELV<-5 && DYE_ELV>5\" compact=no time=TIME no
        """
        self.file_ops.run_command(command)
        command = f"""
maketime {self.file_ops.ehk} {self.file_ops.identifier}_{region}_ehkSelB.gti \"T_SAA_SXS>0 && DYE_ELV>5 && CORTIME>6\" compact=no time=TIME no
        """
        self.file_ops.run_command(command)
        command = f"""
extractor \'filename={evt_file}\' \'eventsout=xa000112000rsl_p0px{self.file_ops.identifier}_{region}_cl_ehkSel.evt\' \'imgfile=NONE\' \'phafile=NONE\' \'fitsbinlc=NONE\' \'regionfile=NONE\' \'timefile={self.file_ops.identifier}_{region}_ehkSelB.gti\' \'xcolf=X\' \'ycolf=Y\' \'tcol=TIME\' \'ecol=PI\' \'xcolh=DETX\' \'ycolh=DETY\'
        """
        self.file_ops.run_command(command)
        command = f"""
extractor \'filename={nxb_evt}\' \'eventsout={self.file_ops.identifier}_{region}_merged_nxb_resolve_gtifix_ehkSel.evt\' \'imgfile=NONE\' \'phafile=NONE\' \'fitsbinlc=NONE\' \'regionfile=NONE\' \'timefile={self.file_ops.identifier}_{region}_ehkSelA.gti\' \'xcolf=X\' \'ycolf=Y\' \'tcol=TIME\' \'ecol=PI\' \'xcolh=DETX\' \'ycolh=DETY\'
        """
        self.file_ops.run_command(command)
        command = f"""
ftselect \'xa000112000rsl_p0px{self.file_ops.identifier}_{region}_cl_ehkSel.evt[EVENTS]\' xa000112000rsl_p0px{self.file_ops.identifier}_{region}_cl_evSel.evt \"(PI>=600) && ((RISE_TIME+0.00075*DERIV_MAX)>46) && ((RISE_TIME+0.00075*DERIV_MAX)<58) && (ITYPE==0) && (STATUS[4]==b0)\"
        """
        self.file_ops.run_command(command)
        if region == 'center':
            regfile = f'{self.file_ops.respdir}/center_det.reg'
            pixels = '-'
        elif region == 'outer':
            regfile = f'{self.file_ops.respdir}/outer_det.reg'
            pixels = '-'
        else:
            regfile = 'NONE'
            pixels = '-'
        
        command = f"""
rslnxbgen infile=xa000112000rsl_p0px{self.file_ops.identifier}_{region}_cl_evSel.evt ehkfile={self.file_ops.ehk} regfile={regfile} regmode=DET innxbfile={self.file_ops.identifier}_{region}_merged_nxb_resolve_gtifix_ehkSel.evt innxbehk={nxb_ehk} database=LOCAL db_location=/Volumes/SUNDISK_SSD/PKS_XRISM/nxb timefirst=-150 timelast=+150 SORTCOL=CORTIME sortbin=\"0,6,8,10,12,99\" expr=\"(PI>=600) && ((RISE_TIME+0.00075*DERIV_MAX)>46) && ((RISE_TIME+0.00075*DERIV_MAX)<58) && (ITYPE==0) && (STATUS[4]==b0)\" outpifile={self.file_ops.identifier}_{region}_rslnxb.pi outnxbfile={self.file_ops.identifier}_{region}_rslnxb.evt outnxbehk={self.file_ops.identifier}_{region}_rslnxb.ehk
"""
        print('----------------------------------')
        print(command)
        self.file_ops.run_command(command)

    def rise_time_screening(self):
        print("----------------------------------")
        print("Rise time screening")
        command = f"""
            ftcopy infile=\"{self.file_ops.infiledir}{self.file_ops.input_file}[EVENTS][(PI>=600) && (((((RISE_TIME+0.00075*DERIV_MAX)>46)&&((RISE_TIME+0.00075*DERIV_MAX)<58))&&ITYPE<4)||(ITYPE==4))&&STATUS[4]==b0]\" outfile={self.file_ops.outfiledir}/{self.file_ops.base_name}_{self.file_ops.reprocounter}_risetime_screening.evt copyall=yes clobber=yes history=yes
        """
        self.file_ops.run_command(command)
        self.file_ops.input_file = f"{self.file_ops.base_name}_{self.file_ops.reprocounter}_risetime_screening.evt"
        self.file_ops.reprocounter += 1

    def make_event_file_without_Ls(self):
        command = f"""
        ftcopy infile=\"{self.file_ops.outfiledir}/{self.file_ops.input_file}[EVENTS][(PI>=4000)&&(PI<=20000)&&(ITYPE<4)]\" outfile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt copyall=yes clobber=yes
"""
        self.file_ops.run_command(command)
        self.rmf_evt_file = f'{self.file_ops.base_name}_without_Lp_for_RMF.evt'

    def edit_header(self, file, keyword, value):
        print("----------------------------------")
        print("Edit Header")
        hdu = fits.open(file, mode='update')  # mode='update' にすることで書き込みモードで開く
        evt = hdu[1]
        print(f"Original value of {keyword}: {evt.header[keyword]}")
        evt.header[keyword] = value
        print(f"New value of {keyword}: {evt.header[keyword]}")
        hdu.flush()  # 変更をファイルに書き込む
        hdu.close()

    def EtoPHA(self,E):
        return int(E*2000)

    def event_selection_without_calpix(self):
        print("----------------------------------")
        print("Event selection")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
filter column "PIXEL=0:11,13:26,28:35"
filter GRADE "0:0"
extr event
save event {self.file_ops.base_name}_{self.file_ops.reprocounter}_Hp_without_calpix.evt
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)
        self.file_ops.input_file = f"{self.file_ops.base_name}_{self.file_ops.reprocounter}_Hp_without_calpix.evt"
        self.without_calpix = f"{self.file_ops.base_name}_{self.file_ops.reprocounter}_Hp_without_calpix.evt"

    def event_selection_allpix(self):
        print("----------------------------------")
        print("Event selection")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.output_dir}
{self.file_ops.base_name}_2.evt
yes
filter GRADE "0:0"
extr event
save event {self.output_dir}/{self.file_ops.base_name}_3.evt
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)

    def select_all_region(self):
        print("----------------------------------")
        print("Select Center Region")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
extr spec 
save spec {self.file_ops.outfiledir}/{self.file_ops.identifier}_all.pi
yes
yes
set image det
extr image
save image {self.file_ops.outfiledir}/{self.file_ops.identifier}_all_img.fits
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)

    def select_center_region(self):
        print("----------------------------------")
        print("Select Center Region")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
select event "PIXEL==0 || PIXEL==17 || PIXEL==18 || PIXEL==35"
extr event
save event {self.file_ops.identifier}_center.evt
yes
yes
extr spec 
save spec {self.file_ops.identifier}_center.pi
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)

    def select_outer_region(self):
        print("----------------------------------")
        print("Select Outer Region")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
select event "PIXEL!=0 && PIXEL!=17 && PIXEL!=18 && PIXEL!=35"
extr event
save event {self.file_ops.identifier}_outer.evt
yes
yes
extr spec 
save spec {self.file_ops.identifier}_outer.pi
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)

    def select_single_pixel(self,pixel_number):
        print("----------------------------------")
        print("Select Center Region")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
select event "PIXEL=={pixel_number}"
extr spec 
save spec {self.file_ops.identifier}_PIXEL_{pixel_number}.pi
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)

#     def exclude_TIME(self,Time_min,Time_max):
#         print("----------------------------------")
#         print("exclude Time")
#         commands = f"""
# xselect << EOF
# xsel
# no
# read event
# {self.file_ops.outfiledir}
# {self.file_ops.input_file}
# yes
# select event "TIME > {Time_max} || TIME < {Time_min}"
# extr event
# save event {self.file_ops.base_name}_{self.file_ops.reprocounter}_exclude_{Time_min}_{Time_max}.evt
# yes
# yes
# exit
# no
# EOF
# """
#         self.file_ops.run_command(commands)
#         self.file_ops.input_file = f"{self.file_ops.base_name}_{self.file_ops.reprocounter}_exclude_{Time_min}_{Time_max}.evt"
#         self.file_ops.reprocounter += 1
    
    def exclude_MXS_phase(self,MXS_phase_threshold):
        '''
        Exclude MXS phase
        This script can use only for the data which is processed by sawadasan's mxs phase pipeline
        '''
        print("----------------------------------")
        print("exclude MXS Phase Time")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
select event "MXS_PHASE > {MXS_phase_threshold}"
extr event
save event {self.file_ops.base_name}_{self.file_ops.reprocounter}_exclude_mxsphase_{MXS_phase_threshold}.evt
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)
        self.file_ops.input_file = f"{self.file_ops.base_name}_{self.file_ops.reprocounter}_exclude_mxsphase_{MXS_phase_threshold}.evt"
        self.file_ops.reprocounter += 1
  
    def exclude_MXS_phase_inphase(self,MXS_phase_threshold):
        '''
        Exclude MXS phase
        This script can use only for the data which is processed by sawadasan's mxs phase pipeline
        '''
        print("----------------------------------")
        print("exclude MXS Phase Time")
        commands = f"""
xselect << EOF
xsel
no
read event
{self.file_ops.outfiledir}
{self.file_ops.input_file}
yes
select event "MXS_PHASE < {MXS_phase_threshold}"
extr event
save event {self.file_ops.base_name}_{self.file_ops.reprocounter}_exclude_mxsphase_{MXS_phase_threshold}.evt
yes
yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)
        self.file_ops.input_file = f"{self.file_ops.base_name}_{self.file_ops.reprocounter}_exclude_mxsphase_{MXS_phase_threshold}.evt"
        self.file_ops.reprocounter += 1

    def get_count(self,Emin,Emax):
        print("----------------------------------")
        print("Get Count")
        PHAmin = self.EtoPHA(Emin)
        PHA_max = self.EtoPHA(Emax)
        commands = f"""
xselect << EOF
xsel
no
read event
{self.output_dir}
{self.file_ops.base_name}_3.evt
yes
yes
filter pha_cutoff {PHAmin} {PHA_max}
extr all

yes
exit
no
EOF
"""
        self.file_ops.run_command(commands)

    def make_exposure_map(self):
        if hasattr(self, 'without_calpix'):
            self.file_ops.input_file = self.without_calpix
        else :
            self.file_ops.input_file = f'xa000112000rsl_p0px{self.file_ops.identifier}_cl_{self.file_ops.reprocounter}_Hp_without_calpix.evt'
            self.file_ops.input_file = f'xa000112000rsl_p0px5000_cl_brt_mxsphase_3_Hp_without_calpix.evt'
            self.file_ops.input_file = f'xa000112000rsl_p0px{self.file_ops.identifier}_cl_2_Hp_without_calpix.evt'
        print("----------------------------------")
        print("Make Exposure Map")
        commands = f"""
xaexpmap ehkfile={self.file_ops.ehk} gtifile={self.file_ops.outfiledir}/{self.file_ops.input_file} instrume=RESOLVE badimgfile=NONE pixgtifile={self.file_ops.pixelgti} outfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo outmaptype=EXPOSURE delta=20.0 numphi=1 stopsys=SKY instmap=CALDB qefile=CALDB contamifile=CALDB vigfile=CALDB obffile=CALDB fwfile=CALDB gvfile=CALDB maskcalsrc=yes fwtype=FILE specmode=MONO specfile=center.pi specform=FITS evperchan=DEFAULT abund=1 cols=0 covfac=1 clobber=yes chatter=1 logfile={self.file_ops.outfiledir}/make_expo_xa000112000rsl_p0px{self.file_ops.identifier}.log
        """
        self.file_ops.run_command(commands)

    def make_rmf_center_outer_X(self):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_X_without_Lp regmode=DET splitrmf=yes elcbinfac=16 whichrmf=X splitcomb=yes resolist=0 eminin=0.0 dein=0.5 nchanin=60000 useingrd=no eminout=0.0 deout=0.5 nchanout=60000 regionfile=NONE pixlist={self.center_pixel_str}",
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_X_without_Lp regmode=DET splitrmf=yes elcbinfac=16 whichrmf=X splitcomb=yes resolist=0 eminin=0.0 dein=0.5 nchanin=60000 useingrd=no eminout=0.0 deout=0.5 nchanout=60000 regionfile=NONE pixlist={self.outer_pixel_str}",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_center_outer(self):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_L_without_Lp regmode=DET whichrmf=L resolist=0 regionfile=NONE pixlist={self.center_pixel_str}",
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_L_without_Lp regmode=DET whichrmf=L resolist=0 regionfile=NONE pixlist={self.outer_pixel_str}",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_center(self):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_L_without_Lp regmode=DET whichrmf=L resolist=0 regionfile=NONE pixlist=0,17,18,35",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_outer(self):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_L_without_Lp regmode=DET whichrmf=L resolist=0 regionfile=NONE pixlist={self.outer_pixel_str}",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_all(self):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_L_without_Lp regmode=DET whichrmf=L resolist=0 regionfile=None pixlist={self.all_pixel_str}",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_all_X(self):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_X_without_Lp regmode=DET splitrmf=yes elcbinfac=16 whichrmf=X splitcomb=yes resolist=0 eminin=0.0 dein=0.5 nchanin=60000 useingrd=no eminout=0.0 deout=0.5 nchanout=60000 regionfile=None pixlist={self.all_pixel_str}",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_single_pixel(self,pixel):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_pix{pixel}_L_without_Lp regmode=DET whichrmf=L resolist=0 regionfile=NONE pixlist={pixel}",
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_rmf_single_pixel_X(self,pixel):
        print("----------------------------------")
        print("Make RMF")
        commands = [
            f"rslmkrmf infile={self.file_ops.outfiledir}/{self.file_ops.base_name}_without_Lp_for_RMF.evt outfileroot={self.file_ops.outfiledir}/{self.file_ops.identifier}_pix{pixel}_X_without_Lp regmode=DET splitrmf=yes elcbinfac=16 whichrmf=X splitcomb=yes resolist=0 eminin=0.0 dein=0.5 nchanin=60000 useingrd=no eminout=0.0 deout=0.5 nchanout=60000 regionfile=None pixlist={pixel}",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_center_outer(self):
        #! 5000のときにこれ使ってないか確認。rmfがLになってる。
        commands = [
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.center_region_file} sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_image_1p8_8keV_1e7.log chatter=3",
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.outer_region_file} sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_image_1p8_8keV_1e7.log chatter=3",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_center(self):
        #! 5000のときにこれ使ってないか確認。rmfがLになってる。
        commands = [
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.center_region_file} sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_center_image_1p8_8keV_1e7.log chatter=3",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_outer(self):
        #! 5000のときにこれ使ってないか確認。rmfがLになってる。
        commands = [
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.outer_region_file} sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_outer_image_1p8_8keV_1e7.log chatter=3",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_all(self):
        commands = [
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.all_region_file} sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_image_1p8_8keV_1e7.log chatter=3",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_all_X(self):
        commands = [
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.all_region_file} sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_X_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_all_image_1p8_8keV_1e7.log chatter=3",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_ssm_center(self, photon_num=10000000):
        pow_factor = int(np.log10(photon_num))
        first_digit = int(photon_num // (10 ** pow_factor))
        photon_str = f"{first_digit}e{pow_factor}"
        commands = [
        f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_center_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_center_to_center_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.center_region_file} sourcetype=IMAGE rmffile=1000_center_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_center.img logfile=arf.log",
        f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_outer_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_outer_to_center_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.center_region_file} sourcetype=IMAGE rmffile=1000_center_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_outerMXS.fits logfile=arf.log",
        f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_exMXS_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_exMXS_to_center_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.center_region_file} sourcetype=IMAGE rmffile=1000_center_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_exMXS.fits logfile=arf.log"
        ]
        # f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_exsource_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_exsource_to_center_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.center_region_file} sourcetype=IMAGE rmffile=1000_center_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_exsource.fits logfile=arf.log",
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_ssm_outer(self, photon_num=10000000):
        pow_factor = int(np.log10(photon_num))
        first_digit = int(photon_num // (10 ** pow_factor))
        photon_str = f"{first_digit}e{pow_factor}"
        print(f"photon = {photon_str}")
        commands = [
        f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_center_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_center_to_outer_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.outer_region_file} sourcetype=IMAGE rmffile=1000_outer_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_center.img logfile=arf.log",
        f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_outer_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_outer_to_outer_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.outer_region_file} sourcetype=IMAGE rmffile=1000_outer_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_outerMXS.fits logfile=arf.log",
        f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_exMXS_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_exMXS_to_outer_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.outer_region_file} sourcetype=IMAGE rmffile=1000_outer_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_exMXS.fits logfile=arf.log"
        ]
        # f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_SSM_exsource_1p8_8keV_{photon_str}.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_SSM_exsource_to_outer_1p8_8keV_{photon_str}.arf numphoton={photon_num} minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/{self.outer_region_file} sourcetype=IMAGE rmffile=1000_outer_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux_exsource.fits logfile=arf.log",
        for command in commands:
            self.file_ops.run_command(command)

    def make_arf_single_pixel(self,pixel):
        commands = [
            f"xaarfgen xrtevtfile={self.file_ops.respdir}/raytrace_image_1p8_8keV_1e7.fits outfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_pix{pixel}_image_1p8_8keV_1e7.arf numphoton=10000000 minphoton=100 source_ra=116.88131195 source_dec=-19.29488623 telescop=XRISM instrume=RESOLVE emapfile={self.file_ops.outfiledir}/xa000112000rsl_p0px{self.file_ops.identifier}.expo regmode=DET regionfile={self.file_ops.respdir}/pixel_by_pixel/pixel_by_pixel_{int(pixel)}_det.reg sourcetype=IMAGE rmffile={self.file_ops.outfiledir}/{self.file_ops.identifier}_pix{pixel}_L_without_Lp.rmf erange=\"0.3 18.0 1.8 8.0\" teldeffile=CALDB qefile=CALDB contamifile=CALDB obffile=CALDB fwfile=CALDB gatevalvefile=CALDB onaxisffile=CALDB onaxiscfile=CALDB mirrorfile=CALDB obstructfile=CALDB frontreffile=CALDB backreffile=CALDB pcolreffile=CALDB scatterfile=CALDB mode=h clobber=yes cleanup=yes seed=7 imgfile={self.file_ops.respdir}/PKS_resolve_arf_image_1.8-8.0_flux.img logfile={self.file_ops.outfiledir}/{self.file_ops.identifier}_pix{pixel}_image_1p8_8keV_1e7.log",
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_center_outer_X(self):
        # self.file_ops.run_heainit()
        #! 5000のときにこれでグルーピングするとcombじゃないから適切にビニングされてないかも。あとで確かめる。
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_center.pi outfile={self.file_ops.identifier}_center_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_center_X_without_Lp.rmf clobber=yes",
            f"ftgrouppha infile={self.file_ops.identifier}_outer.pi outfile={self.file_ops.identifier}_outer_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_outer_X_without_Lp.rmf clobber=yes"
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_center_outer(self):
        # self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_center.pi outfile={self.file_ops.identifier}_center_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_center_L_without_Lp.rmf clobber=yes",
            f"ftgrouppha infile={self.file_ops.identifier}_outer.pi outfile={self.file_ops.identifier}_outer_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_outer_L_without_Lp.rmf clobber=yes"
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_center_outer_X(self):
        # self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_center.pi outfile={self.file_ops.identifier}_center_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_center_X_without_Lp_comb.rmf clobber=yes",
            f"ftgrouppha infile={self.file_ops.identifier}_outer.pi outfile={self.file_ops.identifier}_outer_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_outer_X_without_Lp_comb.rmf clobber=yes"
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_center(self):
        # self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_center.pi outfile={self.file_ops.identifier}_center_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_center_L_without_Lp.rmf clobber=yes",
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_outer(self):
        # self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_outer.pi outfile={self.file_ops.identifier}_outer_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_outer_L_without_Lp.rmf clobber=yes",
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_all(self):
        # self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_all.pi outfile={self.file_ops.identifier}_all_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_all_L_without_Lp.rmf clobber=yes",
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_all_X(self):
        # self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_all.pi outfile={self.file_ops.identifier}_all_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_all_X_without_Lp_comb.rmf clobber=yes",
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def grouping_manual(self, infile, respfile, outfile=None):
        print("----------------------------------")
        print("Grouping")
        if outfile == None:
            outfile = infile.replace('.pi','_merged_b1.pi')
        commands = [
            f"ftgrouppha infile={infile} outfile={outfile} grouptype=optmin groupscale=1 respfile={respfile} clobber=yes",
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def gropuing_single_pixel(self,pixel):
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_PIXEL_{pixel}.pi outfile={self.file_ops.identifier}_PIXEL_{pixel}_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_pix{pixel}_L_without_Lp.rmf clobber=yes" ,
            " "
        ]
        for command in commands:
            self.file_ops.run_command(command)

    def gropuing_single_pixel_X(self,pixel):
        self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_PIXEL_{pixel}.pi outfile={self.file_ops.identifier}_PIXEL_{pixel}_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.identifier}_pix{pixel}_X_without_Lp_comb.rmf clobber=yes",
            " "
        ]
        
        for command in commands:
            self.file_ops.run_command(command)

    def gropuing_single_pixel_only_pi(self,pixel):
        '''
        debug : commでexitも追加しないと、grpphaから抜けれずenter押す必要がある.
        '''
        print("----------------------------------")
        print("Grouping")
        commands = [
            f"ftgrouppha infile={self.file_ops.identifier}_PIXEL_{pixel}.pi outfile={self.file_ops.identifier}_PIXEL_{pixel}_b1.pi grouptype=min groupscale=1 clobber=yes",
            " "
        ]       
        for command in commands:    
            self.file_ops.run_command(command)    

    def group_diag_resp(self, file):
        self.file_ops.run_heainit()
        print("----------------------------------")
        print("Grouping")
        command = f"""
            source /Users/keitatanaka/Init/HEASOFT.sh && ftgrouppha infile={file} outfile={os.path.splitext(os.path.basename(file))[0]}_merged_b1.pi grouptype=optmin groupscale=1 respfile={self.file_ops.respdir}/newdiag.rmf clobber=yes
        """
        
        self.file_ops.run_command(command)

    def process_events(self):
        self.rise_time_screening()
        # f = FitsEdit()
        # file_path = f'{self.file_ops.outfiledir}/{self.file_ops.input_file}'
        # f.exclude_gti(file_path,153171242,153178242)
        # f.divide_gti(file_path,153175083,recycle)
        #self.exclude_TIME(153271679,153407679)
        #self.exclude_TIME(15324678,15394678) # for 55Fe(5000) MXS BRIGHT MODE TIME
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_center_region()
        self.select_outer_region()
        self.select_all_region()
        self.make_rmf_center_outer()
        self.make_rmf_all()
        self.make_exposure_map()
        self.make_arf_center_outer()
        self.make_arf_all()
        self.grouping_all()
        #self.group_diag_resp(file=f'5000_all.pi')
        self.grouping_center_outer()

    def process_events_template(self):
        self.rise_time_screening()
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_center_region()
        self.select_outer_region()
        self.select_all_region()

    def process_events_open(self,recycle='after'):
        self.rise_time_screening()
        f = FitsEdit()
        file_path = f'{self.file_ops.outfiledir}/{self.file_ops.input_file}'
        f.exclude_gti(file_path,153171242,153178242)
        f.divide_gti(file_path,153175083,recycle)
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_center_region()
        self.select_outer_region()
        self.select_all_region()
        self.make_rmf_center_outer()
        self.make_rmf_all()
        self.make_exposure_map()
        self.make_arf_center_outer()
        self.make_arf_all()
        self.grouping_center_outer()
        self.grouping_all()

    def process_events_center_outer(self):
        # self.rise_time_screening()
        # self.make_event_file_without_Ls()
        # self.event_selection_without_calpix()
        # self.select_center_region()
        self.select_outer_region()
        self.make_rmf_center_outer()
        # self.make_exposure_map()
        self.make_arf_center_outer()
        self.grouping_center_outer()

    def process_events_center(self):
        self.rise_time_screening()
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_center_region()
        self.make_rmf_center()
        self.make_exposure_map()
        self.make_arf_center()
        self.grouping_center()

    def process_events_outer(self):
        self.rise_time_screening()
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_outer_region()
        self.make_rmf_outer()
        # self.make_exposure_map()
        self.make_arf_outer()
        self.grouping_outer()

    def process_mxs_events(self, mxs_phase_threshold):
        self.rise_time_screening()
        self.exclude_MXS_phase(mxs_phase_threshold)
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        # self.select_center_region()
        # self.select_outer_region()
        self.select_all_region()
        #self.make_rmf_center_outer_X()
        self.make_rmf_all_X()
        self.make_exposure_map()
        # self.make_arf_center_outer()
        self.make_arf_all()
        self.grouping_all()
        # #self.group_diag_resp(file=f'5000_all.pi')
        # self.grouping_center_outer()

    def process_events_all_region(self):
        self.rise_time_screening()
        # self.exclude_TIME(153171242,153178242)
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_all_region()
        self.make_rmf_all()
        self.make_exposure_map()
        self.make_arf_all()
        self.grouping_all()

    def process_events_for_55Fe(self):
        self.rise_time_screening()
        self.exclude_TIME(153271679,153407679)
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()      
        for pixel in self.ALL_PIXEL:
            self.select_single_pixel(pixel)
            self.group_diag_resp(file=f'{self.file_ops.outfiledir}/{self.file_ops.identifier}_PIXEL_{pixel}.pi')

    def process_events_for_open_occulation(self):
        self.rise_time_screening()
        self.event_selection_without_calpix()      
        for pixel in self.ALL_PIXEL:
            self.select_single_pixel(pixel)
            self.group_diag_resp(file=f'{self.file_ops.outfiledir}/{self.file_ops.identifier}_PIXEL_{pixel}.pi')

    def process_events_for_55Fe_occulation(self):
        self.rise_time_screening()
        self.exclude_TIME(153255500,153257500)
        self.exclude_TIME(153268000,153403000)
        self.event_selection_without_calpix()      
        for pixel in self.ALL_PIXEL:
            self.select_single_pixel(pixel)
            self.group_diag_resp(file=f'{self.file_ops.outfiledir}/{self.file_ops.identifier}_PIXEL_{pixel}.pi')
        
    def process_events_for_55Fe_occulation_all_pixel(self):
        self.rise_time_screening()
        # self.exclude_TIME(153255500,153257500)
        # self.exclude_TIME(153268000,153403000)
        # self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_all_region()
        # self.make_rmf_all_X()
        # self.make_exposure_map()
        # self.make_arf_all()
        # self.grouping_all_X()

    def process_events_for_55Fe_occulation_center_outer(self):
        self.rise_time_screening()
        # self.exclude_TIME(153255500,153257500)
        # self.exclude_TIME(153268000,153403000)
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.select_center_region()
        self.select_outer_region()
        # self.select_all_region()
        # self.make_rmf_center_outer()
        # self.make_rmf_all_X()
        self.make_rmf_center_outer_X()
        # self.make_exposure_map()
        # self.make_arf_center_outer()
        # self.make_arf_all()
        # self.grouping_all()
        # self.group_diag_resp(file=f'5000_all.pi')
        self.grouping_center_outer_X()

    def process_bin(self):
        for pixel in self.ALL_PIXEL:
            self.gropuing_single_pixel(pixel)

    def pixel_by_pixel(self,make_resp=True):
        self.rise_time_screening()
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        self.make_exposure_map()
        for pixel in self.center_pixel:
            self.select_single_pixel(pixel)
            if make_resp == True:
                self.make_rmf_single_pixel(pixel)
                self.make_arf_single_pixel(pixel)
        if make_resp == True:
            for pixel in self.center_pixel:
                self.gropuing_single_pixel(pixel)

        else:
            for pixel in self.ALL_PIXEL:
                self.gropuing_single_pixel_only_pi(pixel)

    def pixel_by_pixel_for_cal(self,t):
        self.rise_time_screening()
        self.exclude_MXS_phase_inphase(t)
        self.make_event_file_without_Ls()
        self.event_selection_without_calpix()
        for pixel in self.ALL_PIXEL:
            self.select_single_pixel(pixel)
            self.gropuing_single_pixel_only_pi(pixel)


class MakeRegion:

    def __init__(self):
        pass

    def make_region(self):
        print("----------------------------------")
        print("Make Region")
        commands = f"""
        ds9 {self.file_ops.outfiledir}/{self.file_ops.input_file} -regions load {self.file_ops.respdir}/center_det.reg
"""
        self.file_ops.run_command(commands)


class FitsEdit:

    def __init__(self):
        pass

    def edit_exposure_time(self, file_path, new_exposure_time):
        print("----------------------------------")
        print("Edit Exposure Time")


        # FITSファイルを読み込み
        with fits.open(file_path, mode='update') as hdul:
            # ヘッダーのEXPOSUREキーワードを更新
            hdul[0].header['EXPOSURE'] = new_exposure_time
            hdul[1].header['EXPOSURE'] = new_exposure_time
            hdul[2].header['EXPOSURE'] = new_exposure_time
            
            # 更新した内容を保存
            hdul.flush()

        print(f"EXPOSURE has been updated to {new_exposure_time} in {file_path}")


    def plot_gti(self, file_path):
        print("----------------------------------")
        print("Take out GTI")
        with fits.open(file_path, mode='update') as hdul:
            # GTIデータを取得
            gti = hdul['GTI'].data
            start = gti['START']
            stop = gti['STOP']
            
            # グラフの描画
            for i in range(len(start)):
                plt.axvspan(start[i], stop[i], color='gray', alpha=0.5)

            
            plt.show()


    def divide_gti(self, file_path, divide_position, option='before', overwrite=True, new_file_path=None):
        print("----------------------------------")
        print("Take out GTI")
        with fits.open(file_path, mode='update') as hdul:
            # GTIデータを取得
            gti = hdul['GTI'].data
            start = gti['START']
            stop = gti['STOP']
            
            print(f"Divide GTI at {divide_position}")
            print(f'max start : {max(start)}')
            print(f'max stop : {max(stop)}')

            # 新しいSTART, STOPを作成
            if option == 'before':
                new_start = start[start < divide_position]
                new_stop = stop[stop < divide_position]
                for i in range(len(start)):
                    if start[i] < divide_position and stop[i] > divide_position:
                        new_stop = np.append(new_stop, divide_position)
            elif option == 'after':
                new_start = start[start > divide_position]
                new_stop = stop[stop > divide_position]
                for i in range(len(start)):
                    if start[i] < divide_position and stop[i] > divide_position:
                        new_start = np.append(divide_position, new_start)
            
            # 新しいGTIデータを作成
            new_gti_data = np.array(list(zip(new_start, new_stop)), dtype=[('START', 'f8'), ('STOP', 'f8')])
            
            if overwrite == True:
            # GTI HDUを更新
                hdul['GTI'].data = new_gti_data
                hdul.flush()  # 更新をFITSファイルに書き込む
                print(f"GTI information in {file_path} has been updated.")

            else:

                if new_file_path == None:
                    new_file_path = file_path.replace('.fits', '_new.fits')
                primary_hdu = fits.PrimaryHDU()
                hdul_new = fits.HDUList([primary_hdu])
                gti_hdu = fits.BinTableHDU(new_gti_data, name='GTI')
                hdul_new.append(gti_hdu)
                hdul_new.writeto(new_file_path, overwrite=True)
                print(f"GTI information in {file_path} has been saved to {new_file_path}.")

    def exclude_gti(self, file_path, exclude_start, exclude_stop):
        print("----------------------------------")
        print("Take out GTI")
        
        with fits.open(file_path, mode='update') as hdul:
            # GTIデータを取得
            gti = hdul['GTI'].data
            print(gti)
            start = gti['START']
            stop = gti['STOP']
            
            # グラフの描画
            for i in range(len(start)):
                plt.axvspan(start[i], stop[i], color='gray', alpha=0.5)

            # 新しいSTART, STOPを作成
            new_start = []
            new_stop = []

            for i in range(len(start)):
                current_start = start[i]
                current_stop = stop[i]
                
                # `exclude_start` と `exclude_stop` がGTI範囲の一部と重なる場合
                if current_start < exclude_stop and current_stop > exclude_start:
                    # 1. 現在の範囲の開始時間が exclude_start より前で、終了時間が exclude_stop より後
                    if current_start < exclude_start and current_stop > exclude_stop:
                        new_start.append(current_start)
                        new_stop.append(exclude_start)  # exclude_start までの範囲
                        new_start.append(exclude_stop)
                        new_stop.append(current_stop)   # exclude_stop 以降の範囲

                    # 2. 現在の範囲の開始時間が exclude_start より前で、終了時間が exclude_stop の前
                    elif current_start < exclude_start and current_stop <= exclude_stop:
                        new_start.append(current_start)
                        new_stop.append(exclude_start)  # exclude_start までの範囲

                    # 3. 現在の範囲の開始時間が exclude_start 以降で、終了時間が exclude_stop より後
                    elif current_start >= exclude_start and current_stop > exclude_stop:
                        new_start.append(exclude_stop)
                        new_stop.append(current_stop)   # exclude_stop 以降の範囲

                    # 4. 現在の範囲の開始時間が exclude_start より前で、終了時間が exclude_start 以降
                    elif current_start < exclude_start and current_stop > exclude_start and current_stop <= exclude_stop:
                        new_start.append(current_start)
                        new_stop.append(exclude_start)  # exclude_start までの範囲

                    # 5. 現在の範囲が `exclude_start` と `exclude_stop` の範囲内に完全に包含されている場合、その範囲を除外
                    elif current_start >= exclude_start and current_stop <= exclude_stop:
                        # この範囲は除外されるため、何も追加しない
                        continue
                else:
                    # 重なっていない場合はそのまま新しいGTIに追加
                    new_start.append(current_start)
                    new_stop.append(current_stop)
            
            # 新しいSTART, STOPを表示
            for i in range(len(new_start)):
                print(f"NEW START: {new_start[i]}, NEW STOP: {new_stop[i]}")
                plt.axvspan(new_start[i], new_stop[i], color='blue', alpha=0.5)
            
            plt.show(block=False)

            # 新しいGTIデータを作成
            new_gti_data = np.array(list(zip(new_start, new_stop)), dtype=[('START', 'f8'), ('STOP', 'f8')])

            # GTI HDUを更新
            hdul['GTI'].data = new_gti_data
            hdul.flush()  # 更新をFITSファイルに書き込む
            
        print(f"GTI information in {file_path} has been updated.")

    def intersect_gti_with_time_range(self, file_path, time_range1_start, time_range1_stop):
        print("----------------------------------")
        print("Extract GTI intervals that intersect with the given time range and update FITS")
        
        with fits.open(file_path, mode='update') as hdul:
            # GTIデータを取得
            gti = hdul['GTI'].data
            start = gti['START']
            stop = gti['STOP']
            
            # グラフの描画
            for i in range(len(start)):
                plt.axvspan(start[i], stop[i], color='gray', alpha=0.5)

            # 新しいSTART, STOPを作成
            new_start = []
            new_stop = []

            for i in range(len(start)):
                current_start = start[i]
                current_stop = stop[i]
                
                # `time_range1` とGTI範囲の共通部分を求める
                common_start = max(current_start, time_range1_start)
                common_stop = min(current_stop, time_range1_stop)

                # もし共通部分が存在する場合、その範囲を新しいSTART, STOPに追加
                if common_start < common_stop:
                    new_start.append(common_start)
                    new_stop.append(common_stop)

            # 新しいSTART, STOPを表示
            for i in range(len(new_start)):
                print(f"NEW START: {new_start[i]}, NEW STOP: {new_stop[i]}")
                plt.axvspan(new_start[i], new_stop[i], color='blue', alpha=0.5)

            plt.show(block=False)

            # 新しいGTIデータを作成
            new_gti_data = np.array(list(zip(new_start, new_stop)), dtype=[('START', 'f8'), ('STOP', 'f8')])

            # GTI HDUを更新
            # hdul['GTI'].data = new_gti_data
            # hdul.flush()  # 更新をFITSファイルに書き込む

        print(f"GTI information in {file_path} has been updated with the common intervals.")

    def edit_gti(self):
        pass


def out_process():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        identifiers = ["2000","3000","4000"]
        # identifiers = ["1000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/raw_data/', outfiledir='/Volumes/SUNDISK_SSD/PKS_XRISM/repro/mxs_scott/center_pixel_by_pixel', identifier=identifier, respdir='/Volumes/SUNDISK_SSD/PKS_XRISM/rmfarf',repro=True,mxs_mode=True)
                processor.pixel_by_pixel()

def out_process_ssm():
    with open("logfile.txt", "w+") as f:
        # 標準出力をリダイレクト
        identifiers = ["1000","2000","3000","4000"]
        # identifiers = ["1000"]
        with redirect_stdout(f):
            for identifier in identifiers:
                processor = EventProcessing(infiledir='/Users/keitatanaka/Dropbox/SSD_backup/PKS_XRISM/repro/mxs_scott/center_outer_all/', outfiledir='/Users/keitatanaka/Dropbox/SSD_backup/PKS_XRISM/repro/mxs_scott/center_outer_all', identifier=identifier, respdir='/Users/keitatanaka/Dropbox/SSD_backup/PKS_XRISM/rmfarf',repro=True,mxs_mode=True)
                processor.make_arf_ssm_center()
                processor.make_arf_ssm_outer()