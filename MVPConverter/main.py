##########################################################################
# MVPAnalyzer/main.py
# Author: Maximilien Wemaere (LMD/CNRS)
# Date: August 2025
#
#
# Simple routines to convert mvp and CTD raw data NETCDF files
#
#
##########################################################################




import numpy as np 
import glob
from datetime import datetime
import os
import gsw
from seabird.cnv import fCNV
from tqdm import tqdm
import xarray as xr
from . import mvp_routines as mvp

class Converter:
    def __init__(self, data_path, output_path=None, subdirs=False,  Yorig=1950):
        """
        Initialize the analyzer with the data path and reference year.
        Args:
            data_path (str): Path to the folder containing MVP files.
            subdirs (bool): Whether to search in subdirectories for MVP files (default False).
            Yorig (int): Reference year for dates (default 1950).
        """
        self.Yorig = Yorig
        self.date_ref = datetime(Yorig, 1, 1)
        self.data_path = data_path
        self.output_path = output_path if output_path is not None else data_path
        self.subdirs = subdirs
        self.mvp = False
        self.ctd = False
        self.speed = False


    def load_mvp_data(self,delp=[],data_path=None,only_new=False):
        """
        Load MVP data from .raw and .log files in the data_path folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            delp (list): Indices of profiles to remove from the list (optional).
            data_path (str): Path to the folder containing MVP files (optional).
        """
        if data_path is not None:
            self.data_path = data_path

        if self.subdirs:
            files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/*.raw', recursive=True)))
        else:
            files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '*.raw', recursive=self.subdirs)))


        if only_new:
            list_output = [f for f in os.listdir(self.output_path) if f.endswith(".nc")]
            files = [f for f in files if not "MVP_"+os.path.basename(f).replace('.raw', '.nc') in list_output]




        print('Found ' + str(len(files)) + ' MVP files in the directory: ' + self.data_path)

        PRES_temp = []
        TEMP_temp = []
        COND_temp = []
        SOUNDVEL_temp = []
        DO_temp = []
        TEMP2_temp = [] # temp from DO sensor
        SUNA_temp = []
        FLUO_temp = [] 
        TURB_temp = []
        PH_temp = [] 
        SALT_temp = []
        TIME_mvp_temp = []
        LAT_temp = []
        LON_temp= []
        DATETIME_mvp = []
        DIR = []
        Label_mvp = []

        delp.sort(reverse=True)
        for i in delp:
            del files[i]

        for mvp_dat_name in files[0:]:

            mvp_log_name=mvp_dat_name[:-4]+'.log'

            # Get start and end time of the cycle

            (mvp_tstart,mvp_tend,cycle_dur, lat, lon, dt_station) = mvp.get_log(mvp_log_name,self.Yorig)

            
            if cycle_dur>1:

                # Read one cycle MVP data  
                (pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw) = mvp.read_mvp_cycle_raw(mvp_dat_name)
                (pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph) = mvp.raw_data_conversion(pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw)
                

                freq_echant = float(len(pres)/cycle_dur)

                DATETIME_mvp.append(dt_station)
                
                if np.nanmax(pres)-np.nanmin(pres)>2:

                    # Allocate time to samples and select the ascending part 
                    (pres_up,soundvel_up,cond_up,temp_up,do_up,temp2_up,suna_up,fluo_up,turb_up,ph_up,time_up) = mvp.time_mvp_cycle_up(pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph,mvp_tstart,mvp_tend)
                    (pres_down,soundvel_down,cond_down,temp_down,do_down,temp2_down,suna_down,fluo_down,turb_down,ph_down,time_down) = mvp.time_mvp_cycle_down(pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph,mvp_tstart,mvp_tend)


                    if len(pres_down)>0:
                        if np.nanmax(pres_down)-np.nanmin(pres_down)>2:
                            PRES_temp.append(pres_down)
                            SOUNDVEL_temp.append(soundvel_down)
                            COND_temp.append(cond_down)
                            TEMP_temp.append(temp_down)
                            DO_temp.append(do_down)
                            TEMP2_temp.append(temp2_down)
                            SUNA_temp.append(suna_down)
                            FLUO_temp.append(fluo_down)
                            TURB_temp.append(turb_down)
                            PH_temp.append(ph_down)
                            SALT_temp.append(gsw.SP_from_C(cond_down, temp_down,pres_down))
                            TIME_mvp_temp.append(time_down)
                            LAT_temp.append(lat)
                            LON_temp.append(lon)

                            DIR.append('down')
                            Label_mvp.append(mvp_dat_name.replace('\\','/').split('/')[-2])

                    else:
                        print('ohohoh no down profile found for file: ' + mvp_dat_name)

                            
                    if len(pres_up)>0:
                        if np.nanmax(pres_up)-np.nanmin(pres_up)>2:
                            PRES_temp.append(pres_up)
                            SOUNDVEL_temp.append(soundvel_up)
                            COND_temp.append(cond_up)
                            TEMP_temp.append(temp_up)
                            DO_temp.append(do_up)
                            TEMP2_temp.append(temp2_up)
                            SUNA_temp.append(suna_up)
                            FLUO_temp.append(fluo_up)
                            TURB_temp.append(turb_up)
                            PH_temp.append(ph_up)
                            SALT_temp.append(gsw.SP_from_C(cond_up, temp_up,pres_up))
                            TIME_mvp_temp.append(time_up)
                            LAT_temp.append(lat)
                            LON_temp.append(lon)

                            DIR.append('up')
                            Label_mvp.append(mvp_dat_name.replace('\\','/').split('/')[-2])

                    else:
                        print('ohohoh no up profile found for file: ' + mvp_dat_name)

                else:
                    print('ohohoh no profile found for file: ' + mvp_dat_name)

                    
                    

        # Re-arange files into matrices
        M_size = 0
        for i in range(len(PRES_temp)):
            M_size = max(M_size, len(PRES_temp[i]))
            
        PRES_mvp = np.zeros(( len(PRES_temp), M_size))
        SOUNDVEL_mvp = np.zeros(( len(PRES_temp), M_size))
        COND_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp = np.zeros(( len(PRES_temp), M_size))
        DO_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp2 = np.zeros(( len(PRES_temp), M_size))
        SUNA_mvp = np.zeros(( len(PRES_temp), M_size))
        FLUO_mvp = np.zeros(( len(PRES_temp), M_size))
        TURB_mvp = np.zeros(( len(PRES_temp), M_size))
        PH_mvp = np.zeros(( len(PRES_temp), M_size))
        SALT_mvp = np.zeros(( len(PRES_temp), M_size))
        TIME_mvp = np.zeros(( len(PRES_temp), M_size))
        LAT_mvp = np.zeros(( len(PRES_temp), M_size))
        LON_mvp = np.zeros(( len(PRES_temp), M_size))
        PRES_mvp[:] = np.nan
        SOUNDVEL_mvp[:] = np.nan
        COND_mvp[:] = np.nan
        TEMP_mvp[:] = np.nan
        DO_mvp[:] = np.nan
        TEMP_mvp2[:] = np.nan
        SUNA_mvp[:] = np.nan
        FLUO_mvp[:] = np.nan
        TURB_mvp[:] = np.nan
        PH_mvp[:] = np.nan
        SALT_mvp[:] = np.nan
        TIME_mvp[:] = np.nan
        LAT_mvp[:] = np.nan
        LON_mvp[:] = np.nan

        del M_size

        for i in range(len(PRES_temp)):
            PRES_mvp[i,0:len(PRES_temp[i])] = PRES_temp[i]
            SOUNDVEL_mvp[i,0:len(SOUNDVEL_temp[i])] = SOUNDVEL_temp[i]
            COND_mvp[i,0:len(COND_temp[i])] = COND_temp[i]
            TEMP_mvp[i,0:len(TEMP_temp[i])] = TEMP_temp[i]
            DO_mvp[i,0:len(DO_temp[i])] = DO_temp[i]
            TEMP_mvp2[i,0:len(TEMP2_temp[i])] = TEMP2_temp[i]
            SUNA_mvp[i,0:len(SUNA_temp[i])] = SUNA_temp[i]
            FLUO_mvp[i,0:len(FLUO_temp[i])] = FLUO_temp[i]
            TURB_mvp[i,0:len(TURB_temp[i])] = TURB_temp[i]
            PH_mvp[i,0:len(PH_temp[i])] = PH_temp[i]
            SALT_mvp[i,0:len(SALT_temp[i])] = SALT_temp[i]
            TIME_mvp[i,0:len(TIME_mvp_temp[i])] = TIME_mvp_temp[i]
            LAT_mvp[i,0:len(PRES_temp[i])] = LAT_temp[i]
            LON_mvp[i,0:len(PRES_temp[i])] = LON_temp[i]

         
        self.PRES_mvp = PRES_mvp
        self.SOUNDVEL_mvp = SOUNDVEL_mvp
        self.COND_mvp = COND_mvp
        self.TEMP_mvp = TEMP_mvp
        self.DO_mvp = DO_mvp
        self.TEMP2_mvp = TEMP_mvp2
        self.SUNA_mvp = SUNA_mvp
        self.FLUO_mvp = FLUO_mvp
        self.TURB_mvp = TURB_mvp
        self.PH_mvp = PH_mvp
        self.SALT_mvp = SALT_mvp
        self.TIME_mvp = TIME_mvp
        self.LAT_mvp = LAT_mvp
        self.LON_mvp = LON_mvp
        self.DATETIME_mvp = DATETIME_mvp
        self.DIR = DIR
        self.label_mvp = Label_mvp
        self.freq_echant = freq_echant
    
        del PRES_temp, SOUNDVEL_temp, DO_temp, TEMP2_temp, SUNA_temp, FLUO_temp, TURB_temp, PH_temp, COND_temp, TEMP_temp, SALT_temp, TIME_mvp_temp, LAT_temp, LON_temp        

        print('MVP data loaded successfully.')
        self.mvp = True



    def load_mvp_data_again(self,data_path,delp=[]):
        """
        Load MVP data from .raw and .log files in the data_path folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path (str): Path to the folder containing MVP files.
            delp (list): Indices of profiles to remove from the list (optional).
        """
        if data_path is not None:
            self.data_path = data_path


        files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/*.raw', recursive=True)))
        print('Found ' + str(len(files)) + ' MVP files in the directory: ' + self.data_path)

        PRES_temp = []
        TEMP_temp = []
        COND_temp = []
        SOUNDVEL_temp = []
        DO_temp = []
        TEMP2_temp = [] # temp from DO sensor
        SUNA_temp = []
        FLUO_temp = [] 
        TURB_temp = []
        PH_temp = [] 
        SALT_temp = []
        TIME_mvp_temp = []
        LAT_temp = []
        LON_temp= []
        DATETIME_mvp = []
        DIR = []
        Label_mvp = []

        delp.sort(reverse=True)
        for i in delp:
            del files[i]

        for mvp_dat_name in files[0:]:

            mvp_log_name=mvp_dat_name[:-4]+'.log'

            # Get start and end time of the cycle

            (mvp_tstart,mvp_tend,cycle_dur, lat, lon, dt_station) = mvp.get_log(mvp_log_name,self.Yorig)

            
            if cycle_dur>1:

                # Read one cycle MVP data  
                (pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw) = mvp.read_mvp_cycle_raw(mvp_dat_name)
                (pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph) = mvp.raw_data_conversion(pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw)


                freq_echant = float(len(pres)/cycle_dur)

                DATETIME_mvp.append(dt_station)
                
                if np.nanmax(pres)-np.nanmin(pres)>2:

                    # Allocate time to samples and select the ascending part 
                    (pres_up,soundvel_up,cond_up,temp_up,do_up,temp2_up,suna_up,fluo_up,turb_up,ph_up,time_up) = mvp.time_mvp_cycle_up(pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph,mvp_tstart,mvp_tend)
                    (pres_down,soundvel_down,cond_down,temp_down,do_down,temp2_down,suna_down,fluo_down,turb_down,ph_down,time_down) = mvp.time_mvp_cycle_down(pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph,mvp_tstart,mvp_tend)


                    if len(pres_down)>0:
                        if np.nanmax(pres_down)-np.nanmin(pres_down)>2:
                            PRES_temp.append(pres_down)
                            SOUNDVEL_temp.append(soundvel_down)
                            COND_temp.append(cond_down)
                            TEMP_temp.append(temp_down)
                            DO_temp.append(do_down)
                            TEMP2_temp.append(temp2_down)
                            SUNA_temp.append(suna_down)
                            FLUO_temp.append(fluo_down)
                            TURB_temp.append(turb_down)
                            PH_temp.append(ph_down)
                            SALT_temp.append(gsw.SP_from_C(cond_down, temp_down,pres_down))
                            TIME_mvp_temp.append(time_down)
                            LAT_temp.append(lat)
                            LON_temp.append(lon)

                            DIR.append('down')
                            Label_mvp.append(mvp_dat_name.replace('\\','/').split('/')[-2])

                    else:
                        print('ohohoh no down profile found for file: ' + mvp_dat_name)

                            
                    if len(pres_up)>0:
                        if np.nanmax(pres_up)-np.nanmin(pres_up)>2:
                            PRES_temp.append(pres_up)
                            SOUNDVEL_temp.append(soundvel_up)
                            COND_temp.append(cond_up)
                            TEMP_temp.append(temp_up)
                            DO_temp.append(do_up)
                            TEMP2_temp.append(temp2_up)
                            SUNA_temp.append(suna_up)
                            FLUO_temp.append(fluo_up)
                            TURB_temp.append(turb_up)
                            PH_temp.append(ph_up)
                            SALT_temp.append(gsw.SP_from_C(cond_up, temp_up,pres_up))
                            TIME_mvp_temp.append(time_up)
                            LAT_temp.append(lat)
                            LON_temp.append(lon)

                            DIR.append('up')
                            Label_mvp.append(mvp_dat_name.replace('\\','/').split('/')[-2])


                    else:
                        print('ohohoh no up profile found for file: ' + mvp_dat_name)

                else:
                    print('ohohoh no profile found for file: ' + mvp_dat_name)

                    
                    

        # Re-arange files into matrices
        M_size = 0
        for i in range(len(PRES_temp)):
            M_size = max(M_size, len(PRES_temp[i]))
            
        if M_size < self.PRES_mvp.shape[1]:
            M_size = self.PRES_mvp.shape[1]
        else:
            nan_cols = np.full((self.PRES_mvp.shape[0], M_size - self.PRES_mvp.shape[1]), np.nan)
            self.PRES_mvp = np.hstack((self.PRES_mvp, nan_cols))
            self.SOUNDVEL_mvp = np.hstack((self.SOUNDVEL_mvp, nan_cols))
            self.COND_mvp = np.hstack((self.COND_mvp, nan_cols))
            self.TEMP_mvp = np.hstack((self.TEMP_mvp, nan_cols))
            self.DO_mvp = np.hstack((self.DO_mvp, nan_cols))
            self.TEMP2_mvp = np.hstack((self.TEMP2_mvp, nan_cols))
            self.SUNA_mvp = np.hstack((self.SUNA_mvp, nan_cols))
            self.FLUO_mvp = np.hstack((self.FLUO_mvp, nan_cols))
            self.TURB_mvp = np.hstack((self.TURB_mvp, nan_cols))
            self.PH_mvp = np.hstack((self.PH_mvp, nan_cols))
            self.SALT_mvp = np.hstack((self.SALT_mvp, nan_cols))
            self.TIME_mvp = np.hstack((self.TIME_mvp, nan_cols))
            self.LAT_mvp = np.hstack((self.LAT_mvp, nan_cols))
            self.LON_mvp = np.hstack((self.LON_mvp, nan_cols))




        PRES_mvp = np.zeros(( len(PRES_temp), M_size))
        SOUNDVEL_mvp = np.zeros(( len(PRES_temp), M_size))
        COND_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp = np.zeros(( len(PRES_temp), M_size))
        DO_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp2 = np.zeros(( len(PRES_temp), M_size))
        SUNA_mvp = np.zeros(( len(PRES_temp), M_size))
        FLUO_mvp = np.zeros(( len(PRES_temp), M_size))
        TURB_mvp = np.zeros(( len(PRES_temp), M_size))
        PH_mvp = np.zeros(( len(PRES_temp), M_size))
        SALT_mvp = np.zeros(( len(PRES_temp), M_size))
        TIME_mvp = np.zeros(( len(PRES_temp), M_size))
        LAT_mvp = np.zeros(( len(PRES_temp), M_size))
        LON_mvp = np.zeros(( len(PRES_temp), M_size))
        PRES_mvp[:] = np.nan
        SOUNDVEL_mvp[:] = np.nan
        COND_mvp[:] = np.nan
        TEMP_mvp[:] = np.nan
        DO_mvp[:] = np.nan
        TEMP_mvp2[:] = np.nan
        SUNA_mvp[:] = np.nan
        FLUO_mvp[:] = np.nan
        TURB_mvp[:] = np.nan
        PH_mvp[:] = np.nan
        SALT_mvp[:] = np.nan
        TIME_mvp[:] = np.nan
        LAT_mvp[:] = np.nan
        LON_mvp[:] = np.nan

        del M_size

        for i in range(len(PRES_temp)):
            PRES_mvp[i,0:len(PRES_temp[i])] = PRES_temp[i]
            SOUNDVEL_mvp[i,0:len(SOUNDVEL_temp[i])] = SOUNDVEL_temp[i]
            COND_mvp[i,0:len(COND_temp[i])] = COND_temp[i]
            TEMP_mvp[i,0:len(TEMP_temp[i])] = TEMP_temp[i]
            DO_mvp[i,0:len(DO_temp[i])] = DO_temp[i]
            TEMP_mvp2[i,0:len(TEMP2_temp[i])] = TEMP2_temp[i]
            SUNA_mvp[i,0:len(SUNA_temp[i])] = SUNA_temp[i]
            FLUO_mvp[i,0:len(FLUO_temp[i])] = FLUO_temp[i]
            TURB_mvp[i,0:len(TURB_temp[i])] = TURB_temp[i]
            PH_mvp[i,0:len(PH_temp[i])] = PH_temp[i]
            SALT_mvp[i,0:len(SALT_temp[i])] = SALT_temp[i]
            TIME_mvp[i,0:len(TIME_mvp_temp[i])] = TIME_mvp_temp[i]
            LAT_mvp[i,0:len(PRES_temp[i])] = LAT_temp[i]
            LON_mvp[i,0:len(PRES_temp[i])] = LON_temp[i]


        self.PRES_mvp = np.concatenate((self.PRES_mvp, PRES_mvp), axis=0)
        self.SOUNDVEL_mvp = np.concatenate((self.SOUNDVEL_mvp, SOUNDVEL_mvp), axis=0)
        self.COND_mvp = np.concatenate((self.COND_mvp, COND_mvp), axis=0)
        self.TEMP_mvp = np.concatenate((self.TEMP_mvp, TEMP_mvp), axis=0)
        self.DO_mvp = np.concatenate((self.DO_mvp, DO_mvp), axis=0)
        self.TEMP2_mvp = np.concatenate((self.TEMP2_mvp, TEMP_mvp2), axis=0)
        self.SUNA_mvp = np.concatenate((self.SUNA_mvp, SUNA_mvp), axis=0)
        self.FLUO_mvp = np.concatenate((self.FLUO_mvp, FLUO_mvp), axis=0)
        self.TURB_mvp = np.concatenate((self.TURB_mvp, TURB_mvp), axis=0)
        self.PH_mvp = np.concatenate((self.PH_mvp, PH_mvp), axis=0)
        self.SALT_mvp = np.concatenate((self.SALT_mvp, SALT_mvp), axis=0)
        self.TIME_mvp = np.concatenate((self.TIME_mvp, TIME_mvp), axis=0)
        self.LAT_mvp = np.concatenate((self.LAT_mvp, LAT_mvp), axis=0)
        self.LON_mvp = np.concatenate((self.LON_mvp, LON_mvp), axis=0)

        self.DATETIME_mvp.extend(DATETIME_mvp)
        self.DIR.extend(DIR)
        self.label_mvp.extend(Label_mvp)
    
        del PRES_temp, SOUNDVEL_temp, DO_temp, TEMP2_temp, SUNA_temp, FLUO_temp, TURB_temp, PH_temp, COND_temp, TEMP_temp, SALT_temp, TIME_mvp_temp, LAT_temp, LON_temp        

        print('MVP data loaded successfully.')
        self.mvp = True


    def load_ctd_data(self,data_path_ctd):
        """
        Load CTD data from .cnv files in the data_path_ctd folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path_ctd (str): Path to the folder containing CTD files.
        """



        list_of_ctd_files = sorted(filter(os.path.isfile,\
                           glob.glob(data_path_ctd + '*.cnv')))

        print('Found ' + str(len(list_of_ctd_files)) + ' CTD files in the directory: ' + data_path_ctd)

        # keys: ['scan', 'timeJ', 'timeQ', 'LATITUDE', 'LONGITUDE', 'PRES', 'TEMP', 'CNDC', 'descentrate', 'flECO-AFL', 'v1', 'wetCDOM', 'v0', 'turbWETntu0', 'v5', 'CStarTr0', 'CStarAt0', 'oxygen_ml_L', 'oxsolML/L', 'v2', 'flag', 'timeS']
        LAT_ctd_temp = []
        LON_ctd_temp = []
        PRES_ctd_temp = []
        TEMP_ctd_temp = []
        COND_ctd_temp = []
        TURB_ctd_temp = []
        OXY_ctd_temp = []
        FLUO_ctd_temp = []
        CDOM_ctd_temp = []
        DATETIME_ctd = []
        SALT_ctd_temp = []

        for ctd_dat_name in tqdm(list_of_ctd_files[0:]):
            ctd_files = ctd_dat_name
            cnv = fCNV(ctd_files)

            Lat_up,Lat_down = split_ctd(cnv['PRES'], cnv['LATITUDE'])
            Lon_up,Lon_down = split_ctd(cnv['PRES'], cnv['LONGITUDE'])
            Pres_up,Pres_down = split_ctd(cnv['PRES'], cnv['PRES'])
            Temp_up,Temp_down = split_ctd(cnv['PRES'], cnv['TEMP'])
            Cond_up,Cond_down = split_ctd(cnv['PRES'], cnv['CNDC']*10)
            Turb_up,Turb_down = split_ctd(cnv['PRES'], cnv['turbWETntu0'])
            Oxy_up,Oxy_down = split_ctd(cnv['PRES'],np.array([a/b*100 for a,b in zip(cnv['oxygen_ml_L'], cnv['oxsolML/L'])]))
            Fluo_up,Fluo_down = split_ctd(cnv['PRES'], cnv['flECO-AFL'])
            Cdom_up,Cdom_down = split_ctd(cnv['PRES'], cnv['wetCDOM'])
            Salt_up,Salt_down = split_ctd(cnv['PRES'], gsw.SP_from_C(cnv['CNDC']*10, cnv['TEMP'], cnv['PRES']))





            LAT_ctd_temp.append(Lat_down)
            LAT_ctd_temp.append(Lat_up)
            LON_ctd_temp.append(Lon_down)
            LON_ctd_temp.append(Lon_up)
            PRES_ctd_temp.append(Pres_down)
            PRES_ctd_temp.append(Pres_up)
            TEMP_ctd_temp.append(Temp_down)
            TEMP_ctd_temp.append(Temp_up)
            COND_ctd_temp.append(Cond_down)
            COND_ctd_temp.append(Cond_up)
            TURB_ctd_temp.append(Turb_down)
            TURB_ctd_temp.append(Turb_up)
            OXY_ctd_temp.append(Oxy_down)
            OXY_ctd_temp.append(Oxy_up)
            FLUO_ctd_temp.append(Fluo_down)
            FLUO_ctd_temp.append(Fluo_up)
            CDOM_ctd_temp.append(Cdom_down)
            CDOM_ctd_temp.append(Cdom_up)
            SALT_ctd_temp.append(Salt_down)
            SALT_ctd_temp.append(Salt_up)




            with open(ctd_dat_name, 'r') as f:
                header_lines = []
                for _ in range(10): 
                    header_lines.append(f.readline().strip())

            line = header_lines[9]
            date_str = line.split('=')[1].strip()
            dt = datetime.strptime(date_str, "%b %d %Y %H:%M:%S")
            DATETIME_ctd.append(dt)

        # Re-arange files into matrices
        M_size = 0
        for i in range(len(PRES_ctd_temp)):
            M_size = max(M_size, len(PRES_ctd_temp[i]))
            
        PRES_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        COND_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        SALT_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        TEMP_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        TURB_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        OXY_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        FLUO_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        CDOM_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        LAT_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        LON_ctd = np.zeros(( len(PRES_ctd_temp), M_size))
        PRES_ctd[:] = np.nan
        COND_ctd[:] = np.nan
        SALT_ctd[:] = np.nan    
        TEMP_ctd[:] = np.nan
        TURB_ctd[:] = np.nan
        OXY_ctd[:] = np.nan
        FLUO_ctd[:] = np.nan
        CDOM_ctd[:] = np.nan
        LAT_ctd[:] = np.nan
        LON_ctd[:] = np.nan
        del M_size
        for i in range(len(PRES_ctd_temp)):
            LAT_ctd[i,0:len(PRES_ctd_temp[i])] = LAT_ctd_temp[i]
            LON_ctd[i,0:len(PRES_ctd_temp[i])] = LON_ctd_temp[i]
            PRES_ctd[i,0:len(PRES_ctd_temp[i])] = PRES_ctd_temp[i]
            TEMP_ctd[i,0:len(PRES_ctd_temp[i])] = TEMP_ctd_temp[i]
            COND_ctd[i,0:len(PRES_ctd_temp[i])] = COND_ctd_temp[i]
            SALT_ctd[i,0:len(PRES_ctd_temp[i])] = SALT_ctd_temp[i]
            TURB_ctd[i,0:len(PRES_ctd_temp[i])] = TURB_ctd_temp[i]
            OXY_ctd[i,0:len(PRES_ctd_temp[i])] = OXY_ctd_temp[i]   
            FLUO_ctd[i,0:len(PRES_ctd_temp[i])] = FLUO_ctd_temp[i]
            CDOM_ctd[i,0:len(PRES_ctd_temp[i])] = CDOM_ctd_temp[i]
        del PRES_ctd_temp, TEMP_ctd_temp, COND_ctd_temp, SALT_ctd_temp, TURB_ctd_temp, OXY_ctd_temp, FLUO_ctd_temp, CDOM_ctd_temp, LAT_ctd_temp, LON_ctd_temp

        self.PRES_ctd = PRES_ctd
        self.TEMP_ctd = TEMP_ctd
        self.COND_ctd = COND_ctd
        self.SALT_ctd = SALT_ctd
        self.TURB_ctd = TURB_ctd
        self.OXY_ctd = OXY_ctd
        self.FLUO_ctd = FLUO_ctd
        self.CDOM_ctd = CDOM_ctd
        self.LAT_ctd = LAT_ctd
        self.LON_ctd = LON_ctd
        self.DATETIME_ctd = DATETIME_ctd

        print('CTD data loaded successfully.')
        self.ctd = True


    def compute_waterflow(self,horizontal_speed):
        """
        Compute the water flow speed (u,v) from the horizontal speed and the direction of the profiles.
        Args:
            horizontal_speed (float): Horizontal speed of the boat in cm/s.
        """
        

        SPEED_MVP = np.zeros((self.PRES_mvp.shape[0], self.PRES_mvp.shape[1]))
        for i in range(self.PRES_mvp.shape[0]):
            SPEED_MVP[i,:] = np.sqrt(np.gradient(self.PRES_mvp[i,:], 1/self.freq_echant)**2+ horizontal_speed**2)

        self.SPEED_mvp = SPEED_MVP
        self.speed = True
        print('Water flow speed computed successfully.')

    def print_profile_metadata(self):
        """
        Print main metadata (date, position, number of samples) for each loaded MVP and CTD profile.
        """

        if self.mvp:
            print('MVP data:')
            print('Number of profiles: ' + str(len(self.DATETIME_mvp)))
            for i in range(0,len(self.DATETIME_mvp)):
                print(f"  Profil down {2*i} - Profil up {2*i+1} - Latitude: {self.LAT_mvp[2*i,0]:.5f}, Longitude: {self.LON_mvp[2*i,0]:.5f}, Date/Heure: {self.DATETIME_mvp[i]}")

        if self.ctd:
            print('CTD data:')
            print('Number of profiles: ' + str(len(self.DATETIME_ctd)))
            for i in range(0,len(self.DATETIME_ctd)):
                print(f"  Profil down {2*i} - Profil up {2*i+1} - Latitude: {self.LAT_ctd[2*i,0]:.5f}, Longitude: {self.LON_ctd[2*i,0]:.5f}, Date/Heure: {self.DATETIME_ctd[i]}")


    def mvp_to_netcdf(self, filepath=None, compression=True, engine=None, per_profile_files=False):
        """
        Export MVP data to a NetCDF file using xarray.

        Args:
            filepath (str): Output NetCDF file path.
            corrected (bool): Also write corrected arrays if present (*_mvp_corr). Default False.
            compression (bool): Enable compression (engine dependent). Default True.
            engine (str|None): One of 'netcdf4', 'h5netcdf', 'scipy'. If None, choose netcdf4.
            per_profile_files (bool): If True, write one .nc per MVP cycle (two rows: down and up).
        """
        if not getattr(self, 'mvp', False):
            raise RuntimeError("No MVP data loaded. Call load_mvp_data() first.")

        engine = 'netcdf4' if engine is None else engine
        if engine == 'scipy' and compression:
            print('Warning: scipy backend does not support compression; writing without compression.')
            compression = False

        # Dimensions
        n_prof, n_samp = self.PRES_mvp.shape

        # Coordinates
        profile_idx = np.arange(n_prof, dtype=np.int32)
        sample_idx = np.arange(n_samp, dtype=np.int32)

        # Direction per profile (down/up)
        direction = None
        if hasattr(self, 'DIR') and len(self.DIR) == n_prof:
            direction = np.array(self.DIR, dtype=object)
        else:
            # Fallback based on even/odd
            direction = np.array(['down' if i % 2 == 0 else 'up' for i in range(n_prof)], dtype=object)

        # Per-sample time as seconds since reference origin
        # TIME_mvp is in days relative to self.date_ref
        time_seconds = None
        if hasattr(self, 'TIME_mvp'):
            time_seconds = self.TIME_mvp * 24.0 * 3600.0
        else:
            time_seconds = np.full((n_prof, n_samp), np.nan)

        # Per-profile datetime (one timestamp per cast pair); map using i//2
        profile_time = None
        if hasattr(self, 'DATETIME_mvp') and len(getattr(self, 'DATETIME_mvp', [])) > 0:
            prof_times = []
            for i in range(n_prof):
                j = i // 2
                if j < len(self.DATETIME_mvp) and self.DATETIME_mvp[j] is not None:
                    prof_times.append(np.datetime64(self.DATETIME_mvp[j]))
                else:
                    prof_times.append(np.datetime64('NaT'))
            profile_time = np.array(prof_times, dtype='datetime64[ns]')
        else:
            profile_time = np.array([np.datetime64('NaT')] * n_prof, dtype='datetime64[ns]')

        # Build dataset variables safely
        data_vars = {}

        def add_var(var_name, arr, units=None, long_name=None):
            if arr is None:
                return
            data_vars[var_name] = (
                ('profile', 'sample'), arr,
                {k: v for k, v in [('units', units), ('long_name', long_name)] if v is not None}
            )
        
        add_var('PRES', getattr(self, 'PRES_mvp', None), units='dbar', long_name='Sea water pressure')
        add_var('TEMP', getattr(self, 'TEMP_mvp', None), units='degC', long_name='In-situ temperature')
        add_var('COND', getattr(self, 'COND_mvp', None), units='mS/cm', long_name='Conductivity')
        add_var('SAL', getattr(self, 'SALT_mvp', None), units='psu', long_name='Practical salinity')
        add_var('SOUNDVEL', getattr(self, 'SOUNDVEL_mvp', None), units='m s-1', long_name='Sound speed')
        add_var('DO', getattr(self, 'DO_mvp', None), units='ml/L', long_name='Dissolved oxygen')
        add_var('TEMP2', getattr(self, 'TEMP2_mvp', None), units='degC', long_name='Oxygen sensor temperature')
        add_var('SUNA', getattr(self, 'SUNA_mvp', None), long_name='SUNA raw/derived')
        add_var('FLUO', getattr(self, 'FLUO_mvp', None), units='ug/L', long_name='Chl fluorescence')
        add_var('TURB', getattr(self, 'TURB_mvp', None), units='NTU', long_name='Turbidity')
        add_var('PH', getattr(self, 'PH_mvp', None), units='1', long_name='pH')
        add_var('TIME_s', getattr(self, 'TIME_mvp', None), units='secondes', long_name='Time relative to beginning')
        if self.speed:
            add_var('SPEED', getattr(self, 'SPEED_mvp', None), units='cm/s', long_name='Water flow speed')

        # Position and time arrays (2D)
        if hasattr(self, 'LAT_mvp'):
            add_var('LATITUDE', self.LAT_mvp, units='degrees_north', long_name='Latitude at sample')
        if hasattr(self, 'LON_mvp'):
            add_var('LONGITUDE', self.LON_mvp, units='degrees_east', long_name='Longitude at sample')
        # Time seconds since reference
        data_vars['TIME'] = (
            ('profile', 'sample'), time_seconds,
            {
                'units': f'seconds since {self.date_ref.strftime("%Y-%m-%d %H:%M:%S")}',
                'long_name': 'Time at sample'
            }
        )

        
        # Coordinates and auxiliary per-profile variables
        coords = {
            'profile': ('profile', profile_idx),
            'sample': ('sample', sample_idx)
        }

        # Encode direction/time according to engine capabilities
        if engine in ('netcdf4', 'h5netcdf'):
            coords['direction'] = ('profile', direction.astype('U'), {'long_name': 'Profile direction'})
            coords['profile_time'] = ('profile', profile_time, {'long_name': 'Profile nominal time'})
            coords['label'] = ('profile', np.array(self.label_mvp, dtype='U'), {'long_name': 'Profile label/source file'})
        else:
            # scipy backend: avoid object strings and datetime; use numeric fallbacks
            dir_flag = np.where(direction.astype('U') == 'down', 0, 1).astype('int8')
            coords['direction_flag'] = (
                'profile', dir_flag, {'long_name': 'Profile direction (0=down,1=up)'}
            )
            ref = np.datetime64(self.date_ref)
            pt = profile_time.astype('datetime64[s]')
            mask = (pt == np.datetime64('NaT'))
            secs = (pt - ref).astype('timedelta64[s]').astype('float64')
            secs[mask] = np.nan
            coords['profile_time_sec'] = (
                'profile', secs,
                {'units': f'seconds since {self.date_ref.strftime("%Y-%m-%d %H:%M:%S")}',
                 'long_name': 'Profile nominal time'}
            )

        # Optional per-profile lat/lon (first valid sample)
        def first_valid(vec):
            # vec shape (n_prof, n_samp)
            out = np.full((vec.shape[0],), np.nan)
            for i in range(vec.shape[0]):
                row = vec[i]
                j = np.where(~np.isnan(row))[0]
                if j.size:
                    out[i] = row[j[0]]
            return out

        if hasattr(self, 'LAT_mvp'):
            coords['profile_lat'] = (
                'profile', first_valid(self.LAT_mvp), {'units': 'degrees_north', 'long_name': 'Profile latitude'}
            )
        if hasattr(self, 'LON_mvp'):
            coords['profile_lon'] = (
                'profile', first_valid(self.LON_mvp), {'units': 'degrees_east', 'long_name': 'Profile longitude'}
            )

        # Global attributes
        attrs = {
            'title': 'MVP profile data',
            'Conventions': 'CF-1.8',
            'institution': 'LMD/CNRS/ENS',
            'source': 'MVPAnalyzer',
            'history': f"Created on {datetime.now().isoformat()}",
            'mvp_Yorig': int(self.Yorig),
            'sampling frequency_hz': float(self.freq_echant)

        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Compression encoding per engine
        encoding = None
        if compression:
            if engine == 'netcdf4':
                encoding = {name: {'zlib': True, 'complevel': 4} for name in data_vars.keys()}
            elif engine == 'h5netcdf':
                encoding = {name: {'compression': 'gzip', 'compression_opts': 4} for name in data_vars.keys()}

        # Determine output base directory
        if filepath is None:
            base_dir = self.output_path if hasattr(self, 'output_path') else os.getcwd() + os.sep
        else:
            # If a full file path was provided and not per_profile_files, honor it
            if (not per_profile_files) and filepath.lower().endswith('.nc'):
                out_path = filepath
                ds.to_netcdf(out_path, encoding=encoding, engine=engine)
                print(f"NetCDF written: {out_path} using engine={engine}")
                return
            base_dir = filepath

        if not base_dir.endswith(os.sep):
            base_dir = base_dir + os.sep

        base_name = "MVP_" + os.path.basename(self.data_path).rstrip(os.sep)
        if per_profile_files:
            # Write one file per pair (down/up)
            total_pairs = (n_prof + 1) // 2
            for i in range(total_pairs):
                idxs = [k for k in (2*i, 2*i+1) if k < n_prof]
                if not idxs:
                    continue
                ds_i = ds.isel(profile=idxs)

                #add i to filename
                fname = f"{base_name}_profile_{i:03d}.nc"
                out_path = os.path.join(base_dir, fname)
                ds_i.to_netcdf(out_path, encoding=encoding, engine=engine)
            print(f"NetCDF written per profile into: {base_dir} using engine={engine}")
        else:
            file_name = f"{base_name}.nc"
            out_path = os.path.join(base_dir, file_name)
            ds.to_netcdf(out_path, encoding=encoding, engine=engine)
            print(f"NetCDF written: {out_path} using engine={engine}")


    def ctd_to_netcdf(self, filepath=None, compression=True, engine=None, per_profile_files=True):
        """
        Export CTD data to a NetCDF file using xarray.

        Args:
            filepath (str): Output NetCDF file path.
            compression (bool): Enable compression (engine dependent). Default True.
            engine (str|None): One of 'netcdf4', 'h5netcdf', 'scipy'. If None, choose netcdf4.
            per_profile_files (bool): If True, write one .nc per CTD cycle (two rows: down and up).
        """
        if not getattr(self, 'ctd', False):
            raise RuntimeError("No CTD data loaded. Call load_ctd_data() first.")

        engine = 'netcdf4' if engine is None else engine
        if engine == 'scipy' and compression:
            print('Warning: scipy backend does not support compression; writing without compression.')
            compression = False

        # Dimensions
        n_prof, n_samp = self.PRES_ctd.shape

        # Coordinates
        profile_idx = np.arange(n_prof, dtype=np.int32)
        sample_idx = np.arange(n_samp, dtype=np.int32)

        # Direction per profile (down/up)
        direction = np.array(['down' if i % 2 == 0 else 'up' for i in range(n_prof)], dtype=object)

        # Per-profile datetime (one timestamp per cast pair); map using i//2
        profile_time = None
        if hasattr(self, 'DATETIME_ctd') and len(getattr(self, 'DATETIME_ctd', [])) > 0:
            prof_times = []
            for i in range(n_prof):
                j = i // 2
                if j < len(self.DATETIME_ctd) and self.DATETIME_ctd[j] is not None:
                    prof_times.append(np.datetime64(self.DATETIME_ctd[j]))
                else:
                    prof_times.append(np.datetime64('NaT'))
            profile_time = np.array(prof_times, dtype='datetime64[ns]')
        else:
            profile_time = np.array([np.datetime64('NaT')] * n_prof, dtype='datetime64[ns]')

        # Build dataset variables safely
        data_vars = {}

        def add_var(var_name, arr, units=None, long_name=None):
            if arr is None:
                return
            data_vars[var_name] = (
                ('profile', 'sample'), arr,
                {k: v for k, v in [('units', units), ('long_name', long_name)] if v is not None}
            )

        
        add_var('PRES', getattr(self, 'PRES_ctd', None), units='dbar', long_name='Sea water pressure')
        add_var('TEMP', getattr(self, 'TEMP_ctd', None), units='degC', long_name='In-situ temperature')
        add_var('COND', getattr(self, 'COND_ctd', None), units='mS/cm', long_name='Conductivity')
        add_var('SAL', getattr(self, 'SALT_ctd', None), units='psu', long_name='Practical salinity')
        add_var('TURB', getattr(self, 'TURB_ctd', None), units='NTU', long_name='Turbidity')
        add_var('OXY', getattr(self, 'OXY_ctd', None), units='ml/L', long_name='Dissolved oxygen')
        add_var('FLUO', getattr(self, 'FLUO_ctd', None), units='ug/L', long_name='Chl fluorescence')
        add_var('CDOM', getattr(self, 'CDOM_ctd', None), units='ppb', long_name='Colored dissolved organic matter')

        # Position arrays (2D)
        if hasattr(self, 'LAT_ctd'):    
            add_var('LATITUDE', self.LAT_ctd, units='degrees_north', long_name='Latitude at sample')
        if hasattr(self, 'LON_ctd'):
            add_var('LONGITUDE', self.LON_ctd, units='degrees_east', long_name='Longitude at sample')
        # Global attributes
        attrs = {
            'Conventions': 'CF-1.6',
            'title': 'CTD Data',
            'institution': 'LMD/ENS/CNRS',
            'source': 'CTD',
            'history': f'Created {np.datetime64("now")}'
        }
        ds = xr.Dataset(data_vars=data_vars, coords={
            'profile': ('profile', profile_idx),
            'sample': ('sample', sample_idx),
            'direction': ('profile', direction.astype('U'), {'long_name': 'Profile direction'}),
            'profile_time': ('profile', profile_time, {'long_name': 'Profile nominal time'}),
            # Optional per-profile lat/lon (first valid sample)
            'profile_lat': (
                'profile', np.array([row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan for row in self.LAT_ctd]),
                {'units': 'degrees_north', 'long_name': 'Profile latitude'}
            ) if hasattr(self, 'LAT_ctd') else None,
            'profile_lon': (
                'profile', np.array([row[~np.isnan(row)][0] if np.any(~np.isnan(row)) else np.nan for row in self.LON_ctd]),
                {'units': 'degrees_east', 'long_name': 'Profile longitude'}
            ) if hasattr(self, 'LON_ctd') else None
        }, attrs=attrs)


        # Compression encoding per engine
        encoding = None
        if compression:
            if engine == 'netcdf4':
                encoding = {name: {'zlib': True, 'complevel': 4} for name in data_vars.keys()}
            elif engine == 'h5netcdf':
                encoding = {name: {'compression': 'gzip', 'compression_opts': 4} for name in data_vars.keys()}
        # Determine output base directory
        if filepath is None:
            base_dir = self.output_path if hasattr(self, 'output_path') else os.getcwd() + os.sep
        else:   
            # If a full file path was provided and not per_profile_files, honor it
            if (not per_profile_files) and filepath.lower().endswith('.nc'):
                out_path = filepath
                ds.to_netcdf(out_path, encoding=encoding, engine=engine)
                print(f"NetCDF written: {out_path} using engine={engine}")
                return
            base_dir = filepath
        if not base_dir.endswith(os.sep):
            base_dir = base_dir + os.sep  
                  
        base_name = "CTD_" + os.path.basename(self.data_path).rstrip(os.sep)
        if per_profile_files:   
            # Write one file per pair (down/up)
            total_pairs = (n_prof + 1) // 2
            for i in range(total_pairs):
                idxs = [k for k in (2*i, 2*i+1) if k < n_prof]
                if not idxs:
                    continue
                ds_i = ds.isel(profile=idxs)

                #add i to filename
                fname = f"{base_name}_profile_{i:03d}.nc"
                out_path = os.path.join(base_dir, fname)
                ds_i.to_netcdf(out_path, encoding=encoding, engine=engine)
            print(f"NetCDF written per profile into: {base_dir} using engine={engine}")
        else:
            file_name = f"{base_name}.nc"
            out_path = os.path.join(base_dir, file_name)
            ds.to_netcdf(out_path, encoding=encoding, engine=engine)
            print(f"NetCDF written: {out_path} using engine={engine}")


    


    def help(self):
        """
        Print all methods of the class with their docstring (header).
        """
        for attr in dir(self):
            if callable(getattr(self, attr)) and not attr.startswith("__"):
                method = getattr(self, attr)
                doc = method.__doc__
                print(f"{attr}:\n{doc}\n{'-'*40}")      


def split_ctd(pres, array):

    ibot = np.min(np.where(pres == pres.max()))

    array_down = array[:ibot]
    array_up = array[ibot:]

    return array_down, array_up