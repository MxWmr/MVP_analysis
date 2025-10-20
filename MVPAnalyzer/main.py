##########################################################################
# MVPAnalyzer/main.py
# Author: Maximilien Wemaere (LMD/CNRS)
# Date: August 2025
#
#
# Simple routines to load, analyze and correct data from a Moving Vessel Profiler (MVP) 300
# Requires numpy, matplotlib, gsw, seabird, tqdm, cartopy
#
#
# Routines to read mvp data are adapted from routines provided by Pierre l'Hegaret (UBO)
# (mvp_routines.py, temporal_lag_correction.py, thermal_mass_correction.py)
#
#
# STILL IN DEVELOPMENT !
#
#
##########################################################################




from math import e
import numpy as np 
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gsw
from seabird.cnv import fCNV
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from . import mvp_routines as mvp
from . import temporal_lag_correction as tlc
from . import thermal_mass_correction_bis as tmc
  
class Analyzer:
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


    def load_mvp_data(self,delp=[],data_path=None,format='raw',only_new=False):
        """
        Load MVP data from .raw and .log files in the data_path folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            delp (list): Indices of profiles to remove from the list (optional).
            data_path (str): Path to the folder containing MVP files (optional).
        """
        if data_path is not None:
            self.data_path = data_path

        if format=='raw':
            if self.subdirs:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/*.raw', recursive=True)))
            else:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '*.raw', recursive=self.subdirs)))


            if only_new:
                list_output = [f for f in os.listdir(self.output_path) if f.endswith(".nc")]
                files = [f for f in files if not "MVP_"+os.path.basename(f).replace('.raw', '.nc') in list_output]

        elif format=='ncdf':
            if self.subdirs:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/MVP*.nc', recursive=True)))
            else:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + 'MVP*.nc', recursive=self.subdirs)))

            if only_new:
                list_output = [f for f in os.listdir(self.output_path) if f.endswith(".nc")]
                files = [f for f in files if not "MVP_"+os.path.basename(f) in list_output]


        print('Found ' + str(len(files)) + ' MVP files in the directory: ' + self.data_path)



        if format=='ncdf':
            
            for f in files:
                nc = xr.open_dataset(f)
                self.PRES_mvp = nc['PRES'].values
                self.TEMP_mvp = nc['TEMP'].values
                self.COND_mvp = nc['COND'].values
                self.SOUNDVEL_mvp = nc['SOUNDVEL'].values
                self.DO_mvp = nc['DO'].values
                self.TEMP2_mvp = nc['TEMP2'].values
                self.SUNA_mvp = nc['SUNA'].values
                self.FLUO_mvp = nc['FLUO'].values
                self.TURB_mvp = nc['TURB'].values
                self.PH_mvp = nc['PH'].values
                self.SALT_mvp = nc['SAL'].values
                self.TIME_mvp = nc['TIME_s'].values
                self.LAT_mvp = nc['LATITUDE'].values
                self.LON_mvp = nc['LONGITUDE'].values
                self.DATETIME_mvp = nc['profile_time'].values
                self.DIR = nc['direction'].values
                self.label_mvp = nc['profile'].values
                self.freq_echant = nc.attrs['sampling frequency_hz']

                nc.close()
                print('MVP data loaded successfully.')
                self.mvp = True

                return

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

            if format=='raw':
                (mvp_tstart,mvp_tend,cycle_dur, lat, lon, dt_station) = mvp.get_log(mvp_log_name,self.Yorig)


            if cycle_dur>1:

                # Read one cycle MVP data  
                (pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw) = mvp.read_mvp_cycle_raw(mvp_dat_name)
                (pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph) = mvp.raw_data_conversion(pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw)
            

                freq_echant = float(len(pres)/cycle_dur)

                DATETIME_mvp.append(dt_station)
                
                if np.nanmax(pres)-np.nanmin(pres)>2:

                    # Allocate time to samples and select the ascending part 
                    (pres_up,soundvel_up,cond_up,temp_up,do_up,temp2_up,suna_up,fluo_up,turb_up,ph_up,time_up) = mvp.time_mvp_cycle_up([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)
                    (pres_down,soundvel_down,cond_down,temp_down,do_down,temp2_down,suna_down,fluo_down,turb_down,ph_down,time_down) = mvp.time_mvp_cycle_down([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)

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




    def load_mvp_data_again(self,data_path=None,format='raw',delp=[]):
        """
        Load MVP data from .raw and .log files in the data_path folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path (str): Path to the folder containing MVP files.
            delp (list): Indices of profiles to remove from the list (optional).
        """
        if data_path is not None:
            self.data_path = data_path
        
        if format=='raw':
            files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/MVP*.raw', recursive=True)))
        elif format=='ncdf':
            files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/MVP*.nc', recursive=True)))
        print('Found ' + str(len(files)) + ' MVP files in the directory: ' + self.data_path)



        if format=='ncdf':
            for f in files:
                nc = xr.open_dataset(f)
                self.PRES_mvp = nc['PRES'].values
                self.TEMP_mvp = nc['TEMP'].values
                self.COND_mvp = nc['COND'].values
                self.SOUNDVEL_mvp = nc['SOUNDVEL'].values
                self.DO_mvp = nc['DO'].values
                self.TEMP2_mvp = nc['TEMP2'].values
                self.SUNA_mvp = nc['SUNA'].values
                self.FLUO_mvp = nc['FLUO'].values
                self.TURB_mvp = nc['TURB'].values
                self.PH_mvp = nc['PH'].values
                self.SALT_mvp = nc['SAL'].values
                self.TIME_mvp = nc['TIME'].values
                self.LAT_mvp = nc['LATITUDE'].values
                self.LON_mvp = nc['LONGITUDE'].values
                self.DATETIME_mvp = nc['profile_time'].values
                self.DIR = nc['direction'].values
                self.Label_mvp = nc['profile'].values
                self.freq_echant = nc.attrs['sampling frequency_hz']

                nc.close()
                print('MVP data loaded successfully.')
                self.mvp = True

                return




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
                    (pres_up,soundvel_up,cond_up,temp_up,do_up,temp2_up,suna_up,fluo_up,turb_up,ph_up,time_up) = mvp.time_mvp_cycle_up([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)
                    (pres_down,soundvel_down,cond_down,temp_down,do_down,temp2_down,suna_down,fluo_down,turb_down,ph_down,time_down) = mvp.time_mvp_cycle_down([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)


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


    def load_ctd_data(self,data_path_ctd,format='cnv'):
        """
        Load CTD data from .cnv files in the data_path_ctd folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path_ctd (str): Path to the folder containing CTD files.
        """


        if format=='cnv':
            list_of_ctd_files = sorted(filter(os.path.isfile,\
                            glob.glob(data_path_ctd + '*.cnv')))
        elif format=='ncdf':
            list_of_ctd_files = sorted(filter(os.path.isfile,\
                            glob.glob(data_path_ctd + 'CTD'+'*.nc')))
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

        if format=='ncdf':
            for f in list_of_ctd_files:
                nc = xr.open_dataset(f)
                PRES_ctd_temp.append(nc['PRES'].values[0])
                PRES_ctd_temp.append(nc['PRES'].values[1])
                TEMP_ctd_temp.append(nc['TEMP'].values[0])
                TEMP_ctd_temp.append(nc['TEMP'].values[1])
                COND_ctd_temp.append(nc['COND'].values[0])
                COND_ctd_temp.append(nc['COND'].values[1])
                SALT_ctd_temp.append(nc['SAL'].values[0])
                SALT_ctd_temp.append(nc['SAL'].values[1])
                TURB_ctd_temp.append(nc['TURB'].values[0])
                TURB_ctd_temp.append(nc['TURB'].values[1])
                OXY_ctd_temp.append(nc['OXY'].values[0])
                OXY_ctd_temp.append(nc['OXY'].values[1])
                FLUO_ctd_temp.append(nc['FLUO'].values[0])
                FLUO_ctd_temp.append(nc['FLUO'].values[1])
                CDOM_ctd_temp.append(nc['CDOM'].values[0])
                CDOM_ctd_temp.append(nc['CDOM'].values[1])
                LAT_ctd_temp.append(nc['LATITUDE'].values[0])
                LAT_ctd_temp.append(nc['LATITUDE'].values[1])
                LON_ctd_temp.append(nc['LONGITUDE'].values[0])
                LON_ctd_temp.append(nc['LONGITUDE'].values[1])
                DATETIME_ctd.append(nc['profile_time'].values[0])

                nc.close()

            self.PRES_ctd = np.array(PRES_ctd_temp)
            self.TEMP_ctd = np.array(TEMP_ctd_temp)
            self.COND_ctd = np.array(COND_ctd_temp)
            self.SALT_ctd = np.array(SALT_ctd_temp)
            self.TURB_ctd = np.array(TURB_ctd_temp)
            self.OXY_ctd = np.array(OXY_ctd_temp)
            self.FLUO_ctd = np.array(FLUO_ctd_temp)
            self.CDOM_ctd = np.array(CDOM_ctd_temp)
            self.LAT_ctd = np.array(LAT_ctd_temp)
            self.LON_ctd = np.array(LON_ctd_temp)
            self.DATETIME_ctd = np.array(DATETIME_ctd)


            print('CTD data loaded successfully.')
            self.ctd = True

            return
            





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


    def compute_waterflow(self,horizontal_speed,corr=False):
        """
        Compute the water flow speed (u,v) from the horizontal speed and the direction of the profiles.
        Args:
            horizontal_speed (float): Horizontal speed of the boat in cm/s.
        """
        
        if corr:
            SPEED_MVP = np.zeros((self.PRES_mvp_corr.shape[0], self.PRES_mvp_corr.shape[1]))
            for i in range(self.PRES_mvp_corr.shape[0]):
                SPEED_MVP[i,:] = np.sqrt(np.gradient(self.PRES_mvp_corr[i,:], 1/self.freq_echant)**2+ horizontal_speed**2)
        else:
            SPEED_MVP = np.zeros((self.PRES_mvp.shape[0], self.PRES_mvp.shape[1]))
            for i in range(self.PRES_mvp.shape[0]):
                SPEED_MVP[i,:] = np.sqrt(np.gradient(self.PRES_mvp[i,:], 1/self.freq_echant)**2+ horizontal_speed**2)

        self.SPEED_mvp = SPEED_MVP
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


    def keep_selected_profiles(self, id_mvp, id_ctd=None):
        """
        Keep only the selected MVP and CTD profiles in the object attributes.
        Args:
            id_mvp (list): Indices of MVP profiles to keep.
            id_ctd (list): Indices of CTD profiles to keep (optional).
        """
        

        # Make a list of all id to keep for MVP profiles
        l_id = []
        l_id2 = []
        for i in id_mvp:
            l_id.append(i)
            l_id.append(i+1)  # Add the next profile for the up profile 
            l_id2.append(i//2) 


  
        # Keep only the selected profiles

        if self.mvp:

            self.PRES_mvp = self.PRES_mvp[l_id,:]
            self.SOUNDVEL_mvp = self.SOUNDVEL_mvp[l_id,:]
            self.COND_mvp = self.COND_mvp[l_id,:]
            self.TEMP_mvp = self.TEMP_mvp[l_id,:]
            self.DO_mvp = self.DO_mvp[l_id,:]
            self.TEMP2_mvp = self.TEMP2_mvp[l_id,:]
            self.SUNA_mvp = self.SUNA_mvp[l_id,:]
            self.FLUO_mvp = self.FLUO_mvp[l_id,:]
            self.TURB_mvp = self.TURB_mvp[l_id,:]
            self.PH_mvp = self.PH_mvp[l_id,:]
            self.SALT_mvp = self.SALT_mvp[l_id,:]
            self.TIME_mvp = self.TIME_mvp[l_id,:]
            self.LAT_mvp = self.LAT_mvp[l_id,:]
            self.LON_mvp = self.LON_mvp[l_id,:]
            self.DATETIME_mvp = np.array(self.DATETIME_mvp)[l_id2]
            self.DIR = np.array(self.DIR)[l_id]
            self.label_mvp = np.array(self.label_mvp)[l_id]

        if self.ctd and id_ctd != None:

            l_id = []
            l_id2 = []
            for i in id_ctd:
                l_id.append(i)
                l_id.append(i+1)  # Add the next profile for the up profile 
                l_id2.append(i//2) 

            self.PRES_ctd = self.PRES_ctd[l_id,:]
            self.TEMP_ctd = self.TEMP_ctd[l_id,:]
            self.SALT_ctd = self.SALT_ctd[l_id,:]
            self.COND_ctd = self.COND_ctd[l_id,:]
            self.TURB_ctd = self.TURB_ctd[l_id,:]
            self.OXY_ctd = self.OXY_ctd[l_id,:]
            self.FLUO_ctd = self.FLUO_ctd[l_id,:]
            self.CDOM_ctd = self.CDOM_ctd[l_id,:]
            self.LAT_ctd = self.LAT_ctd[l_id,:]
            self.LON_ctd = self.LON_ctd[l_id,:]
            self.DATETIME_ctd = np.array(self.DATETIME_ctd)[l_id2]


    def plot_vertical_speed(self,id,mean=False,window=20):
            
        if self.mvp==False:
            print('No MVP data loaded.')
            return
    
        if mean:
            v_z_down = np.gradient(self.PRES_mvp[0::2], 1/self.freq_echant,axis=1)
            v_z_up = np.gradient(self.PRES_mvp[1::2], 1/self.freq_echant,axis=1)

            # smooth speed
            for i in range(v_z_down.shape[0]):
                v_z_down[i,:] = np.convolve(v_z_down[i,:], np.ones(2*window+1)/(2*window+1), mode='same')
                v_z_up[i,:] = np.convolve(v_z_up[i,:], np.ones(2*window+1)/(2*window+1), mode='same')
            
            # take mean profile
            v_z_down = np.nanmean(v_z_down,axis=0)
            v_z_up = np.nanmean(v_z_up,axis=0)

            self.v_z_down = v_z_down
            self.v_z_up = v_z_up


        else:

            v_z_down = np.gradient(self.PRES_mvp[id,:], 1/self.freq_echant)
            v_z_up = np.gradient(self.PRES_mvp[id+1,:], 1/self.freq_echant)

            # smooth speed
            self.v_z_down = np.convolve(v_z_down, np.ones(2*window+1)/(2*window+1), mode='same')
            self.v_z_up = np.convolve(v_z_up, np.ones(2*window+1)/(2*window+1), mode='same')




        plt.figure()

        plt.plot(v_z_down,self.PRES_mvp[id], label='down')
        plt.plot(v_z_up,self.PRES_mvp[id+1], label='up')

        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid()
        plt.xlabel('Vertical speed, m/s')
        plt.ylabel('Pressure, dbar')
        plt.title('Vertical speed profiles')
        plt.legend()


    def plot_profile_map(self):
        """
        Plot a map of the start locations of each profile (MVP and CTD),
        with a land/ocean background and coastlines using cartopy.
        The map is automatically zoomed to the profile area (no excessive margin).
        Requires the cartopy module (pip install cartopy).
        """

        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title('Carte des profils (début de plongée)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        colors = plt.cm.tab10.colors

        # MVP
        if hasattr(self, 'LAT_mvp') and hasattr(self, 'LON_mvp'):

            put_label = True
            c = 0
            for i in range(0,self.LAT_mvp.shape[0],2):
                if i>0:
                    if self.label_mvp[i] == self.label_mvp[i-1]:
                        put_label = False
                    else:
                        put_label = True
                        c+=1

                lat = self.LAT_mvp[i,0] if self.LAT_mvp.ndim == 2 else  self.LAT_mvp[i]
                lon = self.LON_mvp[i,0] if self.LON_mvp.ndim == 2 else  self.LON_mvp[i]
                ax.scatter(lon, lat, color=colors[c], marker='o', label='MVP '+self.label_mvp[i] if put_label else "", transform=ccrs.PlateCarree())

        # CTD
        if hasattr(self, 'LAT_ctd') and hasattr(self, 'LON_ctd'):
            for i in range(0,self.LAT_ctd.shape[0],2):
                lat = self.LAT_ctd[i,0] if self.LAT_ctd.ndim == 2 else self.LAT_ctd[i]
                lon = self.LON_ctd[i,0] if self.LON_ctd.ndim == 2 else self.LON_ctd[i]
                ax.scatter(lon, lat, color='red', marker='^', label='CTD' if i==0 else "", transform=ccrs.PlateCarree())

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def plot_TSprofile(self, id_mvp,id_ctd=None,correction=False):
        """
        Plot temperature and salinity profiles versus pressure for a given profile (MVP and CTD).
        Args:
            id_mvp (int): Index of the MVP profile to plot.
            id_ctd (int, optional): Index of the CTD profile to plot (default: same as id_mvp).
            correction (bool): If True, plot corrected profiles.
        """

        if id_ctd is None:
            id_ctd = id_mvp
            
        
       
        plt.figure()
        if self.mvp:
            if correction:
                plt.plot(self.TEMP_mvp_corr[id_mvp],self.PRES_mvp_corr[id_mvp],label='MVP down corrected')
                plt.plot(self.TEMP_mvp_corr[id_mvp+1],self.PRES_mvp_corr[id_mvp+1],label='MVP up corrected')             
            else:
                plt.plot(self.TEMP_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
                plt.plot(self.TEMP_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.TEMP_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.TEMP_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()    
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Temperature, C')    
        plt.ylabel('Pressure, dbar')


        plt.figure()
        if self.mvp:
            if correction:
                plt.plot(self.SALT_mvp_corr[id_mvp],self.PRES_mvp_corr[id_mvp],label='MVP down corrected')
                plt.plot(self.SALT_mvp_corr[id_mvp+1],self.PRES_mvp_corr[id_mvp+1],label='MVP up corrected')
            else:
                plt.plot(self.SALT_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
                plt.plot(self.SALT_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.SALT_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.SALT_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Salinity, psu')
        plt.ylabel('Pressure, dbar')
    
    def plot_BGCprofile(self, id_mvp,id_ctd=None,):
        """
        Plot raw biogeochemical profiles (O2, turbidity, fluorescence) for a given profile (MVP and CTD).
        Args:
            id_mvp (int): Index of the MVP profile to plot.
            id_ctd (int, optional): Index of the CTD profile to plot (default: same as id_mvp).
        """
    
        if id_ctd is None:
            id_ctd = id_mvp

        

        plt.figure()
        if self.mvp:
            plt.plot(self.DO_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
            plt.plot(self.DO_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.OXY_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.OXY_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()    
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Dissolved Oxygen, %')    
        plt.ylabel('Pressure, dbar')


        plt.figure()
        if self.mvp:
            plt.plot(self.TURB_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
            plt.plot(self.TURB_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.TURB_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.TURB_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Turbidity, NTU')
        plt.ylabel('Pressure, dbar')

        plt.figure()
        if self.mvp:
            plt.plot(self.FLUO_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
            plt.plot(self.FLUO_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.FLUO_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.FLUO_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Fluorescence, ug/L')
        plt.ylabel('Pressure, dbar')

    def plot_diagramTS_raw(self,id_mvp=None,id_ctd=None,correction=False):
        """
        Plot the TS diagram (Salinity vs Temperature) for one or more profiles, with isopycnals.
        Args:
            id_mvp (int, optional): Index of the MVP profile to plot, or None for all profiles.
            id_ctd (int, optional): Index of the CTD profile to plot, or None for all profiles.
            correction (bool): If True, plot corrected profiles.
        """
    

       
    
        plt.figure()
        if id_mvp != None:
            if id_ctd == None:
                id_ctd = id_mvp

            if self.mvp:
                if correction:
                    plt.plot(self.SALT_mvp_corr[id_mvp],self.TEMP_mvp_corr[id_mvp],label='MVP down corrected',linestyle='', marker='.')
                    plt.plot(self.SALT_mvp_corr[id_mvp+1],self.TEMP_mvp_corr[id_mvp+1],label='MVP up corrected',linestyle='', marker='.')
                else:
                    plt.plot(self.SALT_mvp[id_mvp],self.TEMP_mvp[id_mvp],label='MVP down',linestyle='', marker='.')
                    plt.plot(self.SALT_mvp[id_mvp+1],self.TEMP_mvp[id_mvp+1],label='MVP up',linestyle='', marker='.')
            if self.ctd:
                plt.plot(self.SALT_ctd[id_ctd],self.TEMP_ctd[id_ctd],label='CTD down', linestyle='', marker='.')
                plt.plot(self.SALT_ctd[id_ctd+1],self.TEMP_ctd[id_ctd+1],label='CTD up', linestyle='', marker='.')

        else:
            if self.mvp:
                if correction:
                    plt.plot(self.SALT_mvp_corr[0],self.TEMP_mvp_corr[0],linestyle='',color='red', marker='.',label='MVP down corrected')
                    plt.plot(self.SALT_mvp_corr[1],self.TEMP_mvp_corr[1],linestyle='',color='blue', marker='.',label='MVP up corrected')
                    for i in range(2,len(self.PRES_mvp),2):
                        plt.plot(self.SALT_mvp_corr[i],self.TEMP_mvp_corr[i],linestyle='',color='red', marker='.')
                        plt.plot(self.SALT_mvp_corr[i+1],self.TEMP_mvp_corr[i+1],linestyle='',color='blue', marker='.')
                else:
                    plt.plot(self.SALT_mvp[0],self.TEMP_mvp[0],linestyle='',color='red', marker='.',label='MVP down')
                    plt.plot(self.SALT_mvp[1],self.TEMP_mvp[1],linestyle='',color='blue', marker='.',label='MVP up')
                    for i in range(2,len(self.PRES_mvp),2):
                        plt.plot(self.SALT_mvp[i],self.TEMP_mvp[i],linestyle='',color='red', marker='.')
                        plt.plot(self.SALT_mvp[i+1],self.TEMP_mvp[i+1],linestyle='',color='blue', marker='.')
            if self.ctd:
                plt.plot(self.SALT_ctd[0],self.TEMP_ctd[0],color='green', linestyle='', marker='.',label='CTD down')
                plt.plot(self.SALT_ctd[1],self.TEMP_ctd[1],color='orange', linestyle='', marker='.',label='CTD up')
                for i in range(2,len(self.PRES_ctd),2):
                    plt.plot(self.SALT_ctd[i],self.TEMP_ctd[i],color='green', linestyle='', marker='.')
                    plt.plot(self.SALT_ctd[i+1],self.TEMP_ctd[i+1],color='orange', linestyle='', marker='.')


        s_lim = plt.xlim()
        t_lim = plt.ylim()
        SA = np.linspace(s_lim[0], s_lim[1], 100)  # Absolute Salinity [g/kg]
        CT = np.linspace(t_lim[0], t_lim[1], 100)
        SA_grid, CT_grid = np.meshgrid(SA, CT)
        # Calcul de la densité potentielle sigma0 (kg/m³ - 1000)
        sigma0 = gsw.sigma0(SA_grid, CT_grid)
        # Dessiner les contours (les isopycnes)
        contour_plot = plt.contour(SA_grid, CT_grid, sigma0, colors='k', linestyles='dotted')
        # Ajouter les étiquettes (les chiffres) le long des contours
        plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1f')

        plt.legend()  
        plt.xlabel('Salinity, psu') 
        plt.ylabel('Temperature, C')

    def stat_compar(self,id_mvp=[],id_ctd=None,num_sample=5000,cond=False,speed=False,correction=False):
        """
        Statistically compare MVP and CTD profiles (temperature and salinity),
        print statistics and interpolated differences.
        Args:
            id (list): Indices of profiles to compare (all if empty).
            num_sample (int): Number of pressure levels for interpolation.
        """

        if not self.mvp or not self.ctd:
            raise ValueError("MVP or CTD data not loaded.")
        
        if id_mvp == []:
            id_mvp = list(range(0, self.PRES_mvp.shape[0]))
        if id_ctd is None:
            id_ctd = id_mvp

        if len(id_mvp) != len(id_ctd):
            raise ValueError("id_mvp and id_ctd must have the same length.")

        if correction:
            Pres = self.PRES_mvp_corr
            Temp = self.TEMP_mvp_corr
            Salt = self.SALT_mvp_corr
            Cond = self.COND_mvp_corr
        else:
            Pres = self.PRES_mvp
            Temp = self.TEMP_mvp
            Salt = self.SALT_mvp
            Cond = self.COND_mvp
        Do = self.DO_mvp

        # Interpolate MVP and CTD data to match pressure levels
        pmin = np.nanmin(Pres)
        pmax = np.nanmax(Pres)
        pressure_grid = np.linspace(pmin, pmax, num_sample)

        TEMP_mvp_interp = mvp.vertical_interp(Pres[id_mvp,:],Temp[id_mvp,:], pressure_grid)
        SALT_mvp_interp = mvp.vertical_interp(Pres[id_mvp,:], Salt[id_mvp,:], pressure_grid)
        DO_mvp_interp = mvp.vertical_interp(Pres[id_mvp,:], Do[id_mvp,:], pressure_grid)
        COND_mvp_interp = mvp.vertical_interp(Pres[id_mvp,:], Cond[id_mvp,:], pressure_grid) 

        # keep only down profiles
        id_ctd1 = [id_ctd[i] for i in range(len(id_ctd)) if id_ctd[i]%2 == 0]
 
        TEMP_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.TEMP_ctd[id_ctd1,:], pressure_grid)
        SALT_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.SALT_ctd[id_ctd1,:], pressure_grid)
        DO_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.OXY_ctd[id_ctd1,:], pressure_grid)
        COND_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.COND_ctd[id_ctd1,:], pressure_grid)

        # differences study between MVP down and CTD profiles

        # Calcul des différences entre les profils interpolés (MVP - CTD)
        diff_temp_down = TEMP_mvp_interp[0::2] - TEMP_ctd_interp
        diff_temp_up = TEMP_mvp_interp[1::2] - TEMP_ctd_interp
        diff_salt_down = SALT_mvp_interp[0::2] - SALT_ctd_interp
        diff_salt_up = SALT_mvp_interp[1::2] - SALT_ctd_interp
        diff_do_down = DO_mvp_interp[0::2] - DO_ctd_interp
        diff_do_up = DO_mvp_interp[1::2] - DO_ctd_interp
        diff_cond_down = COND_mvp_interp[0::2] - COND_ctd_interp
        diff_cond_up = COND_mvp_interp[1::2] - COND_ctd_interp


        # Plot mean error vs depth for each variable (down/up)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Compute mean error along profiles (axis=0: profiles, axis=1: depth)
        mean_temp_down = np.absolute(np.nanmean(diff_temp_down, axis=0))
        mean_temp_up =  np.absolute(np.nanmean(diff_temp_up, axis=0))
        mean_salt_down =  np.absolute(np.nanmean(diff_salt_down, axis=0))
        mean_salt_up =  np.absolute(np.nanmean(diff_salt_up, axis=0))
        mean_do_down =  np.absolute(np.nanmean(diff_do_down, axis=0))
        mean_do_up =  np.absolute(np.nanmean(diff_do_up, axis=0))
        mean_cond_down =  np.absolute(np.nanmean(diff_cond_down, axis=0))
        mean_cond_up =  np.absolute(np.nanmean(diff_cond_up, axis=0))

        axes[0].plot(mean_temp_down, pressure_grid, label='Down')
        axes[0].plot(mean_temp_up, pressure_grid, label='Up')
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Absolute Mean Error (°C)')
        axes[0].set_ylabel('Pressure (dbar)')
        axes[0].set_title('Temperature Error')
        axes[0].legend()
        axes[0].grid()


        if cond:

            axes[1].plot(mean_cond_down, pressure_grid, label='Down')
            axes[1].plot(mean_cond_up, pressure_grid, label='Up')   
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Absolute Mean Error (S/m)')
            axes[1].set_ylabel('Pressure (dbar)')
            axes[1].set_title('Conductivity Error')
            axes[1].legend()
            axes[1].grid()

        else:

            axes[1].plot(mean_salt_down, pressure_grid, label='Down')
            axes[1].plot(mean_salt_up, pressure_grid, label='Up')
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Absolute Mean Error (psu)')
            axes[1].set_ylabel('Pressure (dbar)')
            axes[1].set_title('Salinity Error')
            axes[1].legend()
            axes[1].grid()
        
        if speed:

            axes[2].plot(self.v_z_down, self.PRES_mvp[0], label='Down')
            axes[2].plot(self.v_z_up, self.PRES_mvp[0], label='Up')
            axes[2].invert_yaxis()
            axes[2].set_xlabel('Vertical Speed (m/s)')
            axes[2].set_ylabel('Pressure (dbar)')
            axes[2].set_title('Vertical Speed')
            axes[2].legend()
            axes[2].grid()


        else:

            axes[2].plot(mean_do_down, pressure_grid, label='Down')
            axes[2].plot(mean_do_up, pressure_grid, label='Up')
            axes[2].invert_yaxis()
            axes[2].set_xlabel('Absolute Mean Error (%)')
            axes[2].set_ylabel('Pressure (dbar)')
            axes[2].set_title('Oxygen Error')
            axes[2].legend()
            axes[2].grid()

        fig.suptitle('Absolute Mean Error (MVP - CTD) vs Depth')
        fig.tight_layout()
        plt.show()
        

        # Compute RMSE

        rmse_temp_down = np.mean(np.sqrt(np.nanmean(diff_temp_down**2, axis=1)))
        rmse_temp_up = np.mean(np.sqrt(np.nanmean(diff_temp_up**2, axis=1)))
        rmse_salt_down = np.mean(np.sqrt(np.nanmean(diff_salt_down**2, axis=1)))
        rmse_salt_up = np.mean(np.sqrt(np.nanmean(diff_salt_up**2, axis=1)))
        rmse_do_down = np.mean(np.sqrt(np.nanmean(diff_do_down**2, axis=1)))
        rmse_do_up = np.mean(np.sqrt(np.nanmean(diff_do_up**2, axis=1)))
        rmse_cond_down = np.mean(np.sqrt(np.nanmean(diff_cond_down**2, axis=1)))
        rmse_cond_up = np.mean(np.sqrt(np.nanmean(diff_cond_up**2, axis=1)))

        # Find index where depth >= 200 dbar; fallback to 0 if not found
        i_200 = 0
        for i in range(len(pressure_grid)):
            if pressure_grid[i] >= 200:
                i_200 = i
                break

        # Slice along depth axis (columns) to keep depths >= 200 dbar
        rmse_temp_down_deep = np.mean(np.sqrt(np.nanmean(diff_temp_down[:, i_200:]**2, axis=1)))
        rmse_temp_up_deep   = np.mean(np.sqrt(np.nanmean(diff_temp_up[:,   i_200:]**2, axis=1)))
        rmse_salt_down_deep = np.mean(np.sqrt(np.nanmean(diff_salt_down[:, i_200:]**2, axis=1)))
        rmse_salt_up_deep   = np.mean(np.sqrt(np.nanmean(diff_salt_up[:,   i_200:]**2, axis=1)))
        rmse_do_down_deep   = np.mean(np.sqrt(np.nanmean(diff_do_down[:,   i_200:]**2, axis=1)))
        rmse_do_up_deep     = np.mean(np.sqrt(np.nanmean(diff_do_up[:,     i_200:]**2, axis=1)))
        rmse_cond_down_deep = np.mean(np.sqrt(np.nanmean(diff_cond_down[:, i_200:]**2, axis=1)))
        rmse_cond_up_deep   = np.mean(np.sqrt(np.nanmean(diff_cond_up[:,   i_200:]**2, axis=1)))    


        # Print statistics + grouped deep RMSE

        temp_rmse = [rmse_temp_down, rmse_temp_up]
        salt_rmse = [rmse_salt_down, rmse_salt_up]
        do_rmse = [rmse_do_down, rmse_do_up]

        temp_rmse_deep = [rmse_temp_down_deep, rmse_temp_up_deep]
        salt_rmse_deep = [rmse_salt_down_deep, rmse_salt_up_deep]
        do_rmse_deep = [rmse_do_down_deep, rmse_do_up_deep]

        labels = ['MVP down', 'MVP up']
        colors = ['blue', 'orange']

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for idx, (ax, data, data_deep, title, ylabel) in enumerate(zip(
            axes,
            [temp_rmse,  salt_rmse,  do_rmse],
            [temp_rmse_deep, salt_rmse_deep, do_rmse_deep],
            ['Temperature', 'Salinity', 'Oxygen'],
            ['RMSE (°C)', 'RMSE (psu)', 'RMSE (%)']
        )):
            x = np.arange(len(labels))
            width = 0.35
            # Side-by-side grouped bars: left = All depths, right = Deep
            label_all = 'All depths' if idx == 0 else None
            label_deep = 'Deep (≥200 dbar)' if idx == 0 else None
            bars_all = ax.bar(x - width/2, data, width=width, color=colors, edgecolor='k', label=label_all)
            bars_deep = ax.bar(x + width/2, data_deep, width=width, color=colors, edgecolor='k', alpha=0.6, label=label_deep)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            ymax = max(max(data), max(data_deep)) * 1.25  # 25% margin above highest
            ax.set_ylim(0, ymax)

            # Annotations
            for b in bars_all:
                h = b.get_height()
                if np.isfinite(h):
                    ax.annotate(f'{h:.3f}', (b.get_x() + b.get_width()/2, h),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
            for b in bars_deep:
                h = b.get_height()
                if np.isfinite(h):
                    ax.annotate(f'{h:.3f}', (b.get_x() + b.get_width()/2, h),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', fontsize=9)

            if idx == 0:
                ax.legend()

        fig.suptitle('RMSE MVP vs CTD')
        fig.tight_layout()
        plt.show()


        if cond:
            print("Conductivity RMSE (MVP - CTD):")
            print(f"  MVP down: {rmse_cond_down:.4f} S/m (deep: {rmse_cond_down_deep:.4f} S/m)")
            print(f"  MVP up:   {rmse_cond_up:.4f} S/m (deep: {rmse_cond_up_deep:.4f} S/m)")

    def correct_oxygen(self,id_mvp=None,id_ctd=None,num_sample=500,plotting=False,correction=False):
        """
        Apply oxygen correction to MVP dissolved oxygen profiles thanks to CTD data.
        Args:
            id_mvp (int): Index of the MVP profile to use for correction.
            id_ctd (int): Index of the CTD profile to use for correction.
            num_sample (int): Number of pressure levels for interpolation.
            plotting (bool): If True, plot the correction results.
            correction (bool): If True, update corrected attributes.
        """

        if not self.mvp or not self.ctd:
            raise ValueError("MVP or CTD data not loaded.")
        

        if id_mvp is None:
            id_mvp,id_ctd = 0,0
            print(f"No profile index provided, using first profiles: MVP {id_mvp} and CTD {id_ctd}.")
        elif id_ctd is None:
            id_ctd = id_mvp
        

        # Interpolate MVP and CTD data to match pressure levels
        pmin = np.nanmin(self.PRES_mvp)
        pmax = np.nanmax(self.PRES_mvp)
        pressure_grid = np.linspace(pmin, pmax, num_sample)


        DO_mvp_interp = mvp.vertical_interp(self.PRES_mvp[id_mvp,:], self.DO_mvp[id_mvp,:], pressure_grid)
        DO_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd,:],self.OXY_ctd[id_ctd,:], pressure_grid)

        mask = ~np.isnan(DO_mvp_interp) & ~np.isnan(DO_ctd_interp)
        pressure_grid = pressure_grid[mask[0]]
        DO_mvp_interp = DO_mvp_interp[mask]
        DO_ctd_interp = DO_ctd_interp[mask]

        diff = DO_mvp_interp-DO_ctd_interp

        A = np.vstack([pressure_grid, np.ones_like(pressure_grid)]).T
        print(A.shape, diff.shape)
        diff = diff.flatten()
        a_estime, b_estime = np.linalg.lstsq(A, diff, rcond=None)[0]

        print(f"Pente estimée (a) : {a_estime:.6f} ")
        print(f"Biais estimé (b) : {b_estime:.6f} ")

        DO_mvp_corr = DO_mvp_interp - (a_estime*pressure_grid + b_estime)



        rmse_before = np.sqrt(np.nanmean((DO_mvp_interp - DO_ctd_interp)**2))
        rmse_after = np.sqrt(np.nanmean((DO_mvp_corr - DO_ctd_interp)**2))
        print(f"RMSE before correction: {rmse_before:.4f}")
        print(f"RMSE after correction: {rmse_after:.4f}")

        DO_mvp_corr_full = self.DO_mvp - (a_estime*self.PRES_mvp + b_estime)

        DO_mvp_corr_full_interp = mvp.vertical_interp(self.PRES_mvp, DO_mvp_corr_full, pressure_grid)
        rmse_after_full = np.mean(np.sqrt(np.nanmean((DO_mvp_corr_full_interp - DO_ctd_interp)**2,axis=1)))
        print(f"RMSE after correction (full profile): {rmse_after_full:.4f}")


        if correction:
            self.DO_mvp = DO_mvp_corr_full

        if plotting:

            plt.figure()
            plt.plot(DO_mvp_interp,pressure_grid,label='MVP')
            plt.plot(DO_ctd_interp,pressure_grid,label='CTD')
            plt.plot(DO_mvp_corr,pressure_grid,label='MVP corrected')
            plt.gca().invert_yaxis()
            plt.xlabel('Dissolved Oxygen, %')
            plt.ylabel('Pressure, dbar')
            plt.title('Oxygen correction')
            plt.legend()
            plt.grid()
            plt.show()


    def viscous_heating_correction(self,correction=False):
        """
        Apply viscous heating correction to MVP temperature profiles.
        Args:
            correction (bool): If True, update corrected attributes.
        """

        self.TEMP_mvp_filt = mvp.viscous_heating(self.TEMP_mvp, self.SALT_mvp,self.PRES_mvp,self.LON_mvp, self.LAT_mvp, self.TIME_mvp)

        if correction:
            self.TEMP_mvp_corr = self.TEMP_mvp_filt
            self.SALT_mvp_corr = gsw.SP_from_C(self.COND_mvp, self.TEMP_mvp,self.PRES_mvp)

    def filtering_surface_waves(self,correction=False):
        """
        Apply filtering to remove surface wave effects from MVP profiles.
        Args:
            correction (bool): If True, update corrected attributes.
        """
        SAMP_TIME = np.zeros((self.TIME_mvp.shape[0],self.TIME_mvp.shape[1]))
        SAMP_TIME[:] = np.nan

        for i in range(self.TIME_mvp.shape[0]):
            SAMP_TIME[i,:] = (self.TIME_mvp[i,:]-np.nanmin(self.TIME_mvp[i,:]))*24*3600
        Time = np.arange(0,np.nanmax(SAMP_TIME), 1/self.freq_echant)


        STime_T_interp = mvp.median(SAMP_TIME, SAMP_TIME, Time)
        Time_T_interp = mvp.median(SAMP_TIME, self.TIME_mvp, Time)
        Pr_T_interp = mvp.median(SAMP_TIME, self.PRES_mvp, Time)
        T_T_interp = mvp.median(SAMP_TIME, self.TEMP_mvp, Time)
        C_T_interp = mvp.median(SAMP_TIME, self.COND_mvp, Time)
        Lon_T_interp = mvp.median(SAMP_TIME, self.LON_mvp, Time)
        Lat_T_interp = mvp.median(SAMP_TIME, self.LAT_mvp, Time)


        sampling_frequency = int(self.freq_echant)
        cutoff_frequency = 1
        TEMP_filt = mvp.remove_surface_waves(T_T_interp,Time_T_interp*24*3600,sampling_frequency,cutoff_frequency,1)
        COND_filt = mvp.remove_surface_waves(C_T_interp,Time_T_interp*24*3600,sampling_frequency,cutoff_frequency,1)
        cutoff_frequency = 1/2
        PRES_filt = mvp.remove_surface_waves(Pr_T_interp,Time_T_interp*24*3600,sampling_frequency,cutoff_frequency,1)
        LAT_filt = mvp.remove_surface_waves(Lat_T_interp,Time_T_interp*24*3600,sampling_frequency,cutoff_frequency,1)
        LON_filt = mvp.remove_surface_waves(Lon_T_interp,Time_T_interp*24*3600,sampling_frequency,cutoff_frequency,1)

        self.TIME0 = mvp.vertical_interp(STime_T_interp, Time_T_interp, Time)
        self.PRES0 = mvp.vertical_interp(STime_T_interp, PRES_filt, Time)
        self.TEMP0 = mvp.vertical_interp(STime_T_interp, TEMP_filt, Time)
        self.COND0 = mvp.vertical_interp(STime_T_interp, COND_filt, Time)
        self.LON0 = mvp.vertical_interp(STime_T_interp, LON_filt, Time)
        self.LAT0 = mvp.vertical_interp(STime_T_interp, LAT_filt, Time)
        self.SALT0 = gsw.SP_from_C(self.COND0, self.TEMP0, self.PRES0)

        if correction:
            self.PRES_mvp_corr = self.PRES0
            self.TEMP_mvp_corr = self.TEMP0
            self.COND_mvp_corr = self.COND0
            self.LON_mvp_corr = self.LON0
            self.LAT_mvp_corr = self.LAT0
            self.SALT_mvp_corr = self.SALT0
            self.TIME_mvp_corr = self.TIME0

    def temporal_lag_correction(self,correction=False):
        """
        Apply temporal lag correction to MVP profiles (T/C alignment).
        Args:
            correction (bool): If True, update corrected attributes.
        """
    
        # Compute distance between profiles

        D = np.zeros_like(self.LON0)
        speed = 1 * 0.514 # knot to m/s
        step = speed/self.freq_echant
        for i in range(self.LON0.shape[0]):
            D[i,:] = np.arange(0,self.LON0.shape[1])*step

        # R = 6373.0

        # lat1 = np.radians(0)
        # lon1 = np.radians(0)
        # lat2 = np.radians(self.LAT_mvp)
        # lon2 = np.radians(self.LON_mvp)

        # dlon = lon2 - lon1
        # dlat = lat2 - lat1

        # a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Dist = R * c * 1e3

        # D = np.zeros((Dist.shape[0], Dist.shape[1]))
        # for i in range(Dist.shape[0]):
        #     D_diff = np.zeros(Dist.shape[1])
        #     D_diff[1::] = np.abs(np.diff(Dist[i,:]))
        #     D[i,:] = np.cumsum(D_diff)
        # del D_diff, R, c, a, dlat, dlon, lon1, lat1, lon2, lat2
        # for i in range(D.shape[0]-1):
        #     D[i+1,:] = D[i+1,:]+np.nanmax(D[i,:])

        self.D0 = D

        SAMP_TIME0 = np.zeros((self.TIME0.shape[0],self.TIME0.shape[1]))
        SAMP_TIME0[:] = np.nan
        for i in range(self.TIME0.shape[0]):
            SAMP_TIME0[i,:] = (self.TIME0[i,:]-np.nanmin(self.TIME0[i,:]))*24*3600

        Time = np.arange(0,np.nanmax(SAMP_TIME0), 1/self.freq_echant)

        # Pr_T_interp = mvp.median(SAMP_TIME0, self.PRES0, Time)
        # T0_T_interp = mvp.median(SAMP_TIME0, self.TEMP0, Time)
        # C0_T_interp = mvp.median(SAMP_TIME0, self.COND0, Time)
        Pr_T_interp = self.PRES0
        T0_T_interp = self.TEMP0
        C0_T_interp = self.COND0

        Samp_time_T_interp = mvp.median(SAMP_TIME0, SAMP_TIME0, Time)
        # Choose direction to be corrected
        sens_corr = 'down'
        bnds = ((0, 0.2), (0.02, 0.05))
        coeff = (0.15, 0.032)

        tau0_down, tauS_down = tlc.facteur_corrections_lag(T0_T_interp, C0_T_interp, Pr_T_interp, Samp_time_T_interp, self.DIR, sens_corr,coeff,bnds,0,1000)
        # print(f"Tau0 down: {tau0_down}, TauS down: {tauS_down}")

        TEMP_align, COND_align = tlc.merge_corrections_lag(self.TEMP0, self.COND0, SAMP_TIME0, self.PRES0, self.D0, tau0_down, tauS_down, self.DIR, sens_corr)
        self.COND_align = self.COND0

        # T0_T_interp = mvp.median(SAMP_TIME0, TEMP_align, Time)
        # C0_T_interp = mvp.median(SAMP_TIME0, COND_align, Time)
        # print(np.sum(np.isnan(T0_T_interp)),'T0_T_interp2',T0_T_interp.shape)

        sens_corr = 'up'

        coeff = (0.135, 0)
        bnds = ((-1, 1), (-2, 2))


        tau0_up, tauS_up = tlc.facteur_corrections_lag(T0_T_interp, C0_T_interp, Pr_T_interp, Samp_time_T_interp, self.DIR, sens_corr,coeff,bnds,0,1000)
        # print(f"Tau0 up: {tau0_up}, TauS up: {tauS_up}")

        self.TEMP_align, COND_align = tlc.merge_corrections_lag(TEMP_align, COND_align, SAMP_TIME0 ,self.PRES0, self.D0, tau0_up, tauS_up, self.DIR, sens_corr)
        self.COND_align[:] = self.COND0[:]
        self.SALT_align = gsw.SP_from_C(self.COND_align, self.TEMP_align, self.PRES0)


        if correction:
            self.TEMP_mvp_corr = self.TEMP_align
            self.COND_mvp_corr = self.COND_align
            self.SALT_mvp_corr = self.SALT_align  
            self.PRES_mvp_corr = self.PRES0
            self.TIME_mvp_corr = self.TIME0
            self.LON_mvp_corr = self.LON0
            self.LAT_mvp_corr = self.LAT0


    def thermal_mass_correction(self,correction=False,save_param=False,load_param=None,sens_corr='down',var_corr='cond',max_depth = 1000):
        """
        Apply thermal mass correction to MVP temperature and conductivity profiles.
        Args:
            correction (bool): If True, update corrected attributes.
        """

        SAMP_TIME0 = np.zeros((self.TIME0.shape[0],self.TIME0.shape[1]))
        SAMP_TIME0[:] = np.nan
        for i in range(self.TIME0.shape[0]):
            SAMP_TIME0[i,:] = (self.TIME0[i,:]-np.nanmin(self.TIME0[i,:]))*24*3600
            
        Pres = np.arange(0, max_depth,0.25)

        Pr_T_interp = mvp.median(self.PRES0, self.PRES0, Pres)
        Samp_time_T_interp = mvp.median(self.PRES0, SAMP_TIME0, Pres)
        Lon_T_interp = mvp.median(self.PRES0, self.LON0, Pres)
        Lat_T_interp = mvp.median(self.PRES0, self.LAT0, Pres)
        T_align_T_interp = mvp.median(self.PRES0, self.TEMP_align, Pres)
        C_align_T_interp = mvp.median(self.PRES0, self.COND_align, Pres)


        sampling_frequency = 4
        cutoff_frequency = 1/4
        T_filt = mvp.remove_surface_waves(T_align_T_interp,Pr_T_interp,sampling_frequency,cutoff_frequency,1)
        C_filt = mvp.remove_surface_waves(C_align_T_interp,Pr_T_interp,sampling_frequency,cutoff_frequency,1)
        P_filt = mvp.remove_surface_waves(Pr_T_interp,Pr_T_interp,sampling_frequency,cutoff_frequency,1)
        S_filt = gsw.SP_from_C(C_filt, T_filt, P_filt)


        if var_corr == 'sal':
            Gamma,T_gamma,C_gamma = tmc.gamma_S(P_filt,T_filt,S_filt, self.DIR,sens_corr,max_depth)
        else:
            Gamma,T_gamma,C_gamma = tmc.gamma_C(P_filt,T_filt,C_filt, self.DIR,sens_corr,max_depth)


        # print("Number of non-NaN values in Gamma:", np.sum(~np.isnan(Gamma)))
        # print("total number of values in Gamma:", Gamma.shape[0]*Gamma.shape[1])

        Pres = np.arange(0, max_depth,0.5)

        self.Pr_T_interp = mvp.median(self.PRES0, self.PRES0, Pres)
        self.Samp_time_T_interp = mvp.median(self.PRES0, SAMP_TIME0, Pres)
        self.Lon_T_interp = mvp.median(self.PRES0, self.LON0, Pres)
        self.Lat_T_interp = mvp.median(self.PRES0, self.LAT0, Pres)
        self.T_align_T_interp = mvp.median(self.PRES0, self.TEMP_align, Pres)
        self.C_align_T_interp = mvp.median(self.PRES0, self.COND_align, Pres)   


        if load_param == None:
            # coeff = (0.043, 1.37, -0.26, 1.53)
            # bnds = ((-0.5, 0.5), (0, 3), (-1, 1), (0, 3.5))
            coeff = (0.0, 0.0, 4)
            # bnds = ((-3, 3), (-1, 1), (-20, 20))
            bnds = ((-0.5, 0.5), (-0.1, 0.1), (-20, 20))

            alphat_apres,alphac_apres,tau_apres,alphat_avant,alphac_avant,tau_avant = tmc.facteur_corrections_TC(T_align_T_interp, C_align_T_interp,Samp_time_T_interp,\
                                                                                            Pr_T_interp, Lon_T_interp, Lat_T_interp,Gamma,T_gamma, C_gamma,\
                                                                                            self.DIR, sens_corr,var_corr,coeff,bnds,max_depth)

            print(f"alphat_apres: {alphat_apres}, \n alphac_apres: {alphac_apres}, \n tau_apres: {tau_apres}")
            print(f"alphat_avant: {alphat_avant}, \n alphac_avant: {alphac_avant}, \n tau_avant: {tau_avant}")
            if save_param:
                tab = np.zeros((6,len(alphat_apres)))
                tab[0,:] = alphat_apres
                tab[1,:] = alphac_apres
                tab[2,:] = tau_apres
                tab[3,:] = alphat_avant
                tab[4,:] = alphac_avant
                tab[5,:] = tau_avant
                np.save('param_slow.npy',tab)
        else:
            tab = np.load(load_param)
            alphat_apres = tab[0,:]
            alphac_apres = tab[1,:]
            tau_apres = tab[2,:]
            alphat_avant = tab[3,:]
            alphac_avant = tab[4,:]
            tau_avant = tab[5,:]


        self.TEMP_final,self.COND_final = tmc.merge_corrections_TC(self.TEMP_align, self.COND_align, SAMP_TIME0, self.PRES0, self.D0, Gamma, T_gamma,C_gamma,\
                                                        alphat_apres, alphac_apres,tau_apres, \
                                                        alphat_avant, alphac_avant, tau_avant ,self.DIR, sens_corr,var_corr)
        self.SALT_final = gsw.SP_from_C(self.COND_final, self.TEMP_final, self.PRES0)

        if correction:
            self.TEMP_mvp_corr = self.TEMP_final
            self.COND_mvp_corr = self.COND_final
            self.SALT_mvp_corr = self.SALT_final
            self.PRES_mvp_corr = self.PRES0
            self.TIME_mvp_corr = self.TIME0
            self.LON_mvp_corr = self.LON0
            self.LAT_mvp_corr = self.LAT0


    def plot_correction(self,id_mvp,id_ctd=None):
        """
        Plot the evolution of temperature and salinity profiles at each correction step for a given profile.
        Args:
            id_mvp (int): Index of the MVP profile to plot.
            id_ctd (int, optional): Index of the CTD profile to plot (default: same as id_mvp).
        """

        if id_ctd is None:
            id_ctd = id_mvp
        
        plt.figure()
        plt.plot(self.TEMP_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP raw down',color='blue', alpha=0.3)
        # plt.plot(self.TEMP_mvp[1],self.PRES_mvp[1],label='MVP raw up',color='orange', alpha=0.3)
        plt.plot(self.TEMP_mvp_filt[id_mvp],self.PRES_mvp[id_mvp],label='MVP filt(VH) down',color='blue', linestyle='--')
        # plt.plot(self.TEMP_mvp_filt[1],self.PRES_mvp[1],label='MVP filt(VH) up',color='orange', linestyle='--')
        plt.plot(self.TEMP0[id_mvp],self.PRES0[id_mvp],label='MVP surf corr down',color='blue', linestyle='-.')
        # plt.plot(self.TEMP0[1],self.PRES0[1],label='MVP surf corr up',color='orange', linestyle='-.')
        plt.plot(self.TEMP_align[id_mvp],self.PRES0[id_mvp],label='MVP lag corr down',color='blue', linestyle=':')
        # plt.plot(self.TEMP_align[1],self.PRES0[1],label='MVP lag corr up',color='orange', linestyle=':')
        # plt.plot(self.TEMP_final[id_mvp],self.PRES0[id_mvp],label='MVP final down',color='blue')
        # plt.plot(self.TEMP_final[1],self.PRES0[1],label='MVP final up',color='orange')
        if self.ctd:
            plt.plot(self.TEMP_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down',color='green')
            plt.plot(self.TEMP_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up',color='green')
        plt.legend()    
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Temperature, C')    
        plt.ylabel('Pressure, dbar')

        plt.figure()
        plt.plot(self.SALT_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP raw down',color='blue', alpha=0.3)
        # plt.plot(self.SALT_mvp[1],self.PRES_mvp[1],label='MVP raw up',color='orange', alpha=0.3)      
        plt.plot(self.SALT_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP filt(VH) down',color='blue', linestyle='--')
        # plt.plot(self.SALT_mvp[1],self.PRES_mvp[1],label='MVP filt(VH) up',color='orange', linestyle='--')
        plt.plot(self.SALT0[id_mvp],self.PRES0[id_mvp],label='MVP surf corr down',color='blue', linestyle='-.')
        # plt.plot(self.SALT0[1],self.PRES0[1],label='MVP surf corr up',color='orange', linestyle='-.')
        plt.plot(self.SALT_align[id_mvp],self.PRES0[id_mvp],label='MVP lag corr down',color='blue', linestyle=':')
        # plt.plot(self.SAL_align[1],self.PRES0[1],label='MVP lag corr up',color='orange', linestyle=':')
        plt.plot(self.SALT_final[id_mvp],self.PRES0[id_mvp],label='MVP final down',color='blue')
        # plt.plot(self.SAL_final[1],self.PRES0[1],label='MVP final up',color='orange')
        if self.ctd:
            plt.plot(self.SALT_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down',color='green')
            plt.plot(self.SALT_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up',color='green')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Salinity, psu')
        plt.ylabel('Pressure, dbar')


    def interpolate_CTD_and_MVPcorrected(self,length):

        """
        Interpolate CTD data onto the corrected MVP pressure levels.
        """
        if not self.ctd:
            raise ValueError("CTD data not loaded.")

        if not hasattr(self, 'PRES_mvp_corr'):
            raise ValueError("Corrected MVP data not available. Apply corrections first.")

        self.TEMP_ctd_on_mvp = np.full(self.PRES_mvp_corr.shape, np.nan)
        self.PRES_ctd_on_mvp = np.full(self.PRES_mvp_corr.shape, np.nan)
        self.COND_ctd_on_mvp = np.full(self.PRES_mvp_corr.shape, np.nan)
        self.SALT_ctd_on_mvp = np.full(self.PRES_mvp_corr.shape, np.nan)
        self.OXY_ctd_on_mvp = np.full(self.PRES_mvp_corr.shape, np.nan)

        pressure_grid = np.linspace(np.nanmin(self.PRES_mvp_corr), np.nanmax(self.PRES_mvp_corr), length)

        self.TEMP_ctd_on_mvp = mvp.vertical_interp(self.PRES_ctd, self.TEMP_ctd, pressure_grid)
        self.PRES_ctd_on_mvp = mvp.vertical_interp(self.PRES_ctd, self.PRES_ctd, pressure_grid)
        self.COND_ctd_on_mvp = mvp.vertical_interp(self.PRES_ctd, self.COND_ctd, pressure_grid)
        self.SALT_ctd_on_mvp = mvp.vertical_interp(self.PRES_ctd, self.SALT_ctd, pressure_grid)
        self.OXY_ctd_on_mvp = mvp.vertical_interp(self.PRES_ctd, self.OXY_ctd, pressure_grid)
        self.TEMP_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp_corr, self.TEMP_mvp_corr, pressure_grid)
        self.PRES_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp_corr, self.PRES_mvp_corr, pressure_grid)
        self.COND_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp_corr, self.COND_mvp_corr, pressure_grid)
        self.SALT_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp_corr, self.SALT_mvp_corr, pressure_grid)
        self.SPEED_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp_corr, self.SPEED_mvp, pressure_grid)
        self.TIME_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp_corr, self.TIME_mvp_corr, pressure_grid)

        print('CTD data interpolated onto corrected MVP pressure levels.')


    def to_netcdf(self, filepath=None, corrected=False, compression=True, engine=None, per_profile_files=False):
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

        # Include corrected arrays if requested and present
        if corrected:
            def add_corr(name, attr, units=None, long_name=None):
                if hasattr(self, attr):
                    data_vars[name] = (
                        ('profile', 'sample'), getattr(self, attr),
                        {k: v for k, v in [('units', units), ('long_name', long_name)] if v is not None}
                    )
            add_corr('pressure_corrected', 'PRES_mvp_corr', units='dbar', long_name='Corrected pressure')
            add_corr('temperature_corrected', 'TEMP_mvp_corr', units='degC', long_name='Corrected temperature')
            add_corr('conductivity_corrected', 'COND_mvp_corr', units='mS/cm', long_name='Corrected conductivity')
            add_corr('salinity_corrected', 'SALT_mvp_corr', units='psu', long_name='Corrected salinity')
            if hasattr(self, 'TIME_mvp_corr'):
                data_vars['time_corrected'] = (
                    ('profile', 'sample'), self.TIME_mvp_corr * 24.0 * 3600.0,
                    {
                        'units': f'seconds since {self.date_ref.strftime("%Y-%m-%d %H:%M:%S")}',
                        'long_name': 'Corrected time at sample'
                    }
                )
            if hasattr(self, 'LAT_mvp_corr'):
                add_corr('latitude_corrected', 'LAT_mvp_corr', units='degrees_north', long_name='Corrected latitude at sample')
            if hasattr(self, 'LON_mvp_corr'):
                add_corr('longitude_corrected', 'LON_mvp_corr', units='degrees_east', long_name='Corrected longitude at sample')

        # Coordinates and auxiliary per-profile variables
        coords = {
            'profile': ('profile', profile_idx),
            'sample': ('sample', sample_idx)
        }

        # Encode direction/time according to engine capabilities
        if engine in ('netcdf4', 'h5netcdf'):
            coords['direction'] = ('profile', direction.astype('U'), {'long_name': 'Profile direction'})
            coords['profile_time'] = ('profile', profile_time, {'long_name': 'Profile nominal time'})
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
            'institution': 'LMD/CNRS',
            'source': 'MVPAnalyzer',
            'history': f"Created on {datetime.now().isoformat()}",
            'mvp_Yorig': int(self.Yorig)
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