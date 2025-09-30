#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################################
#
# Collection of routines for temporal lag correction
#
#
# Function vertical_interp: Interpolate each variable
#
################################################################################

#
# Import libraries
#

import numpy as np 
import scipy.stats as st 
from datetime import date
from datetime import datetime
from scipy import interpolate
from scipy.signal import butter, freqz
from scipy import signal
import gsw
from scipy.optimize import minimize
from scipy import ndimage
from scipy.interpolate import pchip_interpolate
import similaritymeasures



#
################################################################################
#
# Function facteur_corrections_lag: select each profiles in a specific vertical directon, and return 
# the temporal lag with the previous and next profile in opposite direction
#
#   input:
#	TEMP0     : In-situ Temperature
#	PRES0     : Pressure
#	TIME        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#	DEPTH     : Depth
#	DIR     : Vertical direction of each profile 
#	sens_corr     : Chosen direction to be corrected
#
#   output:
#	tau_lag_apres   : temporal lag with the next profile in opposite direction
#	tau_lag_avant   : temporal lag with the previous profile in opposite direction
# 
# 
################################################################################
#

def facteur_corrections_lag(TEMP0, COND0, PRES0, TIME, DIR, sens_corr, coeff,bnds,depth_min,depth_max):

    if sens_corr=='up':
        ind = np.where(np.asarray(DIR)=='up')[0]
    elif sens_corr=='down':
        ind = np.where(np.asarray(DIR)=='down')[0]
    elif sens_corr=='all':
        ind = np.arange(0,len(DIR),1)

    tau0= np.zeros(len(DIR))
    tau0[:] = np.nan
    tauS = np.zeros(len(DIR))
    tauS[:] = np.nan
    for n_rad in range(len(ind)):
        tau0[ind[n_rad]], tauS[ind[n_rad]] = find_coefficients_lag(TEMP0, COND0, PRES0, TIME, DIR, ind[n_rad],coeff,bnds,depth_min,depth_max)

    return tau0, tauS

#
################################################################################
#
# Function find_coefficients_lag: calculate, for a specific profile
# the optimal temporal lag with another one, chosen as a reference, via a minimization
# of the area between two temperature versus pressure cruves
#
#   input:
#	TEMP0     : In-situ Temperature
#	PRES0     : Pressure
#	TIME        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#	DIR     : Vertical direction of each profile 
#	n_rad_corr     : profile to be corrected
#	n_rad_comp     : reference profile
#
#   output:
#	param   : optimal temporal lag
# 
# 
################################################################################
#

def find_coefficients_lag(TEMP0, COND0, PRES0, TIME, DIR, n_rad_corr,coeff,bnds,depth_min,depth_max):
    if DIR[n_rad_corr]=='up':
        ind = np.where(np.isnan(PRES0[n_rad_corr,:])==0)[0]
        TEMP_raw = np.flip(TEMP0[n_rad_corr,ind])
        COND_raw = np.flip(COND0[n_rad_corr,ind])
        TIME_raw = np.flip(TIME[n_rad_corr,ind]-np.nanmin(TIME[n_rad_corr,ind]))
        PRES_raw = np.flip(PRES0[n_rad_corr,ind])
        del ind
        
    elif DIR[n_rad_corr]=='down':
        ind = np.where(np.isnan(PRES0[n_rad_corr,:])==0)[0]
        TEMP_raw = TEMP0[n_rad_corr,ind]
        COND_raw = COND0[n_rad_corr,ind]
        TIME_raw = TIME[n_rad_corr,ind]-np.nanmin(TIME[n_rad_corr,ind])
        PRES_raw = PRES0[n_rad_corr,ind]
        del ind

    w = minimize(Aire_profondeur, coeff,args=(TIME_raw, TEMP_raw, COND_raw, PRES_raw, depth_min, depth_max), method='SLSQP', bounds=bnds, options={'eps': 1e-6})
    param = w.x
    return param

#
################################################################################
#
# Function Aire_profondeur: calculate the area between two curves (temperature vs pressure) with changing temporal lag as an input
#
#   input:
#	params     : temporal lag
#	Aire_ref     : Area of reference (without any correction)
#	TIME_raw     : Temperature of first profile
#	TEMP_raw     : Temperature of first profile
#	PRES_raw     : Pressure of first profile
#	PRES_comp    : Pressure of second profile
#	TEMP_comp    : Temperature of second profile
#
#   output:
#	Aire   : Area between the two curves, normalize by the area of reference
# 
# 
################################################################################
#

def Aire_profondeur(params, TIME_raw, TEMP_raw, COND_raw, PRES_raw, depth_min, depth_max):
    T_corr = correct_lag(TIME_raw, TEMP_raw, PRES_raw, params)
    C_corr = correct_lag(TIME_raw, COND_raw, PRES_raw, -params)
    
    SALT_corr = gsw.SP_from_C(COND_raw,T_corr,PRES_raw)
    SALT_corr2 = gsw.SP_from_C(C_corr,TEMP_raw,PRES_raw)
        
    Aire = np.nanstd(np.abs(np.diff(SALT_corr)))

    return Aire


#
################################################################################
#
# Function correct_lag: shift the temperature profiles by a temporal lag
#
#   input:
#	Time_raw  : Time
#	T_raw     : Temperature profile
#	params    : temporal lag
#
#   output:
#	T0_corr   : Shifted temperature profile
# 
# 
################################################################################
#
    
def correct_lag(Time_raw, T_raw, Pres_raw, params):
    #tau = params
    valid = np.where((Time_raw >= 0) & (np.isnan(T_raw)==0))[0]

    Time_valid = np.zeros(len(valid))
    Time_valid[:] = np.nan
    T0_valid = np.zeros(len(valid))
    T0_valid[:] = np.nan
    P_valid = np.zeros(len(valid))
    P_valid[:] = np.nan
    V = np.zeros(len(valid))
    V[:] = np.nan

    Time_valid[:] = Time_raw[valid]
    T0_valid[:] = T_raw[valid]
    P_valid[:] = Pres_raw[valid]
    V[0:-1] = np.abs(np.diff(P_valid)/np.diff(Time_valid))

    f1 = interpolate.interp1d(Time_valid, T0_valid,'linear',fill_value="extrapolate")
    T0_corr = np.zeros(len(T_raw))
    T0_corr[:] = np.nan
    tau = np.zeros(len(T_raw))
    tau[:] = np.nan
    tau[valid] = params[0]
    tau[valid[np.where(V>0.1)[0]]] = params[0] + params[1]/(V[np.where(V>0.1)[0]])

    T0_corr[valid] = f1(Time_valid + tau[valid])
    return T0_corr


#
################################################################################
#
# Function merge_corrections_lag: apply the temporal lag to the temperature and conductivity profiles
#
#
#   input:
#	TEMP0     : In-situ Temperature
#	COND0     : Conductivity
#	DEPTH0     : Pressure
#	D        : Distance since beginning
#	tau_lag_apres        : temporal lag with the next profile
#	tau_lag_avant        : temporal lag with the previous profile
#	DIR     : Vertical direction of each profile 
#	sens_corr     : Chosen direction to be corrected
#
#   output:
#	T_final   : Corrected temperature profile
#	C_final   : Corrected conductivity profile
# 
# 
################################################################################
#
   

def merge_corrections_lag(TEMP0, COND0, TIME, DEPTH0, D, tau0, tauS, DIR, sens_corr):

    if sens_corr=='up':
        ind_sens = np.where(np.asarray(DIR)=='up')[0]
    elif sens_corr=='down':
        ind_sens = np.where(np.asarray(DIR)=='down')[0]
    elif sens_corr=='all':
        ind_sens = np.arange(0,len(DIR),1)

    T_final = np.zeros((TEMP0.shape[0],TEMP0.shape[1]))
    T_final[:] = np.nan
    C_final = np.zeros((COND0.shape[0],COND0.shape[1]))
    C_final[:] = np.nan
    T_final[:] = TEMP0
    C_final[:] = COND0
    for n_rad_corr in range(len(ind_sens)):
        ind = np.where(np.isnan(TEMP0[ind_sens[n_rad_corr],:])==0)[0]
        T_raw = TEMP0[ind_sens[n_rad_corr],ind]
        C_raw = COND0[ind_sens[n_rad_corr],ind]
        Time_raw = (TIME[ind_sens[n_rad_corr],ind]-np.nanmin(TIME[ind_sens[n_rad_corr],ind]))
        Dep_raw = DEPTH0[ind_sens[n_rad_corr],ind]
        D_raw = D[ind_sens[n_rad_corr],ind]
        
        D_avant = np.zeros(len(Dep_raw))
        D_avant[:] = np.nan
        D_apres = np.zeros(len(Dep_raw))
        D_apres[:] = np.nan

        if ind_sens[n_rad_corr]>0:
            coeff = np.zeros(2)
            coeff[0] = tau0[ind_sens[n_rad_corr]]
            coeff[1] = tauS[ind_sens[n_rad_corr]]
            T_corr_avant = np.zeros(len(Dep_raw))
            T_corr_avant[:] = np.nan
            C_corr_avant = np.zeros(len(Dep_raw))
            C_corr_avant[:] = np.nan
            if DIR[ind_sens[n_rad_corr]]=='down':
                T_corr_avant = correct_lag(Time_raw, T_raw, Dep_raw, coeff)
                C_corr_avant = correct_lag(Time_raw, C_raw, Dep_raw, -coeff)
            elif DIR[ind_sens[n_rad_corr]]=='up':
                T_corr_avant = correct_lag(np.flip(Time_raw), np.flip(T_raw), np.flip(Dep_raw), coeff)
                T_corr_avant = np.flip(T_corr_avant)
                C_corr_avant = correct_lag(np.flip(Time_raw), np.flip(C_raw),np.flip(Dep_raw), -coeff)
                C_corr_avant = np.flip(C_corr_avant)
            del coeff

            for i_L in range(len(Dep_raw)):
                if ind_sens[n_rad_corr]>0:
                    if DIR[ind_sens[n_rad_corr]]=='down':
                        if len(np.where(DEPTH0[ind_sens[n_rad_corr]-1,:]<=Dep_raw[i_L])[0])>0:
                            D_avant[i_L] = D_raw[i_L]-D[ind_sens[n_rad_corr]-1,np.where(DEPTH0[ind_sens[n_rad_corr]-1,:]<=Dep_raw[i_L])[0][0]]
                        else:
                            D_avant[i_L] = np.nan
                    elif DIR[ind_sens[n_rad_corr]]=='up':
                        if len(np.where(DEPTH0[ind_sens[n_rad_corr]-1,:]>=Dep_raw[i_L])[0])>0:
                            D_avant[i_L] = D_raw[i_L]-D[ind_sens[n_rad_corr]-1,np.where(DEPTH0[ind_sens[n_rad_corr]-1,:]>=Dep_raw[i_L])[0][0]]
                        else:
                            D_avant[i_L] = np.nan
            del i_L

        if ind_sens[n_rad_corr]<len(DIR)-1:
            coeff = np.zeros(2)
            coeff[0] = tau0[ind_sens[n_rad_corr]]
            coeff[1] = tauS[ind_sens[n_rad_corr]]
            T_corr_apres = np.zeros(len(Dep_raw))
            T_corr_apres[:] = np.nan
            C_corr_apres = np.zeros(len(Dep_raw))
            C_corr_apres[:] = np.nan
            if DIR[ind_sens[n_rad_corr]]=='down':
                T_corr_apres = correct_lag(Time_raw, T_raw, Dep_raw, coeff)
                C_corr_apres = correct_lag(Time_raw, C_raw, Dep_raw, -coeff)
                #T_corr_apres = correct_lag(Time_raw, T_raw, coeff)
                #C_corr_apres = correct_lag(Time_raw, C_raw, -coeff)
            elif DIR[ind_sens[n_rad_corr]]=='up':
                T_corr_apres = correct_lag(np.flip(Time_raw), np.flip(T_raw), np.flip(Dep_raw), coeff)
                T_corr_apres = np.flip(T_corr_apres)
                C_corr_apres = correct_lag(np.flip(Time_raw), np.flip(C_raw), np.flip(Dep_raw), -coeff)
                C_corr_apres = np.flip(C_corr_apres)
                #T_corr_apres = correct_lag(np.flip(Time_raw), np.flip(T_raw), coeff)
                #T_corr_apres = np.flip(T_corr_apres)
                #C_corr_apres = correct_lag(np.flip(Time_raw), np.flip(C_raw), -coeff)
                #C_corr_apres = np.flip(C_corr_apres)
            del coeff

            for i_L in range(len(Dep_raw)-1):
                if DIR[ind_sens[n_rad_corr]]=='down':
                    if len(np.where(DEPTH0[ind_sens[n_rad_corr]+1,:]<=Dep_raw[i_L])[0])>0:
                        D_apres[i_L] = D[ind_sens[n_rad_corr]+1,np.where(DEPTH0[ind_sens[n_rad_corr]+1,:]<=Dep_raw[i_L])[0][0]]-D_raw[i_L]
                    else:
                        D_apres[i_L] = np.nan
                elif DIR[ind_sens[n_rad_corr]]=='up':
                    if len(np.where(DEPTH0[ind_sens[n_rad_corr]+1,:]>=Dep_raw[i_L])[0])>0:
                        D_apres[i_L] = D[ind_sens[n_rad_corr]+1,np.where(DEPTH0[ind_sens[n_rad_corr]+1,:]>=Dep_raw[i_L])[0][0]]-D_raw[i_L]
                    else:
                        D_apres[i_L] = np.nan
            del i_L

        if (ind_sens[n_rad_corr]>0) & (ind_sens[n_rad_corr]<len(DIR)-1):
            M = np.nanmax([np.nanmax(D_avant) , np.nanmax(D_apres)])
            D_avant = M-D_avant
            D_apres = M-D_apres
            del M

        
        T_corr = np.zeros(len(T_raw))
        T_corr[:] = np.nan
        C_corr = np.zeros(len(T_raw))
        C_corr[:] = np.nan
        for i_L in range(len(T_raw)):
            if (ind_sens[n_rad_corr]>0) & (ind_sens[n_rad_corr]<len(DIR)-1):
                if (np.isnan(D_apres[i_L])==1) & (np.isnan(D_avant[i_L])==0):
                    T_corr[i_L] = T_corr_avant[i_L]
                    C_corr[i_L] = C_corr_avant[i_L]
                elif (np.isnan(D_apres[i_L])==0) & (np.isnan(D_avant[i_L])==1):
                    T_corr[i_L] = T_corr_apres[i_L]
                    C_corr[i_L] = C_corr_apres[i_L]
                else:
                    T_corr[i_L] = (T_corr_apres[i_L]*(D_apres[i_L])+T_corr_avant[i_L]*(D_avant[i_L]))/((D_apres[i_L])+(D_avant[i_L]))
                    C_corr[i_L] = (C_corr_apres[i_L]*(D_apres[i_L])+C_corr_avant[i_L]*(D_avant[i_L]))/((D_apres[i_L])+(D_avant[i_L]))
            elif ind_sens[n_rad_corr]==0:
                T_corr[i_L] = T_corr_apres[i_L]
                C_corr[i_L] = C_corr_apres[i_L]
            elif ind_sens[n_rad_corr]>=len(DIR)-1:
                T_corr[i_L] = T_corr_avant[i_L]
                C_corr[i_L] = C_corr_avant[i_L]
        del i_L
        
        T_final[ind_sens[n_rad_corr],ind] = T_corr
        C_final[ind_sens[n_rad_corr],ind] = C_corr
        del ind, T_corr, C_corr, T_raw, C_raw, Time_raw, Dep_raw, D_raw, D_apres, D_avant

    return T_final,C_final

