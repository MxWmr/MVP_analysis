################################################################################
#
# Collection of routines for thermal mass correction
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
from fastdtw import fastdtw


#
################################################################################
#
# Function gamma_C: calculate the coefficient of sensitivity of conductivity to temperature at fixed pressure
#
#   input:
#	PRES0     : Pressure
#	TEMP0     : In-situ Temperature
#	COND0     : Conductivity
#	DIR     : Vertical direction of each profile 
#	sens_corr     : Chosen direction to be used as reference
#
#   output:
#	Gamma_C_filt   : coefficient of sensitivity
#	Tem   : range of temperature for the coefficient
#	Con   : range of conductivity for the coefficient
# 
################################################################################
#

def gamma_C(PRES0, TEMP0, COND0, DIR, sens_corr,max_depth):



    # mask = PRES0 <= max_depth
    # TEMP0 = np.where(mask, TEMP0, np.nan)
    # PRES0 = np.where(mask, PRES0, np.nan)
    # COND0 = np.where(mask, COND0, np.nan)

    if sens_corr=='up':
        ind = np.where(np.asarray(DIR)=='up')[0]
    elif sens_corr=='down':
        ind = np.where(np.asarray(DIR)=='down')[0]
    elif sens_corr=='all':
        ind = np.arange(0,len(DIR),1)

    Pr = np.reshape(PRES0[ind,:],len(ind)*PRES0.shape[1])
    T0 = np.reshape(TEMP0[ind,:],len(ind)*TEMP0.shape[1])
    C0 = np.reshape(COND0[ind,:],len(ind)*COND0.shape[1])
    del ind

    ind = np.where((np.isnan(Pr)==0) | (np.isnan(T0)==0) | (np.isnan(C0)==0))[0]
    Pr = Pr[ind]
    T0 = T0[ind]
    C0 = C0[ind]
    del ind

    sigma = np.where(np.abs(Pr-Pr[0])>=0.5)[0][0]
    sz = 2*np.where(np.abs(Pr-Pr[0])>=0.5)[0][0]    # length of gaussFilter vector
    x = np.linspace(-sz / 2, sz / 2, sz)
    gaussFilter = np.exp(-(x*x)/(2*sigma*sigma))
    gaussFilter = gaussFilter/np.sum(gaussFilter)
    T_filt = np.convolve(T0, gaussFilter,mode='same')
    C_filt = np.convolve(C0, gaussFilter,mode='same')
    P_filt = np.convolve(Pr, gaussFilter,mode='same')

    pas_p = 5 #10*np.nanmedian(np.abs(np.diff(Pr)))
    pas_t = (np.nanmax(T_filt) - np.nanmin(T_filt)) / 25
    pas_c = (np.nanmax(C_filt) - np.nanmin(C_filt)) / 25

    Pre = np.arange(np.floor(np.nanmin(Pr)), np.ceil(np.nanmax(Pr)), pas_p)
    Tem = np.arange(np.floor(np.nanmin(T0)), np.ceil(np.nanmax(T0)), pas_t)
    Con = np.arange(np.floor(np.nanmin(C0)), np.ceil(np.nanmax(C0)), pas_c)
    gamma_c = np.zeros((len(Tem), len(Con)))
    N = np.zeros((len(Tem), len(Con)))

    for i_p in range(len(Pre)):
        ind_both = np.where((P_filt>=Pre[i_p]) & (P_filt<Pre[i_p]+pas_p) & (np.isnan(T_filt)==0))[0]
        if len(ind_both)>=1:
            coefs = np.polyfit(T_filt[ind_both], C_filt[ind_both], 0)
            if np.isnan(coefs[0])==0:
                C_temp = np.unique(np.round(C_filt[ind_both]/pas_c)*pas_c)
                T_temp = np.unique(np.round(T_filt[ind_both]/pas_t)*pas_t)
                for i_c in range(len(C_temp)):
                    ind_C = np.where((C_temp[i_c]>=Con) & (C_temp[i_c]<Con+pas_c))[0]
                    # ind_C = np.where(np.abs(Con - C_temp[i_c]) < pas_c/2)[0]
                    for i_t in range(len(T_temp)):
                        ind_T = np.where((T_temp[i_t]>=Tem) & (T_temp[i_t]<Tem+pas_t))[0]
                        # ind_T = np.where(np.abs(Tem - T_temp[i_t]) < pas_t/2)[0]
                        gamma_c[ind_T,ind_C] = gamma_c[ind_T,ind_C]+coefs[0]
                        N[ind_T,ind_C] = N[ind_T,ind_C]+1
                        del ind_T
                    del i_t, ind_C
                del i_c, C_temp, T_temp
            del coefs
        del ind_both
    del i_p

    Gamma_C = np.zeros((gamma_c.shape[0],gamma_c.shape[1]))
    Gamma_C[:] = np.nan
    Gamma_C = gamma_c/N
    Gamma_C[N<=2] = np.nan


    sigma_x = np.where(np.abs(Tem-Tem[0])>=2*pas_t)[0][0]
    sxz = 5*np.where(np.abs(Tem-Tem[0])>=2*pas_t)[0][0]
    x = np.linspace(-sxz / 2, sxz / 2, sxz)
    sigma_y = np.where(np.abs(Con-Con[0])>=2*pas_c)[0][0]
    syz = 5*np.where(np.abs(Con-Con[0])>=2*pas_c)[0][0]
    y = np.linspace(-syz / 2, syz / 2, syz)
    [X,Y] = np.meshgrid(x,y)
    gaussFilter = np.exp(-((X*X)/(2*sigma_x*sigma_x))-((Y*Y)/(2*sigma_y*sigma_y)))
    gaussFilter = gaussFilter/np.nansum(gaussFilter)

    Gamma_C_filt = np.zeros((Gamma_C.shape[0], Gamma_C.shape[1]))
    Gamma_C_filt[:] = np.nan
    # Gamma_C_filt = ndimage.convolve(Gamma_C, gaussFilter,mode='constant')
    Gamma_C_filt = nan_convolve2d(Gamma_C, gaussFilter)

    return Gamma_C_filt,Tem,Con


def nan_convolve2d(mat, kernel):
    mask = np.isnan(mat)
    mat_filled = np.where(mask, 0, mat)
    valid = (~mask).astype(float)
    conv = ndimage.convolve(mat_filled, kernel, mode='constant', cval=0.0)
    norm = ndimage.convolve(valid, kernel, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = conv / norm
    result[norm == 0] = np.nan
    return result

#
################################################################################
#
# Function gamma_S: calculate the coefficient of sensitivity of salinity to temperature at fixed pressure
#
#   input:
#	PRES0     : Pressure
#	TEMP0     : In-situ Temperature
#	SAL0     : Practical Salinity
#	DIR     : Vertical direction of each profile 
#	sens_corr     : Chosen direction to be used as reference
#
#   output:
#	Gamma_S_filt   : coefficient of sensitivity
#	Tem   : range of temperature for the coefficient
#	Sal   : range of salinity for the coefficient
# 
################################################################################
#

def gamma_S(PRES0, TEMP0, SAL0, DIR, sens_corr,max_depth):

    mask = PRES0 <= max_depth
    TEMP0 = np.where(mask, TEMP0, np.nan)
    PRES0 = np.where(mask, PRES0, np.nan)
    SAL0 = np.where(mask, SAL0, np.nan)
    

    if sens_corr=='up':
        ind = np.where(np.asarray(DIR)=='up')[0]
    elif sens_corr=='down':
        ind = np.where(np.asarray(DIR)=='down')[0]
    elif sens_corr=='all':
        ind = np.arange(0,len(DIR),1)

    Pr = np.reshape(PRES0[ind,:],len(ind)*PRES0.shape[1])
    T0 = np.reshape(TEMP0[ind,:],len(ind)*TEMP0.shape[1])
    S0 = np.reshape(SAL0[ind,:],len(ind)*SAL0.shape[1])
    del ind

    ind = np.where((np.isnan(Pr)==0) | (np.isnan(T0)==0) | (np.isnan(S0)==0))[0]
    Pr = Pr[ind]
    T0 = T0[ind]
    S0 = S0[ind]
    del ind

    sigma = np.where(np.abs(Pr-Pr[0])>=0.5)[0][0]
    sz = 2*np.where(np.abs(Pr-Pr[0])>=0.5)[0][0]    # length of gaussFilter vector
    x = np.linspace(-sz / 2, sz / 2, sz)
    gaussFilter = np.exp(-(x*x)/(2*sigma*sigma))
    gaussFilter = gaussFilter/np.sum(gaussFilter)
    T_filt = np.convolve(T0, gaussFilter,mode='same')
    P_filt = np.convolve(Pr, gaussFilter,mode='same')
    S_filt = np.convolve(S0, gaussFilter,mode='same')

    pas_p = 0.25#10*np.nanmedian(np.abs(np.diff(Pr)))
    pas_t = 0.1#10*np.nanmedian(np.abs(np.diff(T0)))
    pas_s = 0.01#10*np.nanmedian(np.abs(np.diff(S0)))
    Pre = np.arange(np.floor(np.nanmin(Pr)), np.ceil(np.nanmax(Pr)), pas_p)
    Tem = np.arange(np.floor(np.nanmin(T0)), np.ceil(np.nanmax(T0)), pas_t)
    Sal = np.arange(np.floor(np.nanmin(S0)), np.ceil(np.nanmax(S0)), pas_s)

    gamma_s = np.zeros((len(Tem), len(Sal)))
    N = np.zeros((len(Tem), len(Sal)))

    for i_p in range(len(Pre)):
        ind_both = np.where((P_filt>=Pre[i_p]) & (P_filt<Pre[i_p]+pas_p) & (np.isnan(T_filt)==0))[0]
        if len(ind_both)>=1:
            coefs = np.polyfit(T_filt[ind_both], S_filt[ind_both], 1)
            if np.isnan(coefs[0])==0:
                S_temp = np.unique(np.round(S_filt[ind_both]/pas_s)*pas_s)
                T_temp = np.unique(np.round(T_filt[ind_both]/pas_t)*pas_t)
                for i_s in range(len(S_temp)):
                    ind_S = np.where((S_temp[i_s]>=Sal) & (S_temp[i_s]<Sal+pas_s))[0]
                    # ind_S = np.where(np.abs(Sal - S_temp[i_s]) < pas_s/2)[0]
                    for i_t in range(len(T_temp)):
                        ind_T = np.where((T_temp[i_t]>=Tem) & (T_temp[i_t]<Tem+pas_t))[0]
                        # ind_T = np.where(np.abs(Tem - T_temp[i_t]) < pas_t/2)[0]
                        gamma_s[ind_T,ind_S] = gamma_s[ind_T,ind_S]+coefs[0]
                        N[ind_T,ind_S] = N[ind_T,ind_S]+1
                        del ind_T
                    del i_t, ind_S
                del i_s, S_temp, T_temp
            del coefs
        del ind_both
    del i_p

    Gamma_S = np.zeros((gamma_s.shape[0],gamma_s.shape[1]))
    Gamma_S[:] = np.nan
    Gamma_S = gamma_s/N
    Gamma_S[N<=2] = np.nan

    sigma_x = np.where(np.abs(Tem-Tem[0])>=2*pas_t)[0][0]
    sxz = 5*np.where(np.abs(Tem-Tem[0])>=2*pas_t)[0][0]
    x = np.linspace(-sxz / 2, sxz / 2, sxz)
    sigma_y = np.where(np.abs(Sal-Sal[0])>=2*pas_s)[0][0]
    syz = 5*np.where(np.abs(Sal-Sal[0])>=2*pas_s)[0][0]
    y = np.linspace(-syz / 2, syz / 2, syz)
    [X,Y] = np.meshgrid(x,y)
    gaussFilter = np.exp(-((X*X)/(2*sigma_x*sigma_x))-((Y*Y)/(2*sigma_y*sigma_y)))
    gaussFilter = gaussFilter/np.nansum(gaussFilter)

    Gamma_S_filt = np.zeros((Gamma_S.shape[0], Gamma_S.shape[1]))
    Gamma_S_filt[:] = np.nan
    Gamma_S_filt = ndimage.convolve(Gamma_S, gaussFilter,mode='constant')
    
    return Gamma_S_filt,Tem,Sal



#
################################################################################
#
# Function facteur_corrections_TC: select each profiles in a specific vertical directon, and return 
# the temporal lag with the previous and next profile in opposite direction
#
#   input:
#	TEMP0     : In-situ Temperature
#	COND0     : Conductivity
#	TIME        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#	PRES0     : Pressure
#	LON0     : Longitude
#	LAT0     : Latitude
#	Gamma     : Coefficient of sensitivity
#	T_gamma     : range of temperature
#	C_gamma     : range of conducitivity or salinity depending on which coefficient of sensitivity is chosen
#	DIR     : Vertical direction of each profile 
#	sens_corr     : Chosen direction to be corrected
#	var_corr     : variable to correct either 'cond' or 'sal'
#
#   output:
#	alphat_apres   : error of amplitude for temperature (calculated with the next profile in opposite direction)
#	alphac_apres   : error of amplitude for conductivity or salinity (calculated with the next profile in opposite direction)
#	tau_apres   : relaxation time (calculated with the next profile in opposite direction)
#	alphat_avant   : error of amplitude for temperature (calculated with the previous profile in opposite direction)
#	alphac_avant   : error of amplitude for conductivity or salinity (calculated with the previous profile in opposite direction)
#	tau_avant   : relaxation time (calculated with the next profile in opposite direction)
# 
################################################################################
#


def facteur_corrections_TC(TEMP0,COND0,TIME,PRES0,LON0,LAT0,Gamma,T_gamma, C_gamma, DIR, sens_corr,var_corr,coeff,bnds,max_depth):
    


    if sens_corr=='up':
        ind = np.where(np.asarray(DIR)=='up')[0]
    elif sens_corr=='down':
        ind = np.where(np.asarray(DIR)=='down')[0]
    elif sens_corr=='all':
        ind = np.arange(0,len(DIR),1)

    tau_lag_avant = np.zeros(len(DIR))
    tau_lag_avant[:] = np.nan
    tau_lag_apres = np.zeros(len(DIR))
    tau_lag_apres[:] = np.nan
    
    alphat_apres = np.zeros(len(DIR))
    alphat_apres[:] = np.nan
    alphac_apres = np.zeros(len(DIR))
    alphac_apres[:] = np.nan
    tau_apres = np.zeros(len(DIR))
    tau_apres[:] = np.nan
    alphat_avant = np.zeros(len(DIR))
    alphat_avant[:] = np.nan
    alphac_avant = np.zeros(len(DIR))
    alphac_avant[:] = np.nan
    tau_avant = np.zeros(len(DIR))
    tau_avant[:] = np.nan
    import time
    for n_rad in range(len(ind)):
        t1 = time.time()
        if ind[n_rad]<len(DIR)-1:
            if len(np.where(np.isnan(TEMP0[ind[n_rad]+1,:])==0)[0])>0:
                alphat_apres[ind[n_rad]], alphac_apres[ind[n_rad]], tau_apres[ind[n_rad]] =\
                    find_coefficients_TC(TEMP0, COND0, TIME, PRES0, LON0, LAT0, Gamma, T_gamma, C_gamma, DIR, ind[n_rad],ind[n_rad]+1,var_corr,coeff,bnds,max_depth)
        if ind[n_rad]>0:
            if len(np.where(np.isnan(TEMP0[ind[n_rad]-1,:])==0)[0])>0:
                alphat_avant[ind[n_rad]], alphac_avant[ind[n_rad]], tau_avant[ind[n_rad]] =\
                    find_coefficients_TC(TEMP0, COND0, TIME, PRES0, LON0, LAT0, Gamma, T_gamma, C_gamma, DIR, ind[n_rad],ind[n_rad]-1,var_corr,coeff,bnds,max_depth)
        print(time.time()-t1," secondes pour le profil ", n_rad)
    del n_rad
    
    return alphat_apres,alphac_apres,tau_apres,alphat_avant,alphac_avant,tau_avant 


#
################################################################################
#
# Function find_coefficients_TC: calculate, for a specific profile
# the optimal thermal mass correction coefficients with another one, chosen as a reference, via a minimization
# of the area between two temperature versus conductivity or salinity curves
#
#   input:
#	TEMP0     : In-situ Temperature
#	COND0     : Conductivity
#	TIME        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#	PRES0     : Pressure
#	LON0     : Longitude
#	LAT0     : Latitude
#	Gamma     : Coefficient of sensitivity
#	T_gamma     : range of temperature
#	C_gamma     : range of conducitivity or salinity depending on which coefficient of sensitivity is chosen
#	DIR     : Vertical direction of each profile 
#	n_rad_corr     : profile to be corrected
#	n_rad_comp     : reference profile
#	var_corr     : variable to correct either 'cond' or 'sal'
#
#   output:
#	param   : optimal coefficients of correction
# 
# 
################################################################################
#

def find_coefficients_TC(TEMP0, COND0, TIME, PRES0, LON0, LAT0, Gamma, T_gamma, C_gamma,DIR, n_rad_corr,n_rad_comp,var_corr,coeff,bnds,max_depth):
    if DIR[n_rad_corr]=='down':
        ind = np.where(np.isnan(PRES0[n_rad_corr,:])==0)[0]
        ind = ind[PRES0[n_rad_corr,ind]<=max_depth]
        TEMP_raw = TEMP0[n_rad_corr,ind]
        COND_raw = COND0[n_rad_corr,ind]
        TIME_raw = TIME[n_rad_corr,ind]
        PRES_raw = PRES0[n_rad_corr,ind]
        LAT_raw = LAT0[n_rad_corr,ind]
        LON_raw = LON0[n_rad_corr,ind]
        del ind

        ind = np.where(np.isnan(PRES0[n_rad_comp,:])==0)[0]
        ind = ind[PRES0[n_rad_comp,ind]<=max_depth]
        TEMP_comp = np.flip(TEMP0[n_rad_comp,ind])
        COND_comp = np.flip(COND0[n_rad_comp,ind])
        TIME_comp = np.flip(TIME[n_rad_comp,ind])
        PRES_comp = np.flip(PRES0[n_rad_comp,ind])
        LAT_comp = np.flip(LAT0[n_rad_comp,ind])
        LON_comp = np.flip(LON0[n_rad_comp,ind])
        del ind

    if DIR[n_rad_corr]=='up':
        ind = np.where(np.isnan(PRES0[n_rad_corr,:])==0)[0]
        ind = ind[PRES0[n_rad_corr,ind]<=max_depth]
        TEMP_raw = np.flip(TEMP0[n_rad_corr,ind])
        COND_raw = np.flip(COND0[n_rad_corr,ind])
        TIME_raw = np.flip(TIME[n_rad_corr,ind])
        PRES_raw = np.flip(PRES0[n_rad_corr,ind])
        LAT_raw = np.flip(LAT0[n_rad_corr,ind])
        LON_raw = np.flip(LON0[n_rad_corr,ind])
        del ind

        ind = np.where(np.isnan(PRES0[n_rad_comp,:])==0)[0]
        ind = ind[PRES0[n_rad_comp,ind]<=max_depth]

        TEMP_comp = TEMP0[n_rad_comp,ind]
        COND_comp = COND0[n_rad_comp,ind]
        TIME_comp = TIME[n_rad_comp,ind]
        PRES_comp = PRES0[n_rad_comp,ind]
        LAT_comp = LAT0[n_rad_comp,ind]
        LON_comp = LON0[n_rad_comp,ind]
        del ind
    

    Aire_ref = calcul_Aire_ref(TEMP_raw, TEMP_comp, COND_raw, COND_comp, TIME_raw, TIME_comp, PRES_raw, PRES_comp, LON_raw, LON_comp, LAT_raw, LAT_comp, Gamma, T_gamma, C_gamma, var_corr)
    if Aire_ref==0:
        param = (0.0, 0.0, 4)
    else:
        w = minimize(calcul_Aire, coeff,args=(Aire_ref, TEMP_raw, TEMP_comp, COND_raw, COND_comp, TIME_raw, TIME_comp, PRES_raw, PRES_comp, LON_raw, LON_comp, LAT_raw, LAT_comp, Gamma, T_gamma, C_gamma, var_corr),\
                    method='SLSQP', bounds=bnds)

        param = w.x
        # print("Nombre d'itérations :", w.nit)
        # print("Nombre d'évaluations de la fonction :", w.nfev)
        # print("Nombre d'évaluations du gradient :", w.get('njev', None))
    return param

#
################################################################################
#
# Function calcul_Aire_ref: calculate the area between two curves (temperature vs pressure)
#
#   input:
#	T_raw     : Temperature of first profile
#	T_comp    : Temperature of second profile
#	C_raw     : Conductivity of first profile
#	C_comp    : Conductivity of second profile
#	Time_raw     : Time of first profile
#	Time_comp    : Time of second profile
#	Pr_raw     : Pressure of first profile
#	Pr_comp    : Pressure of second profile
#	Lon_raw     : Longitude of first profile
#	Lon_comp    : Longitude of second profile
#	Lat_raw     : Latitude of first profile
#	Lat_comp    : Latitude of second profile
#	Gamma     : Coefficient of sensitivity
#	T_gamma     : range of temperature
#	C_gamma     : range of conducitivity or salinity depending on which coefficient of sensitivity is chosen
#	var_corr     : variable to correct either 'cond' or 'sal'
#
#   output:
#	Aire_ref   : Area between the two curves
# 
# 
################################################################################
#

def calcul_Aire_ref(T_raw, T_comp, C_raw, C_comp, Time_raw, Time_comp, Pr_raw, Pr_comp, Lon_raw, Lon_comp, Lat_raw, Lat_comp, Gamma, T_gamma, C_gamma, var_corr):
    
    if var_corr=='cond':
        S_raw = gsw.SP_from_C(C_raw,T_raw,Pr_raw)
        SA_raw = gsw.SA_from_SP(S_raw,Pr_raw,Lon_raw,Lat_raw)
        CT_raw = gsw.CT_from_t(SA_raw,T_raw,Pr_raw)
        Sigma_raw = gsw.rho(SA_raw,CT_raw,Pr_raw)
        del CT_raw, SA_raw
    
        S_comp = gsw.SP_from_C(C_comp,T_comp,Pr_comp)
        SA_comp = gsw.SA_from_SP(S_comp,Pr_comp,Lon_comp,Lat_comp)
        CT_comp = gsw.CT_from_t(SA_comp,T_comp,Pr_comp)
        Sigma_comp = gsw.rho(SA_comp,CT_comp,Pr_comp)
        del CT_comp, SA_comp
    elif var_corr=='sal':
        SA_raw = gsw.SA_from_SP(C_raw,Pr_raw,Lon_raw,Lat_raw)
        CT_raw = gsw.CT_from_t(SA_raw,T_raw,Pr_raw)
        Sigma_raw = gsw.rho(SA_raw,CT_raw,Pr_raw)
        del CT_raw, SA_raw
    
        SA_comp = gsw.SA_from_SP(C_comp,Pr_comp,Lon_comp,Lat_comp)
        CT_comp = gsw.CT_from_t(SA_comp,T_comp,Pr_comp)
        Sigma_comp = gsw.rho(SA_comp,CT_comp,Pr_comp)
        del CT_comp, SA_comp
    dens_min = np.nanmax([np.nanmin(Sigma_raw), np.nanmin(Sigma_comp)])
    dens_max = np.nanmin([np.nanmax(Sigma_raw), np.nanmax(Sigma_comp)])
    dens_mask1 = np.where((dens_min <= Sigma_raw) & (Sigma_raw <= dens_max))[0]
    dens_mask2 = np.where((dens_min <= Sigma_comp) & (Sigma_comp <= dens_max))[0]
    if (len(dens_mask1)<2) | (len(dens_mask2)<2):
        Aire_ref=0
    else:
        min_ref1 = dens_mask1[0]
        min_ref2 = dens_mask2[0]
        max_ref1 = dens_mask1[-1]
        max_ref2 = dens_mask2[-1]
        del dens_min, dens_max, dens_mask1, dens_mask2

        T_temp1 = T_raw[min_ref1:max_ref1]
        C_temp1 = C_raw[min_ref1:max_ref1]
        T_temp2 = T_comp[min_ref2:max_ref2]
        C_temp2 = C_comp[min_ref2:max_ref2]

        x = T_temp1
        y = C_temp1
        exp_data = np.zeros((len(T_temp1), 2))
        exp_data[:, 0] = x
        exp_data[:, 1] = y

        # Generate random numerical data
        x = T_temp2
        y = C_temp2
        num_data = np.zeros((len(T_temp2), 2))
        num_data[:, 0] = x
        num_data[:, 1] = y
        
        # A1, d1 = similaritymeasures.dtw(exp_data, num_data)
        A1, _ = fastdtw(exp_data, num_data)
        del exp_data, num_data, x, y

        Aire_ref = A1
        del A1
    return Aire_ref


#
################################################################################
#
# Function calcul_Aire: calculate the area between two set of curves (temperature vs conductivity or salinity, and density versus conductivity or salinity) with changing thermal mass correction coefficients as an input
#
#   input:
#	params     : temporal lag
#	Aire_ref     : Area of reference (without any correction)
#	T_raw     : Temperature of first profile
#	T_comp    : Temperature of second profile
#	C_raw     : Conductivity of first profile
#	C_comp    : Conductivity of second profile
#	Time_raw     : Time of first profile
#	Time_comp    : Time of second profile
#	Pr_raw     : Pressure of first profile
#	Pr_comp    : Pressure of second profile
#	Lon_raw     : Longitude of first profile
#	Lon_comp    : Longitude of second profile
#	Lat_raw     : Latitude of first profile
#	Lat_comp    : Latitude of second profile
#	Gamma     : Coefficient of sensitivity
#	T_gamma     : range of temperature
#	C_gamma     : range of conducitivity or salinity depending on which coefficient of sensitivity is chosen
#	var_corr     : variable to correct either 'cond' or 'sal'

#   output:
#	Aire   : Area between the two curves, normalize by the area of reference
# 
# 
################################################################################
#


def calcul_Aire(params, Aire_ref, T_raw, T_comp, C_raw, C_comp, Time_raw, Time_comp, Pr_raw, Pr_comp, Lon_raw, Lon_comp, Lat_raw, Lat_comp, Gamma, T_gamma, C_gamma, var_corr):
    if params[2]==0:
        Aire = Aire_ref+1
    else:
        T_corr, C_corr = T_C_corrected(T_raw, C_raw, Time_raw,Gamma,T_gamma,C_gamma,params)
        if var_corr=='cond':
            S_raw = gsw.SP_from_C(C_raw,T_raw,Pr_raw)
            S_corr = gsw.SP_from_C(C_corr,T_raw,Pr_raw)
        elif var_corr=='sal':
            S_raw = C_raw
            S_corr = C_corr
        if (np.nanmax(T_corr)>np.nanmax(T_raw)+1) | (np.nanmax(C_corr)>np.nanmax(C_raw)+1) | (np.nanmax(S_corr)>np.nanmax(S_raw)+0.2) | (np.nanmin(T_corr)<np.nanmin(T_raw)-1) |(np.nanmin(C_corr)<np.nanmin(C_raw)-1)  |(np.nanmin(S_corr)<np.nanmin(S_raw)-0.2):
            Aire = Aire_ref+1
        else:
            if var_corr=='cond':
                S_corr = gsw.SP_from_C(C_corr,T_raw,Pr_raw)
                SA_corr = gsw.SA_from_SP(S_corr,Pr_raw,Lon_raw,Lat_raw)
                CT_corr = gsw.CT_from_t(SA_corr,T_raw,Pr_raw)
                Sigma_corr1 = gsw.rho(SA_corr,CT_corr,Pr_raw)
                del CT_corr, SA_corr   

                S_corr = gsw.SP_from_C(C_raw,T_corr,Pr_raw)
                SA_corr = gsw.SA_from_SP(S_corr,Pr_raw,Lon_raw,Lat_raw)
                CT_corr = gsw.CT_from_t(SA_corr,T_corr,Pr_raw)
                Sigma_corr2 = gsw.rho(SA_corr,CT_corr,Pr_raw)
                del CT_corr, SA_corr

                S_raw = gsw.SP_from_C(C_raw,T_raw,Pr_raw)
                SA_raw = gsw.SA_from_SP(S_raw,Pr_raw,Lon_raw,Lat_raw)
                CT_raw = gsw.CT_from_t(SA_raw,T_raw,Pr_raw)
                Sigma_raw = gsw.rho(SA_raw,CT_raw,Pr_raw)
                del CT_raw, SA_raw

                S_comp = gsw.SP_from_C(C_comp,T_comp,Pr_comp)
                SA_comp = gsw.SA_from_SP(S_comp,Pr_comp,Lon_comp,Lat_comp)
                CT_comp = gsw.CT_from_t(SA_comp,T_comp,Pr_comp)
                Sigma_comp = gsw.rho(SA_comp,CT_comp,Pr_comp)
                del CT_comp, SA_comp
            elif var_corr=='sal':
                SA_corr = gsw.SA_from_SP(C_corr,Pr_raw,Lon_raw,Lat_raw)
                CT_corr = gsw.CT_from_t(SA_corr,T_raw,Pr_raw)
                Sigma_corr1 = gsw.rho(SA_corr,CT_corr,Pr_raw)
                del CT_corr, SA_corr   

                SA_corr = gsw.SA_from_SP(C_corr,Pr_raw,Lon_raw,Lat_raw)
                CT_corr = gsw.CT_from_t(SA_corr,T_corr,Pr_raw)
                Sigma_corr2 = gsw.rho(SA_corr,CT_corr,Pr_raw)
                del CT_corr, SA_corr

                SA_raw = gsw.SA_from_SP(C_raw,Pr_raw,Lon_raw,Lat_raw)
                CT_raw = gsw.CT_from_t(SA_raw,T_raw,Pr_raw)
                Sigma_raw = gsw.rho(SA_raw,CT_raw,Pr_raw)
                del CT_raw, SA_raw

                SA_comp = gsw.SA_from_SP(C_comp,Pr_comp,Lon_comp,Lat_comp)
                CT_comp = gsw.CT_from_t(SA_comp,T_comp,Pr_comp)
                Sigma_comp = gsw.rho(SA_comp,CT_comp,Pr_comp)
                del CT_comp, SA_comp

            dens_min = np.nanmax([np.nanmin(Sigma_corr1), np.nanmin(Sigma_corr2)])
            dens_min = np.nanmax([np.nanmin(dens_min), np.nanmin(Sigma_comp)])
            dens_max = np.nanmin([np.nanmax(Sigma_corr1), np.nanmax(Sigma_corr2)])
            dens_max = np.nanmin([np.nanmax(dens_max), np.nanmax(Sigma_comp)])
            dens_mask1 = np.where((dens_min <= Sigma_corr1) & (Sigma_corr1 <= dens_max))[0]
            dens_mask2 = np.where((dens_min <= Sigma_corr2) & (Sigma_corr2 <= dens_max))[0]
            dens_mask3 = np.where((dens_min <= Sigma_comp) & (Sigma_comp <= dens_max))[0]
            if (len(dens_mask1)<2) | (len(dens_mask2)<2) | (len(dens_mask3)<2):
                Aire=Aire_ref
            else:
                min_idx1 = dens_mask1[0]
                min_idx2 = dens_mask2[0]
                min_idx3 = dens_mask3[0]
                max_idx1 = dens_mask1[-1]
                max_idx2 = dens_mask2[-1]
                max_idx3 = dens_mask3[-1]
                del dens_min, dens_max, dens_mask1, dens_mask2, dens_mask3

                T_temp1 = T_raw[min_idx1:max_idx1]
                C_temp1 = C_corr[min_idx1:max_idx1]
                T_temp2 = T_corr[min_idx2:max_idx2]
                C_temp2 = C_raw[min_idx2:max_idx2] 
                T_temp3 = T_comp[min_idx3:max_idx3]
                C_temp3 = C_comp[min_idx3:max_idx3]

                x = T_temp1
                y = C_temp1
                exp_data = np.zeros((len(T_temp1), 2))
                exp_data[:, 0] = x
                exp_data[:, 1] = y

                # Generate random numerical data
                x = T_temp3
                y = C_temp3
                num_data = np.zeros((len(T_temp3), 2))
                num_data[:, 0] = x
                num_data[:, 1] = y

                # A1, d1 = similaritymeasures.dtw(exp_data, num_data)
                A1, _ = fastdtw(exp_data, num_data)
                del exp_data, num_data, x, y

                x = T_temp2
                y = C_temp2
                exp_data = np.zeros((len(T_temp2), 2))
                exp_data[:, 0] = x
                exp_data[:, 1] = y

                # Generate random numerical data
                x = T_temp3
                y = C_temp3
                num_data = np.zeros((len(T_temp3), 2))
                num_data[:, 0] = x
                num_data[:, 1] = y

                # A2, d2 = similaritymeasures.dtw(exp_data, num_data)
                A2, _ = fastdtw(exp_data, num_data)
                del exp_data, num_data, x, y

                Aire = A1*A2/(Aire_ref*Aire_ref)
                del A1, A2

                del T_temp1, T_temp2, T_temp3, C_temp1, C_temp2, C_temp3, 

    return Aire


#
################################################################################
#
# Function T_C_corrected: shift the temperature profiles by a temporal lag
#
#   input:
#	T_raw     : Temperature profile
#	C_raw     : Conductivity or salinity profile
#	Time_raw  : Time
#	Gamma     : Coefficient of sensitivity
#	T_gamma     : range of temperature
#	C_gamma     : range of conducitivity or salinity depending on which coefficient of sensitivity is chosen
#	params    : thermal mass correction coefficients
#
#   output:
#	T_corr   : corrected temperature profile
#	C_corr   : corrected conductivity profile
# 
# 
################################################################################
#

def T_C_corrected(T_raw, C_raw, Time_raw,Gamma,T_gamma,C_gamma,params):
    valid = np.where((Time_raw >= 0) & (np.isnan(T_raw)==0))[0]

    Time_valid = Time_raw[valid]
    T_valid = T_raw[valid]
    C_valid = C_raw[valid]
    timestamp_unique, index_from = np.unique(Time_valid, return_index=True)
    T_unique = T_valid[index_from]
    C_unique = C_valid[index_from]
    
    Time_interp = np.arange(np.nanmin(Time_raw),np.nanmax(Time_raw),np.nanmedian(np.abs(np.diff(Time_raw))))
    
    f1 = interpolate.interp1d(timestamp_unique, T_unique,'linear',fill_value="extrapolate")
    f2 = interpolate.interp1d(timestamp_unique, C_unique,'linear',fill_value="extrapolate")
    T_interp = np.zeros(len(Time_interp))
    T_interp[:] = np.nan
    C_interp = np.zeros(len(Time_interp))
    C_interp[:] = np.nan
    T_interp = f1(Time_interp)
    C_interp = f2(Time_interp)
    
    #T_interp = pchip_interpolate(timestamp_unique, T_unique, Time_interp)
    #C_interp = pchip_interpolate(timestamp_unique, C_unique, Time_interp)


    dTime = np.diff(Time_interp)
    dTemp = np.diff(T_interp)

    alpha_t = params[0]
    alpha_c = params[1]
    tau = params[2]

    at = 2* alpha_t / (2 + dTime / tau)
    ac = 2* alpha_c / (2 + dTime / tau)
    b = 1 - 4 / (2 + dTime / tau)


    #Definition du filtre pour lisser Gamma = dC/dT
    if np.abs(np.max(Time_interp)-Time_interp[0])>25:
        sigma = np.where(np.abs(Time_interp-Time_interp[0])>=1)[0][0]
        sz = 2*np.where(np.abs(Time_interp-Time_interp[0])>=1)[0][0]    # length of gaussFilter vector
        x = np.linspace(-sz / 2, sz / 2, sz)
        gaussFilter = np.exp(-(x*x) / (2 * sigma * sigma))
        gaussFilter = gaussFilter / np.sum(gaussFilter)

    gamma = np.zeros(len(Time_interp))
    gamma[:] = np.nan
    for i_z in range(len(Time_interp)):
        if (len(np.where(T_gamma>=T_interp[i_z])[0])>0) & (len(np.where(C_gamma>=C_interp[i_z])[0])>0):
            gamma[i_z] = Gamma[np.where(T_gamma>=T_interp[i_z])[0][0], np.where(C_gamma>=C_interp[i_z])[0][0]]
    del i_z
    if np.abs(np.max(Time_interp)-Time_interp[0])>25:
        gamma = np.convolve(gamma, gaussFilter,mode='same')



    T_t = np.zeros(len(Time_interp))
    C_t = np.zeros(len(Time_interp))

    for i_z in range(len(Time_interp)-1):
        T_t[i_z+1] = -b[i_z]*T_t[i_z] + at[i_z]*dTemp[i_z]
        C_t[i_z+1] = -b[i_z]*T_t[i_z] + ac[i_z]*gamma[i_z]*dTemp[i_z]
    del i_z

    T_t[np.where(np.isnan(T_t)==1)[0]] = 0
    C_t[np.where(np.isnan(C_t)==1)[0]] = 0

    T_corr_interp = T_interp - T_t
    C_corr_interp = C_interp + C_t
    
    T_corr = np.zeros(len(Time_raw))
    T_corr[:] = np.nan
    C_corr = np.zeros(len(Time_raw))
    C_corr[:] = np.nan

    if (np.nanmax(T_corr_interp)>np.nanmax(T_raw)+10) | (np.nanmax(C_corr_interp)>np.nanmax(C_raw)+10) | (np.nanmin(T_corr_interp)<np.nanmin(T_raw)-10) | (np.nanmin(C_corr_interp)<np.nanmin(C_raw)-10):
        T_corr[:] = T_raw
        C_corr[:] = C_raw
    else:

        f1 = interpolate.interp1d(Time_interp, T_corr_interp,'linear',fill_value="extrapolate")
        f2 = interpolate.interp1d(Time_interp, C_corr_interp,'linear',fill_value="extrapolate")
        T_corr[valid[index_from]] = f1(timestamp_unique)
        C_corr[valid[index_from]] = f2(timestamp_unique)
        #T_corr[index_from] = pchip_interpolate(Time_interp, T_corr_interp, timestamp_unique)
        #C_corr[index_from] = pchip_interpolate(Time_interp, C_corr_interp, timestamp_unique)
    return T_corr, C_corr


#
################################################################################
#
# Function merge_corrections_TC: apply the temporal lag to the temperature and conductivity profiles
#
#
#   input:
#	TEMP0     : In-situ Temperature
#	COND0     : Conductivity
#	TIME     : Time
#	DEPTH0     : Pressure
#	D        : Distance since beginning
#	Gamma     : Coefficient of sensitivity
#	T_gamma     : range of temperature
#	C_gamma     : range of conducitivity or salinity depending on which coefficient of sensitivity is chosen
#	alphat_apres   : error of amplitude for temperature (calculated with the next profile in opposite direction)
#	alphac_apres   : error of amplitude for conductivity or salinity (calculated with the next profile in opposite direction)
#	tau_apres   : relaxation time (calculated with the next profile in opposite direction)
#	alphat_avant   : error of amplitude for temperature (calculated with the previous profile in opposite direction)
#	alphac_avant   : error of amplitude for conductivity or salinity (calculated with the previous profile in opposite direction)
#	tau_avant   : relaxation time (calculated with the next profile in opposite direction)
#	tau_lag_apres        : temporal lag with the next profile
#	tau_lag_avant        : temporal lag with the previous profile
#	DIR     : Vertical direction of each profile 
#	sens_corr     : Chosen direction to be corrected
#	var_corr     : variable to correct either 'cond' or 'sal'
#
#   output:
#	T_final   : Corrected temperature profile
#	C_final   : Corrected conductivity profile
# 
# 
################################################################################
#

def merge_corrections_TC(TEMP0, COND0, TIME, DEPTH0, D, Gamma, T_gamma, C_gamma, alphat_apres, alphac_apres, tau_apres, alphat_avant, alphac_avant, tau_avant, DIR, sens_corr,var_corr):
    if sens_corr=='up':
        ind_sens = np.where(np.asarray(DIR)=='up')[0]
    elif sens_corr=='down':
        ind_sens = np.where(np.asarray(DIR)=='down')[0]
    elif sens_corr=='all':
        ind_sens = np.arange(0,len(DIR),1)

    T_final = np.zeros((TEMP0.shape[0],TEMP0.shape[1]))
    T_final[:] = np.nan
    C_final = np.zeros((TEMP0.shape[0],TEMP0.shape[1]))
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
            coeff = np.zeros(3)
            coeff[0] = alphat_avant[ind_sens[n_rad_corr]]
            coeff[1] = alphac_avant[ind_sens[n_rad_corr]]
            coeff[2] = tau_avant[ind_sens[n_rad_corr]]
            if DIR[ind_sens[n_rad_corr]]=='down':
                T_corr, C_corr_avant = T_C_corrected(T_raw, C_raw, Time_raw,Gamma,T_gamma,C_gamma,coeff)
                T_corr_avant = T_raw
                del T_corr
            elif DIR[ind_sens[n_rad_corr]]=='up':
                T_corr, C_corr_avant = T_C_corrected(np.flip(T_raw), np.flip(C_raw), np.flip(Time_raw),Gamma,T_gamma,C_gamma,coeff)
                C_corr_avant = np.flip(C_corr_avant)
                T_corr_avant = T_raw
                del T_corr
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
            coeff = np.zeros(3)
            coeff[0] = alphat_apres[ind_sens[n_rad_corr]]
            coeff[1] = alphac_apres[ind_sens[n_rad_corr]]
            coeff[2] = tau_apres[ind_sens[n_rad_corr]]
            if DIR[ind_sens[n_rad_corr]]=='down':
                T_corr, C_corr_apres = T_C_corrected(T_raw, C_raw, Time_raw,Gamma,T_gamma,C_gamma,coeff)
                T_corr_apres = T_raw
                del T_corr
            elif DIR[ind_sens[n_rad_corr]]=='up':
                T_corr, C_corr_apres = T_C_corrected(np.flip(T_raw), np.flip(C_raw), np.flip(Time_raw),Gamma,T_gamma,C_gamma,coeff)
                C_corr_apres = np.flip(C_corr_apres)
                T_corr_apres = T_raw
                del T_corr
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