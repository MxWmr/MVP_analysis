import numpy as np 
import gsw
from scipy.optimize import minimize
from fastdtw import fastdtw



def compute_vertical_velocity(P, fs, window=20):
    v_z = np.gradient(P, 1/fs,axis=1)


    # smooth speed
    for i in range(v_z.shape[0]):
        v_z[i,:] = np.convolve(v_z[i,:], np.ones(2*window+1)/(2*window+1), mode='same')
    
    return v_z

def T_correction(T,V,Fn,params_r):

    alpha_o = params_r[0]
    alpha_s = params_r[1]
    tau_o = params_r[2]
    tau_s = params_r[3]

    alpha = alpha_o + alpha_s*np.reciprocal(V)
    tau = tau_o + tau_s*np.sqrt(np.reciprocal(V))

    params = np.zeros(2,len(alpha))
    params[0] = 4*Fn*alpha*tau*np.reciprocal(1+4*Fn*tau)
    params[1] = 1 -  2*params[0]*np.reciprocal(alpha)

    T_corr = np.copy(T)
    T_corr[1:] = -params[1]*T[:-2] + params[0]*(T[1:]-T[:-2])

    return T_corr


def data_correction(T,V,Fn,params_r):

    alpha_o = params_r[:,0]
    alpha_s = params_r[:,1]
    tau_o = params_r[:,2]
    tau_s = params_r[:,3]
    T_corr = np.copy.deepcopy(T)

    for i in range(len(V)):

        alpha = alpha_o[i] + alpha_s[i]*np.reciprocal(V[i])
        tau = tau_o[i] + tau_s[i]*np.sqrt(np.reciprocal(V[i]))

        params = np.zeros(2,len(alpha))
        params[0] = 4*Fn*alpha*tau*np.reciprocal(1+4*Fn*tau)
        params[1] = 1 -  2*params[0]*np.reciprocal(alpha)

        T_corr[i,1:] = -params[1]*T[i,:-2] + params[0]*(T[i,1:]-T[i,:-2])





def find_params(T,C,P,Fn,V,init_val,bnds):

    params = np.zeros((len(V),4))
    params[:] = np.nan
    for i in range(len(P)):

        result = minimize(objective_function, init_val,args=(T[i],C[i],P[i],V[i],Fn), bounds=bnds, method='SLSQP')
        params[i] = result.x


    return params


def objective_function(params,T,C,P,V,Fn):

    T_corr = T_correction(T,V,Fn,params)
    S = gsw.SP_from_C(T,C,P)
    S_corr = gsw.SP_from_C(T_corr,C,P)
    TS_corr = np.vstack((T_corr,S_corr)).T
    TS_raw = np.vstack((T,S)).T
    dist, _ = fastdtw(TS_corr, TS_raw)
    
    return dist