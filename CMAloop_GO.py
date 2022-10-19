# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:00:10 2021

@author: Rajdeep Dutta
"""

from ScatteringMatrix import ComputeRT
from mixtureGO import gauss_mix, gausserfi_mix
import numpy as np
import pandas as pd
import random
import cma
import pickle
from sklearn.metrics import r2_score

nk_df = pd.read_csv('Refraction_Index_Library.csv', delimiter=',', header = 0,
                       skipinitialspace = True)

global Mo, Pv, indxRef
Mo, Pv = 0, 1            # Choose the type of film material

if Mo==1 and Pv==0:      # Load spectral data for metal oxide films
    global X_train
    TrainTestData = np.load('TrainTestData.npz')
    [X_train, X_val, X_test, y_train, y_val, y_test] = [TrainTestData[s] for s in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']]
    X_train = X_train[:,:,:2]
    y_train = [y_train[:,:651], y_train[:,651:-1], y_train[:,-1][:,np.newaxis]]
elif Mo==0 and Pv==1:    # Load spectral data for perovskite films
    global RT_list
    data_perv = np.load('ExpData.npz')
    RT_list = data_perv['X']
    
    
def totLossRT_g(decn_var, return_RT=False):

    coeff = decn_var[:-2]
    nbar0 = decn_var[-2]
    t = decn_var[-1]
    lam =  np.linspace(350, 1000, num=651) # [nm] wavelength range
    omega = np.true_divide(3*1e3,lam)
    omega = omega[::-1]
    
    def cal_RT():
        Phi = 0 # [Degree] incident angle
        LayerStructure = [
        ## [Material ID, thickness[nm],  coherence(i.e.True or False)]
                    ['Air', 0, False], # Top Medium
                    ['SiN', 500, False], # Placeholder
                    ['glass',1000000, True], # Substrate Glass
                    ['Air', 0, False]  # Bottom Medium
                ]
        structure = []
        for layer_i in range(len(LayerStructure)):
            #print(layer_i)
            mat_label = LayerStructure[layer_i][0]
            wl = nk_df[mat_label+'_wl']+0.0001
            n  = nk_df[mat_label+'_n']
            k  = nk_df[mat_label+'_k']
            n_interp = np.interp(lam, wl, n)
            k_interp = np.interp(lam, wl, k)
            Ref_Idx = n_interp + 1j*k_interp
            Thickness = LayerStructure[layer_i][1]
            Incoherence = LayerStructure[layer_i][2]
            Roughness = 0
            if layer_i ==0 or layer_i == len(LayerStructure)-1:
                Thickness = 10000
                Incoherence = False
            structure.append([Thickness, Ref_Idx, Incoherence, Roughness])
        
        # thin film of concern
        OsciModel_Layer1 = True
        # Generate R_T
        k_layer1_new = gauss_mix(omega, *coeff)
        k_layer1_new = k_layer1_new[::-1]
        n_layer1_new = nbar0 + gausserfi_mix(omega, *coeff)
        n_layer1_new = n_layer1_new[::-1]  
        if OsciModel_Layer1 == True:
            n_interp = n_layer1_new
            k_interp = k_layer1_new
            Ref_Idx = n_interp + 1j*k_interp
            Thickness = t
            Incoherence = False
            Roughness = 0
            structure[1] = [Thickness, Ref_Idx, Incoherence, Roughness]
            R_cal, T_cal = ComputeRT(structure,lam,Phi/180*np.pi) # Calculate the R and T spectra
        return R_cal, T_cal
    
    if Mo==1 and Pv==0:
        ## Metal oxide film's spectral data 
        R_exp, T_exp = X_train[indxRef,:,0], X_train[indxRef,:,1]
    elif Mo==0 and Pv==1:
        ## Perovskite film's spectral data 
        R_exp, T_exp = RT_list[indxRef,:,0], RT_list[indxRef,:,1] 

    R_cal, T_cal = cal_RT()              
    Loss = np.dot(R_cal-R_exp,R_cal-R_exp) + np.dot(T_cal-T_exp,T_cal-T_exp) 
            
    if not return_RT:
        return Loss
    else:
        try:
            return R_cal, T_cal
        except:
            R_cal, T_cal = cal_RT()
            return R_cal, T_cal
        
## Apply CMA-ES to the inverse problem
opts = cma.CMAOptions()
opts.set('MaxIter', 1500)
opts['ftarget'] = 5e-2
opts['tolfun'] = 1e-20  # opts['tolflatfitness'] = 5
opts['verb_disp'] = 0
opts['CMA_active'] = 1
numGO = 3       # Number of Gaussian Oscillators
opts.set('bounds', [np.hstack([np.tile([-0.1,3,0.2],numGO),np.array([0.1,20])]), 
                    np.hstack([np.tile([5,9,1.1],numGO),np.array([5,2000])])])
opts.set('CMA_stds',np.hstack([np.tile([5/5,6/5,1/5],numGO), np.array([5/5,2000/5])]))
x_startCMA=np.hstack([np.tile([1,3,0.5],numGO), np.array([0.1,100])])

# =============================================================================
# Added code: initialize a table to collect results (FYI: I used d as the notation of thickness instead of t)
# =============================================================================
result = {
    'error_hat': [], # estimation error    
    'indxRef': [], # input reference index
    'R_hat': [], # calculated R from the best solution found
    'T_hat': [], # calculated T from the best solution found
    'R': [], # simulated R (our label)
    'T': [], # simulated T (our label)
    'n_hat': [], # calculated n from the best solution found
    'k_hat': [], # calculated k from the best solution found
    'n': [], # simulated n (our label)
    'k': [], # simulated k (our label)
    'd_hat': [], # predicted d from the best solution found
    'd': [], # actual d (our label)
    'feval': [], # no. of function evaluations
    }
# =============================================================================
# 
# =============================================================================
lam =  np.linspace(350, 1000, num=651) # [nm] wavelength range
omega = np.true_divide(3*1e3,lam)
omega = omega[::-1]
sigma_start=1
#rec_cma=[]
nsamples=100
nruns=5
global indxRef
sd=1
for i in range(nsamples*nruns):
    sd = sd + np.where(i % nruns > 0, 0, 100)
    random.seed(sd)
    indxRef = random.randint(100,35000)
    res_CMAES=cma.fmin2(totLossRT_g, x_startCMA, sigma_start, opts) # restarts=8, bipop=True)
#    rec_cma.append([indxRef, y_train[2][indxRef,0], res_CMAES[0][-1], res_CMAES[1].result[1], res_CMAES[1].result[4], res_CMAES[1].timer.elapsed, res_CMAES[1].opts['seed']])   #  data_perv['y'][indxRef,-1]
    # Save details in a Dictionary
    # n, k =  y_train[0][indxRef], y_train[1][indxRef]       
    n, k = data_perv['y'][indxRef,0:651], data_perv['y'][indxRef,651:1302]     
    coeff = res_CMAES[0][:-2]
    nbar = res_CMAES[0][-2]
    k_soln, n_soln = gauss_mix(omega, *coeff), nbar+ gausserfi_mix(omega, *coeff) 
    n_hat, k_hat = n_soln[::-1], k_soln[::-1]
    # R, T =  X_train[indxRef, :, 0], X_train[indxRef, :, 1]      
    R, T = data_perv['X'][indxRef,:,0], data_perv['X'][indxRef,:,1]    
    R_hat, T_hat = totLossRT_g(res_CMAES[0], return_RT=True)
    # d =   y_train[2][indxRef][0]       
    d = data_perv['y'][indxRef,-1]   
    d_hat = res_CMAES[0][-1]
    error_hat = res_CMAES[1].result[1]
    feval = res_CMAES[1].result[4]
    
    if np.abs(d-d_hat)< 0.1*d and error_hat< 0.17:
        result['error_hat'].append(error_hat)
        result['indxRef'].append(indxRef)
        result['R_hat'].append(R_hat)
        result['T_hat'].append(T_hat)
        result['R'].append(R)
        result['T'].append(T)
        result['n_hat'].append(n_hat)
        result['k_hat'].append(k_hat)
        result['n'].append(n)
        result['k'].append(k)
        result['d_hat'].append(d_hat)
        result['d'].append(d)
        result['feval'].append(feval)
    
        # Save results using pickle
        pickle.dump(result, open('resultB_GO3.pickle', 'wb'))
    
# =============================================================================
# Can read the pickle file by this following line
# =============================================================================
result = pickle.load(open('resultB_GO3.pickle', 'rb'))

# =============================================================================
# Can obatin the metrics for thickness by the following lines
# =============================================================================
result_df = pd.DataFrame(result)
#result_df = result_df.loc[(np.abs(result_df['d']-result_df['d_hat'])< 0.1*result_df['d']) & (result_df['error_hat']< 0.1)]
result_df= result_df.loc[result_df.groupby('indxRef').error_hat.idxmin()]



r2_d = r2_score(result_df['d'], result_df['d_hat'])
print(f'R-square score for predicted thickness is {r2_d}')

r2_n = []
for index, row in result_df.iterrows():
    r2_n.append(r2_score(row['n'], row['n_hat']))
print(f'median R-square score for predicted n is {np.median(r2_n)}') 

r2_k = []
for index, row in result_df.iterrows():
    r2_k.append(r2_score(row['k'], row['k_hat']))
print(f'median R-square score for predicted k is {np.median(r2_k)}') 

r2_R = []
for index, row in result_df.iterrows():
    r2_R.append(r2_score(row['R'], row['R_hat']))
print(f'mean R-square score for predicted R is {np.mean(r2_R)}')

r2_T = []
for index, row in result_df.iterrows():
    r2_T.append(r2_score(row['T'], row['T_hat']))
print(f'mean R-square score for predicted T is {np.mean(r2_T)}')   

print('average number of function evaluations:') 
print(np.mean(result_df['feval']))  

print(f'success rate is {len(r2_T)/nsamples}')