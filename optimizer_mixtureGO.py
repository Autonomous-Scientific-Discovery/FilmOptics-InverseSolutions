# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:09:15 2021

@author: Rajdeep Dutta
"""

from ScatteringMatrix import ComputeRT
from mixtureGO import gauss_mix, gausserfi_mix
import numpy as np
import pandas as pd
import cma
#from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import minimize, Bounds, leastsq
#import matplotlib.pyplot as plt


nk_df = pd.read_csv('Refraction_Index_Library.csv', delimiter=',', header = 0,
                       skipinitialspace = True)

global Mo, Pv, indxRef
Mo, Pv = 0, 1            # Choose the type of film material
indxRef = 5            # Select an index from the data set

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


def totLossRT_g(decn_var):
    if Mo==1 and Pv==0:        
        ## Metal oxide film's spectral data
        R_exp, T_exp = X_train[indxRef,:,0], X_train[indxRef,:,1]
    elif Mo==0 and Pv==1:    
        ## Perovskite film's spectral data  ## Experimental data
        R_exp, T_exp = RT_list[indxRef,:,0], RT_list[indxRef,:,1]
    
    coeff = decn_var[:-2]
    nbar0 = decn_var[-2]
    t = decn_var[-1]
    lam =  np.linspace(350, 1000, num=651) # [nm] wavelength range
    omega = np.true_divide(3*1e3,lam)
    omega = omega[::-1]
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
    k_layer1_new = gauss_mix(omega, *coeff)    # Call Gaussian mixture function 
    k_layer1_new = k_layer1_new[::-1]
    n_layer1_new = nbar0 + gausserfi_mix(omega, *coeff)     # Call Gaussian*erfi mixture function
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
               
    Loss = np.dot(R_cal-R_exp,R_cal-R_exp)+np.dot(T_cal-T_exp,T_cal-T_exp) 
        
    return Loss

## Apply CMA-ES to the inverse problem
opts = cma.CMAOptions()
opts.set('MaxIter', 1000)
# opts['popsize'] = 100
opts['ftarget'] = 5e-2
opts['tolfun'] = 1e-20  # opts['tolflatfitness'] = 5
opts['tolfunhist'] = 1e-20
#opts['verb_disp'] = 1
# opts['seed'] =   # 611333  # 651707 # 470705
# 718528  # 885508 # 943357 # 943859 # 1051871
opts['CMA_active'] = 1
# opts['AdaptSigma'] = 'CMAAdaptSigmaCSA'
numGO = 3       # Number of Gaussian Oscillators
opts.set('bounds', [np.hstack([np.tile([-0.1,3,0.2],numGO),np.array([0.1,10])]), 
                    np.hstack([np.tile([5,9,1.1],numGO),np.array([5,2000])])])
opts.set('CMA_stds',np.hstack([np.tile([5/5,6/5,1/5],numGO), np.array([5/5,2000/5])]))
x_startCMA=np.hstack([np.tile([1,3,0.5],numGO), np.array([0.1,100])])
sigma_start=0.7
res_CMAES=cma.fmin2(totLossRT_g, x_startCMA, sigma_start, opts)#, noise_handler=cma.NoiseHandler(11)) # restarts=8, bipop=True)
#res_CMAES=cma.CMAEvolutionStrategy(x_start, sigma_start, opts).optimize(totLossRT)
cma.plot()

#-------------------------------------------------------------
from comparePlot import CalcRTnk_GO
import matplotlib.pyplot as plt
lam =  np.linspace(350, 1000, num=651)     # [nm] wavelength range
R_exp, T_exp = RT_list[indxRef,:,0], RT_list[indxRef,:,1]  
Soln_CMAES_TLO = res_CMAES[0]    # inverse solution with CMAES
R_cmatl, T_cmatl, n_cmatl, k_cmatl = CalcRTnk_GO(lam, Soln_CMAES_TLO)

plt.figure()
plt.plot(lam, R_exp,'r:', label='R$_{exp}$ existing', linewidth=2.7)
plt.plot(lam, T_exp,'k:', label='T$_{exp}$ existing', linewidth=2.7)
plt.plot(lam, R_cmatl, c='orange', label='R$_{cal}$ GO+CMAES')
plt.plot(lam, T_cmatl, c='slategrey', label='T$_{cal}$ GO+CMAES')
plt.minorticks_on()
plt.xlim((350,1000))
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R), Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

plt.figure()
plt.plot(lam, n_cmatl, c='navy', label='n$_{cal}$ GO+CMAES')
plt.plot(lam, k_cmatl, c='forestgreen', label='k$_{cal}$ GO+CMAES')
plt.minorticks_on()
plt.xlim((350,1000))
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive Index (n), Extinction Coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()