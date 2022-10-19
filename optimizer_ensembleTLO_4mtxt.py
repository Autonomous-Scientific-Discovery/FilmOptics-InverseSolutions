# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:41:36 2022

@author: Rajdeep Dutta
"""


from ScatteringMatrix import ComputeRT
from VensembleTL import nk_VensembleTL
from evaluateModel import CalcRTnk_TL
import numpy as np
import pandas as pd
import cma
# from scipy.optimize import minimize, Bounds, leastsq
import matplotlib.pyplot as plt
np.random.seed(0)

nk_df = pd.read_csv('Refraction_Index_Library.csv', delimiter=',', header = 0,
                       skipinitialspace = True)

global dataR, dataT
## Load the experimental measurement data from .txt
dataR = np.loadtxt('Rcal-CGO.txt', delimiter=',', dtype=np.float64)
dataT = np.loadtxt('Tcal-CGO.txt', delimiter=',', dtype=np.float64)

def totLossRT_tl(decn_var):

    coeff = decn_var[:-1]
    t = decn_var[-1]
    lam =  np.linspace(400, 900, num=651)     # [nm] wavelength range

    indTL=int(len(coeff[:-2])/3)
    indicator1 = np.max(np.where(np.abs(coeff[indTL:2*indTL]) > coeff[-2], 0, 1))
    indicator2 = np.max(np.where(np.abs(coeff[indTL:2*indTL]) > coeff[2*indTL:3*indTL]/1.414, 0, 1))

    
    if indicator1==0 and indicator2==0:
 
        ## Experimental spectra from a .txt 
        R_exp, T_exp = np.interp(lam, dataR[:,0], dataR[:,1])/100, np.interp(lam, dataT[:,0], dataT[:,1])/100  
          
    
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
        n_layer1_new, k_layer1_new = nk_VensembleTL(lam,coeff)
        if OsciModel_Layer1 == True:
            n_interp = n_layer1_new
            k_interp = k_layer1_new
            Ref_Idx = n_interp + 1j*k_interp
            Thickness = t
            Incoherence = False
            Roughness = 0
            structure[1] = [Thickness, Ref_Idx, Incoherence, Roughness]
            R_cal, T_cal = ComputeRT(structure,lam,Phi/180*np.pi)   # Calculate the R and T spectra
               
        Loss = np.dot(R_cal-R_exp,R_cal-R_exp) + np.dot(T_cal-T_exp,T_cal-T_exp) 

    elif indicator1==1 or indicator2==1:
        Loss=1e3
    
    return Loss 

## Call CMA-ES population-based optimization routine
opts = cma.CMAOptions()
opts.set('MaxIter', 1000)
opts.set('popsize', 50)
opts['ftarget'] = 2e-2
opts['tolfun'] = 1e-20  # opts['tolflatfitness'] = 5
opts['tolfunhist'] = 1e-20
#opts['verb_disp'] = 1
opts['seed'] = 173618
opts['CMA_active'] = 1
# opts['AdaptSigma'] =  'CMAAdaptSigmaMedianImprovement' # 'CMAAdaptSigmaCSA'  # CMAAdaptSigmaTPA  # 
numTLO = 3       # Number of Tauc-Lorentz Oscillators
opts.set('bounds', [np.hstack([np.repeat([0,-5,0],numTLO),np.array([0,0,20])]), 
                    np.hstack([np.repeat([100,5,5],numTLO),np.array([5,5,2000])])])
opts.set('CMA_stds',np.hstack([np.repeat([100/5,10/5,5/5],numTLO), np.array([5/5,5/5,2000/5])]))
x_startCMA=np.hstack([np.repeat([10,0,0],numTLO), np.array([0,0,20])])
sigma_start=0.5
res_CMAES=cma.fmin2(totLossRT_tl, x_startCMA, sigma_start, opts)#, noise_handler=cma.NoiseHandler(12)) # restarts=8, bipop=True)
##res_CMAES=cma.CMAEvolutionStrategy(x_start, sigma_start, opts).optimize(totLossRT)
cma.plot()


lam =  np.linspace(400, 900, num=651)     # [nm] wavelength range
R_exp, T_exp = np.interp(lam, dataR[:,0], dataR[:,1])/100, np.interp(lam, dataT[:,0], dataT[:,1])/100  
Soln_CMAES_TLO = res_CMAES[0]    # inverse solution with CMAES
R_cmatl, T_cmatl, n_cmatl, k_cmatl = CalcRTnk_TL(lam, Soln_CMAES_TLO)

plt.figure()
plt.plot(lam, R_exp,'r:', label='R$_{exp}$ existing', linewidth=2.7)
plt.plot(lam, T_exp,'k:', label='T$_{exp}$ existing', linewidth=2.7)
plt.plot(lam, R_cmatl, c='orange', label='R$_{cal}$ TLO+CMAES')
plt.plot(lam, T_cmatl, c='slategrey', label='T$_{cal}$ TLO+CMAES')
plt.minorticks_on()
plt.xlim((400,900))
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R), Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

data_kCGO = np.loadtxt('k-CGO.txt', delimiter=',', dtype=np.float64)
data_nCGO = np.loadtxt('n-CGO.txt', delimiter=',', dtype=np.float64)
n_CGO, k_CGO = np.interp(lam, data_nCGO[:,0], data_nCGO[:,1]), np.interp(lam, data_kCGO[:,0], data_kCGO[:,1]) 
plt.figure()
plt.plot(lam, n_CGO, c='lightblue', label='n$_{cal}$ CGO')
plt.plot(lam, k_CGO, c='greenyellow',  label='k$_{cal}$ CGO')
plt.plot(lam, n_cmatl, c='navy', label='n$_{cal}$ TLO+CMAES')
plt.plot(lam, k_cmatl, c='forestgreen', label='k$_{cal}$ TLO+CMAES')
plt.minorticks_on()
plt.xlim((400,900))
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive Index (n), Extinction Coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
