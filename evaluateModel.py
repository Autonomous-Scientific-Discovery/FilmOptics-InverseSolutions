# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:35:01 2020

@author: Rajdeep Dutta
"""

from ScatteringMatrix import ComputeRT
from mixtureGO import gauss_mix, gausserfi_mix
from VensembleTL import nk_VensembleTL 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nk_df = pd.read_csv('Refraction_Index_Library.csv', delimiter=',', header = 0,
                       skipinitialspace = True)
def CalcRTnk_GO(lam, decn_var):
#
    coeff = decn_var[:-2]
    nbar = decn_var[-2]
    t = decn_var[-1]
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
    
    k_layer1_new = gauss_mix(omega, *coeff)
    k_layer1_new = k_layer1_new[::-1]
    n_layer1_new = nbar + gausserfi_mix(omega, *coeff)
    n_layer1_new = n_layer1_new[::-1]  
    if OsciModel_Layer1 == True:
        n_interp = n_layer1_new
        k_interp = k_layer1_new
        Ref_Idx = n_interp + 1j*k_interp
        Thickness = t
        Incoherence = False
        Roughness = 0
        structure[1] = [Thickness, Ref_Idx, Incoherence, Roughness]
        R_cal, T_cal=ComputeRT(structure,lam,Phi/180*np.pi) # Calculate the R and T spectra
    
    return  R_cal, T_cal, n_layer1_new, k_layer1_new


def CalcRTnk_TL(lam, decn_var):
#
    coeff = decn_var[:-1]
    t = decn_var[-1]
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
    
    n_layer1_new, k_layer1_new = nk_VensembleTL(lam,coeff)
    if OsciModel_Layer1 == True:
        n_interp = n_layer1_new
        k_interp = k_layer1_new
        Ref_Idx = n_interp + 1j*k_interp
        Thickness = t
        Incoherence = False
        Roughness = 0
        structure[1] = [Thickness, Ref_Idx, Incoherence, Roughness]
        R_cal, T_cal=ComputeRT(structure,lam,Phi/180*np.pi) # Calculate the R and T spectra
    
    return  R_cal, T_cal, n_layer1_new, k_layer1_new

lam =  np.linspace(350, 1000, num=651) # [nm] wavelength range

# optimal solution 
opt_var = np.array([1.38511550e+00, 8.27064008e+00, 1.11449891e+00, 8.96727041e-02,   # ExpData ref 14 seed 470705
                    4.50749585e+00, 2.97058475e-01, 3.59329446e-01, 5.72605122e+00,
                    7.85310391e-01, 9.37500562e-02, 4.10470828e+00, 1.42897151e-01,
                    2.49619223e-01, 6.31225471e+00, 4.56477710e-01, 1.96519042e+00,
                    9.93545421e+01])

R_cal, T_cal, n_film, k_film = CalcRTnk_GO(lam, opt_var)

# m = 14 # an arbitrary number from library

# #TrainTestData = np.load('TrainTestData.npz')
# #[X_train, X_val, X_test, y_train, y_val, y_test] = [TrainTestData[s] for s in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']]
# #X_train = X_train[:,:,:2]
# #y_train = [y_train[:,:651], y_train[:,651:-1], y_train[:,-1][:,np.newaxis]]

# data_perv = np.load('ExpData.npz')
# RT_list = data_perv['X']

# #nkt_list = data_perv['y']
# #n_selected = nkt_list[m,0:651]
# #k_selected = nkt_list[m,651:1302]
# #t_selected = nkt_list[m,-1]


# # Plot an arbitrary wavelength-resolved RT and n, k
# plt.figure()

# #plt.plot(lam, X_train[n,:,0],'r:',label='R_actual',linewidth=2.7)
# #plt.plot(lam, X_train[n,:,1],'k:',label='T_actual',linewidth=2.7)

# plt.plot(lam, RT_list[m,:,0],'r:',label='R_actual',linewidth=2.7)
# plt.plot(lam, RT_list[m,:,1],'k:',label='T_actual',linewidth=2.7)
# plt.plot(lam, R_cal,'r',label='R_estimate')
# plt.plot(lam, T_cal,'k',label='T_estimate')

# plt.xlabel('Wavelength [nm]')
# plt.ylabel('Reflectance & Transmittance')
# plt.legend()
# plt.show()

# plt.figure()
# #plt.plot(lam, y_train[0][n,:],'b:',label='n_actual',linewidth=2.7)
# #plt.plot(lam, y_train[1][n,:],'g:',label='k_actual',linewidth=2.7)

# #plt.plot(lam, n_selected,'b:',label='n_actual',linewidth=2.7)
# #plt.plot(lam, k_selected,'g:',label='k_actual',linewidth=2.7)

# plt.plot(lam, n_film,'b',label='n_estimate')
# plt.plot(lam, k_film,'g',label='k_estimate')

# #plt.title('simulated, achieved t: {} nm'.format({y_train[2][n,0], opt_var[-1]}))
# #plt.title('achieved, real t: {} mm'.format({t_selected, opt_var[-1]}))

# plt.xlabel('Wavelength [nm]')
# plt.ylabel('Refractive Index')
# plt.legend()
# plt.show()