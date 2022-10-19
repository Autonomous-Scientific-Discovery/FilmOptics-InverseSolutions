# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:31:33 2022

@author: Rajdeep Dutta
"""

import numpy as np
import pandas as pd
from ScatteringMatrix import ComputeRT
from mixtureGO import gauss_mix, gausserfi_mix
from VensembleTL import nk_VensembleTL 
import matplotlib.pyplot as plt

nk_df = pd.read_csv('Refraction_Index_Library.csv', delimiter=',', header = 0,
                       skipinitialspace = True)

def CalcRTnk_GO(decn_var):
#
    coeff = decn_var[:-2]
    nbar = decn_var[-2]
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


def CalcRTnk_TL(decn_var):
#
    coeff = decn_var[:-1]
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


def totLossRT_GO(decn_var):
    ## Define gauss and gausserfi mixture functions
    
    indxRef = 11
    ## Tauc-Lorentz model data
#    R_exp, T_exp = X_train[indxRef,:,0], X_train[indxRef,:,1]
    
#    ## Perovskite model data  ## Experimental data
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
               
    Loss = np.dot(R_cal-R_exp,R_cal-R_exp)+np.dot(T_cal-T_exp,T_cal-T_exp) 
        
    return Loss