# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:17:50 2021

@author: Rajdeep Dutta
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

##---------------------------------------------------------------------------------------
# Load data from the saved results .pickle for all type-A films
result_df1 = pd.DataFrame(pickle.load(open('resultA_TLO2.pickle', 'rb')))
result_df2 = pd.DataFrame(pickle.load(open('resultA_GO3.pickle', 'rb')))
# Plot: estimated thickness vs. original thickness for all type-A thin-films
plt.figure()
plt.scatter(np.array(result_df1['d']),np.array(result_df1['d_hat']), color='y', label='d_estimate TLO')
plt.scatter(np.array(result_df2['d']),np.array(result_df2['d_hat']), color= 'g', label='d_estimate GO')
plt.xlabel('Actual Thickness [nm]', fontsize=12)
plt.ylabel('Estimated Thickness [nm]', fontsize=12)
plt.legend(fontsize=11)
plt.show()

# Define the wavelength independent axis 
lam = np.linspace(350, 1000, 651)

# Choose a solution index
indb=77
# Plot estimated vs. original refractive index & extinction coefficient
plt.figure()
plt.plot(lam, list(result_df1['n'])[indb], 'b:', label='n_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['n_hat'])[indb], 'b', label='n_estimate')
plt.plot(lam, list(result_df1['k'])[indb], 'g:', label='k_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['k_hat'])[indb], 'g', label='k_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive index (n) & Extinction coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
print('Actual and Estimated Thickness:', list(result_df1['d'])[indb],',', list(result_df1['d_hat'])[indb])
print('Total estimation error:', list(result_df1['error_hat'])[indb])
# Plot measured vs. obtained reflectance & transmittance
plt.figure()
plt.plot(lam, list(result_df1['R'])[indb], 'r:', label='R_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['R_hat'])[indb], 'r', label='R_estimate')
plt.plot(lam, list(result_df1['T'])[indb], 'k:', label='T_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['T_hat'])[indb], 'k', label='T_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R) & Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

# Find the index referring to the best result
indb=27 # 24  # 18
# Plot estimated vs. original refractive index & extinction coefficient
plt.figure()
plt.plot(lam, list(result_df2['n'])[indb], 'b:', label='n_actual', linewidth=2.7) 
plt.plot(lam, list(result_df2['n_hat'])[indb], 'b', label='n_estimate')
plt.plot(lam, list(result_df2['k'])[indb], 'g:', label='k_actual', linewidth=2.7) 
plt.plot(lam, list(result_df2['k_hat'])[indb], 'g', label='k_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive index (n) & Extinction coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
print('Actual and Estimated Thickness:', list(result_df2['d'])[indb],',', list(result_df2['d_hat'])[indb])
print('Total estimation error:', list(result_df2['error_hat'])[indb])
# Plot measured vs. obtained reflectance & transmittance
plt.figure()
plt.plot(lam, list(result_df2['R'])[indb], 'r:', label='R_actual', linewidth=2.7) 
plt.plot(lam, list(result_df2['R_hat'])[indb], 'r', label='R_estimate')
plt.plot(lam, list(result_df2['T'])[indb], 'k:', label='T_actual', linewidth=2.7) 
plt.plot(lam, list(result_df2['T_hat'])[indb], 'k', label='T_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R) & Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

# Find the index referring to the best result
indb= 29 # 38 
print('indb')
# Plot estimated vs. original refractive index & extinction coefficient
plt.figure()
plt.plot(lam, list(result_df1['n'])[indb], 'b:', label='n_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['n_hat'])[indb], 'b', label='n_estimate')
plt.plot(lam, list(result_df1['k'])[indb], 'g:', label='k_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['k_hat'])[indb], 'g', label='k_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive index (n) & Extinction coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
print('Actual and Estimated Thickness:', list(result_df1['d'])[indb],',', list(result_df1['d_hat'])[indb])
print('Total estimation error:', list(result_df1['error_hat'])[indb])
# Plot measured vs. obtained reflectance & transmittance
plt.figure()
plt.plot(lam, list(result_df1['R'])[indb], 'r:', label='R_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['R_hat'])[indb], 'r', label='R_estimate')
plt.plot(lam, list(result_df1['T'])[indb], 'k:', label='T_actual', linewidth=2.7) 
plt.plot(lam, list(result_df1['T_hat'])[indb], 'k', label='T_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R) & Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()



##----------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------
# Load data from the saved results .pickle for all type-B films
result_df3 = pd.DataFrame(pickle.load(open('resultB_TLO4.pickle', 'rb')))
result_df4 = pd.DataFrame(pickle.load(open('resultB_GO5.pickle', 'rb')))
# Plot: estimated thickness vs. original thickness for all type-B thin-films
plt.figure()
plt.scatter(np.array(result_df3['d']),np.array(result_df3['d_hat']), color='c', label='d_estimate TLO')
plt.scatter(np.array(result_df4['d']),np.array(result_df4['d_hat']), color='b', label='d_estimate GO')
plt.xlabel('Actual Thickness [nm]', fontsize=12)
plt.ylabel('Estimated Thickness [nm]', fontsize=12)
plt.legend(fontsize=11)
plt.show()

# Find the index referring to the best result
indb=60 
# Plot estimated vs. original refractive index & extinction coefficient
plt.figure()
plt.plot(lam, list(result_df4['n'])[indb], 'b:', label='n_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['n_hat'])[indb], 'b', label='n_estimate')
plt.plot(lam, list(result_df4['k'])[indb], 'g:', label='k_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['k_hat'])[indb], 'g', label='k_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive index (n) & Extinction coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
print('Actual and Estimated Thickness:', list(result_df4['d'])[indb],',', list(result_df4['d_hat'])[indb])
print('Total estimation error:', list(result_df4['error_hat'])[indb])
# Plot measured vs. obtained reflectance & transmittance
plt.figure()
plt.plot(lam, list(result_df4['R'])[indb], 'r:', label='R_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['R_hat'])[indb], 'r', label='R_estimate')
plt.plot(lam, list(result_df4['T'])[indb], 'k:', label='T_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['T_hat'])[indb], 'k', label='T_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R) & Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

# Find the index referring to the best result
indb=38  # np.argmin(np.array(result_df4['error_hat'])) 
# Plot estimated vs. original refractive index & extinction coefficient
plt.figure()
plt.plot(lam, list(result_df4['n'])[indb], 'b:', label='n_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['n_hat'])[indb], 'b', label='n_estimate')
plt.plot(lam, list(result_df4['k'])[indb], 'g:', label='k_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['k_hat'])[indb], 'g', label='k_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive index (n) & Extinction coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
print('Actual and Estimated Thickness:', list(result_df4['d'])[indb],',', list(result_df4['d_hat'])[indb])
print('Total estimation error:', list(result_df4['error_hat'])[indb])
# Plot measured vs. obtained reflectance & transmittance
plt.figure()
plt.plot(lam, list(result_df4['R'])[indb], 'r:', label='R_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['R_hat'])[indb], 'r', label='R_estimate')
plt.plot(lam, list(result_df4['T'])[indb], 'k:', label='T_actual', linewidth=2.7) 
plt.plot(lam, list(result_df4['T_hat'])[indb], 'k', label='T_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R) & Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

# Find the index referring to the best result
indb= 78  # np.argmin(np.array(result_df3['error_hat'])) 
# Plot estimated vs. original refractive index & extinction coefficient
plt.figure()
plt.plot(lam, list(result_df3['n'])[indb], 'b:', label='n_actual', linewidth=2.7) 
plt.plot(lam, list(result_df3['n_hat'])[indb], 'b', label='n_estimate')
plt.plot(lam, list(result_df3['k'])[indb], 'g:', label='k_actual', linewidth=2.7) 
plt.plot(lam, list(result_df3['k_hat'])[indb], 'g', label='k_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive index (n) & Extinction coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()
print('Actual and Estimated Thickness:', list(result_df3['d'])[indb],',', list(result_df3['d_hat'])[indb])
print('Total estimation error:', list(result_df3['error_hat'])[indb])
# Plot measured vs. obtained reflectance & transmittance
plt.figure()
plt.plot(lam, list(result_df3['R'])[indb], 'r:', label='R_actual', linewidth=2.7) 
plt.plot(lam, list(result_df3['R_hat'])[indb], 'r', label='R_estimate')
plt.plot(lam, list(result_df3['T'])[indb], 'k:', label='T_actual', linewidth=2.7) 
plt.plot(lam, list(result_df3['T_hat'])[indb], 'k', label='T_estimate')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Reflectance (R) & Transmittance (T)', fontsize=12)
plt.legend(fontsize=11)
plt.show()

##-------------------------------------------------------------------------------------