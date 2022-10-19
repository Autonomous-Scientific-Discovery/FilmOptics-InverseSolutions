# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:07:08 2021

@author: Rajdeep Dutta
"""

import numpy as np
from evaluateModel import CalcRTnk_GO, CalcRTnk_TL
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as mcolors

data_perv = np.load('ExpData.npz')
RT_list = data_perv['X']
ind_exp=5

result=pickle.load(open('result_ExpData050.pickle','rb'))

lam =  np.linspace(350, 1000, num=651) # [nm] wavelength range

Soln_CMAES_GO = result['CMA_solution_GO'][0]
R_cmagm, T_cmagm, n_cmagm, k_cmagm = CalcRTnk_GO(lam, Soln_CMAES_GO)


Soln_CMAES_TLO = result['CMA_solution_TLO'][0]
R_cmatl, T_cmatl, n_cmatl, k_cmatl = CalcRTnk_TL(lam, Soln_CMAES_TLO)


TRfit = np.loadtxt("PUMA1_TR_film0099.txt")
nkfit = np.loadtxt("PUMA1_nk_film0099.txt")


lam = np.linspace(350, 1000, 651)
plt.figure()
plt.plot(lam, RT_list[ind_exp,:,0],':', c='maroon', label='R$_{exp}$', linewidth=2.7)
plt.plot(lam, RT_list[ind_exp,:,1],':', c='black', label='T$_{exp}$', linewidth=2.7)
plt.plot(lam, R_cmatl, c='orange', label='R$_{cal}$ TLO+CMAES')
plt.plot(lam, T_cmatl, c='grey', label='T$_{cal}$ TLO+CMAES')
plt.plot(lam, R_cmagm, c='tomato', label='R$_{cal}$ GO+CMAES')
plt.plot(lam, T_cmagm, c='slategrey', label='T$_{cal}$ GO+CMAES')
# plt.scatter(TRfit[:,0],TRfit[:,2], c='gold', s=10, label='R$_{cal}$ PUMA')
# plt.scatter(TRfit[:,0],TRfit[:,1], c='lightgrey', s=10, label='T$_{cal}$ PUMA')
plt.xlabel('Wavelength [nm]', fontsize=13)
plt.ylabel('Reflectance (R), Transmittance (T)', fontsize=13)
plt.legend(fontsize=11)
plt.show()


plt.figure()
plt.plot(lam, n_cmatl, c='slateblue', label='n$_{cal}$ TLO+CMAES')
plt.plot(lam, k_cmatl, c='forestgreen', label='k$_{cal}$ TLO+CMAES')
plt.plot(lam, n_cmagm, c='navy', label='n$_{cal}$ GO+CMAES')
plt.plot(lam, k_cmagm, c='darkolivegreen', label='k$_{cal}$ GO+CMAES')
# plt.scatter(nkfit[:,0],nkfit[:,1], c='powderblue', s=10, label='n$_{cal}$ PUMA')
# plt.scatter(nkfit[:,0],nkfit[:,2], c='greenyellow', s=10, label='k$_{cal}$ PUMA')
plt.xlabel('Wavelength [nm]', fontsize=12)
plt.ylabel('Refractive Index (n), Extinction Coefficient (k)', fontsize=12)
plt.legend(fontsize=11)
plt.show()


plt.figure()
plt.plot(result['CMA_evolution_TLO'][0], label= 'TLO+CMAES')
plt.plot(result['CMA_evolution_GO'][0], label= 'GO+CMAES')
plt.plot(result['GA_evolution_TLO'][0], label= 'TLO+GA')
plt.plot(result['GA_evolution_GO'][0], label= 'GO+GA')
plt.xlim((-2,500))
plt.ylim((-2,100))
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Estimation Error', fontsize=12)
plt.legend(fontsize=11)
plt.show()

