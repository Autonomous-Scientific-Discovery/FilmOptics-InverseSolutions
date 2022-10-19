# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:37:06 2022

@author: Rajdeep Dutta
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

lam=np.linspace(350,1000,651)
result = pickle.load(open('resultB_noise.pickle', 'rb'))

fig=plt.figure()
ax1=fig.add_subplot(231)
ax1.plot(lam,result['R'][0],'k:')
ax1.plot(lam,result['T'][0],'g:')
ax1.plot(lam,result['R_hat'][0])
ax1.plot(lam,result['T_hat'][0])
# ax1.set_xlabel('Wavelength (nm)', fontsize=9)
ax1.set_ylabel('Reflectance & Transmittance')
ax1=fig.add_subplot(232)
ax1.plot(lam,result['R'][1],'k:', label='R_actual')
ax1.plot(lam,result['T'][1],'g:', label='T_actual')
ax1.plot(lam,result['R_hat'][1], label='R_estimate')
ax1.plot(lam,result['T_hat'][1], label='T_estimate')
plt.legend(fontsize=9)
ax1.set_xlabel('Wavelength (nm)')
ax1=fig.add_subplot(233)
ax1.plot(lam,result['R'][2],'k:')
ax1.plot(lam,result['T'][2],'g:')
ax1.plot(lam,result['R_hat'][2])
ax1.plot(lam,result['T_hat'][2])
# ax1.set_xlabel('Wavelength (nm)', fontsize=9)
ax4=fig.add_subplot(234)
ax4.plot(result['error_evol'][0],'k')
ax4.set_xlabel('Iterations', fontsize=9)
ax4.set_ylabel('Estimation Error')
ax5=fig.add_subplot(235)
ax5.plot(result['error_evol'][1],'k')
ax5.set_xlabel('Iterations', fontsize=9)
ax6=fig.add_subplot(236)
ax6.plot(result['error_evol'][2],'k')
ax6.set_xlabel('Iterations', fontsize=9)
plt.show()



plt.figure()
plt.plot(lam, result['R_hat'][0], label='R_estimate without noise$_m$')
plt.plot(lam, result['T_hat'][0], label='T_estimate without noise$_m$')
plt.plot(lam, result['R_hat'][1], label='R_estimate with 5% noise$_m$')
plt.plot(lam, result['T_hat'][1], label='T_estimate with 5% noise$_m$')
plt.plot(lam, result['R_hat'][2], label='R_estimate with 10% noise$_m$')
plt.plot(lam, result['T_hat'][2], label='T_estimate with 10% noise$_m$')
plt.xlim([350,700])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance & Transmittance')
plt.legend()
plt.show()

plt.figure()
plt.plot(lam, result['n_hat'][0], label='n_estimate without noise$_m$')
plt.plot(lam, result['k_hat'][0], label='k_estimate without noise$_m$')
plt.plot(lam, result['n_hat'][1], label='n_estimate with 5% noise$_m$')
plt.plot(lam, result['k_hat'][1], label='k_estimate with 5% noise$_m$')
plt.plot(lam, result['n_hat'][2], label='n_estimate with 10% noise$_m$')
plt.plot(lam, result['k_hat'][2], label='k_estimate with 10% noise$_m$')
plt.xlim([350,700])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Refractive index & Extinction coefficient')
plt.legend()
plt.show()