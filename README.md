# Optical-Inverse-Solutions-
Extracting Film Thickness and Optical Constants from Spectrophotometric Data by Evolutionary Optimization


**1.** Input Data: These data files contain the spectral data, reflectance (R) and transmittance (T), and the film characteristics, thickness (d) and optical constants (n, k) for various materials. 'TrainTestData.npz' contains fully synthetic data, 'perovData.npz' contains semi-synthetic data, and 'ExpData.npz' contains experimentally measured data. Note that the file: 'Refractive_Index_Library' contains a wide range of references for different refractive index and extinction coefficient profiles.  

**2.** The optical dispersion models, ensembles of Tauc-Lorentz and Gaussian oscillators, are implemented in 'VensembleTL.py', 'mixtureGO.py', while 'ScatteringMatrix.py' is used to execute the transfer matrix method. For known film thickness (d) and optical constants (n, k), 'evaluateModel.py' emulates the outpout reflectance and transmittance. 

**3.** The application of the evolutionary algorithm, covariance matrix adaptation evolution strategy (CMAES), is carried out using the codes: 'optimizer_ensembleTL.py' (single run), 'optimizer_ensembleTLO_4mtxt.py' (single run), 'CMAloop_TL.py' (multiple runs), 'optimizer_mixtureGO.py' (single run), 'CMAloop_GO.py' (multiple runs).  
