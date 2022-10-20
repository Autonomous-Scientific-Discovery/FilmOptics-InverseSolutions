# Optical-Inverse-Solutions-
**Extracting Film Thickness and Optical Constants from Spectrophotometric Data by Evolutionary Optimization**


**1.** _Input Data:_ These data files contain the spectral data, reflectance (R) and transmittance (T), and the film characteristics, thickness (d) and optical constants (n, k) for various materials. 'TrainTestData.npz' contains fully synthetic data, 'perovData.npz' contains semi-synthetic data, and 'ExpData.npz' contains experimentally measured data. Note that the file: 'Refractive_Index_Library' contains a wide range of references for different refractive index and extinction coefficient profiles. Also, 'Rcal-CGO.txt' and 'Tcal-CGO.txt' files contain spectral data for performance comparison with existing methods.   

**2.** _Adopted Models:_ The optical dispersion models, ensembles of Tauc-Lorentz and Gaussian oscillators, are implemented in 'VensembleTL.py', 'mixtureGO.py', while 'ScatteringMatrix.py' is used to execute the transfer matrix method. For known film thickness (d) and optical constants (n, k), 'evaluateModel.py' emulates the outpout reflectance and transmittance. Further details are available in: https://github.com/PV-Lab/thicknessML/tree/main/data/utils . 

**3.** _Evolutionary Optimization:_ The application of the evolutionary algorithm, covariance matrix adaptation evolution strategy (CMAES), is carried out using the codes: 'optimizer_ensembleTL.py' (single run), 'optimizer_ensembleTLO_4mtxt.py' (single run), 'CMAloop_TL.py' (multiple runs), 'optimizer_mixtureGO.py' (single run), 'CMAloop_GO.py' (multiple runs). The CMAES algorithm is implemented using the package 'cma' available at: https://pypi.org/project/cma/ .  
