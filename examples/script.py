from cv_sdp import CV_Bell_inequality_LP, CV_Bell_Ineq_state
import cv_sdp.utils.moments as moments

## Import state
import numpy as np
import scipy.io as sio

rho7=sio.loadmat('/Users/celopetegui/Documents/PhD/Work/Bell_ineqs/CV_contextuality_SDP/cv_sdp/data/rho_dim_7_SB_max_CHSH.mat')['rho_dim_7_SB_max_CHSH']
eigv7,eigvects7=np.linalg.eig(rho7)
C=eigvects7[:,np.argmax(eigv7)]
Nmax=int(np.sqrt(C.shape[0])-1)
n1=np.kron(np.arange(Nmax+1),np.ones(Nmax+1))
n2=np.kron(np.ones(Nmax+1),np.arange(Nmax+1))

nvarsA=2
varsA=[1.1872,2.4058]
nvarsB=2
varsB=[1.7331,2.9517]
cv_bell_state=CV_Bell_Ineq_state([C,n1,n2],varsA,varsB,1024,7,method="LP",N_bins_eff=32)

cv_bell_state.evaluate()