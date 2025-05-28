#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:49:44 2024

@author: celopetegui
"""
import numpy as np
from scipy.sparse import coo_array

def a_matrix(Nmax):
    row=[i for i in range(Nmax)]
    col=[i+1 for i in range(Nmax)]
    data=[np.sqrt(i+1) for i in range(Nmax)]
    
    a=coo_array((data,(row,col)),shape=(Nmax+1,Nmax+1),dtype=np.complex128)
    return a
    
def a_dag_matrix(Nmax):
    row=[i+1 for i in range(Nmax)]
    col=[i for i in range(Nmax)]
    data=[np.sqrt(i+1) for i in range(Nmax)]
    
    adag=coo_array((data,(row,col)),shape=(Nmax+1,Nmax+1),dtype=np.complex128)
    return adag

def quadrature_operator(Nmax,context):
    a=a_matrix(Nmax)
    a_dag=a_dag_matrix(Nmax)
    
    quad=(a*np.exp(-1j*context)+a_dag*np.exp(1j*context))/np.sqrt(2)  # vacuum noise =1/2 consistent witht the current implementation of get_psi_q_theta
    return quad
def pre_compute_all_quadrature_powers(context,k,Nmax):
    quad_matrix=quadrature_operator(Nmax+2*k+1,context)
    aux=coo_array(np.eye(quad_matrix.shape[0],dtype=np.complex128))
    matr_array=[]
    for i in range(2*k+1):
        matr_array.append(aux.toarray()[:Nmax+1,:Nmax+1])
        aux=quad_matrix@aux
    return matr_array
        
    
