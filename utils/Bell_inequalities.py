#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:29:16 2024

@author: celopetegui
"""
import numpy as np
from . import operators as ops
from . import monomials as monomials

def Bell_inequality(MCs,XCs):
    """
    Parameters
    ----------
    MCs : moment matrices (n_contexts,n_monoms,n_monoms)
    XCs : Matrices (same shape as MCs)
        encode the coefficients of each moment in the Bell inequality. The correspond to variables 2->C+1 of the Dual problem. (indexing from 1 on)

    Returns
    -------
    value of the Bell inequality for the given empirical model

    """
    
    
    return 1-np.einsum('nij,nji',MCs,XCs)


def Bell_inequality_operator(XCs,Nmax,varsA,varsB,k):
    """
    

    Parameters
    ----------
    XCs : real tensor (n_context,n_monoms,n_monoms)
        Encode the coefficients of each moment in the Bell inequality.
    Nmax : INT
        Maximum number of photons in the Fock representation of the quadrature operators
    varsA,varsB: list 
        set of variables measured in each mode
        contexts : list (n_context,2)
        list of all the contexts contexts[i]=(varsA[i/nvarsB],varsB[i mod nvarsB]) 
    Returns
    -------
    Bell : Matrix (Nmax+1,Nmax+1)
        Matrix corresponding to the linear Bell inequality defined by XCs. 
        Tr[\rho Bell] is the value of the Bell inequality applied on state rho 
        

    """
    #Precompute all quadrature powers for each 
    
    quadsApowers=[ops.pre_compute_all_quadrature_powers(varsA[i], k,Nmax) for i in range(len(varsA))]
    quadsBpowers=[ops.pre_compute_all_quadrature_powers(varsB[i], k,Nmax) for i in range(len(varsB))]

    contexts=[]
    for i in range(len(varsA)):
        for j in range(len(varsB)):
            contexts.append([varsA[i],varsB[j]])
    Bell=np.eye((Nmax+1)*(Nmax+1),dtype=np.complex128)
    indexMonomials=monomials.moment_matrix_coordinates_indexed(2,k,sparse=False)
    monoms2K=monomials.get_monomial_basis(2,2*k)
    for i in range(len(contexts)):
        for j in range(len(indexMonomials)):
            aux=0-np.kron((quadsApowers[int(i/len(varsB))][monoms2K[j][0]]),(quadsBpowers[np.mod(i,len(varsB))][monoms2K[j][1]]))*np.sum(np.array(indexMonomials[j])*XCs[i])
            
            Bell+=aux
    return Bell