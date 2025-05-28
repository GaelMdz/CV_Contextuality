import numpy as np
import scipy as sp
def num_monoms(d,K):
    """
    # d:      number of variables
    # K:      highest order to consider
    # return: number of monomials in the basis of polynomials up to order K, given by Comb(d+K,K)
    # Notice that the actual number of variables involved in the optimization includes moments of monomials up to order 2K
    # This is... comb(d+2K,2K)
    """
    return sp.special.comb(d+K,K,exact=True)

def combinations_with_replacement_nvar(nvar, r):
    """
     combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
     nvar :   number of variables to combine
     r    : order of the monomial (number of variables in each combination counting repetitions)

     This function is barely modified from the itertools implementation to allow for the the number of variables to come as a parameter
    """
    pool = ['x%d' % (i+1) for i in range(nvar)]  # list of the type ['x1','x2',...,'xnvar']
    if not nvar and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != nvar - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)

def get_monomial_basis(d,K):
    """
    # d       : number of variables
    # K       : highest order of the monomials to consider
    # returns : all monomials of nvar variables up to order K
    #           the output shape is (s(K,d),d) where s(K,d)=C_{K}^{K+d} is the number of possible monomials
    #           each element of the output [alpha1,...,alphad] is so that the corresponding monomial, for variables 
    #           x1,x2,...,xd is given by x1^alpha1 ... xd^alphad
    """
    exps=[]
    vars=['x%d' % (i+1) for i in range(d)]
    for k in range(K+1):
        for i in combinations_with_replacement_nvar(d,k):
          count=[0]*d
          iter=0
          for j in i:
            if j==vars[iter]:
              count[iter]+=1
            else:
              for l in range(iter+1,d):
                if j==vars[l]:
                  count[l]+=1
                  iter=l
                  l=d
          exps.append(count)
    return np.array(exps)


def project_monomials(monoms,i,j):
    """
    # monoms : set of monomials in all variables up to order K 
    # i,j    : index of the two variables to which we want to project... i.e we 
    #          want to keep only terms of the kind xi^{alpha_i} x_j^{alpha_j}
    # return : list containing the indices of the projected monomials in the global monomials array,
    #          the index list is so that the corresponding monomials are sorted lexicographically 
    """
    return np.where(np.sum(np.delete(monoms,[i,j],1),1)==0)[0]


#sparse version included
def moment_matrix_components_dictionary(monoms,sparse=True):
    """
    # monoms: list of monomials up to order K 
    # return: dictionary whose keys are monomials of order up to 2K and their values are the corresponding "coordenate" on the moment matrix
    #         example: dict['[0 1 2 3]']=A, so that A_{alpha,beta}=1 if alpha + beta=[0,1,2,3] and zero otherwise 
    #         sum_{alpha} dict[alpha] moment[alpha] corresponnds to the moment matrix
    # if sparse True the output will be consistent with the sparse representation scipy.sparse.coo_array
    # the shape will be (values,Coord_i,Coord_j) so that matr[coord_i[k],coord_j[k]]=value[k]
    # given that these matrices are quite sparse this is the most convenient way to store them
    """
    dictionary=dict()

    if sparse: 
        for i in range(len(monoms)):
          for j in range(i,len(monoms)):
            pattern=monoms[i]+monoms[j]
            patternstr=str(pattern)
            if patternstr in dictionary.keys():
              dictionary[patternstr][0].append(1)
              dictionary[patternstr][1].append(i)
              dictionary[patternstr][2].append(j)
              if i<j:
                dictionary[patternstr][0].append(1)
                dictionary[patternstr][1].append(j)
                dictionary[patternstr][2].append(i)
            else:
              dictionary[patternstr]=[[1],[i],[j]] #format [value],[location_i],[location_j]
              if i<j:
                dictionary[patternstr][0].append(1)
                dictionary[patternstr][1].append(j)
                dictionary[patternstr][2].append(i)
        return dictionary 
    #if not sparse   
    
    for i in range(len(monoms)):
      for j in range(i,len(monoms)):
        pattern=monoms[i]+monoms[j]
        patternstr=str(pattern)
        if patternstr in dictionary.keys():
          dictionary[patternstr][i,j]=1
          if i<j:
            dictionary[patternstr][j,i]=1
        else:
          dictionary[patternstr]=np.zeros([len(monoms),len(monoms)])
          dictionary[patternstr][i,j]=1
          if i<j:
            dictionary[patternstr][j,i]=1
    return dictionary


def moment_matrix_components_dictionary_two_variables(K,sparse=True):
    """
    # K:      highest order for the polynomial... order of the relaxation 
    # return: dictionary whose keys are monomials of two variables of order up to 2K and their values are the corresponding "coordenate" on the moment matrix
    #         example: dict['[0 1]']=A, so that A_{alpha,beta}=1 if alpha + beta=[0,1] and zero otherwise 
    #         sum_{alpha} dict[alpha] moment[alpha] corresponnds to the moment matrix
    """
    monoms=get_monomial_basis(2,K) #consider d=2
    dictionary=dict()
    if sparse: 
        for i in range(len(monoms)):
          for j in range(i,len(monoms)):
            pattern=monoms[i]+monoms[j]
            patternstr=str(pattern)
            if patternstr in dictionary.keys():
              dictionary[patternstr][0].append(1)
              dictionary[patternstr][1].append(i)
              dictionary[patternstr][2].append(j)
              if i<j:
                dictionary[patternstr][0].append(1)
                dictionary[patternstr][1].append(j)
                dictionary[patternstr][2].append(i)
            else:
              dictionary[patternstr]=[[1],[i],[j]] #format [value],[location_i],[location_j]
              if i<j:
                dictionary[patternstr][0].append(1)
                dictionary[patternstr][1].append(j)
                dictionary[patternstr][2].append(i)
        return dictionary 
    for i in range(len(monoms)):
      for j in range(i,len(monoms)):
        pattern=monoms[i]+monoms[j]
        patternstr=str(pattern)
        if patternstr in dictionary.keys():
          dictionary[patternstr][i,j]=1
          if i<j:
            dictionary[patternstr][j,i]=1
        else:
          dictionary[patternstr]=np.zeros([len(monoms),len(monoms)])
          dictionary[patternstr][i,j]=1
          if i<j:
            dictionary[patternstr][j,i]=1
    return dictionary


def moment_matrix(K,moments,separate=True,sparse=True):
    """
    # d:       2 (only compute moments of up to 2 variables)
    # K:       order of the relaxation (highest order moments)
    # moments: list of moments of all monomials of the two variables up to order 2K 
    # separate: bool, if False returns the corresponding matrix of moments
    #                 if True returns the list of matrices corresponding to the "coordinates" of each monomial
    #                         on the moment matrix multiplied by the value of the moment. This we can link to 
    #                         each optimization variable and should be the good way to get the input for SDPA optimization 
    # return: matrix of moments (see the form of the output above)
    #         shape: if separate=True -> List of numpy arrays [num_monoms(2,2K)][num_monoms(2,K),num_monoms(2,K)] 
    #                if separate=False-> np array [num_monoms(2,K),num_monoms(2,K)]
    """
    monoms2K=get_monomial_basis(2,2*K) # monomial basis up to order 2K (directly mapeable to the moments array)
    #first we get the dictionary
    dict_2K=moment_matrix_components_dictionary_two_variables(K,sparse)

    if sparse:
        if separate: 
            coordinates_monoms_2K=[]
            # now we can link the elements of the dictionary to the corresponding index in the list of moments (monomials of order 2K)
            for i in range(len(monoms2K)):
                dict_2K[str(monoms2K[i])][0]=np.array(dict_2K[str(monoms2K[i])][0])*moments[i]
                coordinates_monoms_2K.append(dict_2K[str(monoms2K[i])])
            return coordinates_monoms_2K
        else: 
            moment_matrix=np.zeros(dict_2K['[0 0]'].shape)
            for i in range(len(monoms2K)):
                moment_matrix+=moments[i]*dict_2K[str(monoms2K[i])].toarray()
            return moment_matrix
    
    if separate: 
        coordinates_monoms_2K=[]
        # now we can link the elements of the dictionary to the corresponding index in the list of moments (monomials of order 2K)
        for i in range(len(monoms2K)):
            coordinates_monoms_2K.append(dict_2K[str(monoms2K[i])]*moments[i])
        return coordinates_monoms_2K
    else: 
        moment_matrix=np.zeros(dict_2K['[0 0]'].shape)
        for i in range(len(monoms2K)):
            moment_matrix+=moments[i]*dict_2K[str(monoms2K[i])]
        return moment_matrix
        
        
def moment_matrix_coordinates_indexed(d,K,sparse=True): 
    """
    # d:      number of variables 
    # K:      order of the relaxation
    # return: indexed contribution of each optimization variable for the moment matrix 
    """
    monoms2K=get_monomial_basis(d,2*K) # monomial basis up to order 2K (directly mapeable to the moments array)
    monomsK=get_monomial_basis(d,K)
    
    dict_2K=moment_matrix_components_dictionary(monomsK,sparse)

    coordinates_monoms_2K=[]
    # now we can link the elements of the dictionary to the corresponding index in the list of moments (monomials of order 2K)
    for i in range(len(monoms2K)):
        coordinates_monoms_2K.append(dict_2K[str(monoms2K[i])])
    return coordinates_monoms_2K
#sparse version included
def localizing_matrix_components_dictionary(monoms,p,sparse=True):
    """
    # monoms: list of monomials up to order Kdims 
    # p:      localizing polynomial. Ex: A-x_1^2-x_2^2-...-x_N^2 is the localizing polynomial for a ball in N-dimensions. 
    #         format of the input: p=[coeffs,monoms] Ex: for the polynomial of the ball 
    #                              coeffs=[A,-1,-1,...,-1]
    #                              monoms=[[0,...,0],[2,0,...,0],[0,2,...,0],...,[0,...,0,2]]
    # return: dictionary whose keys are monomials of order up to 2K and their values are the corresponding "coordenate" on the moment matrix
    #         example: dict['[0 1 2 3]']=A, so that A_{alpha,beta}=1 if alpha + beta=[0,1,2,3] and zero otherwise 
    #         sum_{alpha} dict[alpha] moment[alpha] corresponnds to the moment matrix
    # if sparse True the output will be consistent with the sparse representation scipy.sparse.coo_array
    # the shape will be (values,Coord_i,Coord_j) so that matr[coord_i[k],coord_j[k]]=value[k]
    # given that these matrices are quite sparse this is the most convenient way to store them
    """
    dictionary=dict()

    if sparse: 
        for k in range(len(p[0])):
            for i in range(len(monoms)):
              for j in range(i,len(monoms)):
                pattern=monoms[i]+monoms[j]+p[1][k]
                patternstr=str(pattern)
                if patternstr in dictionary.keys():
                  dictionary[patternstr][0].append(p[0][k])
                  dictionary[patternstr][1].append(i)
                  dictionary[patternstr][2].append(j)
                  if i<j:
                    dictionary[patternstr][0].append(p[0][k])
                    dictionary[patternstr][1].append(j)
                    dictionary[patternstr][2].append(i)
                else:
                  dictionary[patternstr]=[[p[0][k]],[i],[j]] #format [value],[location_i],[location_j]
                  if i<j:
                    dictionary[patternstr][0].append(p[0][k])
                    dictionary[patternstr][1].append(j)
                    dictionary[patternstr][2].append(i)
        return dictionary 
    #if not sparse   
    
    for i in range(len(monoms)):
      for j in range(i,len(monoms)):
        pattern=monoms[i]+monoms[j]+p[1][k]
        patternstr=str(pattern)
        if patternstr in dictionary.keys():
          dictionary[patternstr][i,j]=p[0][k]
          if i<j:
            dictionary[patternstr][j,i]=p[0][k]
        else:
          dictionary[patternstr]=np.zeros([len(monoms),len(monoms)])
          dictionary[patternstr][i,j]=p[0][k]
          if i<j:
            dictionary[patternstr][j,i]=p[0][k]
    return dictionary

def localizing_matrix_coordinates_indexed(d,K,p,sparse=True): 
    """
    # d:      number of variables 
    # K:      order of the relaxation
    # p:      localizing polynomial. Ex: A-x_1^2-x_2^2-...-x_N^2 is the localizing polynomial for a ball in N-dimensions. 
    #         format of the input: p=[coeffs,monoms] Ex: for the polynomial of the ball 
    #                              coeffs=[A,-1,-1,...,-1]
    #                              monoms=[[0,...,0],[2,0,...,0],[0,2,...,0],...,[0,...,0,2]]
    # return: indexed contribution of each optimization variable for the moment matrix 
    """
    #relevant properties of the localizing polynomial: 
    Kpol=np.max(np.sum(np.array(p[1]),axis=1))  # order of the polynomial 

    Kdims=K-int(np.ceil(Kpol/2))  # order of the monomials leading to the formation of the localizing matrix
    
    monoms2K=get_monomial_basis(d,2*K) # monomial basis up to order 2K (directly mapeable to the moments array)
    #monomsKdims=get_monomial_basis(d,Kdims)
    monomsK=get_monomial_basis(d,K)
    dict_2K=localizing_matrix_components_dictionary(monomsK,p,sparse)

    coordinates_monoms_2K=[]
    # now we can link the elements of the dictionary to the corresponding index in the list of moments (monomials of order 2K)
    for i in range(len(monoms2K)):
        coordinates_monoms_2K.append(dict_2K[str(monoms2K[i])])
    return coordinates_monoms_2K


def project_moment_matrix_coordinates_indexied(d,K,i,j,sparse=True,sparse_global=True):
    """
    # d:      number of variables 
    # K:      order of the relaxation
    # i,j:    index of the variables of the context to which we are projecting the variables. 
    # return: indexed contribution of each optimization variable for the projected moment matrix... zero for everyone but for the projected 
    #         monomials, indexed by the output of project_monomials
    #         if sparse=True:
    #                the output consists of the list of indices to the "projected variables" (it is the set of monomials corresponding to the relevant context)
    #                and the list of their contribution to the context moment matrix
    #         if sparse=False: 
    #                returns a 3D array of shape [num_monoms(d,2K),num_monoms(2,K),num_monoms(2,K)], with a matrix of zeros [i,:,:] if the monomial is not in the relevant context 
    """
    n_monoms=num_monoms(d,2*K)
    n_monomsK=num_monoms(2,K)
    monoms2K=get_monomial_basis(d,2*K)
    
    proj_indices=project_monomials(monoms2K,i,j)
    moment_matrix_context=moment_matrix_coordinates_indexed(2,K,sparse_global)
    
    if sparse_global:
        coordinates_monoms=[sp.sparse.coo_array((n_monomsK,n_monomsK))]*n_monoms   #  check whether is better to pass as a matrix or with the references to indices and values cause I think to pass to the sdpa the second is better 
        if sparse:
            return proj_indices,moment_matrix_context
        for i in range(len(proj_indices)):
            coordinates_monoms[proj_indices[i]]=sp.sparse.coo_array((np.array(moment_matrix_context[i][0]),(np.array(moment_matrix_context[i][1]),np.array(moment_matrix_context[i][2]))))
        return coordinates_monoms
    if sparse: 
        return proj_indices,moment_matrix_context
    
    coordinates_monoms=np.zeros([n_monoms,n_monomsK,n_monomsK]) 
    for i in range(len(proj_indices)):
        coordinates_monoms[proj_indices[i],:,:]=moment_matrix_context[i]
    return coordinates_monoms  
    
    
