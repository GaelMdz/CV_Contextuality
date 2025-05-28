import numpy as np
from scipy.special import factorial as fact
from math import sqrt 
from scipy.special import factorial2 as fact2
import scipy.integrate as integrate
from math import pow
import sympy as sp 


##Note: The moments that are computed in the Fock basis correspond to the convention x=(a+a^\dagger)
##      For the moments computed using Hermite polynomials and half rescaling the convention is x=(a+a^\dagger)/sqrt(2)
##      For the moments computed using Hermite polynomials the convention is x=(a+a^\dagger)/sqrt(2)

### To compute the moments of general Fock superpositions 

def compute_kl(n,k,l):
    if np.abs(np.mod(n,2)-np.mod(np.abs(k-l),2))==1:
        return 0
    if np.abs(k-l)>n: 
        return 0
    liminf=int((n-k-l)/2)
    liminf=np.max([0,liminf])
    limsup=int((n-np.abs(l-k))/2)
    
    s=0
    for i in range(liminf,limsup+1):
        s+=1/(pow(2,i)*fact(i)*fact(int((n+l-k)/2)-i)*fact(int((n+k-l)/2)-i)*fact((l+k-n)/2+i))
    return s*fact(n)*np.sqrt(fact(k)*fact(l))

def moment_superp(theta1,theta2,n1,n2,C,n,m):
    """
    I consider a general superposition of the form sum_{i} C_{i} |n_i,m_i>
    Input about the moment: 
    theta1,theta2: quadratures of which the moments are computed 
    n1,n2 order of the moment
    
    Input about the state: 
    C=[c_1,...,c_M]
    n=[n_1,...,n_M]
    m=[m_1,...,m_M]
    """
    mom=0
    for i in range(len(C)):
        for j in range(len(C)):
            mom+=(np.conj(C[i])*C[j])*(1j*np.sin(((n[i]-n[j])*theta1 + (m[i]-m[j])*theta2))+np.cos(((n[i]-n[j])*theta1 + (m[i]-m[j])*theta2)))*compute_kl(n1,n[i],n[j])*compute_kl(n2,m[i],m[j])
             
            
    return mom

def moment_superp_half_scale(theta1,theta2,n1,n2,C,n,m):
    """
    I consider a general superposition of the form sum_{i} C_{i} |n_i,m_i>
    Input about the moment: 
    theta1,theta2: quadratures of which the moments are computed 
    n1,n2 order of the moment
    
    Input about the state: 
    C=[c_1,...,c_M]
    n=[n_1,...,n_M]
    m=[m_1,...,m_M]
    """
    mom=0
    for i in range(len(C)):
        for j in range(len(C)):
            mom+=(np.conj(C[i])*C[j])*(1j*np.sin(((n[i]-n[j])*theta1 + (m[i]-m[j])*theta2))+np.cos(((n[i]-n[j])*theta1 + (m[i]-m[j])*theta2)))*compute_kl(n1,n[i],n[j])*compute_kl(n2,m[i],m[j])/(pow(np.sqrt(2),n1)*pow(np.sqrt(2),n2))
             
            
    return mom

## PR box 
def moment_PR(n1,n2,a,b,C):
    if C==1: 
        return 1/2*(((-a)**n1)*((-b)**n2)+(a**n1)*(b**n2))
    return 1/2*((-a)**n1*(b)**n2+a**n1*(-b)**n2)

def moment_PR_N(n1,n2,a,b,C):
    mom=0
    if C==1: 
        for i in range(len(a)):
            mom+=1/(2*len(a))*(((-a[i])**n1)*((-b[i])**n2)+(a[i]**n1)*(b[i]**n2))
        return mom
    for i in range(len(a)):
        mom+=1/(2*len(a))*(((-a[i])**n1)*((b[i])**n2)+(a[i]**n1)*((-b[i])**n2))
    return mom


def displ_Gaussian(mu,sigma,x_list):
    return np.exp(-(x_list-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def PR_Gaussian(a,b,C,sigma,x_list):
    """
    

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.
    sq : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    if C==1:
        pr=1/2*(np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(b, sigma, x_list))+np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(-b, sigma, x_list)))
    else:
        pr=1/2*(np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(b, sigma, x_list)))
    
    
    
    return pr
def moment_PR_Gaussian(n1,n2,a,b,C,sigma,x_list):
    pr=PR_Gaussian(a, b, C, sigma, x_list)
    return np.sum(np.einsum('i,j->ij',(x_list**n1),(x_list**n2))*pr)*(x_list[1]-x_list[0])**2

def moment_not_PR(n1,n2,a,b,C):
    #if C==1: 
    #    return 1/2*(((-a)**n1)*((-b)**n2)+(a**n1)*(b**n2))
    return 1/2*((-a)**n1*(b)**n2+a**n1*(-b)**n2)

def pr_CHSH(a,b,C,eta1,sigma,x_list):
    """
    

    Parameters
    ----------
    a : float, peaks of the measurement prob distribution in Alice's side are in +- A
    b : float, peaks of the measurement prob distribution in Bob's side are in +- A
    C : int \in {1,2,3,4}, label for the context
    eta1 : float \in [0,1/2],  defined in PE's thesis eta2=1/2-eta1
    sigma : float \in [0,infinity], variance of each peak
    x_list : list of values defining the grid for each mode (the multimode grid is obtained through einsum('i,j->ij'))
    
    Returns
    -------
    pr : probability distribution corresponding to the CHSH PR like (page 38, table 1.2 PE's PhD thesis (changed 1<->4)')
         0->-a(b)
         1-> a(b)
    """
    eta2=1/2-eta1
    if C==1: 
        pr=eta2*np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+eta1*np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(b, sigma, x_list))+eta1*np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+eta2*np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(b, sigma, x_list))
    else:
        pr=eta1*np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+eta2*np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(b, sigma, x_list))+eta2*np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+eta1*np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(b, sigma, x_list))

    return pr
def moments_CHSH(n1,n2,a,b,C,eta1,sigma,x_list):
    """
    

    Parameters
    ----------
    n1 : int
        order of the variable from Alice's side in the  monomial
    n2 : int
        order of the variable from Bob's side in the  monomial
    a : float, peaks of the measurement prob distribution in Alice's side are in +- A
    b : float, peaks of the measurement prob distribution in Bob's side are in +- A
    C : int \in {1,2,3,4}, label for the context
    eta1 : float \in [0,1/2],  defined in PE's thesis eta2=1/2-eta1
    sigma : float \in [0,infinity], variance of each peak
    x_list : list of values defining the grid for each mode (the multimode grid is obtained through einsum('i,j->ij'))
    

    Returns
    -------
    moment: value of the moment

    """
    eta2=1/2-eta1
    if sigma==0:
        if C==1:
            return eta2*(-a)**n1*(-b)**n2+eta1*(-a)**n1*b**n2+eta1*(a)**n1*(-b)**n2+eta2*(a)**n1*b**n2
        return eta1*(-a)**n1*(-b)**n2+eta2*(-a)**n1*b**n2+eta2*(a)**n1*(-b)**n2+eta1*(a)**n1*b**n2
    
    pr=pr_CHSH(a, b, C,eta1, sigma, x_list)
    return np.sum(np.einsum('i,j->ij',(x_list**n1),(x_list**n2))*pr)*(x_list[1]-x_list[0])**2
def pr_general_table(a,b,C,table,sigma,x_list):
    """
    

    Parameters
    ----------
    a : float, peaks of the measurement prob distribution in Alice's side are in +- A
    b : float, peaks of the measurement prob distribution in Bob's side are in +- A
    C : int \in {1,2,3,4}, label for the context
    table : Probabily table 
    sigma : float \in [0,infinity], variance of each peak
    x_list : list of values defining the grid for each mode (the multimode grid is obtained through einsum('i,j->ij'))
    
    Returns
    -------
    pr : probability distribution corresponding to the CHSH PR like (page 38, table 1.2 PE's PhD thesis (changed 1<->4)')
         0->-a(b)
         1-> a(b)
    """
    
    pr=table[C][0]*np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+table[C][1]*np.einsum('i,j->ij',displ_Gaussian(-a,sigma,x_list),displ_Gaussian(b, sigma, x_list))+table[C][2]*np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(-b, sigma, x_list))+table[C][3]*np.einsum('i,j->ij',displ_Gaussian(a,sigma,x_list),displ_Gaussian(b, sigma, x_list))

    return pr
def moments_general_table(n1,n2,a,b,C,table,sigma,x_list):
    """
    

    Parameters
    ----------
    n1 : int
        order of the variable from Alice's side in the  monomial
    n2 : int
        order of the variable from Bob's side in the  monomial
    a : float, peaks of the measurement prob distribution in Alice's side are in +- A
    b : float, peaks of the measurement prob distribution in Bob's side are in +- A
    C : int \in {1,2,3,4}, label for the context
    table : prob outcome table, like table 1 PRL 119,050504
    sigma : float \in [0,infinity], variance of each peak
    x_list : list of values defining the grid for each mode (the multimode grid is obtained through einsum('i,j->ij'))
    
    
    Returns
    -------
    moment: value of the moment
    
    """
    table_probs=[[1/2,0,0,1/2],[3/8,1/8,1/8,3/8],[3/8,1/8,1/8,3/8],[1/8,3/8,3/8,1/8]]
    if sigma==0:
        return table_probs[C][0]*(-a)**n1*(-b)**n2+table_probs[C][1]*(-a)**n1*b**n2+table_probs[C][2]*(a)**n1*(-b)**n2+table_probs[C][3]*(a)**n1*b**n2
    
    pr=pr_general_table(a, b, C,table, sigma, x_list)
    return np.sum(np.einsum('i,j->ij',(x_list**n1),(x_list**n2))*pr)*(x_list[1]-x_list[0])**2

    
    

## moment from marginal prob distribution 
def hermite_functions(n_max, x):
    """
    Subrutine to calculate Hermite functions used for quadrature distribution
    Parameters
    ----------
    n_max : Integer
        Index of largest Hermite function.
    x : array
        x-axis.
    Returns
    -------
    Hlist : list
        List of Hermite functions.
    """
    Hlist = []
    Hlist.append(np.pi**(-1/4)*np.exp(-1/2*x**2))
    if n_max > 0:
        Hlist.append(np.sqrt(2)*x*np.pi**(-1/4)*np.exp(-1/2*x**2))
        for n in range(1, n_max):
            Hlist.append((x*Hlist[n] - np.sqrt(n/2)*Hlist[n-1])/np.sqrt((n + 1)/2))
    return Hlist



def get_psi_q_theta(coeff,n1,n2,nmax,theta1,theta2,p_list):
    """
    

    Parameters
    ----------
    coeff : array of complex numbers
        The coefficients of the Fock state expansion
    cutoff : integer
        cutoff of the representation
    p_list : array of doubles
        Ddiscretization of the p quadrature axes

    Returns
    -------
    psi : array of complex numbers
        p-wave function of the corresponding state 

        The ordering is such that the first index (i) runs over the first mode
        and the second index (j) runs over the second mode. To have an ordering equivalent to the
        real axis representation (firs mode second index, and second mode fliped) need to apply flip(transpose(psi),axis=0)

    """
    Hermite_list=np.array(hermite_functions(nmax,p_list))
    psi=np.zeros((p_list.shape[0],p_list.shape[0]), dtype='complex')
    
    for i in range(len(coeff)):
        psi += (np.exp(-1j*(n1[i]*theta1+n2[i]*theta2)))*coeff[i]*np.einsum('i,j->ij',Hermite_list[n1[i]],Hermite_list[n2[i]])
    return psi

def get_psi_sigmoid_q_theta(coeff,n1,n2,nmax,theta1,theta2,p_list):
    """
    

    Parameters
    ----------
    coeff : array of complex numbers
        The coefficients of the Fock state expansion
    cutoff : integer
        cutoff of the representation
    p_list : array of doubles
        Ddiscretization of the p quadrature axes

    Returns
    -------
    psi : array of complex numbers
        p-wave function of the corresponding state 

    """
    inv_sigmoid=np.vectorize(lambda x: np.log(x/(1-x)))
    Hermite_list=np.array(hermite_functions(nmax,inv_sigmoid(p_list)))
    psi=np.zeros((p_list.shape[0],p_list.shape[0]), dtype='complex')
    
    for i in range(len(coeff)):
        psi += (np.exp(-1j*(n1[i]*theta1+n2[i]*theta2)))*coeff[i]*np.einsum('i,j->ij',Hermite_list[n1[i]],Hermite_list[n2[i]])
    return psi
def get_psi_q_theta_norm(coeff,n1,n2,nmax,theta1,theta2,p_list):
    """
    

    Parameters
    ----------
    coeff : array of complex numbers
        The coefficients of the Fock state expansion
    cutoff : integer
        cutoff of the representation
    p_list : array of doubles
        Ddiscretization of the p quadrature axes

    Returns
    -------
    psi : array of complex numbers
        p-wave function of the corresponding state 

    """
    Hermite_list=np.array(hermite_functions(nmax,p_list))
    psi=np.zeros((p_list.shape[0],p_list.shape[0]), dtype='complex')
    
    for i in range(len(coeff)):
        psi += (np.exp(-1j*(n1[i]*theta1+n2[i]*theta2)))*coeff[i]*np.einsum('i,j->ij',Hermite_list[n1[i]],Hermite_list[n2[i]])
    return psi/np.sqrt(np.sum(np.abs(psi)**2)*(p_list[1]-p_list[0])**2)

def moment_from_marginal(n1,n2,marginal,x_list1,x_list2):
    return np.sum(np.einsum('i,j->ij',(x_list1**n1),(x_list2**n2))*np.abs(marginal)**2)*(x_list1[1]-x_list1[0])*(x_list2[1]-x_list2[0])
def moment_from_marginal_prob_circ(n1,n2,marginal,x_list1,x_list2,R):
    region=np.zeros((len(x_list1),len(x_list2)))
    for i in range(len(x_list1)):
        for j in range(len(x_list2)):
            if x_list1[i]**2+x_list2[j]**2<=R**2:
                region[i,j]=1
    return np.sum(np.einsum('i,j->ij',(x_list1**n1),(x_list2**n2))*region*marginal)
def moment_from_marginal_prob(n1,n2,marginal,x_list1,x_list2):
    
    return np.sum(np.einsum('i,j->ij',(x_list1**n1),(x_list2**n2))*marginal)### corrected the order of the variables to match that in the marginal 

def Sigmoid_moment_from_marginal(n1,n2,marginal,x_list1,x_list2,sigm_factor):
    sigmoid=np.vectorize(lambda x: 2*(1/(1+np.exp(-sigm_factor*x)))-1)
    return np.sum(np.einsum('i,j->ij',(sigmoid(x_list1)**n1),(sigmoid(x_list2)**n2))*np.abs(marginal)**2)*(x_list1[1]-x_list1[0])*(x_list2[1]-x_list2[0])
def Sigmoid_moment_from_marginal_prob(n1,n2,marginal,x_list1,x_list2,sigm_factor):
    sigmoid=np.vectorize(lambda x: 2*(1/(1+np.exp(-sigm_factor*x)))-1)
    return np.sum(np.einsum('i,j->ij',(sigmoid(x_list1)**n1),(sigmoid(x_list2)**n2))*marginal)

def tanh_moment_from_marginal(n1,n2,marginal,x_list1,x_list2,skew_factor):
    
    return np.sum(np.einsum('i,j->ji',(np.cos(skew_factor*x_list1)**n1),(np.cos(skew_factor*x_list2)**n2))*np.abs(marginal)**2)*(x_list1[1]-x_list1[0])*(x_list2[1]-x_list2[0])
def tanh_moment_from_marginal_prob(n1,n2,marginal,x_list1,x_list2,skew_factor):
    
    return np.sum(np.einsum('i,j->ji',(np.tanh(skew_factor*x_list1)**n1),(np.tanh(skew_factor*x_list2)**n2))*marginal)

def cos_moment_from_marginal(n1,n2,marginal,x_list1,x_list2,skew_factor):
    
    return np.sum(np.einsum('i,j->ij',(np.tanh(skew_factor*x_list1)**n1),(np.tanh(skew_factor*x_list2)**n2))*np.abs(marginal)**2)*(x_list1[1]-x_list1[0])*(x_list2[1]-x_list2[0])

def displ_squeezed_state(mu,r,x_list):
    return np.exp(-(x_list-mu)**2/(2*np.exp(-2*r)))/np.sqrt(2*np.pi*np.exp(-2*r))

def superp_squeezed_states(N,r,alpha,coeff,x_list):## add the peak coefficients
    aux=np.zeros(len(x_list))
    n_list=np.arange(N)-(N-1)/2
    delta_x=x_list[1]-x_list[0]
    for i in range(len(n_list)):
        aux+=coeff[i]*displ_squeezed_state(alpha*n_list[i], r, x_list)
    return aux/np.sqrt(np.sum((np.abs(aux)**2)*delta_x))

def superp_squeezed_states_2m(r,alpha1,alpha2,x_list1,x_list2):## add the peak coefficients
    aux=np.zeros((len(x_list1),len(x_list2)))
    
    delta_x=x_list1[1]-x_list1[0]
    for i in range(2):
        aux+=np.einsum('i,j->ij',displ_squeezed_state(alpha1*(-1)**i, r, x_list1),displ_squeezed_state(alpha2*(-1)**i, r, x_list2))
    return aux/np.sqrt(np.sum((np.abs(aux)**2)*delta_x**2))

def get_psi_q_theta_from_q_vector2(psix,x_list,p_list,theta,cutoff):
    """
    ## Works (but be careful about the sign in the exponents and the conventions)

    Parameters
    ----------
    psix : array of complex numbers
        q-wave function of the corresponding state 
    p_list : p_list : array of doubles
        Discretization of the q_theta quadrature axes
    x_list : array of doubles
        Discretization of the x quadrature axes
    theta: angle of the quadrature we want to target

    Returns
    -------
    psi_p: array of complex numbers
        q_theta-wave function of the state

    """
    psi_p=np.zeros(p_list.shape,dtype='complex')
    delta_p=(p_list[1]-p_list[0])
    delta_x=(x_list[1]-x_list[0])
    Hermite_list=hermite_functions(cutoff,x_list)
    for i in range(len(p_list)):
        hermite_aux=hermite_functions(cutoff,p_list[i])
        hermite_prod_sum=np.sum(np.einsum('i,ij->ij',np.array(hermite_aux)*np.exp(1j*theta*np.arange(cutoff+1)),np.array(Hermite_list)),axis=0)
        psi_p[i]=np.sum(delta_x*psix*hermite_prod_sum)
    return psi_p/np.sum(np.abs(psi_p)**2*delta_p)
def get_psi_q_theta_from_q_vector2modes(psix,x_list,p_list,theta1,theta2,cutoff):
    """
    ## Works (but be careful about the sign in the exponents and the conventions)

    Parameters
    ----------
    psix : array of complex numbers
        q-wave function of the corresponding state 
    p_list : p_list : array of doubles
        Ddiscretization of the q_theta quadrature axes
    x_list : array of doubles
        Ddiscretization of the x quadrature axes
    theta: angle of the quadrature we want to target

    Returns
    -------
    psi_p: array of complex numbers
        q_theta-wave function of the state

    """
    
    delta_p=(p_list[1]-p_list[0])
    delta_x=(x_list[1]-x_list[0])
    Hermite_list=hermite_functions(cutoff,x_list)
    Hermite_matrix=np.einsum('ij,kl->ikjl',Hermite_list,Hermite_list)
    exp_term=np.einsum('i,j->ij',np.exp(1j*theta1*np.arange(cutoff+1)),np.exp(1j*theta2*np.arange(cutoff+1)))
    coeffs=np.einsum('ij,klij->kl',psix*delta_x**2,Hermite_matrix)
    wave_f_out=np.einsum('ijkl,ij->kl',Hermite_matrix,coeffs*exp_term)
    
    return wave_f_out/np.sqrt(np.sum(np.abs(wave_f_out)**2*delta_p**2))
def superp_squeezed_states_2m_context(r,alpha1,alpha2,x_list1,x_list2,theta1,theta2,cutoff=40):## add the peak coefficients
    aux=np.zeros((len(x_list1),len(x_list2)),dtype='complex')
    
    delta_x=x_list1[1]-x_list1[0]
    for i in range(2):
        displ_sq_1=get_psi_q_theta_from_q_vector2(displ_squeezed_state(alpha1*(-1)**i, r, x_list1),x_list1,x_list1,theta1,cutoff)
        displ_sq_2=get_psi_q_theta_from_q_vector2(displ_squeezed_state(alpha2*(-1)**i, r, x_list2),x_list2,x_list2,theta2,cutoff)
        
        aux+=np.einsum('i,j->ij',displ_sq_1,displ_sq_2)
    return aux/np.sqrt(np.sum((np.abs(aux)**2)*delta_x**2))
## more peaks 
def superp_squeezed_states_2m_N(N,r,alpha1,alpha2,x_list1,x_list2):## add the peak coefficients
    aux=np.zeros((len(x_list1),len(x_list2)))
    n_list=np.arange(N)-(N-1)/2
    
    delta_x=x_list1[1]-x_list1[0]
    for i in range(len(n_list)):
        aux+=np.einsum('i,j->ij',displ_squeezed_state(n_list[i]*alpha1, r, x_list1),displ_squeezed_state(n_list[i]*alpha2, r, x_list2))
    return aux/np.sqrt(np.sum((np.abs(aux)**2)*delta_x**2))

def superp_squeezed_states_2m_context_N(N,r,alpha1,alpha2,x_list1,x_list2,theta1,theta2,cutoff=40):## add the peak coefficients
    aux=np.zeros((len(x_list1),len(x_list2)),dtype='complex')
    n_list=np.arange(N)-(N-1)/2
    
    delta_x=x_list1[1]-x_list1[0]
    for i in range(len(n_list)):
        displ_sq_1=get_psi_q_theta_from_q_vector2(displ_squeezed_state(n_list[i]*alpha1, r, x_list1),x_list1,x_list1,theta1,cutoff)
        displ_sq_2=get_psi_q_theta_from_q_vector2(displ_squeezed_state(n_list[i]*alpha2, r, x_list2),x_list2,x_list2,theta2,cutoff)
        
        aux+=np.einsum('i,j->ij',displ_sq_1,displ_sq_2)
    return aux/np.sqrt(np.sum((np.abs(aux)**2)*delta_x**2))


def entangled_GKP_N_peaks_context(N,r,alpha1,alpha2,x_list1,x_list2,theta1,theta2,cutoff=40):
    aux=np.zeros((len(x_list1),len(x_list2)),dtype='complex')
    n_list=np.arange(N)-(N-1)/2
    delta_x=x_list1[1]-x_list1[0]
    for i in range(len(n_list)):
        for j in range(len(n_list)):
            if theta1==0:
                displ_sq_1=displ_squeezed_state((n_list[i]*alpha1+n_list[j]*alpha2)/np.sqrt(2), r, x_list1)
            else:
                displ_sq_1=get_psi_q_theta_from_q_vector2(displ_squeezed_state((n_list[i]*alpha1+n_list[j]*alpha2)/np.sqrt(2), r, x_list1),x_list1,x_list1,theta1,cutoff)
            if theta2==0:
                displ_sq_2=displ_squeezed_state((-n_list[i]*alpha1+n_list[j]*alpha2)/np.sqrt(2), r, x_list2)
            else: 
                displ_sq_2=get_psi_q_theta_from_q_vector2(displ_squeezed_state((-n_list[i]*alpha1+n_list[j]*alpha2)/np.sqrt(2), r, x_list2),x_list2,x_list2,theta2,cutoff)
        
            aux+=np.einsum('i,j->ij',displ_sq_1,displ_sq_2)
    return aux/np.sqrt(np.sum((np.abs(aux)**2)*delta_x**2))
    

## To compute the moments of photon subtracted states 

def symplecticStructure(m):
    """Symplectic Structure Matrix."""
    J = np.zeros((m, m))
    for i in range(m):
        if i<int(m/2):
            J[i,i+int(m/2)]=-1
        else: 
            J[i,i-int(m/2)]=1
    
    return J

def tensorProduct(g, h):
    """Tensor Product of two vectors."""
    C = np.einsum('i,j->ij',g, h)
    return C

def As(V, g):
    """Photon Subtraction Correction Matrix A."""
    P = tensorProduct(g, g)
    J=symplecticStructure(V.shape[0])
    PJ= tensorProduct(np.einsum('ij,j',J,g),np.einsum('ij,j',J,g))
    I = np.eye(V.shape[0]) 
    A = (2 * np.einsum('ij,jk,kl->il',(V - I) ,P+PJ,(V - I))) / np.trace(np.einsum('ij,jk->ik',(V - I) , P+PJ))
    return A

def Vsq(s):
    """Covariance matrix of a two mode squeezed vacuum state (with no xp correlations)"""
    return np.array([
    [1 / np.sqrt(2) * (s + 1 / s), 1 / np.sqrt(2) * (s - 1 / s), 0, 0],
    [1 / np.sqrt(2) * (s - 1 / s), 1 / np.sqrt(2) * (s + 1 / s), 0, 0],
    [0, 0, 1 / np.sqrt(2) * (s + 1 / s), 1 / np.sqrt(2) * (1 / s - s)],
    [0, 0, 1 / np.sqrt(2) * (1 / s - s), 1 / np.sqrt(2) * (s + 1 / s)]
    ])

def grad_sympy(f,var):
    grad=[sp.diff(f,var[i]) for i in range(len(var))]
    return np.array(grad)


def chi(a1, a2, a3, a4,V,g):
    """Characteristic function of a two mode photon subtracted state, obtained by subtracting one photon in mode g 
    from a two mode Gaussian state with Covariance matrix V
    """
    var=np.array([a1,a2,a3,a4])
    return (1-np.einsum('i,ij,j',var,As(V,g),var)/2)*sp.exp(-np.einsum('i,ij,j',var,V,var)/2)
    #return ((1-(var.T*sp.Matrix(As(V,g))*var)[0,0]/2))*sp.exp(-(var.T*sp.Matrix(V)*var)[0,0]/2)


def moment(n, m, context,V,g):
    """
    moment x_1^n x_2^m of a photon subtracted state (photon subtracted in mode g from Gaussian state with covariance matrix V)
    with x_1,x_2 different quadratures given by 
    context=(theta_1,theta_2)
    x_1=cos(theta_1)q_A+sin(theta_1)p_A
    x_2=cos(theta_2)q_B+sin(theta_2)p_B
    
    
    """
    v = [
        [np.cos(context[0]), 0, np.sin(context[0]), 0] for _ in range(n)
    ] + [
        [0, np.cos(context[1]), 0, np.sin(context[1])] for _ in range(m)
    ]

    a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
    f = chi(a1, a2, a3, a4,V,g)

    for k in range(len(v)):
        f = np.einsum('i,i',np.array(v[k]),(grad_sympy(f, [a1, a2, a3, a4])))
    
    return (sp.I)**(-len(v)) * f.subs({a1: 0, a2: 0, a3: 0, a4: 0})
def moment_from_char_function(ns, context,f):
    """
    moment x_1^n x_2^m of a photon subtracted state (photon subtracted in mode g from Gaussian state with covariance matrix V)
    with x_1,x_2 different quadratures given by 
    context=(theta_1,theta_2)
    x_1=cos(theta_1)q_A+sin(theta_1)p_A
    x_2=cos(theta_2)q_B+sin(theta_2)p_B
    
    
    """
    n=ns[0]
    m=ns[1]
    v = [
        [np.cos(context[0]), 0, np.sin(context[0]), 0] for _ in range(n)
    ] + [
        [0, np.cos(context[1]), 0, np.sin(context[1])] for _ in range(m)
    ]

    a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
    

    for k in range(len(v)):
        f = np.einsum('i,i',np.array(v[k]),(np.array(grad_sympy(f, [a1, a2, a3, a4]))))
    
    return (sp.I)**(-len(v)) * f.subs({a1: 0, a2: 0, a3: 0, a4: 0})
def moment_from_char_function_tuples(tupl,f):
    """
    moment x_1^n x_2^m of a photon subtracted state (photon subtracted in mode g from Gaussian state with covariance matrix V)
    with x_1,x_2 different quadratures given by 
    context=(theta_1,theta_2)
    x_1=cos(theta_1)q_A+sin(theta_1)p_A
    x_2=cos(theta_2)q_B+sin(theta_2)p_B
    
    
    """
    ns, context=tupl
    n=int(ns[0])
    m=int(ns[1])
    v = [
        [np.cos(context[0]), 0, np.sin(context[0]), 0] for _ in range(n)
    ] + [
        [0, np.cos(context[1]), 0, np.sin(context[1])] for _ in range(m)
    ]

    a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
    

    for k in range(len(v)):
        f = np.einsum('i,i',np.array(v[k]),(np.array(grad_sympy(f, [a1, a2, a3, a4]))))
    
    return (sp.I)**(-len(v)) * f.subs({a1: 0, a2: 0, a3: 0, a4: 0})

def f(x,y): 
    return x[0]*x[1]+y

def Wigner_photon_subtracted(V,g):
    lengV=V.shape[0]
    detV=np.linalg.det(V)
    invV=np.linalg.inv(V)
    
    return lambda x1,x2,p1,p2: 1/(pow(2,lengV/2+1)*pow(np.pi,lengV/2)*np.sqrt(detV))*(np.einsum('i,ij,jk,kl,l',np.array([x1,x2,p1,p2]),invV,As(
    V, g),invV,np.array([x1,x2,p1,p2]))+2)*np.exp(-1/2*np.einsum('i,ij,j',np.array([x1,x2,p1,p2]),invV,np.array([x1,x2,p1,p2])))

def Wigner_photon_subtracted_sympy(V,g):
    lengV=V.shape[0]
    detV=np.linalg.det(V)
    invV=np.linalg.inv(V)
    x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    return  1/(pow(2,lengV/2+1)*pow(np.pi,lengV/2)*np.sqrt(detV))*(np.einsum('i,ij,jk,kl,l',np.array([x1,x2,p1,p2]),invV,As(
    V, g),invV,np.array([x1,x2,p1,p2]))+2-np.einsum('ij,ji',invV,As(
    V, g)))*sp.exp(-1/2*np.einsum('i,ij,j',np.array([x1,x2,p1,p2]),invV,np.array([x1,x2,p1,p2])))

#Can probably improve the function bellow by using a direct projection onto the right quadratures (as in the function in Mathematica)
def wig_marginals_photon_subtr(V,alpha,g1,f):
    """

    Parameters
    ----------
    V :     numpy matrix   -> covariance matrix of the input Gaussian state
    alpha : numpy array    -> mean field of the input Gaussian state
    g1:     numpy array    -> mode in which the photon was subtracted 
    f:      numpy array    -> quadratures in which we want to project the Wigner function
                              if we choose [[1,0,0,0],[0,1,0,0]] we get the marginal p(x1,x2)
                             
    
    Returns
    -------
    W(x1,x2,p1,p2) projected on quadratures defined by f 
    
    """
    J=symplecticStructure(V.shape[0])
    Vf=np.einsum('ij,jk,lk->il',f,V,f)
    alphaf=np.einsum('ij,j->i',f,alpha)
    Identity=np.identity(2)
    
    gcompl=np.einsum('ij,j->i',J,g1)
    gfull=np.array([g1,gcompl])
    Vg1=np.einsum('ij,jk,lk->il',gfull,V,gfull)
    Vg1f=np.einsum('ij,jk,lk->il',gfull,V,f)
    
    idg1f=np.einsum('ij,kj->ik',gfull,f)
    alphag1=np.einsum('ij,j',gfull,alpha)
    
    x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    
    var=np.array([x1,x2,p1,p2])
    
    vareff=var[:f.shape[0]]  # constrain to the number of variables that we want to project to 
    
    invVf=np.linalg.inv(Vf)
    xi1=np.einsum('ij,jk,k->i',Vg1f-idg1f,invVf,vareff-alphaf)+alphag1
    
    X11=Vg1-Identity-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg1f-idg1f)
    
    WigG=sp.exp(-1/2*np.einsum('i,ij,j',vareff-alphaf,invVf,vareff-alphaf))/((2*np.pi)**(f.shape[0]/2)*np.sqrt(np.linalg.det(Vf)))
    ExpA=np.einsum('i,i',alphag1,alphag1)+np.trace(Vg1-Identity)
    ExpACond=np.einsum('i,i',xi1,xi1)+np.trace(X11)
    
    return ExpACond/ExpA*WigG

def wig_marginals_photon_subtr_numeric(x1list,x2list,V,alpha,g1,f):
    """

    Parameters
    ----------
    V :     numpy matrix   -> covariance matrix of the input Gaussian state
    alpha : numpy array    -> mean field of the input Gaussian state
    g1:     numpy array    -> mode in which the photon was subtracted 
    f:      numpy array    -> quadratures in which we want to project the Wigner function
                              if we choose [[1,0,0,0],[0,1,0,0]] we get the marginal p(x1,x2)
                             
    
    Returns
    -------
    W(x1,x2,p1,p2) projected on quadratures defined by f 
    
    """
    J=symplecticStructure(V.shape[0])
    Vf=np.einsum('ij,jk,lk->il',f,V,f)
    alphaf=np.einsum('ij,j->i',f,alpha)
    Identity=np.identity(2)
    
    gcompl=np.einsum('ij,j->i',J,g1)
    gfull=np.array([g1,gcompl])
    Vg1=np.einsum('ij,jk,lk->il',gfull,V,gfull)
    Vg1f=np.einsum('ij,jk,lk->il',gfull,V,f)
    
    idg1f=np.einsum('ij,kj->ik',gfull,f)
    alphag1=np.einsum('ij,j',gfull,alpha)
    
    #x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    
    x1,x2=np.meshgrid(x1list,x2list)
    var=np.concatenate((np.reshape(x1,(x1.shape[0],x1.shape[1],1)),np.reshape(x2,(x2.shape[0],x2.shape[1],1))),axis=2)

    
    
    vareff=var  # constrain to the number of variables that we want to project to 
    
    invVf=np.linalg.inv(Vf)
    xi1=np.einsum('ij,jk,lmk->lmi',Vg1f-idg1f,invVf,vareff-alphaf)+alphag1
    
    X11=Vg1-Identity-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg1f-idg1f)
    
    WigG=np.exp(-1/2*np.einsum('lmi,ij,lmj->lm',vareff-alphaf,invVf,vareff-alphaf))/((2*np.pi)**(f.shape[0]/2)*np.sqrt(np.linalg.det(Vf)))
    ExpA=np.einsum('i,i',alphag1,alphag1)+np.trace(Vg1-Identity)
    ExpACond=np.einsum('lmi,lmi->lm',xi1,xi1)+np.trace(X11)
    
    return ExpACond/(ExpA)*WigG
    
def moment_photon_subtr_from_wigner_marginal(tupl,A,V,alpha,g1):
    """

    Parameters
    ----------
    tupl : tuple
        (n,context):
            n: exponents of the monomial of which the moment is computed 
            context: context in which the moment is computed
    wigner : lambda function 
        Describes the Wigner function of the state from which we want to compute the moments
    A: cutoff in all quadratures, i.e. q_i \in [-A,A]
    
    Returns
    -------
    <q_context1^n1 q_context2^n2> = \int wigner(x1,x2,p1,p2) (x1 cos theta1 +p1 sin theta1)^n1 (x2 cos theta2 +p2 sin theta2)^n1 

    """
    n,context=tupl
    f=np.array([[np.cos(context[0]),0,np.sin(context[0]),0],[0,np.cos(context[1]),0,np.sin(context[1])]])    # quadratures to which we project the wigner function
    
    
    x1, x2 = sp.symbols('x1 x2')
    
    marginal=wig_marginals_photon_subtr(V,alpha,g1,f)
    
    return sp.N(sp.integrate(x1**n[0]*x2**n[1]*marginal,(x1,-A,A),(x2,-A,A)))

def moment_photon_subtr_from_wigner_marginal_numeric(tupl,A,V,alpha,g1):
    """

    Parameters
    ----------
    tupl : tuple
        (n,context):
            n: exponents of the monomial of which the moment is computed 
            context: context in which the moment is computed
    wigner : lambda function 
        Describes the Wigner function of the state from which we want to compute the moments
    A: cutoff in all quadratures, i.e. q_i \in [-A,A]
    
    Returns
    -------
    <q_context1^n1 q_context2^n2> = \int wigner(x1,x2,p1,p2) (x1 cos theta1 +p1 sin theta1)^n1 (x2 cos theta2 +p2 sin theta2)^n1 

    """
    n,context=tupl
    f=np.array([[np.cos(context[0]),0,np.sin(context[0]),0],[0,np.cos(context[1]),0,np.sin(context[1])]])    # quadratures to which we project the wigner function
    
    
    
    marginal=lambda x1,x2: wig_marginals_photon_subtr_numeric(x1,x2,V,alpha,g1,f)
    
    return integrate.nquad(lambda x1,x2: x1**n[0]*x2**n[1]*marginal(x1,x2),[[-A,A],[-A,A]])[0]   
def normalization(V,alpha,g1,A):
        """

        Parameters
        ----------
        V :     numpy matrix   -> covariance matrix of the input Gaussian state
        alpha : numpy array    -> mean field of the input Gaussian state
        g1:     numpy array    -> mode in which the photon was subtracted 
        
                                 
        
        Returns
        -------
        W(x1,x2,p1,p2) projected on quadratures defined by f 
        
        """
        f=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        J=symplecticStructure(V.shape[0])
        Vf=np.einsum('ij,jk,lk->il',f,V,f)
        alphaf=np.einsum('ij,j->i',f,alpha)
        Identity=np.identity(2)
        
        
        gcompl=np.einsum('ij,j->i',J,g1)
        gfull=np.array([g1,gcompl])
        Vg1=np.einsum('ij,jk,lk->il',gfull,V,gfull)
        Vg1f=np.einsum('ij,jk,lk->il',gfull,V,f)
        
        idg1f=np.einsum('ij,kj->ik',gfull,f)
        alphag1=np.einsum('ij,j',gfull,alpha)
        
        x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
        
        var=np.array([x1,x2,p1,p2])
        
        vareff=var[:f.shape[0]]  # constrain to the number of variables that we want to project to 
        
        invVf=np.linalg.inv(Vf)
        xi1=np.einsum('ij,jk,k->i',Vg1f-idg1f,invVf,vareff-alphaf)+alphag1
        
        X11=Vg1-Identity-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg1f-idg1f)
        
        WigG=sp.exp(-1/2*np.einsum('i,ij,j',vareff-alphaf,invVf,vareff-alphaf))/((2*np.pi)**(f.shape[0]/2)*np.sqrt(np.linalg.det(Vf)))
        ExpA=np.einsum('i,i',alphag1,alphag1)+np.trace(Vg1-Identity)
        ExpACond=np.einsum('i,i',xi1,xi1)+np.trace(X11)
        
        return integrate.nquad(sp.lambdify([x1,x2,p1,p2],ExpACond/ExpA*WigG),[[-A,A],[-A,A],[-A,A],[-A,A]])
    
    


def moment_photon_subtr_from_wigner(tupl,wigner,A,norm):
    """

    Parameters
    ----------
    tupl : tuple
        (n,context):
            n: exponents of the monomial of which the moment is computed 
            context: context in which the moment is computed
    wigner : lambda function 
        Describes the Wigner function of the state from which we want to compute the moments
    A: cutoff in all quadratures, i.e. q_i \in [-A,A]
    
    Returns
    -------
    <q_context1^n1 q_context2^n2> = \int wigner(x1,x2,p1,p2) (x1 cos theta1 +p1 sin theta1)^n1 (x2 cos theta2 +p2 sin theta2)^n1 

    """
    n,context=tupl
    
    qA=lambda x1,p1: x1*np.cos(context[0])+p1*np.sin(context[0])
    qB=lambda x2,p2: x2*np.cos(context[1])+p2*np.sin(context[1])
    x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    wignerlambd=sp.lambdify([x1,x2,p1,p2],wigner)
    
    return integrate.nquad(lambda x1,x2,p1,p2: pow(qA(x1,p1),n[0])*pow(qB(x2,p2),n[1])*wignerlambd(x1,x2,p1,p2)/norm,[[-A,A],[-A,A],[-A,A],[-A,A]])[0]

def moment_photon_subtr_from_wigner_sympy(tupl,wigner,A):
    """

    Parameters
    ----------
    tupl : tuple
        (n,context):
            n: exponents of the monomial of which the moment is computed 
            context: context in which the moment is computed
    wigner : lambda function 
        Describes the Wigner function of the state from which we want to compute the moments
    A: cutoff in all quadratures, i.e. q_i \in [-A,A]
    
    Returns
    -------
    <q_context1^n1 q_context2^n2> = \int wigner(x1,x2,p1,p2) (x1 cos theta1 +p1 sin theta1)^n1 (x2 cos theta2 +p2 sin theta2)^n1 

    """
    n,context=tupl
    x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    qA= x1*np.cos(context[0])+p1*np.sin(context[0])
    qB= x2*np.cos(context[1])+p2*np.sin(context[1])
    
    
    return sp.N(sp.integrate(qA**n[0]*qB**n[1]*wigner,(x1,-9,9),(x2,-9,9),(p1,-9,9),(p2,-9,9)))

###  Two photon subtracted states ############################################################################################

def normalization_2_photon_subtracted_states(V,alpha,g1,g2,A):
        """

        Parameters
        ----------
        V :     numpy matrix   -> covariance matrix of the input Gaussian state
        alpha : numpy array    -> mean field of the input Gaussian state
        g1:     numpy array    -> mode in which the first photon was subtracted 
        g2:     numpy array    -> mode in which the second photon was subtracted 
        
        
        Returns
        -------
        measure of the wigner function inside the hypercube of length A
        
        """
        f=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        J=symplecticStructure(V.shape[0])
        Vf=np.einsum('ij,jk,lk->il',f,V,f)
        alphaf=np.einsum('ij,j->i',f,alpha)
        Identity=np.identity(2)
        
        
        g1compl=np.einsum('ij,j->i',J,g1)
        g1full=np.array([g1,g1compl])
        Vg1=np.einsum('ij,jk,lk->il',g1full,V,g1full)
        Vg1f=np.einsum('ij,jk,lk->il',g1full,V,f)
        
        g2compl=np.einsum('ij,j->i',J,g2)
        g2full=np.array([g2,g2compl])
        Vg2=np.einsum('ij,jk,lk->il',g2full,V,g2full)
        Vg2f=np.einsum('ij,jk,lk->il',g2full,V,f)
        
        Vg1g2=np.einsum('ij,jk,lk->il',g2full,V,g1full)
        
        idg1f=np.einsum('ij,kj->ik',g1full,f)
        alphag1=np.einsum('ij,j',g1full,alpha)
        
        idg2f=np.einsum('ij,kj->ik',g2full,f)
        alphag2=np.einsum('ij,j',g2full,alpha)
        
        idg1g2=np.einsum('ij,kj->ik',g1full,g2full)
        
        x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
        
        var=np.array([x1,x2,p1,p2])
        
        vareff=var[:f.shape[0]]  # constrain to the number of variables that we want to project to 
        
        invVf=np.linalg.inv(Vf)
        xi1=np.einsum('ij,jk,k->i',Vg1f-idg1f,invVf,vareff-alphaf)+alphag1
        xi2=np.einsum('ij,jk,k->i',Vg2f-idg2f,invVf,vareff-alphaf)+alphag2
        
        
        X11=Vg1-Identity-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg1f-idg1f)
        X22=Vg2-Identity-np.einsum('ij,jk,lk->il',Vg2f-idg2f,invVf,Vg2f-idg2f)
        X12=Vg1g2-idg1g2-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg2f-idg2f)
        
       
        WigG=sp.exp(-1/2*np.einsum('i,ij,j',vareff-alphaf,invVf,vareff-alphaf))/((2*np.pi)**(f.shape[0]/2)*np.sqrt(np.linalg.det(Vf)))
        ExpA=(np.einsum('i,i',alphag1,alphag1)+np.trace(Vg1-Identity))*(np.einsum('i,i',alphag2,alphag2)+np.trace(Vg2-Identity))+2*np.einsum('ij,ij',Vg1g2-idg1g2,Vg1g2-idg1g2)+4*np.einsum('i,ij,j',alphag1,Vg1g2-idg1g2,alphag2)
        
        ExpACond=(np.einsum('i,i',xi1,xi1)+np.trace(X11))*(np.einsum('i,i',xi2,xi2)+np.trace(X22))+2*np.einsum('ji,ji',X12,X12)+4*np.einsum('i,ij,j',xi1,X12,xi2)
       


        return integrate.nquad(sp.lambdify([x1,x2,p1,p2],ExpACond/ExpA*WigG),[[-A,A],[-A,A],[-A,A],[-A,A]])
    

#wphotonsubt=Wigner_photon_subtracted(np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]))
#wphotonsubt(0,0,0,0)
#moment_photon_subtr_from_wigner(([0,0],[2,2]), wphotonsubt, 7)
#x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
#wphotonsubt(x1, x2, p1, p2)
#integrate.nquad(lambda x1,x2,p1,p2: wphotonsubt(x1,x2,p1,p2),[[-4,4],[-4,4],[-4,4],[-4,4]])
#Wigner_photon_subtracted_sympy(np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]))
#ton_subtr(np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([0,0,0,0]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]),np.identity(4))


# chi(x1,x2,p1,p2,np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]))
# sp.exp(x1)
# faux=Wigner_photon_subtracted_sympy(np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0])) 
# faux2=sp.lambdify([x1,x2,p1,p2],faux)
# faux2(0,0,0,0)
# sp.N(sp.integrate(faux2(x1,x2,p1,p2),(x1,-9,9),(x2,-9,9),(p1,-9,9),(p2,-9,9)))
# sp.N(sp.integrate(faux,(x1,-9,9),(x2,-9,9),(p1,-9,9),(p2,-9,9)))

# func=wig_marginals_photon_subtr(np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([0,0,0,0]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]),np.array([[1,0,0,0],[0,1,0,0]]))


# moment_photon_subtr_from_wigner_sympy(([2,2],[0.785,0.785]),faux,9)

# moment_photon_subtr_from_wigner_marginal(([8,8],[0.785,0.785]),9,np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([0,0,0,0]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]))

# moment_photon_subtr_from_wigner_marginal_numeric(([8,8],[0.785,0.785]),9,np.array([[1/2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,1/2]]),np.array([0,0,0,0]),np.array([1/np.sqrt(2),1/np.sqrt(2),0,0]))


def wig_marginals_2_photon_subtr_numeric(x1list,x2list,V,alpha,g1,g2,f):
    """

    Parameters
    ----------
    V :     numpy matrix   -> covariance matrix of the input Gaussian state
    alpha : numpy array    -> mean field of the input Gaussian state
    g1:     numpy array    -> mode in which the photon was subtracted 
    f:      numpy array    -> quadratures in which we want to project the Wigner function
                              if we choose [[1,0,0,0],[0,1,0,0]] we get the marginal p(x1,x2)
                             
    
    Returns
    -------
    W(x1,x2,p1,p2) projected on quadratures defined by f 
    
    """
    J=symplecticStructure(V.shape[0])
    Vf=np.einsum('ij,jk,lk->il',f,V,f)
    alphaf=np.einsum('ij,j->i',f,alpha)
    Identity=np.identity(2)
    
    gcompl1=np.einsum('ij,j->i',J,g1)
    gcompl2=np.einsum('ij,j->i',J,g2)
    g1full=np.array([g1,gcompl1])
    g2full=np.array([g2,gcompl2])
    Vg1=np.einsum('ij,jk,lk->il',g1full,V,g1full)
    Vg1f=np.einsum('ij,jk,lk->il',g1full,V,f)
    
    idg1f=np.einsum('ij,kj->ik',g1full,f)
    alphag1=np.einsum('ij,j',g1full,alpha)
    
    Vg2=np.einsum('ij,jk,lk->il',g2full,V,g2full)
    Vg2f=np.einsum('ij,jk,lk->il',g2full,V,f)
    
    idg2f=np.einsum('ij,kj->ik',g2full,f)
    alphag2=np.einsum('ij,j',g2full,alpha)

    Vg1g2=np.einsum('ij,jk,lk->il',g2full,V,g1full)
        
        
    idg1g2=np.einsum('ij,kj->ik',g1full,g2full)
        
    #x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    
    x1,x2=np.meshgrid(x1list,x2list)
    var=np.concatenate((np.reshape(x1,(x1.shape[0],x1.shape[1],1)),np.reshape(x2,(x2.shape[0],x2.shape[1],1))),axis=2)

    
    
    vareff=var # constrain to the number of variables that we want to project to 
    
    invVf=np.linalg.inv(Vf)
    xi1=np.einsum('ij,jk,lmk->lmi',Vg1f-idg1f,invVf,vareff-alphaf)+alphag1  ## Correct according to ordering in vareff
    xi2=np.einsum('ij,jk,lmk->lmi',Vg2f-idg2f,invVf,vareff-alphaf)+alphag2  ## Correct according to ordering in vareff
 
    
        
    X11=Vg1-Identity-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg1f-idg1f)
    X22=Vg2-Identity-np.einsum('ij,jk,lk->il',Vg2f-idg2f,invVf,Vg2f-idg2f)
    X12=Vg1g2-idg1g2-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg2f-idg2f)
    WigG=np.exp(-1/2*np.einsum('lmi,ij,lmj->lm',vareff-alphaf,invVf,vareff-alphaf))/((2*np.pi)**(f.shape[0]/2)*np.sqrt(np.linalg.det(Vf))) ## Correct up to the ordering of vars
    ExpA=(np.einsum('i,i',alphag1,alphag1)+np.trace(Vg1-Identity))*(np.einsum('i,i',alphag2,alphag2)+np.trace(Vg2-Identity))+2*np.einsum('ij,ij',Vg1g2-idg1g2,Vg1g2-idg1g2)+4*np.einsum('i,ij,j',alphag1,Vg1g2-idg1g2,alphag2)
        
    ExpACond=(np.einsum('lmi,lmi->lm',xi1,xi1)+np.trace(X11))*(np.einsum('lmi,lmi->lm',xi2,xi2)+np.trace(X22))+2*np.einsum('ji,ji',X12,X12)+4*np.einsum('lmi,ij,lmj->lm',xi1,X12,xi2)
       

    return ExpACond/(ExpA)*WigG

def wig_marginals_photon_subtr_numeric(x1list,x2list,V,alpha,g1,f):
    """

    Parameters
    ----------
    V :     numpy matrix   -> covariance matrix of the input Gaussian state
    alpha : numpy array    -> mean field of the input Gaussian state
    g1:     numpy array    -> mode in which the photon was subtracted 
    f:      numpy array    -> quadratures in which we want to project the Wigner function
                              if we choose [[1,0,0,0],[0,1,0,0]] we get the marginal p(x1,x2)
                             
    
    Returns
    -------
    W(x1,x2,p1,p2) projected on quadratures defined by f 
    
    """
    J=symplecticStructure(V.shape[0])
    Vf=np.einsum('ij,jk,lk->il',f,V,f)
    alphaf=np.einsum('ij,j->i',f,alpha)
    Identity=np.identity(2)
    
    gcompl=np.einsum('ij,j->i',J,g1)
    gfull=np.array([g1,gcompl])
    Vg1=np.einsum('ij,jk,lk->il',gfull,V,gfull)
    Vg1f=np.einsum('ij,jk,lk->il',gfull,V,f)
    
    idg1f=np.einsum('ij,kj->ik',gfull,f)
    alphag1=np.einsum('ij,j',gfull,alpha)
    
    #x1, x2, p1, p2 = sp.symbols('x1 x2 p1 p2')
    
    x1,x2=np.meshgrid(x1list,x2list)
    var=np.concatenate((np.reshape(x1,(x1.shape[0],x1.shape[1],1)),np.reshape(x2,(x2.shape[0],x2.shape[1],1))),axis=2)

    
    
    vareff=var  # constrain to the number of variables that we want to project to 
    
    invVf=np.linalg.inv(Vf)
    xi1=np.einsum('ij,jk,lmk->lmi',Vg1f-idg1f,invVf,vareff-alphaf)+alphag1
    
    X11=Vg1-Identity-np.einsum('ij,jk,lk->il',Vg1f-idg1f,invVf,Vg1f-idg1f)
    
    WigG=np.exp(-1/2*np.einsum('lmi,ij,lmj->lm',vareff-alphaf,invVf,vareff-alphaf))/((2*np.pi)**(f.shape[0]/2)*np.sqrt(np.linalg.det(Vf)))
    ExpA=np.einsum('i,i',alphag1,alphag1)+np.trace(Vg1-Identity)
    ExpACond=np.einsum('lmi,lmi->lm',xi1,xi1)+np.trace(X11)
    
    return ExpACond/(ExpA)*WigG



### Modular measurement on GKP 
def moment_from_marginal_prob_modular(marginal,x_list1,x_list2):

    return np.sum(np.einsum('i,j->ij',-(2*np.mod(np.round(x_list1/np.sqrt(np.pi)),2)-1),-(2*np.mod(np.round(x_list2/np.sqrt(np.pi)),2)-1))*marginal)
