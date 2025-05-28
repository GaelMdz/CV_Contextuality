import sys 
from .moments import get_psi_q_theta,moment_from_marginal_prob,moment_superp_half_scale
from .monomials import (get_monomial_basis, moment_matrix_coordinates_indexed, 
                        num_monoms, localizing_matrix_coordinates_indexed, project_moment_matrix_coordinates_indexied)
from .Bell_inequalities import Bell_inequality_operator
from .operators import pre_compute_all_quadrature_powers
import matplotlib.pyplot as plt
import numpy as np 
import picos as pc
import cvxopt as cvxopt

## To do: 
##  - Add the computation of histograms for density matrices to include mixed states examples



class CV_Bell_inequality_LP:
    def __init__(self, histograms, N_bins,A):
        """
        Initialize the CV_Bell_inequality_LP class.

        Parameters:
        histograms (list of arrays): A set of arrays representing histograms.
        N_bins (int): The number of bins to be used.
        """
        self.histograms = histograms
        self.N_bins = N_bins
        self.A=A
        self.Nc = len(histograms)


    def evaluate(self):
        """
        INPUT: 
            histograms -> list of histograms to consider
            N_bins       -> number of bins in the coarse grainned histogram on which the LP is evaluated
            
        RETURN: 
            CF           -> double: contextual fraction associated to the given empirical model
        """
        
        #contexts=config.get("contexts")
        
        self.Nc = len(self.histograms)
        
        
        ######### Linear SDP  ##################
        sdp_non_loc = pc.Problem()
        
        # Define y variable to be the distribution of dimension N_bins x N_bins x ... x N_bins (depending on the number of contexts). Reshape to a one-dimensional array 
        # create variable 
        num_vars = (self.N_bins**2) * 4
        self.y = pc.RealVariable("y", shape=(num_vars))
        
        # Add positivity constraint
        sdp_non_loc.add_constraint(self.y <= 1 / self.Nc)

        # Add context constraints. 
        arrs = []
        dict_perms = {'1': 'kilj', '2': 'kijl', '3': 'iklj', '4': 'ikjl'}
        for c in range(self.Nc):
            x2_dup = (self.y[c * (self.N_bins**2):(c + 1) * (self.N_bins**2)]).dupvec(self.N_bins**2)  # shape (N^4,1), repeats each x2d[x1,x2] N^2 times
            x4_flat = x2_dup.reshuffled(
            permutation=dict_perms[str(c + 1)],
            dimensions=(self.N_bins, self.N_bins, self.N_bins, self.N_bins))
            arrs.append(x4_flat)
        sdp_non_loc.add_constraint(pc.sum(arrs) <= 0)

        obj = pc.sum([(pc.Constant(self.histograms[c].reshape((self.N_bins * self.N_bins))) | self.y[c * (self.N_bins**2):(c + 1) * (self.N_bins**2)]) for c in range(self.Nc)])

        sdp_non_loc.set_objective('max', obj)

        print("Solving LP...")
        solution = sdp_non_loc.solve(solver='mosek', verbosity=0, dualize=False)
        print("LP solved.")
        print("Status:", sdp_non_loc.status)
        print("Objective value:", sdp_non_loc.value)

        return sdp_non_loc.value
    
    def get_and_visualize_Bell_inequality(self): 
        """
        Visualize the filters against which each histogram is evaluated for the definition of the Bell inequality. 
        The Bell inequality is defined as the sum of the element wise products of the histograms with the filters.
        """
        matrices_Bell_ineq=[np.array(self.y.value)[i*self.N_bins**2:(i+1)*self.N_bins**2].reshape((self.N_bins,self.N_bins)) 
                            for i in range(self.Nc)]
        def plot_matrices(my_matrices,title,A):
            # Create a figure with 4 subplots arranged in a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(title,fontsize=16)
            # Flatten the axes array for easy iteration
            axes = axes.ravel()

            # Plot each matrix with its own colorbar
            for i, matrix in enumerate(my_matrices):
                im = axes[i].imshow(np.flip(np.transpose(matrix),axis=0), cmap='viridis',extent=(-A,A,-A,A))  # You can change the colormap as needed
                #axes[i].set_title(f'Matrix {i+1}')
                fig.colorbar(im, ax=axes[i],fraction=0.036)  # Add a colorbar to each subplot

            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the plot
            plt.show()
        plot_matrices(matrices_Bell_ineq,"Bell inequality filters",self.A)
        return matrices_Bell_ineq
    
    def visualize_histograms(self):
        """
        Visualize the histograms.
        """
        def plot_matrices(my_matrices,title,A):
            # Create a figure with 4 subplots arranged in a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(title,fontsize=16)
            # Flatten the axes array for easy iteration
            axes = axes.ravel()

            # Plot each matrix with its own colorbar
            for i, matrix in enumerate(my_matrices):
                im = axes[i].imshow(np.flip(np.transpose(matrix),axis=0), cmap='viridis',extent=(-A,A,-A,A))
                #axes[i].set_title(f'Matrix {i+1}')
                fig.colorbar(im, ax=axes[i],fraction=0.036)  # Add a colorbar to each subplot
            # Adjust layout to prevent overlap
            plt.tight_layout()
            # Show the plot
            plt.show()
        plot_matrices(self.histograms,"Histograms",self.A)

class CV_Bell_inequality_moments:
    def __init__(self, moment_vectors, varsA,varsB,K,A,localizing_constraints):
        """
        Initialize the CV_Bell_inequality_moments class.

        Parameters:
        moments (array): An array representing the moments.
        K (int): The parameter K.
        Nvars (int): The number of variables.
        """
        self.moment_vectors = moment_vectors
        self.K = K
        self.A=A
        self.varsA = varsA
        self.varsB = varsB
        self.nvarsA = len(varsA)
        self.nvarsB = len(varsB)
        self.Nvars = self.nvarsA+self.nvarsB
        self.localizing_constraints = localizing_constraints
    def evaluate(self):
        """
        Placeholder for the evaluate function. To be defined later.
        """
        monoms_2_vars=get_monomial_basis(2,2*self.K)

        self.sdp_non_loc=pc.Problem()
        sdp_non_loc=self.sdp_non_loc
        # create variable 
        num_vars=num_monoms(self.Nvars,2*self.K)
        self.y=pc.RealVariable("y",num_monoms(self.Nvars,2*self.K))
        y=self.y
        # First condition: matrix of moments positive definite (sum y_i coord_y_i_Mom)\
        dim_M = num_monoms(self.nvarsA + self.nvarsB, self.K)
        M_coords = moment_matrix_coordinates_indexed(self.nvarsA + self.nvarsB, self.K)
        Ms = [pc.Constant('M{0}'.format(i), 
              cvxopt.spmatrix(M_coords[i][0], M_coords[i][1], M_coords[i][2], 
              size=(num_monoms(self.nvarsA + self.nvarsB, self.K), num_monoms(self.nvarsA + self.nvarsB, self.K)))) 
              for i in range(num_vars)]

        # Explicit the constraint
        sdp_non_loc.add_constraint(pc.sum([y[i] * Ms[i] for i in range(num_vars)]) >> 0)

        # Add context constraints
        dim_MC = num_monoms(2, self.K)
        Mcss = []
        for a in range(self.nvarsA):
            for b in range(self.nvarsB):
                num_contex = self.nvarsB * a + b
                indices, MC_coords = project_moment_matrix_coordinates_indexied(self.nvarsA + self.nvarsB, self.K, a, b + self.nvarsA)
                MCs = [pc.Constant('MC[{0},{1}][{2}]'.format(a, b + self.nvarsA, i), 
                                   cvxopt.spmatrix(MC_coords[i][0], MC_coords[i][1], MC_coords[i][2], size=(dim_MC, dim_MC))) 
                                   for i in range(len(MC_coords))]
                Mcss.append(pc.sum([(self.moment_vectors[num_contex][i]) * MCs[i] for i in range(len(monoms_2_vars))]))
                sdp_non_loc.add_constraint(pc.sum([(self.moment_vectors[num_contex][i] - y[indices[i]]) * MCs[i] 
                                                   for i in range(len(monoms_2_vars))]) >> 0)

        if self.localizing_constraints:
            loc_polys=[[],[]]

            for i in range(self.Nvars):
                coeff1=[self.A,-1]
                coeff2=[self.A,1]
                monoms=[[0]*(self.Nvars)]*2
                aux=[0]*(self.Nvars)
                aux[i]=1
                monoms[1]=aux
                loc_polys[0].append(coeff1)
                loc_polys[0].append(coeff2)
                loc_polys[1].append(monoms)
                loc_polys[1].append(monoms)


            # Add localizing constraints on the SDP definition
            for i in range(len(loc_polys[0])):
                loc_matrix_dims = num_monoms(self.nvarsA + self.nvarsB, self.K)
                loc_matrix_coords = localizing_matrix_coordinates_indexed(self.nvarsA + self.nvarsB, self.K, [loc_polys[0][i], loc_polys[1][i]])
                B = [pc.Constant('B{0}'.format(i), cvxopt.spmatrix(loc_matrix_coords[k][0], loc_matrix_coords[k][1], loc_matrix_coords[k][2], 
                                                                size=(loc_matrix_dims, loc_matrix_dims))) for k in range(num_vars)]
                
                sdp_non_loc.add_constraint(pc.sum([y[k] * B[k] for k in range(num_vars)]) >> 0)

        sdp_non_loc.set_objective('min', -y[0])
        print("Solving SDP...")
        solution = sdp_non_loc.solve(solver='mosek',dualize=True,duals=True,verbosity=0)
        print("SDP solved.")
        print("Status:", sdp_non_loc.status)
        print("Objective value:", -sdp_non_loc.value)

        return sdp_non_loc.value+np.min(self.moment_vectors[:,0]) # compare to the mass of the distribution of each context
    
    def extract_Bell_inequality(self,NmaxBell):
        """
        Extract dual variables from the SDP solution.
        """
        dual = self.sdp_non_loc.dual
        if dual.status != 'solved':
            dual.solve()
        vars_dual = []
        for key in dual.variables.keys():
            vars_dual.append(dual.get_valued_variable(key))
        self.XCs=vars_dual[1:self.Nvars+1]
        self.XCs=np.array(self.XCs)

        Nmax=11 ## Fock space cutoff
        self.bellEx=Bell_inequality_operator(self.XCs,NmaxBell,  np.array(self.varsA),np.array(self.varsB),self.K)
        return self.bellEx

class CV_Bell_Ineq_state:
    def __init__(self, state, settings_A, settings_B, N_bins,A, method, K=None, N_bins_eff=None,localizing_constraints=None):
        """
        Initialize the CV_Bell_Ineq_state class.

        Parameters:
        state: The state object.
        settings_A (list): A list of settings for party A.
        settings_B (list): A list of settings for party B.
        N_bins (int): The number of bins to be used for the full histogram
        N_bins_eff (int): The number of bins to be used for the evaluation of the LP
        method (str): The method to be used ("moment_based" or "LP").
        K (int, optional): The parameter K, required if method is "moment_based".
        """
        self.state = state   # For now the state is a list of three elements: [C,n1,n2]
        self.settings_A = settings_A
        self.settings_B = settings_B
        self.A=A
        self.N_bins = N_bins
        self.N_bins_eff = N_bins_eff
        self.method = method
        self.K = K
        self.localizing_constraints = localizing_constraints

        if method == "moment_based" and K is None:
            raise ValueError("K must be specified for the 'moment_based' method.")
        if method == "moment_based" and localizing_constraints is None:
            raise ValueError("It must be specified whether to consider localizing constraints or not for the 'moment_based' method.")
        if method == "LP" and N_bins_eff is None:
            raise ValueError("N_bins_eff must be specified for the 'LP' method.")

    def generate_histograms(self):
        """
        Placeholder for the generate_histograms function. To be defined later.
        """
        contexts=[]
        for varA in self.settings_A:
            for varB in self.settings_B:
                contexts.append([varA,varB])

        x_list=np.linspace(-self.A,self.A,self.N_bins)

        

        #Extract the state parameters
        if isinstance(self.state, list) and len(self.state)==3: #Checks if the state is specified by a list [C,n1,n2]
            C=self.state[0]
            n1=self.state[1]
            n2=self.state[2]
        else:
            raise ValueError("State must be a list of three elements: [C,n1,n2] to use this method, instead use the method 'compute_moments_from_density_matrix'.")

        #Compute the histograms for each context
        self.histograms=[]
        for i in range(len(contexts)):
            prob_context=((x_list[1]-x_list[0])**2)*np.abs(get_psi_q_theta(np.array(C),np.array(n1,dtype=int),np.array(n2,dtype=int),int(np.max(n1)),contexts[i][0],contexts[i][1],x_list))**2
            self.histograms.append(prob_context)
        self.histograms=np.array(self.histograms)

        if self.method == "LP":
            if self.N_bins % self.N_bins_eff != 0:
                raise ValueError("N_bins should be divisible by N_bins_eff for LP method.")
            histograms2=np.zeros((len(self.histograms),self.N_bins_eff,self.N_bins_eff))
            for i in range(self.N_bins_eff):
                for j in range(self.N_bins_eff):
                    histograms2[:,i,j]=np.einsum('ijk->i',self.histograms[:,i*int(self.N_bins/self.N_bins_eff):(i+1)*int(self.N_bins/self.N_bins_eff),j*int(self.N_bins/self.N_bins_eff):(j+1)*int(self.N_bins/self.N_bins_eff)])
            self.histograms=histograms2 
    def compute_moments_from_histograms(self):
        """
        Compute the moments from the histograms.
        """
        if not hasattr(self, 'histograms') or self.histograms is None:
            self.generate_histograms()
        histograms = self.histograms
        monoms_2_vars=get_monomial_basis(2,2*self.K)
        x_list=np.linspace(-self.A,self.A,self.N_bins)
        moment_vectors=[]
        for k in range(len(histograms)):
            moment_context_i=[moment_from_marginal_prob(monoms_2_vars[j][0],monoms_2_vars[j][1], histograms[k], x_list, x_list) 
                                                        for j in range(len(monoms_2_vars))]
            moment_vectors.append(moment_context_i)
        moment_vectors=np.array(moment_vectors,dtype=np.double)
        return moment_vectors
    def compute_moments_from_Fock_coeffs(self):
        """
        Compute the moments from the Fock coefficients (C).
        """
        #Extract the state parameters
        if isinstance(self.state, list) and len(self.state)==3: #Checks if the state is specified by a list [C,n1,n2]
            C=self.state[0]
            n1=self.state[1]
            n2=self.state[2]
        else:
            raise ValueError("State must be a list of three elements: [C,n1,n2] to use this method, instead use the method 'compute_moments_from_density_matrix'.")

        if C.shape != ((self.Nmax+1)**2,):
            new_C = np.zeros((self.Nmax+1,self.Nmax+1), dtype=complex)
            for i in range(len(C)):
                new_C[self.n1[i], self.n2[i]] = C[i]
            C = new_C.flatten()
    
        contexts=[]
        for i in range(len(self.settings_A)):
            for j in range(len(self.settings_B)):
                contexts.append([self.settings_A[i],self.settings_B[j]])
        monoms_2_vars=get_monomial_basis(2,2*self.K)
        moment_vectors=[]
        for i in range(len(contexts)):
            moment_context_i=[moment_superp_half_scale(contexts[i][0],contexts[i][1],monoms_2_vars[j][0],monoms_2_vars[j][1],C,n1,n2) 
                              for j in range(len(monoms_2_vars))]
            moment_vectors.append(moment_context_i)
        moment_vectors=np.array(moment_vectors,dtype=np.double)
        return moment_vectors
    
    def compute_moments_from_density_matrix(self):
        """
        Compute the moments from the density matrix.
        """
        # Extract the state parameters
        if isinstance(self.state, np.ndarray):
            if self.state.shape[0] != self.state.shape[1]:
                raise ValueError("Density matrix must be square.")
            if self.state.shape[0] != (self.Nmax+1)**2:
                raise ValueError("Density matrix must be of size (Nmax+1)^2.")
            rho= self.state
        else:
            raise ValueError("State must be a density matrix (numpy array), consider using the method 'compute_moments_from_Fock_coeffs' or 'compute_moments_from_histogram' instead.")       

        #Compute the exact moments
        moment_vectors=[]
        nvarsA=len(self.settings_A)
        nvarsB=len(self.settings_B)
        monoms_2_vars=get_monomial_basis(2,2*self.K)
        for i in range(len(self.settings_A)+len(self.settings_B)):
            # Compute the quadrature powers for each variable
            quadsApowersaux=pre_compute_all_quadrature_powers( self.settings_A[int(i/nvarsB)], self.K,self.Nmax) 
            quadsBpowersaux=pre_compute_all_quadrature_powers( self.settings_B[int(np.mod(i,nvarsB))], self.K,self.Nmax) 
            #moment_context_i=[moments.moment_superp(contexts[i][0],contexts[i][1],monoms_2_vars[j][0],monoms_2_vars[j][1],np.array(C),np.array(n1,dtype=int),np.array(n2,dtype=int)) for j in range(len(monoms_2_vars))]
            moment_context_i2=[np.einsum('ij,ji',rho,np.kron(quadsApowersaux[monoms_2_vars[j][0]],quadsBpowersaux[monoms_2_vars[j][1]])) for j in range(len(monoms_2_vars))]
            moment_vectors2.append(moment_context_i2)
        moment_vectors2=np.array(moment_vectors2,dtype=np.double)
        return moment_vectors2
    
    def evaluate(self):
        """
        Evaluate the Bell inequality based on the specified method.
        """
        if not hasattr(self, 'histograms') or self.histograms is None:
            self.generate_histograms()
        histograms = self.histograms

        if self.method == "LP":
            self.lp_solver = CV_Bell_inequality_LP(histograms, self.N_bins_eff,self.A)
            self.CF=self.lp_solver.evaluate() # returns the value of the contextual fraction
        elif self.method == "moment_based":
            moment_vectors=self.compute_moments_from_histograms()
            self.moment_solver = CV_Bell_inequality_moments(moment_vectors, self.settings_A,self.settings_B, self.K,self.A,self.localizing_constraints)    
            self.CF=self.moment_solver.evaluate()   # returns the value of the contextual fraction
        else:
            raise ValueError("Invalid method. Choose either 'moment_based' or 'LP'.")

        return self.CF
    def get_and_visualize_Bell_inequality(self):
        """
        Delegate visualization to the LP solver.
        """
        if self.lp_solver is None:
            raise RuntimeError("LP solver has not been initialized. Run evaluate() first.")
        return self.lp_solver.get_and_visualize_Bell_inequality()
    def visualize_histograms(self):
        """
        Visualize the histograms.
        """
        if not hasattr(self, 'histograms') or self.histograms is None:
            self.generate_histograms()
        
        def plot_matrices(my_matrices,title,A):
            # Create a figure with 4 subplots arranged in a 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(title,fontsize=16)
            # Flatten the axes array for easy iteration
            axes = axes.ravel()

            # Plot each matrix with its own colorbar
            for i, matrix in enumerate(my_matrices):
                im = axes[i].imshow(np.flip(np.transpose(matrix),axis=0), cmap='viridis',extent=(-A,A,-A,A))
                #axes[i].set_title(f'Matrix {i+1}')
                fig.colorbar(im, ax=axes[i],fraction=0.036)  # Add a colorbar to each subplot
            # Adjust layout to prevent overlap
            plt.tight_layout()
            # Show the plot
            plt.show()
        plot_matrices(self.histograms,"Histograms",self.A)