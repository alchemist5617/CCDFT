import numpy as np

def logistic(x0,a,tmax):
    '''
    Logistic map
    
    Parameters
    ----------
    x0 : float
         Initial value
    a : float
        bifurcation parameter
    tmax : int
            time step
    
    Returns
    -------
    x    : vector of length tmax, float
            
    """
    '''
    x = np.zeros(tmax)
    x[0] = x0
    for t in range(0,tmax-1):
        x[t+1] = a*x[t]*(1-x[t])
    return x


def logistic_map(z,a):
    '''
    univariate logistic map

    Parameters
    ----------
    z : float, 
        initial value.
    a : float, 
        bifurcation parameter.

    Returns
    -------
    f : float, 
        next time step.

    '''
    f = a*z*(1-z)
    return f


def logistic_map_N(z, C_ij, parameters):
    '''
    ring of logistic maps

    Parameters
    ----------
    z          :  vector of float, 
                  initializations of node values.
    C_ij       :  adjacency matrix, int.
    parameters :  dictionary of parameters
                  s = sigma, 
                  a = bifurcation parameter for logistic map,
                  T = time series length.
                  
                

    Returns
    -------
    z_T : matrix of shape (T, number of nodes).
        
    Source:
    Bonsen, A., Omelchenko, I., Zakharova, A. et al. 
    Chimera states in networks of logistic maps with hierarchical connectivities. 
    Eur. Phys. J. B 91, 65 (2018). 
    https://doi.org/10.1140/epjb/e2018-80630-y

    '''
    
    s = parameters['s']
    a = parameters['a']
    T = parameters['T']
    
    M = np.sum(C_ij)
    
    z_T = z.copy()
    t=1
    while t < T:
        f_zj = logistic_map(z, a) # vector
        z_next = []
        # coupling
        for i in np.arange(len(z)):
            f_zi = logistic_map(z[i], a) # number
            z_i = f_zi + s/M * (np.sum(C_ij * (f_zj - f_zi)))
            z_next.append(z_i)
        z_T = np.vstack((z_T, z_next))
        # update z's
        z = np.array(z_next)
        t+=1
    return z_T
