import numpy as np

def adjacency_maker(row):
    """Constructing a circulant adjacency matrix from a row.
    Shifting the row cyclically one position to the right in 
    order to obtain successive rows.
    Parameters
    ----------
    row : ndarray
        The first row of the matrix
    Returns
    -------
    adjacency_matrix : circulant adjacency matrix
    """
    
    #initialization of the matrix
    N = len(row)
    adjacency_matrix = np.zeros((N,N))
    
    #shifting the input row to the right 
    for z in range(N):
        adjacency_matrix[z,:] = np.roll(row,z)
        
    return(adjacency_matrix)
    
def hierarchical_connectivities(base,n,m):
    """Construction of the hierarchical network connectivity. 
    Parameters
    ----------
    base : ndarray
        The base pattern is containing zeros and non-zero values.
    n : int
        The last hierarchical step which defines the size of the resulting network
    m : int
        The number of hierarchical steps (m <= n).
    Returns
    -------
    C    : Coupling matrix (Adjacency matrix).
    c_nm : The number of links for the network at hierarchical step m.
    """
    
    #converting base to boolean
    base = np.array(base).astype('bool')
    
    #length of the base pattern
    b = len(base)
    
    #The number of ones in the base pattern and the number of links in the network.
    c_1 = np.count_nonzero(base)
    c_nm = (c_1**m) * (b**(n-m))
    
    #initialization of the row of the coupling matrix
    row = list(np.copy(base))
    
    #performing the fractal construction algorithm of a Cantor set
    for i in range(1,m):
        temp = []
        for j in range(len(row)):
            if row[j]:
                temp = temp + list(base)
            if not row[j]:
                temp = temp + ([False]*len(base))
        row = list(np.copy(temp))
    

    if m < n:
        final_n = b**(n-m)
        temp = []
        for j in range(len(row)):
            if row[j]:
                temp = temp + ([True]*final_n)
            if not row[j]:
                temp = temp + ([False]*final_n)
        row = list(np.copy(temp))
    
    #adding an additional zero corresponds to the self-coupling
    row.insert(0,False) 
    
    #constructing the coupling matrix
    C = adjacency_maker(row)
    
    return(C, c_nm)
