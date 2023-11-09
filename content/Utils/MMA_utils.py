import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from scipy.spatial.distance import cdist


def sparse_assem(elements, mats, nodes, neq, assem_op, kloc):
    """
    Assembles the global stiffness matrix
    using a sparse storing scheme

    The scheme used to assemble is COOrdinate list (COO), and
    it converted to Compressed Sparse Row (CSR) afterward
    for the solution phase [1]_.

    Parameters
    ----------
    elements : ndarray (int)
      Array with the number for the nodes in each element.
    mats    : ndarray (float)
      Array with the material profiles.
    nodes    : ndarray (float)
      Array with the nodal numbers and coordinates.
    assem_op : ndarray (int)
      Assembly operator.
    neq : int
      Number of active equations in the system.
    uel : callable function (optional)
      Python function that returns the local stiffness matrix.

    Returns
    -------
    kglob : sparse matrix (float)
      Array with the global stiffness matrix in a sparse
      Compressed Sparse Row (CSR) format.

    References
    ----------
    .. [1] Sparse matrix. (2017, March 8). In Wikipedia,
        The Free Encyclopedia.
        https://en.wikipedia.org/wiki/Sparse_matrix

    """
    rows = []
    cols = []
    stiff_vals = []
    nels = elements.shape[0]
    for ele in range(nels):
        kloc_ = kloc * mats[elements[ele, 0], 2]
        ndof = kloc.shape[0]
        dme = assem_op[ele, :ndof]
        for row in range(ndof):
            glob_row = dme[row]
            if glob_row != -1:
                for col in range(ndof):
                    glob_col = dme[col]
                    if glob_col != -1:
                        rows.append(glob_row)
                        cols.append(glob_col)
                        stiff_vals.append(kloc_[row, col])

    stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()

    return stiff

def sensi_el(els, UC, kloc):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in range(len(els)):
        node_el = els[el, -4:]
        U_el = UC[node_el]
        U_el = np.reshape(U_el, (8,1))
        x_i = -U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(x_i)
    sensi_number = np.array(sensi_number)

    return sensi_number

def volume(length, height, nx, ny):
    """
    Volume calculation.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements.
    length : ndarray
        Length of the beam.
    height : ndarray
        Height of the beam.
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Return 
    ----------
    V: float

    """

    dy = length / nx
    dx = height / ny
    V = dy*dx

    return V

def x_star(lamb, q_o, L_j, v_j, alpha, x_max):
    x_t = L_j + np.sqrt(q_o / (lamb * v_j))
    x_star = np.clip(x_t, alpha, x_max)
    return x_star

def objective_function(lamb, r_o, v_max, q_o, L_j, v_j, alpha, x_max):
    x_star_value = x_star(lamb, q_o, L_j, v_j, alpha, x_max)
    return -(r_o - lamb*v_max + (q_o/(x_star_value-L_j) + lamb*v_j*x_star_value).sum())

def gradient(lamb, r_o, v_max, q_o, L_j, v_j, alpha, x_max):
    x_star_value = x_star(lamb, q_o, L_j, v_j, alpha, x_max)
    return (v_j * x_star_value).sum() - v_max