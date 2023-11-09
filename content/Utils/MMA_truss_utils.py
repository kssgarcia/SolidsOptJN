import numpy as np
import matplotlib.pyplot as plt
from solidspy.preprocesor import rect_grid
import solidspy.postprocesor as pos
import solidspy.assemutil as ass
import solidspy.solutil as sol

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

plt.style.use("ggplot")
plt.rcParams["grid.linestyle"] = "dashed"

def is_equilibrium(nodes, elements, mats, loads):
    """
    Check if the system is in equilibrium
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
        
    Returns
    -------
    equil : bool
        Variable True when the system is in equilibrium and False when it doesn't
    """   
    equil = True
    DME, IBC , neq = ass.DME(nodes[:,-2:], elements)
    stiff, _ = ass.assembler(elements, mats, nodes[:,:-2], neq, DME)
    load_vec = ass.loadasem(loads, IBC, neq)
    disp = sol.static_sol(stiff, load_vec)
    if not(np.allclose(stiff.dot(disp)/stiff.max(), load_vec/stiff.max())):
        equil = False

    return equil

def sparse_assem(elements, mats, nodes, neq, assem_op):
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
        kloc = ass.retriever(elements, mats, nodes, ele)[0] * mats[ele, 2]
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
                        stiff_vals.append(kloc[row, col])

    stiff = coo_matrix((stiff_vals, (rows, cols)), shape=(neq, neq)).tocsr()

    return stiff

def fem_sol(nodes, els, mats, loads):
    """
    Compute the FEM solution for a given problem.

    Parameters
    ----------
    nodes : array
        Array with nodes
    els : array
        Array with element information.
    mats : array
        Array with material els. We need a material profile
        for each element for the optimization process.
    loads : array
        Array with loads.

    Returns
    -------
    disp_comp : array
        Displacement for each node.
    """
    assem_op, bc_array, neq = ass.DME(nodes[:,-2:], els)
    stiff_mat = sparse_assem(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)

    return disp, UC, rhs_vec


def lengths(els, nodes):
    """
    Compute the volume of the truss
    Parameters
    -------
    nodes: ndarray
        Array with models nodes
    els: ndarray
        Array with els information.
    Return
    -------
    """
    ini = els[:, 3]
    end = els[:, 4]
    lengths = np.linalg.norm(nodes[end, 1:3] - nodes[ini, 1:3], axis=1)
    return lengths

def grid_truss(length, height, nx, ny):
    """
    Generate a grid made of vertical, horizontal and diagonal
    members
    """
    nels = (nx - 1)*ny +  (ny - 1)*nx + 2*(nx - 1)*(ny - 1)
    x, y, _ = rect_grid(length, height, nx - 1, ny - 1)
    nodes = np.zeros((nx*ny, 5))
    nodes[:, 0] = range(nx*ny)
    nodes[:, 1] = x
    nodes[:, 2] = y
    elements = np.zeros((nels, 5), dtype=int)
    elements[:, 0] = range(nels)
    elements[:, 1] = 6
    elements[:, 2] = range(nels)
    hor_bars =  [[cont, cont + 1] for cont in range(nx*ny - 1)
                 if (cont + 1)%nx != 0]
    vert_bars =  [[cont, cont + nx] for cont in range(nx*(ny - 1))]
    diag1_bars =  [[cont, cont + nx + 1] for cont in range(nx*(ny - 1))
                   if  (cont + 1)%nx != 0]
    diag2_bars =  [[cont, cont + nx - 1] for cont in range(nx*(ny - 1))
                   if  cont%nx != 0]
    bars = hor_bars + vert_bars + diag1_bars + diag2_bars
    elements[:len(bars), 3:] = bars
    return nodes, elements, nels, x, y


def plot_truss(nodes, elements, mats, stresses, tol=1e-5):
    """
    Plot a truss and encodes the stresses in a colormap
    """
    mask = (mats[:,1]==1e-8)
    if mask.sum() > 0:
        stresses[mask] = 0

    max_stress = max(-stresses.min(), stresses.max())
    scaled_stress = 0.5*(stresses + max_stress)/max_stress
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    max_val = 4
    min_val = 0.5
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    for el in elements:
        if areas[el[2]] > tol:
            ini, end = el[3:]
            color = plt.cm.seismic(scaled_stress[el[0]])
            plt.plot([nodes[ini, 1], nodes[end, 1]],
                     [nodes[ini, 2], nodes[end, 2]],
                     color=color, lw=widths[el[2]])
    plt.axis("image")

def plot_truss_del(nodes, elements, mats, stresses):
    """
    nodes: ndarray
        Array with models nodes
    nodes: ndarray
        Array with models nodes
    """
    max_stress = max(-stresses.min(), stresses.max())
    scaled_stress = 0.5*(stresses + max_stress)/max_stress
    min_area = mats[:, 1].min()
    max_area = mats[:, 1].max()
    areas = mats[:, 1].copy()
    max_val = 4
    min_val = 0.5
    if max_area - min_area > 1e-6:
        widths = (max_val - min_val)*(areas - min_area)/(max_area - min_area)\
            + min_val
    else:
        widths = 3*np.ones_like(areas)
    for el in elements:
        ini, end = el[3:]
        plt.plot([nodes[ini, 1], nodes[end, 1]],
                    [nodes[ini, 2], nodes[end, 2]],
                    color=(1.0, 0.0, 0.0, 1.0), lw=widths[el[2]])
    plt.axis("image")

def protect_els(els, loads, BC, mask_del):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes
    mask_del : ndarray 
        Mask array with the elements that must be deleted.
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = mask_del.copy()
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -2:] == p)[:,0]
        mask_els[protect_index] = False

        for protect in protect_index:
            a_ = np.argwhere(els[np.logical_not(mask_els), -2] == els[protect, -2])[:,0]
            b_ = np.argwhere(els[np.logical_not(mask_els), -1] == els[protect, -1])[:,0]
            if len(a_)==1 or len(b_)==1:
                mask_els[protect] = True

    return mask_els

def del_node(nodes, els):
    """
    Retricts nodes dof that aren't been used.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements

    Returns
    -------
    """   
    n_nodes = nodes.shape[0]
    for n in range(n_nodes):
        if n not in els[:, -4:]:
            nodes[n, -2:] = -1


def sensi_el(els, mats, nodes, UC):
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
        kloc, _ = ass.retriever(els, mats, nodes, el)
        node_el = els[el, -2:]
        U_el = UC[node_el]
        U_el = np.reshape(U_el, (4,1))
        x_i = -U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(x_i)
    sensi_number = np.array(sensi_number)

    return sensi_number

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