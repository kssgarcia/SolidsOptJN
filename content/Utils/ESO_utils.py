import numpy as np
import solidspy.uelutil as uel 
import solidspy.postprocesor as pos 
import solidspy.assemutil as ass    
import solidspy.solutil as sol      

def is_equilibrium(nodes, mats, els, loads):
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
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)
    disp = sol.static_sol(stiff_mat, rhs_vec)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(), rhs_vec/stiff_mat.max()):
        equil = False

    return equil
    
def preprocessing(nodes, mats, els, loads):
    """
    Compute IBC matrix and the static solve.
    
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
    bc_array : ndarray 
        Boundary conditions array
    disp : ndarray 
        Static displacement solve.
    """   

    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    print("Number of elements: {}".format(els.shape[0]))

    # System assembly
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = sol.static_sol(stiff_mat, rhs_vec)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(),
                       rhs_vec/stiff_mat.max()):
        print("The system is not in equilibrium!")
    return bc_array, disp


def postprocessing(nodes, mats, els, bc_array, disp):
    """
    Compute the nodes displacements, strains and stresses.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    IBC : ndarray 
        Boundary conditions array
    UG : ndarray 
        Static solve.
        
    Returns
    -------
    disp_complete : ndarray 
        Displacements at elements.
    strain_nodes : ndarray 
        Strains at elements.
    stress_nodes : ndarray 
        Stresses at elements.
    """   
    
    disp_complete = pos.complete_disp(bc_array, nodes, disp)
    strain_nodes, stress_nodes = None, None
    strain_nodes, stress_nodes = pos.strain_nodes(nodes, els, mats, disp_complete)
    
    return disp_complete, strain_nodes, stress_nodes

def protect_els(els, loads, BC):
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
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = np.ones_like(els[:,0], dtype=bool)
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -4:] == p)[:,0]
        mask_els[protect_index] = False
        
    return mask_els

def del_node(nodes, els):
    """
    Retricts nodes dof that aren't been used.
    
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


def volume(els, length, height, nx, ny):
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

    ny : float
    Return 
    ----------
    V: float

    """

    dy = length / nx
    dx = height / ny
    V   = dx * dy * els.shape[0]

    return V

def sensi_el(nodes, mats, els, UC):
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
        params = tuple(mats[els[el, 2], :])
        elcoor = nodes[els[el, -4:], 1:3]
        kloc, _ = uel.elast_quad4(elcoor, params)

        node_el = els[el, -4:]
        U_el = UC[node_el]
        U_el = np.reshape(U_el, (8,1))
        a_i = 0.5 * U_el.T.dot(kloc.dot(U_el))[0,0]
        sensi_number.append(a_i)
    sensi_number = np.array(sensi_number)
    sensi_number = sensi_number/sensi_number.max()

    return sensi_number

def strain_els(els, E_nodes, S_nodes):
    """
    Compute the elements strains and stresses.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    E_nodes : ndarray
        Strains at nodes.
    S_nodes : ndarray
        Stresses at nodes.
        
    Returns
    -------
    E_els : ndarray (nnodes, 3)
        Strains at elements.
    S_els : ndarray (nnodes, 3)
        Stresses at elements.
    """   
    
    E_els = []
    S_els = []
    for el in els:
        strain_nodes = np.take(E_nodes, list(el[3:]), 0)
        stress_nodes = np.take(S_nodes, list(el[3:]), 0)
        strain_elemt = (strain_nodes[0] + strain_nodes[1] + strain_nodes[2] + strain_nodes[3]) / 4
        stress_elemt = (stress_nodes[0] + stress_nodes[1] + stress_nodes[2] + stress_nodes[3]) / 4
        E_els.append(strain_elemt)
        S_els.append(stress_elemt)
    E_els = np.array(E_els)
    S_els = np.array(S_els)
    
    return E_els, S_els