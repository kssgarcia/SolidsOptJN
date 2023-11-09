import numpy as np
import solidspy.assemutil as ass    
import solidspy.postprocesor as pos 
import solidspy.solutil as sol      
import solidspy.uelutil as uel 

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
        Static displacement solve
    rh_vec : ndarray 
        Vector of loads
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
    return bc_array, disp, rhs_vec


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
    bc_array : ndarray 
        Boundary conditions array
    disp : ndarray 
        Static solve
        
    Returns
    -------
    disp_complete : ndarray 
        Displacements at elements.
    strain_nodes : ndarray 
        Strains at elements
    stress_nodes : ndarray 
        Stresses at elements
    """   
    
    disp_complete = pos.complete_disp(bc_array, nodes, disp)
    strain_nodes, stress_nodes = None, None
    strain_nodes, stress_nodes = pos.strain_nodes(nodes, els, mats, disp_complete)
    
    return disp_complete, strain_nodes, stress_nodes


def protect_els(els, nels, loads, BC):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    nels : ndarray
        Number of elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = np.zeros(nels, dtype=bool)
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -4:] == p)[:,0]
        mask_els[els[protect_index,0]] = True
        
    return mask_els

def del_node(nodes, els, loads, BC):
    """
    Retricts nodes dof that aren't been used and free up the nodes that are in use.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes

    Returns
    -------
    """   
    protect_nodes = np.hstack((loads[:,0], BC)).astype(int)
    for n in nodes[:,0]:
        if n not in els[:, -4:]:
            nodes[int(n), -2:] = -1
        elif n not in protect_nodes and n in els[:, -4:]:
            nodes[int(n), -2:] = 0


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
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Return 
    ----------
    V: float

    """

    dy = length / nx
    dx = height / ny
    V = dx * dy * np.ones(els.shape[0])

    return V

def sensitivity_els(nodes, mats, els, mask, UC):
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
    mask : ndarray
        Mask of optimal estructure
    UC : ndarray
        Displacements at nodes

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = []
    for el in range(els.shape[0]):
        if mask[el] == False:
            sensi_number.append(0)
            continue
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

def adjacency_nodes(nodes, els):
    """
    Create an adjacency matrix for the elements connected to each node.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    adj_nodes : ndarray, nodes.shape[0]
        Adjacency elements for each node.
    """
    adj_nodes = []
    for n in nodes[:, 0]:
        adj_els = np.argwhere(els[:, -4:] == n)[:,0]
        adj_nodes.append(adj_els)
    return adj_nodes

def center_els(nodes, els):
    """
    Calculate the center of each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    centers : ndarray, nodes.shape[0]
        Adjacency elements for each node.
    """
    centers = []
    for el in els:
        n = nodes[el[-4:], 1:3]
        center = np.array([n[1,0] + (n[0,0] - n[1,0])/2, n[2,1] + (n[0,1] - n[2,1])/2])
        centers.append(center)
    centers = np.array(centers)
    return centers

def sensitivity_nodes(nodes, adj_nodes, centers, sensi_els):
    """
    Calculate the sensitivity of each node.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    adj_nodes : ndarray
        Adjacency matrix of nodes
    centers : ndarray
        Array with center of elements
    sensi_els : ndarra
        Sensitivity of each element without filter
        
    Returns
    -------
    sensi_nodes : ndarray
        Sensitivity of each nodes
    """
    sensi_nodes = []
    for n in nodes:
        connected_els = adj_nodes[int(n[0])]
        if connected_els.shape[0] > 1:
            delta = centers[connected_els] - n[1:3]
            r_ij = np.linalg.norm(delta, axis=1) # We can remove this line and just use a constant because the distance is always the same
            w_i = 1/(connected_els.shape[0] - 1) * (1 - r_ij/r_ij.sum())
            sensi = (w_i * sensi_els[connected_els]).sum(axis=0)
        else:
            sensi = sensi_els[connected_els[0]]
        sensi_nodes.append(sensi)
    sensi_nodes = np.array(sensi_nodes)

    return sensi_nodes

def sensitivity_filter(nodes, centers, sensi_nodes, r_min):
    """
    Performe the sensitivity filter.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    sensi_nodes : ndarray
        Array with nodal sensitivity
    centers : ndarray
        Array with center of elements
    r_min : ndarra
        Minimum distance 
        
    Returns
    -------
    sensi_els : ndarray
        Sensitivity of each element with filter
    """
    sensi_els = []
    for i, c in enumerate(centers):
        delta = nodes[:,1:3]-c
        r_ij = np.linalg.norm(delta, axis=1)
        omega_i = (r_ij < r_min)
        w = 1/(omega_i.sum() - 1) * (1 - r_ij[omega_i]/r_ij[omega_i].sum())
        sensi_els.append((w*sensi_nodes[omega_i]).sum()/w.sum())
        
    sensi_els = np.array(sensi_els)
    sensi_els = sensi_els/sensi_els.max()

    return sensi_els

def plot_mesh(elements, nodes, disp, E_nodes=None):
    """
    Plot contours for model

    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py

    Parameters
    ----------
    nodes : ndarray (float)
        Array with number and nodes coordinates:
         `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each element.
    disp : ndarray (float)
        Array with the displacements.
    E_nodes : ndarray (float)
        Array with strain field in the nodes.

    """
    # Check for structural elements in the mesh
    struct_pos = 5 in elements[:, 1] or \
             6 in elements[:, 1] or \
             7 in elements[:, 1]
    if struct_pos:
        # Still not implemented visualization for structural elements
        print(disp)
    else:
        pos.plot_node_field(disp, nodes, elements, title=[r"$u_x$", r"$u_y$"],
                        figtitle=["Horizontal displacement",
                                  "Vertical displacement"])
        if E_nodes is not None:
            pos.plot_node_field(E_nodes, nodes, elements,
                            title=[r"",
                                   r"",
                                   r"",],
                            figtitle=["Strain epsilon-xx",
                                      "Strain epsilon-yy",
                                      "Strain gamma-xy"])