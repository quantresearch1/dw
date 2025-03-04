"""
Direct solver for block-angular problems for comparison with Dantzig-Wolfe.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.linalg import block_diag


def direct_solve(c, F, client_blocks):
    """
    Solve the problem directly (without decomposition)
    
    Parameters:
    -----------
    c : numpy.ndarray
        Objective function coefficients
    F : numpy.ndarray
        Complicating constraints matrix
    client_blocks : list of dict
        Client blocks data
        
    Returns:
    --------
    dict
        Solution information including:
        - 'obj_value': Objective value
        - 'x': Solution vector
        - 'status': Solution status
        - 'solve_time': Solution time
    """
    # Create block diagonal A matrix
    A_blocks = [client_block["A"] for client_block in client_blocks]
    A = block_diag(*A_blocks)
    
    # Concatenate b vectors
    b = np.hstack([client_block['b'] for client_block in client_blocks])
    
    # Create and solve the model
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    
    # Add variables
    x = model.addMVar(c.shape[0], lb=0.0)
    
    # Add constraints
    model.addMConstr(F, x, "=", np.zeros(F.shape[0]))
    model.addMConstr(A, x, '<', b)
    
    # Set objective
    model.setObjective(c @ x, sense=GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    # Return solution
    if model.status == GRB.OPTIMAL:
        return {
            'status': 'Optimal',
            'obj_value': model.objVal,
            'x': x.X,
            'solve_time': model.Runtime
        }
    else:
        return {
            'status': f'Not optimal: {model.status}',
            'obj_value': float('inf'),
            'x': None,
            'solve_time': model.Runtime
        }