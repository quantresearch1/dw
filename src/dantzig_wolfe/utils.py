"""
Utility functions for Dantzig-Wolfe decomposition.
"""
import numpy as np


def is_optimal(total_reduced_cost, tol=1e-6):
    """
    Check if the current solution is optimal
    
    Parameters:
    -----------
    total_reduced_cost : float
        Sum of negative reduced costs
    tol : float
        Optimality tolerance
        
    Returns:
    --------
    bool
        True if solution is optimal, False otherwise
    """
    return abs(total_reduced_cost) < tol


def validate_problem_structure(c, F, client_blocks):
    """
    Validate that the problem has the correct structure
    
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
    bool
        True if valid, False otherwise
    
    Raises:
    -------
    ValueError
        If problem structure is invalid
    """
    n = c.shape[0]  # Total number of variables
    
    # Check that F has correct dimension
    if F.shape[1] != n:
        raise ValueError(f"F has {F.shape[1]} columns but c has {n} elements")
    
    # Check that client blocks cover all variables
    all_indices = set()
    for i, block in enumerate(client_blocks):
        if 'indices' not in block:
            raise ValueError(f"Client block {i} missing 'indices'")
        if 'A' not in block or 'b' not in block:
            raise ValueError(f"Client block {i} missing 'A' or 'b'")
        if 'lb' not in block or 'ub' not in block:
            raise ValueError(f"Client block {i} missing 'lb' or 'ub'")
            
        indices = block['indices']
        
        # Check for duplicates within this block
        if len(indices) != len(set(indices)):
            raise ValueError(f"Client block {i} has duplicate indices")
        
        # Check for overlap with other blocks
        if any(idx in all_indices for idx in indices):
            raise ValueError(f"Client block {i} overlaps with other blocks")
            
        all_indices.update(indices)
        
        # Check dimensions of A and b
        A_i = block['A']
        b_i = block['b']
        
        if A_i.shape[1] != len(indices):
            raise ValueError(
                f"Client block {i}: A has {A_i.shape[1]} columns but indices has {len(indices)} elements"
            )
        
        if A_i.shape[0] != b_i.shape[0]:
            raise ValueError(
                f"Client block {i}: A has {A_i.shape[0]} rows but b has {b_i.shape[0]} elements"
            )
        
        # Check bounds dimensions
        lb_i = block['lb']
        ub_i = block['ub']
        
        if len(lb_i) != len(indices):
            raise ValueError(
                f"Client block {i}: lb has {len(lb_i)} elements but indices has {len(indices)} elements"
            )
        
        if len(ub_i) != len(indices):
            raise ValueError(
                f"Client block {i}: ub has {len(ub_i)} elements but indices has {len(indices)} elements"
            )
    
    # Check that all variables are covered
    if len(all_indices) != n:
        raise ValueError(f"Client blocks cover {len(all_indices)} variables but c has {n} elements")
    
    # Check that indices are in range
    if max(all_indices) >= n or min(all_indices) < 0:
        raise ValueError(f"Client block indices out of range [0, {n-1}]")
    
    return True


def verify_solution(c, F, client_blocks, x, tol=1e-6):
    """
    Verify that a solution is feasible and calculate its objective value
    
    Parameters:
    -----------
    c : numpy.ndarray
        Objective function coefficients
    F : numpy.ndarray
        Complicating constraints matrix
    client_blocks : list of dict
        Client blocks data
    x : numpy.ndarray
        Solution vector
    tol : float
        Feasibility tolerance
        
    Returns:
    --------
    dict
        Verification results including:
        - 'is_feasible': bool
        - 'obj_value': float
        - 'violations': dict
    """
    result = {
        'is_feasible': True,
        'obj_value': np.dot(c, x),
        'violations': {}
    }
    
    # Check complicating constraints: F @ x = 0
    complicating_violation = F @ x
    if not np.allclose(complicating_violation, 0, atol=tol):
        result['is_feasible'] = False
        result['violations']['complicating'] = complicating_violation
    
    # Check client block constraints: A_i @ x_i = b_i (equality constraints)
    for i, block in enumerate(client_blocks):
        indices = block['indices']
        A_i = block['A']
        b_i = block['b']
        
        x_i = x[indices]
        constraint_values = A_i @ x_i
        
        # Check for equality constraint satisfaction
        if not np.allclose(constraint_values, b_i, atol=tol):
            result['is_feasible'] = False
            result['violations'][f'client_{i}'] = constraint_values - b_i
    
    # Check variable bounds
    for i, block in enumerate(client_blocks):
        indices = block['indices']
        x_i = x[indices]
        
        # Check lower and upper bounds
        lb_i = block['lb']
        ub_i = block['ub']
        
        if not np.all(x_i >= lb_i - tol):
            result['is_feasible'] = False
            if 'bounds' not in result['violations']:
                result['violations']['bounds'] = {}
            result['violations']['bounds'][f'client_{i}_lb'] = np.minimum(0, x_i - lb_i)
        
        if not np.all(x_i <= ub_i + tol):
            result['is_feasible'] = False
            if 'bounds' not in result['violations']:
                result['violations']['bounds'] = {}
            result['violations']['bounds'][f'client_{i}_ub'] = np.maximum(0, x_i - ub_i)
    
    return result