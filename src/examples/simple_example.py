"""
Simple example demonstrating the Dantzig-Wolfe decomposition.
"""
import numpy as np
import time
from dantzig_wolfe import DantzigWolfeDecomposition, verify_solution
from solvers import direct_solve


def run_example():
    """
    Example demonstrating the Dantzig-Wolfe decomposition on a simple problem
    with 2 client blocks and a coupling constraint
    """
    print("Running simple Dantzig-Wolfe example...")
    
    # Problem dimensions
    n1 = 2  # Variables in block 1
    n2 = 3  # Variables in block 2
    n = n1 + n2  # Total variables

    # Objective coefficients: c = [c1, c2]
    c = np.array([5, 3, 2, 7, 4])

    # Complicating constraints: F @ x = 0
    # One constraint linking blocks: x0 + x1 - x2 - x3 - x4 = 0
    F = np.array([[1, 1, -1, -1, -1]])

    # Block 1 constraints: A1 @ x1 <= b1
    A1 = np.array([[1, 2], [3, 1]])  # x0 + 2*x1 <= 10  # 3*x0 + x1 <= 15
    b1 = np.array([10, 15])

    # Block 2 constraints: A2 @ x2 <= b2
    A2 = np.array(
        [
            [1, 1, 1],  # x2 + x3 + x4 <= 20
            [2, 1, 0],  # 2*x2 + x3 <= 8
            [0, 0, 1],  # x4 <= 5
        ]
    )
    b2 = np.array([20, 8, 5])

    # Define client blocks
    client_blocks = [
        {'A': A1, 'b': b1, 'indices': np.array([0, 1])},
        {'A': A2, 'b': b2, 'indices': np.array([2, 3, 4])},
    ]

    # Create and solve using Dantzig-Wolfe
    print("\nSolving with Dantzig-Wolfe decomposition...")
    start_time = time.time()
    
    dw = DantzigWolfeDecomposition(c, F, client_blocks)
    solution = dw.solve(verbose=True, use_stabilization=True, use_parallel=True, column_management=True)
    
    dw_time = time.time() - start_time

    # Print results
    print("\nFinal solution (Dantzig-Wolfe):")
    print(f"Status: {solution['status']}")
    print(f"Objective value: {solution['obj_value']:.6f}")
    print(f"Solution vector: {solution['x']}")
    print(f"Solution time: {dw_time:.4f} seconds")
    print(f"Number of iterations: {solution['iterations']}")

    # Verify solution
    print("\nSolution verification:")
    verification = verify_solution(c, F, client_blocks, solution['x'])
    print(f"Is feasible: {verification['is_feasible']}")
    print(f"Objective value: {verification['obj_value']:.6f}")
    if not verification['is_feasible']:
        print(f"Constraint violations: {verification['violations']}")
    
    # Solve directly for comparison
    print("\nSolving directly (without decomposition)...")
    start_time = time.time()
    
    direct_solution = direct_solve(c, F, client_blocks)
    
    direct_time = time.time() - start_time
    
    print("\nFinal solution (Direct):")
    print(f"Status: {direct_solution['status']}")
    print(f"Objective value: {direct_solution['obj_value']:.6f}")
    if direct_solution['x'] is not None:
        print(f"Solution vector: {direct_solution['x']}")
    print(f"Solution time: {direct_time:.4f} seconds")
    
    # Compare solutions
    if solution['x'] is not None and direct_solution['x'] is not None:
        dw_obj = solution['obj_value']
        direct_obj = direct_solution['obj_value']
        
        print("\nComparison:")
        print(f"Objective difference: {abs(dw_obj - direct_obj):.6f}")
        print(f"Time speedup: {direct_time / dw_time:.2f}x")


if __name__ == "__main__":
    run_example()