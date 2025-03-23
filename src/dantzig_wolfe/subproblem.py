"""
Subproblem solving module for Dantzig-Wolfe decomposition.
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import concurrent.futures


class SubproblemSolver:
    """
    Solves subproblems for the Dantzig-Wolfe decomposition algorithm.
    
    Handles both sequential and parallel solving of subproblems.
    """
    
    def __init__(self, c, F, client_blocks, optimality_tol=1e-6):
        """
        Initialize the subproblem solver
        
        Parameters:
        -----------
        c : numpy.ndarray
            Objective function coefficients
        F : numpy.ndarray
            Complicating constraints matrix
        client_blocks : list of dict
            Each dict contains:
            - 'A': Constraint matrix for client i
            - 'b': RHS vector for client i
            - 'indices': Indices of variables for client i
            - 'lb': Lower bounds for variables in this client block
            - 'ub': Upper bounds for variables in this client block
        optimality_tol : float
            Optimality tolerance
        """
        self.c = c
        self.F = F
        self.client_blocks = client_blocks
        self.num_clients = len(client_blocks)
        self.optimality_tol = optimality_tol
    
    def generate_initial_point(self, client_idx):
        """
        Generate an initial extreme point for a client
        
        Parameters:
        -----------
        client_idx : int
            Client index
            
        Returns:
        --------
        tuple
            (point, cost, complicating_contribution)
        """
        # Extract client data
        client_indices = self.client_blocks[client_idx]['indices']
        A_i = self.client_blocks[client_idx]['A']
        b_i = self.client_blocks[client_idx]['b']
        lb_i = self.client_blocks[client_idx]['lb']
        ub_i = self.client_blocks[client_idx]['ub']

        # Client-specific costs
        c_i = self.c[client_indices]

        # Client variables in complicating constraints
        F_i = self.F[:, client_indices]

        # Create and solve model to find initial extreme point
        model = gp.Model()
        model.setParam('OutputFlag', 0)

        # Add variables with appropriate bounds
        x = model.addMVar(shape=len(client_indices), lb=lb_i, ub=ub_i, name="x")

        # Add equality constraints instead of inequality
        model.addMConstr(A_i, x, sense="=", b=b_i, name="feasible_region")

        # Set objective to get a basic feasible solution
        model.setObjective(c_i @ x, GRB.MINIMIZE)

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract solution
            x_values = x.X

            # Calculate cost
            cost_i = np.dot(c_i, x_values)

            # Calculate contribution to complicating constraints
            complicating_i = F_i @ x_values
            
            return x_values, cost_i, complicating_i
        else:
            # If no solution is found, the system might be infeasible
            raise ValueError(f"Could not find initial extreme point for client {client_idx}. Status: {model.status}")
    
    def solve_subproblem(self, client_idx, duals_complicating, dual_convexity):
        """
        Solve subproblem for a specific client
        
        Parameters:
        -----------
        client_idx : int
            Client index
        duals_complicating : numpy.ndarray
            Dual values for complicating constraints
        dual_convexity : float
            Dual value for convexity constraint
            
        Returns:
        --------
        tuple
            (reduced_cost, point, objective_value)
        """
        # Extract client data
        client_indices = self.client_blocks[client_idx]['indices']
        A_i = self.client_blocks[client_idx]['A']
        b_i = self.client_blocks[client_idx]['b']
        lb_i = self.client_blocks[client_idx]['lb']
        ub_i = self.client_blocks[client_idx]['ub']

        # Client-specific costs
        c_i = self.c[client_indices]

        # Client variables in complicating constraints
        F_i = self.F[:, client_indices]

        # Create modified cost (reduced cost)
        modified_cost = c_i.copy()
        for k in range(self.F.shape[0]):
            modified_cost = modified_cost + duals_complicating[k] * F_i[k, :]

        # Create and solve subproblem
        model = gp.Model()
        model.setParam('OutputFlag', 0)

        # Add variables with appropriate bounds
        x = model.addMVar(shape=len(client_indices), lb=lb_i, ub=ub_i, name="x")

        # Add equality constraints
        model.addMConstr(A_i, x, sense="=", b=b_i, name="feasible_region")

        # Set objective
        model.setObjective(modified_cost @ x, GRB.MINIMIZE)

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract solution
            x_values = x.X

            # Calculate objective value with original costs
            obj_value = np.dot(c_i, x_values)

            # Calculate reduced cost
            reduced_cost = model.objVal - dual_convexity

            return reduced_cost, x_values, obj_value
        else:
            raise ValueError(f"Subproblem {client_idx} could not be solved optimally. Status: {model.status}")
    
    def solve_all_subproblems(self, duals_complicating, duals_convexity, use_parallel=True, max_workers=None):
        """
        Solve all subproblems, either sequentially or in parallel
        
        Parameters:
        -----------
        duals_complicating : numpy.ndarray
            Dual values for complicating constraints
        duals_convexity : list
            Dual values for convexity constraints
        use_parallel : bool
            Whether to solve subproblems in parallel
        max_workers : int or None
            Maximum number of parallel workers (None = use default)
            
        Returns:
        --------
        list
            List of results (reduced_cost, point, objective_value) for each client
        """
        if use_parallel:
            return self._solve_all_subproblems_parallel(
                duals_complicating, 
                duals_convexity, 
                max_workers
            )
        else:
            results = []
            for i in range(self.num_clients):
                result = self.solve_subproblem(i, duals_complicating, duals_convexity[i])
                results.append(result)
            return results
    
    def _solve_all_subproblems_parallel(self, duals_complicating, duals_convexity, max_workers=None):
        """
        Solve all subproblems in parallel
        
        Parameters:
        -----------
        duals_complicating : numpy.ndarray
            Dual values for complicating constraints
        duals_convexity : list
            Dual values for convexity constraints
        max_workers : int or None
            Maximum number of parallel workers
            
        Returns:
        --------
        list
            List of results (reduced_cost, point, objective_value) for each client
        """
        results = [None] * self.num_clients  # Initialize with None placeholders
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a mapping of future to client index
            future_to_client = {}
            
            # Submit all subproblems to thread pool
            for i in range(self.num_clients):
                future = executor.submit(
                    self.solve_subproblem, 
                    i, 
                    duals_complicating, 
                    duals_convexity[i]
                )
                future_to_client[future] = i
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_client):
                client_idx = future_to_client[future]
                try:
                    result = future.result()
                    results[client_idx] = result
                except Exception as e:
                    print(f"Error in subproblem {client_idx}: {e}")
                    results[client_idx] = None
        
        return results