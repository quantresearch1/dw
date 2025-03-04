"""
Main implementation of Dantzig-Wolfe decomposition for block-angular structured problems.
"""
import numpy as np
import time

from dantzig_wolfe.master_problem import MasterProblemManager
from dantzig_wolfe.subproblem import SubproblemSolver
from dantzig_wolfe.utils import is_optimal


class DantzigWolfeDecomposition:
    """
    Implementation of Dantzig-Wolfe decomposition for block-angular structured problems
    
    Problem structure:
    min c.T @ x
    s.t. F @ x = 0  (complicating constraints)
         A_i @ x_i <= b_i for each client i
         x >= 0
    """

    def __init__(self, c, F, client_blocks, max_iterations=100, optimality_tol=1e-6):
        """
        Initialize the Dantzig-Wolfe decomposition solver

        Parameters:
        -----------
        c : numpy.ndarray
            Objective function coefficients
        F : numpy.ndarray
            Complicating constraints matrix of shape (m, n)
        client_blocks : list of dict
            Each dict contains:
            - 'A': Constraint matrix for client i
            - 'b': RHS vector for client i
            - 'indices': Indices of variables for client i in the original problem
        max_iterations : int
            Maximum number of column generation iterations
        optimality_tol : float
            Optimality tolerance
        """
        self.c = c
        self.F = F
        self.client_blocks = client_blocks
        self.num_clients = len(client_blocks)
        self.max_iterations = max_iterations
        self.optimality_tol = optimality_tol

        # Verify dimensions
        self.n = c.shape[0]  # Total number of variables
        self.m = F.shape[0]  # Number of complicating constraints

        # Initialize storage for extreme points
        self.extreme_points = [[] for _ in range(self.num_clients)]
        self.extreme_point_costs = [[] for _ in range(self.num_clients)]
        self.extreme_point_complicating = [[] for _ in range(self.num_clients)]
        
        # Track column usage for column management
        self.column_usage = [[] for _ in range(self.num_clients)]
        self.iteration_count = 0
        
        # Stabilization parameters
        self.use_stabilization = True
        self.stabilization_center = None
        self.stabilization_weight = 0.5
        self.stabilization_weight_decrease = 0.9
        self.min_stabilization_weight = 0.01
        
        # Parallel processing parameters
        self.use_parallel = True
        self.max_workers = None  # None means the default (CPU count)

        # Initialize problem managers
        self.master_manager = None
        self.subproblem_solver = SubproblemSolver(c, F, client_blocks, optimality_tol)

    def solve(self, verbose=True, use_stabilization=True, use_parallel=True, 
              column_management=True, column_removal_threshold=5):
        """
        Solve the problem using Dantzig-Wolfe decomposition

        Parameters:
        -----------
        verbose : bool
            Whether to print progress information
        use_stabilization : bool
            Whether to use dual stabilization techniques
        use_parallel : bool
            Whether to solve subproblems in parallel
        column_management : bool
            Whether to remove non-basic columns periodically
        column_removal_threshold : int
            Remove columns not used in the basis for this many iterations
            
        Returns:
        --------
        dict
            Solution information including:
            - 'obj_value': Objective value
            - 'x': Solution vector
            - 'iterations': Number of iterations
            - 'time': Solution time
            - 'status': Solution status
        """
        start_time = time.time()
        self.use_stabilization = use_stabilization
        self.use_parallel = use_parallel
        
        # Initialize extreme points
        self._initialize_extreme_points()

        # Create initial master problem
        self.master_manager = MasterProblemManager(
            self.num_clients, 
            self.m, 
            self.extreme_points,
            self.extreme_point_costs,
            self.extreme_point_complicating
        )
        
        # Initialize column usage tracking
        if column_management:
            for i in range(self.num_clients):
                self.column_usage[i] = [0] * len(self.extreme_points[i])

        # Column generation loop
        iteration = 0
        prev_obj = float('inf')
        
        # For stabilization
        if use_stabilization:
            self.stabilization_center = np.zeros(self.m)
        
        while iteration < self.max_iterations:
            iteration += 1
            self.iteration_count = iteration

            if verbose:
                print(f"\nIteration {iteration}")

            # Solve restricted master problem
            status, obj_val = self.master_manager.solve_master()

            if not status:
                if verbose:
                    print(f"Master problem could not be solved optimally.")
                break

            # Get dual prices
            duals_complicating, duals_convexity = self.master_manager.get_dual_values()
            
            # Update stabilization center if using stabilization
            if use_stabilization:
                self._update_stabilization(duals_complicating, prev_obj, obj_val)
                # Apply stabilization
                stabilized_duals = self._stabilize_duals(duals_complicating)
            else:
                stabilized_duals = duals_complicating

            # Generate columns by solving subproblems
            new_columns_added = 0
            total_reduced_cost = 0
            
            # Update column usage for tracking
            if column_management:
                self._update_column_usage()
            
            # Solve subproblems
            subproblem_results = self.subproblem_solver.solve_all_subproblems(
                stabilized_duals, 
                duals_convexity,
                use_parallel,
                self.max_workers
            )
            
            # Process results and add columns
            for i, result in enumerate(subproblem_results):
                if result is not None:
                    reduced_cost, new_point, subproblem_obj = result
                    if reduced_cost < -self.optimality_tol:
                        self._add_column(i, new_point, subproblem_obj)
                        new_columns_added += 1
                        total_reduced_cost += reduced_cost
            
            # Perform column management if enabled
            if column_management and iteration % 5 == 0 and iteration > 10:
                removed = self._remove_unused_columns(column_removal_threshold)
                if verbose and removed > 0:
                    print(f"  Removed {removed} unused columns")

            if verbose:
                print(f"  Master objective: {obj_val:.6f}")
                print(f"  New columns added: {new_columns_added}")
                print(f"  Sum of negative reduced costs: {total_reduced_cost:.6f}")
                if use_stabilization:
                    print(f"  Stabilization weight: {self.stabilization_weight:.4f}")

            # Save current objective for next iteration
            prev_obj = obj_val
            
            # Check termination criteria
            if new_columns_added == 0 or abs(total_reduced_cost) < self.optimality_tol:
                if verbose:
                    print(f"Optimization terminated after {iteration} iterations")
                break

        # Construct solution
        solution = self._construct_solution()
        solution['time'] = time.time() - start_time
        solution['iterations'] = iteration

        return solution
        
    def _update_stabilization(self, current_duals, prev_obj, current_obj):
        """
        Update stabilization parameters based on algorithm progress
        
        Parameters:
        -----------
        current_duals : numpy.ndarray
            Current dual values for complicating constraints
        prev_obj : float
            Previous master problem objective value
        current_obj : float
            Current master problem objective value
        """
        # Update stabilization center using a convex combination
        if self.stabilization_center is None:
            self.stabilization_center = current_duals.copy()
        else:
            # If objective is improving, put more weight on current duals
            if current_obj < prev_obj - self.optimality_tol:
                alpha = 0.7  # More weight to current duals when improving
            else:
                alpha = 0.3  # More weight to previous center when stalling
                
            self.stabilization_center = (1 - alpha) * self.stabilization_center + alpha * current_duals
            
        # Decrease stabilization weight over time
        if self.iteration_count % 5 == 0:
            self.stabilization_weight = max(
                self.min_stabilization_weight, 
                self.stabilization_weight * self.stabilization_weight_decrease
            )
    
    def _stabilize_duals(self, duals):
        """
        Apply stabilization to dual values
        
        Parameters:
        -----------
        duals : numpy.ndarray
            Current dual values
            
        Returns:
        --------
        numpy.ndarray
            Stabilized dual values
        """
        if self.stabilization_center is None:
            return duals
            
        # Apply proximal point stabilization
        weight = self.stabilization_weight
        return weight * self.stabilization_center + (1 - weight) * duals
    
    def _update_column_usage(self):
        """Update which columns are in the basis for column management"""
        lambda_values = self.master_manager.get_lambda_values()
        
        for i in range(self.num_clients):
            for j in range(len(self.extreme_points[i])):
                if (i, j) in lambda_values and lambda_values[i, j] > self.optimality_tol:
                    self.column_usage[i][j] = self.iteration_count
    
    def _remove_unused_columns(self, threshold):
        """
        Remove columns that haven't been in the basis for several iterations
        
        Parameters:
        -----------
        threshold : int
            Remove columns not used in the basis for this many iterations
            
        Returns:
        --------
        int
            Number of columns removed
        """
        total_removed = 0
        
        for i in range(self.num_clients):
            # Skip if we have too few columns
            if len(self.extreme_points[i]) <= 2:
                continue
                
            cols_to_remove = []
            
            # Find columns to remove
            for j in range(len(self.extreme_points[i])):
                # Skip if column is recently used or never used (might be new)
                age = self.iteration_count - self.column_usage[i][j]
                if age > threshold and self.column_usage[i][j] > 0:
                    cols_to_remove.append(j)
            
            # Remove columns in reverse order (to maintain correct indices)
            if cols_to_remove:
                # Call the master problem manager to handle the column removal
                removed = self.master_manager.remove_columns(i, cols_to_remove)
                
                # Update our tracking data structures
                for j in sorted(cols_to_remove, reverse=True):
                    self.extreme_points[i].pop(j)
                    self.extreme_point_costs[i].pop(j)
                    self.extreme_point_complicating[i].pop(j)
                    self.column_usage[i].pop(j)
                
                total_removed += removed
        
        return total_removed
            
    def set_initial_columns(self, initial_points):
        """
        Set initial columns for warm start
        
        Parameters:
        -----------
        initial_points : dict
            Dictionary mapping client indices to lists of points
            Format: {client_idx: [point1, point2, ...]}
        """
        # Check if we already have initialized the master problem
        if self.master_manager is not None:
            raise ValueError("Cannot set initial columns after master problem is created")
            
        for client_idx, points in initial_points.items():
            if client_idx >= self.num_clients:
                raise ValueError(f"Invalid client index: {client_idx}")
                
            # Add each point as an extreme point
            for point in points:
                # Extract client data
                client_indices = self.client_blocks[client_idx]['indices']
                
                # Validate point dimensions
                if len(point) != len(client_indices):
                    raise ValueError(
                        f"Point dimension {len(point)} does not match client {client_idx} "
                        f"dimension {len(client_indices)}"
                    )
                
                # Client-specific costs
                c_i = self.c[client_indices]
                
                # Client variables in complicating constraints
                F_i = self.F[:, client_indices]
                
                # Calculate cost
                cost_i = np.dot(c_i, point)
                
                # Calculate contribution to complicating constraints
                complicating_i = F_i @ point
                
                # Store the point
                self.extreme_points[client_idx].append(point)
                self.extreme_point_costs[client_idx].append(cost_i)
                self.extreme_point_complicating[client_idx].append(complicating_i)

    def _initialize_extreme_points(self):
        """Generate initial extreme points for each client subproblem"""
        for i in range(self.num_clients):
            # Let the subproblem solver handle generation of initial points
            point, cost, complicating = self.subproblem_solver.generate_initial_point(i)
            
            # Store the point
            self.extreme_points[i].append(point)
            self.extreme_point_costs[i].append(cost)
            self.extreme_point_complicating[i].append(complicating)

    def _add_column(self, client_idx, new_point, obj_value):
        """
        Add a new column to the master problem

        Parameters:
        -----------
        client_idx : int
            Client index
        new_point : numpy.ndarray
            New extreme point
        obj_value : float
            Objective value contribution
        """
        # Store the new extreme point
        self.extreme_points[client_idx].append(new_point)
        self.extreme_point_costs[client_idx].append(obj_value)

        # Calculate contribution to complicating constraints
        client_indices = self.client_blocks[client_idx]['indices']
        F_i = self.F[:, client_indices]
        complicating_contribution = F_i @ new_point
        self.extreme_point_complicating[client_idx].append(complicating_contribution)

        # Add to master problem
        self.master_manager.add_column(
            client_idx, 
            new_point, 
            obj_value, 
            complicating_contribution
        )
        
        # Add the new column to column usage tracking
        if len(self.column_usage) > client_idx:
            self.column_usage[client_idx].append(0)

    def _construct_solution(self):
        """
        Construct the original space solution from the master problem solution

        Returns:
        --------
        dict
            Solution information
        """
        status, obj_val = self.master_manager.get_status_and_objective()
        if not status:
            return {'status': "Not Optimal", 'obj_value': float('inf'), 'x': None}

        # Get lambda values
        lambda_values = self.master_manager.get_lambda_values()

        # Construct solution in original space
        x = np.zeros(self.n)
        for i in range(self.num_clients):
            client_indices = self.client_blocks[i]['indices']
            for j, point in enumerate(self.extreme_points[i]):
                lambda_val = lambda_values.get((i, j), 0)
                if lambda_val > 1e-10:
                    x[client_indices] += lambda_val * point

        return {
            'status': "Optimal",
            'obj_value': obj_val,
            'x': x,
            'lambda_values': lambda_values,
        }