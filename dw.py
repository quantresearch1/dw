import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import concurrent.futures


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

        # Initialize master problem
        self.master = None
        self.lambda_vars = None
        self.convexity_constrs = None
        self.complicating_constrs = None

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
        self._create_master_problem()
        
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
            self.master.optimize()

            if self.master.status != GRB.OPTIMAL:
                if verbose:
                    print(
                        f"Master problem could not be solved optimally. Status: {self.master.status}"
                    )
                break

            # Get dual prices
            duals_complicating = np.array([
                self.complicating_constrs[i].Pi for i in range(self.m)
            ])
            duals_convexity = [
                self.convexity_constrs[i].Pi for i in range(self.num_clients)
            ]
            
            # Update stabilization center if using stabilization
            if use_stabilization:
                self._update_stabilization(duals_complicating, prev_obj, self.master.objVal)
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
            
            # Solve subproblems (in parallel if enabled)
            if use_parallel:
                subproblem_results = self._solve_all_subproblems_parallel(
                    stabilized_duals, duals_convexity
                )
                
                # Process results and add columns
                for i, result in enumerate(subproblem_results):
                    if result is not None:
                        reduced_cost, new_point, subproblem_obj = result
                        if reduced_cost < -self.optimality_tol:
                            self._add_column(i, new_point, subproblem_obj)
                            new_columns_added += 1
                            total_reduced_cost += reduced_cost
            else:
                for i in range(self.num_clients):
                    # Solve subproblem i
                    reduced_cost, new_point, subproblem_obj = self._solve_subproblem(
                        i, stabilized_duals, duals_convexity[i]
                    )

                    if reduced_cost < -self.optimality_tol:
                        # Add new column to master problem
                        self._add_column(i, new_point, subproblem_obj)
                        new_columns_added += 1
                        total_reduced_cost += reduced_cost
            
            # Perform column management if enabled
            if column_management and iteration % 5 == 0 and iteration > 10:
                removed = self._remove_unused_columns(column_removal_threshold)
                if verbose and removed > 0:
                    print(f"  Removed {removed} unused columns")

            if verbose:
                print(f"  Master objective: {self.master.objVal:.6f}")
                print(f"  New columns added: {new_columns_added}")
                print(f"  Sum of negative reduced costs: {total_reduced_cost:.6f}")
                if use_stabilization:
                    print(f"  Stabilization weight: {self.stabilization_weight:.4f}")

            # Save current objective for next iteration
            prev_obj = self.master.objVal
            
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
        for i in range(self.num_clients):
            for j in range(len(self.extreme_points[i])):
                if (i, j) in self.lambda_vars:
                    # Check if column is basic (has non-zero value)
                    if self.lambda_vars[i, j].x > self.optimality_tol:
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
            for j in sorted(cols_to_remove, reverse=True):
                if (i, j) in self.lambda_vars:
                    # Remove variable from model
                    self.master.remove(self.lambda_vars[i, j])
                    del self.lambda_vars[i, j]
                    
                    # Remove column data
                    self.extreme_points[i].pop(j)
                    self.extreme_point_costs[i].pop(j)
                    self.extreme_point_complicating[i].pop(j)
                    self.column_usage[i].pop(j)
                    
                    # Update indices of remaining variables
                    new_lambda_vars = {}
                    for (client, col), var in self.lambda_vars.items():
                        if client == i and col > j:
                            new_lambda_vars[(client, col-1)] = var
                        else:
                            new_lambda_vars[(client, col)] = var
                    self.lambda_vars = new_lambda_vars
                    
                    total_removed += 1
        
        if total_removed > 0:
            # Update the model to reflect changes
            self.master.update()
            
        return total_removed
    
    def _solve_all_subproblems_parallel(self, duals_complicating, duals_convexity):
        """
        Solve all subproblems in parallel
        
        Parameters:
        -----------
        duals_complicating : numpy.ndarray
            Dual values for complicating constraints
        duals_convexity : list
            Dual values for convexity constraints
            
        Returns:
        --------
        list
            List of results (reduced_cost, new_point, objective_value) for each client
        """
        results = [None] * self.num_clients  # Initialize with None placeholders
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a mapping of future to client index
            future_to_client = {}
            
            # Submit all subproblems to thread pool
            for i in range(self.num_clients):
                future = executor.submit(
                    self._solve_subproblem, 
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
            
    def set_initial_columns(self, initial_points):
        """
        Set initial columns for warm start
        
        Parameters:
        -----------
        initial_points : dict
            Dictionary mapping client indices to lists of points
            Format: {client_idx: [point1, point2, ...]}
        """
        # Check if we already have a master problem
        if self.master is not None:
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
            # Extract client data
            client_indices = self.client_blocks[i]['indices']
            A_i = self.client_blocks[i]['A']
            b_i = self.client_blocks[i]['b']

            # Client-specific costs
            c_i = self.c[client_indices]

            # Client variables in complicating constraints
            F_i = self.F[:, client_indices]

            # Create and solve model to find initial extreme point
            model = gp.Model()
            model.setParam('OutputFlag', 0)

            # Add variables
            n_i = len(client_indices)
            x = model.addMVar(n_i, lb=0, name="x")

            # Add constraints
            model.addMConstr(A_i, x, sense="<", b=b_i, name="feasible_region")

            # Set objective to get a basic feasible solutiom
            model.setObjective(c_i @ x, GRB.MINIMIZE)

            # Solve
            model.optimize()

            if model.status == GRB.OPTIMAL:
                # Extract solution
                x_values = x.X

                # Store extreme point
                self.extreme_points[i].append(x_values)

                # Calculate cost
                cost_i = np.dot(c_i, x_values)
                self.extreme_point_costs[i].append(cost_i)

                # Calculate contribution to complicating constraints
                complicating_i = F_i @ x_values
                self.extreme_point_complicating[i].append(complicating_i)
            else:
                raise ValueError(f"Could not find initial extreme point for client {i}")

    def _create_master_problem(self):
        """Create the initial restricted master problem"""
        self.master = gp.Model()
        self.master.setParam('OutputFlag', 0)

        # Create lambda variables for each extreme point as a 2D dictionary
        self.lambda_vars = {}
        for i in range(self.num_clients):
            for j in range(len(self.extreme_points[i])):
                var_name = f"lambda[{i},{j}]"
                self.lambda_vars[i, j] = self.master.addVar(lb=0, name=var_name)

        # Convexity constraints
        self.convexity_constrs = self.master.addConstrs(
            (
                gp.quicksum(
                    self.lambda_vars[i, j] for j in range(len(self.extreme_points[i]))
                )
                == 1
                for i in range(self.num_clients)
            ),
            name="convexity",
        )

        # Complicating constraints
        self.complicating_constrs = self.master.addConstrs(
            (
                gp.quicksum(
                    self.lambda_vars[i, j] * self.extreme_point_complicating[i][j][k]
                    for i in range(self.num_clients)
                    for j in range(len(self.extreme_points[i]))
                )
                == 0
                for k in range(self.m)
            ),
            name=f"complicating",
        )

        # Objective function
        self.master.setObjective(
            gp.quicksum(
                self.extreme_point_costs[i][j] * self.lambda_vars[i, j]
                for i in range(self.num_clients)
                for j in range(len(self.extreme_points[i]))
            ),
            GRB.MINIMIZE,
        )

    def _solve_subproblem(self, client_idx, duals_complicating, dual_convexity):
        """
        Solve subproblem for client_idx with given dual values

        Parameters:
        -----------
        client_idx : int
            Client index
        duals_complicating : list or numpy.ndarray
            Dual values for complicating constraints
        dual_convexity : float
            Dual value for convexity constraint

        Returns:
        --------
        tuple
            (reduced_cost, new_point, objective_value)
        """
        # Extract client data
        client_indices = self.client_blocks[client_idx]['indices']
        A_i = self.client_blocks[client_idx]['A']
        b_i = self.client_blocks[client_idx]['b']

        # Client-specific costs
        c_i = self.c[client_indices]

        # Client variables in complicating constraints
        F_i = self.F[:, client_indices]

        # Create modified cost (reduced cost)
        modified_cost = c_i.copy()
        for k in range(self.m):
            modified_cost = modified_cost + duals_complicating[k] * F_i[k, :]

        # Create and solve subproblem
        model = gp.Model()
        model.setParam('OutputFlag', 0)

        # Add variables
        n_i = len(client_indices)
        x = model.addMVar(n_i, lb=0, name="x")

        # Add constraints
        model.addMConstr(A_i, x, sense="<", b=b_i, name="feasible_region")

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
            raise ValueError(f"Subproblem {client_idx} could not be solved optimally")

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

        # Add new lambda variable to master problem
        col_idx = (
            len(self.extreme_points[client_idx]) - 1
        )  # Index of the newly added point
        var_name = f"lambda_{client_idx}_{col_idx}"
        self.lambda_vars[client_idx, col_idx] = self.master.addVar(lb=0, name=var_name)
        
        # Add the new column to column usage tracking
        if hasattr(self, 'column_usage') and len(self.column_usage) > client_idx:
            if len(self.column_usage[client_idx]) < len(self.extreme_points[client_idx]):
                self.column_usage[client_idx].append(0)

        # Update convexity constraint
        self.master.chgCoeff(
            self.convexity_constrs[client_idx],
            self.lambda_vars[client_idx, col_idx],
            1.0,
        )

        # Update complicating constraints
        for k in range(self.m):
            coeff = complicating_contribution[k]
            if abs(coeff) > 1e-10:
                self.master.chgCoeff(
                    self.complicating_constrs[k],
                    self.lambda_vars[client_idx, col_idx],
                    coeff,
                )

        # Update objective
        self.master.chgCoeff(
            self.master.getObjective(), self.lambda_vars[client_idx, col_idx], obj_value
        )

    def _construct_solution(self):
        """
        Construct the original space solution from the master problem solution

        Returns:
        --------
        dict
            Solution information
        """
        if self.master.status != GRB.OPTIMAL:
            return {'status': self.master.status, 'obj_value': float('inf'), 'x': None}

        # Get lambda values
        lambda_values = {}
        for i in range(self.num_clients):
            for j in range(len(self.extreme_points[i])):
                if (i, j) in self.lambda_vars:
                    lambda_values[(i, j)] = self.lambda_vars[i, j].x

        # Construct solution in original space
        x = np.zeros(self.n)
        for i in range(self.num_clients):
            client_indices = self.client_blocks[i]['indices']
            for j, point in enumerate(self.extreme_points[i]):
                lambda_val = lambda_values.get((i, j), 0)
                if lambda_val > 1e-10:
                    x[client_indices] += lambda_val * point

        return {
            'status': self.master.status,
            'obj_value': self.master.objVal,
            'x': x,
            'lambda_values': lambda_values,
        }


# Example usage
def run_example():
    """
    Example demonstrating the Dantzig-Wolfe decomposition on a simple problem
    with 2 client blocks and a coupling constraint
    """
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
    dw = DantzigWolfeDecomposition(c, F, client_blocks)
    
    # Demonstrate warm starting with initial columns
    # initial_points = {
    #     0: [np.array([1.0, 4.5])],
    #     1: [np.array([2.0, 4.0, 1.0])]
    # }
    # dw.set_initial_columns(initial_points)
    
    solution = dw.solve(verbose=True, use_stabilization=True, use_parallel=True, column_management=True)

    # Print results
    print("\nFinal solution:")
    print(f"Status: {solution['status']}")
    print(f"Objective value: {solution['obj_value']:.6f}")
    print(f"Solution vector: {solution['x']}")
    print(f"Solution time: {solution['time']:.4f} seconds")
    print(f"Number of iterations: {solution['iterations']}")

    # Verify solution
    x_opt = solution['x']
    obj_value = np.dot(c, x_opt)
    complicating_constr = np.dot(F, x_opt)

    print("\nSolution verification:")
    print(f"Objective value: {obj_value:.6f}")
    print(f"Complicating constraint: {complicating_constr}")

    # Verify block constraints
    block1_feasible = np.all(A1 @ x_opt[0:n1] <= b1 + 1e-6)
    block2_feasible = np.all(A2 @ x_opt[n1:n] <= b2 + 1e-6)

    print(f"Block 1 constraints satisfied: {block1_feasible}")
    print(f"Block 2 constraints satisfied: {block2_feasible}")


if __name__ == "__main__":
    run_example()


def simple_solve(c, F, client_blocks):
    """Check we get the same solution as DW"""
    from scipy.linalg import block_diag

    A = block_diag(*[client_block["A"] for client_block in client_blocks])
    m = gp.Model()
    x = m.addMVar(sum(len(client_block['indices']) for client_block in client_blocks))
    m.addMConstr(F, x, "=", np.array([0]))
    m.addMConstr(
        A, x, '<', np.hstack([client_block['b'] for client_block in client_blocks])
    )
    m.setObjective(c.T @ x, sense=gp.GRB.MINIMIZE)
    m.optimize()
    print(f"Optimal value of simple solve: {m.objVal}")
    print(f"Optimal solution of simple solve: {x.X}")


if __name__ == "__main__":
    run_example()
