"""
Master problem implementation for Dantzig-Wolfe decomposition.
"""
import gurobipy as gp
import numpy as np
import numpy.typing as npt
from gurobipy import GRB


class MasterProblemManager:
    """
    Manages the restricted master problem for Dantzig-Wolfe decomposition.

    Handles creation, solving, and updating of the master problem.
    """

    def __init__(
        self,
        num_clients: int,
        m: int,
        extreme_points: list[list[npt.ArrayLike]],
        extreme_point_costs: list[list[float]],
        extreme_point_complicating: list[list[npt.ArrayLike]],
    ):
        """
        Initialize the master problem

        Parameters:
        -----------
        num_clients : int
            Number of client subproblems
        m : int
            Number of complicating constraints
        extreme_points : list of lists
            Extreme points for each client
        extreme_point_costs : list of lists
            Costs of extreme points
        extreme_point_complicating : list of lists
            Contributions to complicating constraints
        """
        self.num_clients = num_clients
        self.m = m
        self.extreme_points = extreme_points
        self.extreme_point_costs = extreme_point_costs
        self.extreme_point_complicating = extreme_point_complicating

        # Create the master problem
        self.master = gp.Model()
        self.master.setParam("OutputFlag", 0)

        # Create lambda variables for each extreme point as a 2D dictionary
        self.lambda_vars: dict[tuple[int, int], gp.Var] = {}
        for i in range(self.num_clients):
            for j in range(len(extreme_points[i])):
                var_name = f"lambda[{i},{j}]"
                self.lambda_vars[i, j] = self.master.addVar(lb=0, name=var_name)

        # Convexity constraints
        self.convexity_constrs = self.master.addConstrs(
            (
                gp.quicksum(
                    self.lambda_vars[i, j] for j in range(len(extreme_points[i]))
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
                    self.lambda_vars[i, j] * extreme_point_complicating[i][j][k]
                    for i in range(self.num_clients)
                    for j in range(len(extreme_points[i]))
                )
                == 0
                for k in range(self.m)
            ),
            name="complicating",
        )

        # Objective function
        self.master.setObjective(
            gp.quicksum(
                extreme_point_costs[i][j] * self.lambda_vars[i, j]
                for i in range(self.num_clients)
                for j in range(len(extreme_points[i]))
            ),
            GRB.MINIMIZE,
        )

    def solve_master(self) -> tuple[bool, float]:
        """
        Solve the master problem

        Returns:
        --------
        tuple
            (is_optimal, objective_value)
        """
        self.master.optimize()

        is_optimal = self.master.Status == GRB.OPTIMAL
        obj_val = self.master.ObjVal if is_optimal else float("inf")

        return is_optimal, obj_val

    def get_dual_values(self) -> tuple[npt.ArrayLike, list[float]]:
        """
        Get dual values from the master problem

        Returns:
        --------
        tuple
            (duals_complicating, duals_convexity)
        """
        # Get dual prices for complicating constraints
        duals_complicating = np.array(
            [self.complicating_constrs[i].Pi for i in range(self.m)]
        )

        # Get dual prices for convexity constraints
        duals_convexity = [
            self.convexity_constrs[i].Pi for i in range(self.num_clients)
        ]

        return duals_complicating, duals_convexity

    def get_lambda_values(self) -> dict[tuple[int, int], float]:
        """
        Get the values of lambda variables

        Returns:
        --------
        dict
            Dictionary mapping (client_idx, col_idx) to lambda values
        """
        lambda_values = {}

        for i in range(self.num_clients):
            for j in range(len(self.extreme_points[i])):
                if (i, j) in self.lambda_vars:
                    lambda_values[(i, j)] = self.lambda_vars[i, j].X

        return lambda_values

    def get_status_and_objective(self) -> tuple[bool, float]:
        """
        Get the solution status and objective value

        Returns:
        --------
        tuple
            (is_optimal, objective_value)
        """
        is_optimal = self.master.Status == GRB.OPTIMAL
        obj_val = self.master.ObjVal if is_optimal else float("inf")

        return is_optimal, obj_val

    def add_column(
        self,
        client_idx: int,
        obj_value: float,
        complicating_contribution: npt.ArrayLike,
    ) -> None:
        """
        Add a new column to the master problem

        Parameters:
        -----------
        client_idx : int
            Client index
        obj_value : float
            Objective value contribution
        complicating_contribution : numpy.ndarray
            Contribution to complicating constraints
        """
        # Add new lambda variable to master problem
        col_idx = (
            len(self.extreme_points[client_idx]) - 1
        )  # Index of the newly added point
        var_name = f"lambda_{client_idx}_{col_idx}"
        self.lambda_vars[client_idx, col_idx] = self.master.addVar(lb=0, name=var_name)

        # Update convexity constraint
        self.master.chgCoeff(
            self.convexity_constrs[client_idx],
            self.lambda_vars[client_idx, col_idx],
            1.0,
        )

        # Update complicating constraints
        for k in range(self.m):
            coeff = complicating_contribution[k]
            if abs(coeff) > 1e-10:  # Only add non-zero coefficients
                self.master.chgCoeff(
                    self.complicating_constrs[k],
                    self.lambda_vars[client_idx, col_idx],
                    coeff,
                )

        # Update objective
        self.master.chgCoeff(
            self.master.getObjective(), self.lambda_vars[client_idx, col_idx], obj_value
        )

    def remove_columns(self, client_idx: int, cols_to_remove: list[int]) -> int:
        """
        Remove columns from the master problem

        Parameters:
        -----------
        client_idx : int
            Client index
        cols_to_remove : list of int
            List of column indices to remove

        Returns:
        --------
        int
            Number of columns actually removed
        """
        removed = 0

        # Remove columns in reverse order (to maintain correct indices)
        for j in sorted(cols_to_remove, reverse=True):
            if (client_idx, j) in self.lambda_vars:
                # Remove variable from model
                self.master.remove(self.lambda_vars[client_idx, j])
                del self.lambda_vars[client_idx, j]

                # Update indices of remaining variables
                new_lambda_vars = {}
                for (client, col), var in self.lambda_vars.items():
                    if client == client_idx and col > j:
                        new_lambda_vars[(client, col - 1)] = var
                    else:
                        new_lambda_vars[(client, col)] = var
                self.lambda_vars = new_lambda_vars

                removed += 1

        if removed > 0:
            # Update the model to reflect changes
            self.master.update()

        return removed
