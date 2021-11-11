from abc import ABC
from abc import abstractmethod
import numpy as np
import pulp


class MIP(ABC):
    """Super class for all Mixed Integer Linear Programming models.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, label, minimize=True):
        # Initialize problem
        if minimize:
            self._problem = pulp.LpProblem(label, pulp.LpMinimize)
        else:
            self._problem = pulp.LpProblem(label, pulp.LpMaximize)

    @property
    def problem(self):
        return self._problem

    def solve(self, solver):
        """Solve the LP.

        Parameters
        ----------
        solver : {'glpk', 'gurobi', 'cplex'}
            The solver that will be used to solve the LP constructed from
            the provided instance.

        Notes
        -----
        Possibly move the solver configuration to the calling script in
        the future, and requiring a solver object to be passed as a
        parameter.
        """
        # TODO: move this solver selection outside of the model
        if solver == 'glpk':
            slv = pulp.GLPK(
                keepFiles=True,
                mip=True,
                msg=True,
                options=None,
                path=None,
                timeLimit=None
            )
        elif solver == 'cplex':
            slv = pulp.CPLEX(
                gapAbs=None,
                gapRel=None,
                keepFiles=True,
                logPath=None,
                maxMemory=None,
                maxNodes=None,
                mip=True,
                msg=False,
                options=None,
                path=None,
                threads=None,
                timeLimit=3600,
                warmStart=False
            )
        elif solver == 'gurobi':
            slv = pulp.GUROBI(
                gapAbs=None,
                gapRel=None,
                keepFiles=True,
                logPath=None,
                mip=True,
                msg=True,
                options=None,
                path=None,
                threads=None,
                timeLimit=None,
                warmStart=False
            )
        else:
            raise ValueError(f"Solver type {solver} is not supported.")

        self._problem.solve(slv)

    def relax_problem(self):
        """Relax all decision variables in the LP. All integer variables
        (including binary) will be turned into continuous variables.
        """
        for var in self._problem.variables():
            var.cat = pulp.LpContinuous

    @abstractmethod
    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        pass


class JumpPointMIP(MIP):
    """Abstract class with some common jump point based model properties.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, instance, label):
        self._totaltime = instance['resource_availability'].shape[0]
        self._njobs = instance['jump_points'].shape[0]
        self._kjumps = instance['jump_points'].shape[1] - 1

        super().__init__(label, minimize=False)

    @property
    def njobs(self):
        return self._njobs

    @property
    def kjumps(self):
        return self._kjumps

    @property
    def totaltime(self):
        return self._totaltime


class TimeIndexedNoDeadline(JumpPointMIP):
    """Class implementing a time-indexed Mixed Integer Linear Programming
    model.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, instance, label):
        super().__init__(instance, label)
        # Define variables
        p = np.zeros(shape=(self._njobs, self._totaltime),
                     dtype=object)
        c = np.zeros(shape=(self._njobs, self._totaltime),
                     dtype=object)
        s = np.zeros(shape=(self._njobs, self._totaltime),
                     dtype=object)
        x = np.zeros(shape=(self._njobs, self._totaltime),
                     dtype=object)
        # Note that no job will ever be completed at t = 0, so the first
        # element of each row is always empty.
        for j in range(self._njobs):
            for t in range(instance['jump_points'][j, 0],
                           self._totaltime):
                # A job is completed at time t (index t)
                # if c[j, t] == 1.
                # This can only happen for t >= r_j
                c[j, t] = pulp.LpVariable(
                    f"c_{j},{t}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpInteger
                )
            for t in range(instance['jump_points'][j, 0],
                           self._totaltime):
                # A job is started at time t (index t)
                # if s[j, t] == 1.
                # This can only happen for r_j <= t <= T - p_j
                s[j, t] = pulp.LpVariable(
                    f"s_{j},{t}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpInteger
                )
            for t in range(instance['jump_points'][j, 0],
                           self._totaltime):
                # A job is processed at time t (index t)
                # if x[j, t] == 1.
                # This can only happen for t >= r_j
                x[j, t] = pulp.LpVariable(
                    f"x_{j},{t}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpInteger
                )
                # A job consumes p[j, t] amount of resources at time t
                # (index t).
                # This can only happen for t >= r_j
                p[j, t] = pulp.LpVariable(
                    f"p_{j},{t}",
                    lowBound=0,
                    cat=pulp.LpContinuous
                )

        # We need a matrix of weights w_j,t of equal dimensions to c that
        # defines the reward of finishing job j at exactly time t.
        # Note that jump points are inclusive. They represent the last time
        # index a job can be completed at to obtain the corresponding reward,
        # not the first time index of the next reward-interval.
        w = np.zeros(shape=(self._njobs, self._totaltime), dtype=np.int32)
        for j in range(self._njobs):
            # We only need to update the non-zero weights, so no need to
            # manipulate anything beyond the last jump point, or before
            # the release date.
            w[j, 0:instance['jump_points'][j, 0] + 1] = \
                instance['weights'][j, 0]
            for k in range(1, self._kjumps + 1):
                w[
                    j,
                    instance['jump_points'][j, k - 1] + 1:
                    instance['jump_points'][j, k] + 1
                ] = instance['weights'][j, k - 1]

        # Add objective
        self._problem += pulp.lpSum([
            w[j, t] * c[j, t]
            for j in range(self._njobs)
            for t in range(instance['jump_points'][j, 0],
                           self._totaltime)
        ])

        # TODO \/
        # Add constraints
        # 1. Ensure every job is completed exactly once
        for j in range(self._njobs):
            self._problem += pulp.lpSum([
                c[j, t]
                for t in range(instance['jump_points'][j, 0],
                               self._totaltime)
            ]) == 1, f"Complete_once_job_{j}"

        # 2. Ensure every job is started exactly once
        for j in range(self._njobs):
            self._problem += pulp.lpSum([
                s[j, t]
                for t in range(instance['jump_points'][j, 0],
                               self._totaltime)
            ]) == 1, f"Start_once_job_{j}"

        # 3. Define the "processed" variable
        for j in range(self._njobs):
            rj = instance['jump_points'][j, 0]
            self._problem += (s[j, rj] - c[j, rj] - x[j, rj]
                              == 0), f"Processing_job_{j}_at_{rj}"
            for t in range(instance['jump_points'][j, 0] + 1,
                           self._totaltime):
                self._problem += (x[j, t-1] + s[j, t] - c[j, t] - x[j, t]
                                  == 0), f"Processing_job_{j}_at_{t}"

        # 4. Resource requirement
        for j in range(self._njobs):
            self._problem += pulp.lpSum([
                p[j, t]
                for t in range(instance['jump_points'][j, 0],
                               self._totaltime)
            ]) == instance['resource_requirement'][j], \
                f"Resource_requirement_job_{j}"

        # 5. Resource availability
        for t in range(self._totaltime):
            self._problem += pulp.lpSum([
                p[j, t]
                for j in range(self._njobs)
            ]) <= instance['resource_availability'][t], \
                f"Resource_availability_time_{t}"

        # 6 & 7. Bound resource consumption
        for j in range(self._njobs):
            for t in range(instance['jump_points'][j, 0],
                           self._totaltime):
                self._problem += (instance['bounds'][j, 0] * x[j, t] - p[j, t]
                                  <= 0), f"Lower_bound_{j}_at_{t}"
                self._problem += (instance['bounds'][j, 1] * x[j, t] - p[j, t]
                                  >= 0), f"Upper_bound_{j}_at_{t}"

    def print_solution(self):
        """Print a human readable version of the (current) solution.

        Notes
        -----
        The current implementation is very minimal, only returning the
        objective value.
        """
        print("Total profit = ", pulp.value(self._problem.objective))
