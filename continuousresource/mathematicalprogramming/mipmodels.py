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

    def solve(self, solver, timelimit=None):
        """Solve the LP.

        Parameters
        ----------
        solver : {'glpk', 'gurobi', 'cplex'}
            The solver that will be used to solve the LP constructed from
            the provided instance.
        timelimit : int
            Optional value indicating the timelimit set on solving the
            problem. By default, no timelimit is enforced.

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
                timeLimit=timelimit
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
                timeLimit=timelimit,
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
                timeLimit=timelimit,
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


class ContinuousResourceMIP(MIP):
    """Class implementing a Mixed Integer Linear Programming model for a
    resource scheduling problem with continuous time and resource.

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
        super().__init__(label, minimize=True)

        self._njobs = len(instance['jobs'])
        self._nevents = self._njobs * 2
        resource = instance['constants']['resource_availability']
        # TODO: fix to some sensible value
        bigM = 100000

        # Define variables
        p = np.zeros(shape=(self._njobs, self._nevents),
                     dtype=object)
        t = np.zeros(shape=(self._nevents),
                     dtype=object)
        a = np.zeros(shape=(self._nevents, self._nevents),
                     dtype=object)
        b = np.zeros(shape=(self._nevents, self._nevents),
                     dtype=object)

        # For the numbering of jobs and events we will maintain the
        # following relation: a job j is represented by two events in the
        # order, at index i (start event) and i + 1 (completion event),
        # where i = j * 2.

        for j in range(self._njobs):
            for i in range(self._nevents):
                p[j, i] = pulp.LpVariable(
                    f"p_{j},{i}",
                    lowBound=0,
                    cat=pulp.LpContinuous
                )

            # No job can recieve resources in the interval it completes
            p[j, 2 * j + 1].upBound = 0

        for i in range(self._nevents):
            t[i] = pulp.LpVariable(
                f"t_{i}",
                lowBound=0,
                cat=pulp.LpContinuous
            )
            for i2 in range(self._nevents):
                if i != i2:
                    a[i, i2] = pulp.LpVariable(
                        f"a_{i},{i2}",
                        lowBound=0,
                        upBound=1,
                        cat=pulp.LpInteger
                    )
                    b[i, i2] = pulp.LpVariable(
                        f"b_{i},{i2}",
                        lowBound=0,
                        upBound=1,
                        cat=pulp.LpInteger
                    )

        # Add objective
        self._problem += pulp.lpSum([
            instance['jobs'][j, 5] * t[j * 2 + 1] + instance['jobs'][j, 6]
            for j in range(self._njobs)
        ])

        # Add constraints
        for i in range(self._nevents):
            for i2 in range(self._nevents):
                if i != i2:
                    # 1. Define the order amongst the events
                    self._problem += t[i] - t[i2] - bigM * a[i2, i] <= 0, \
                        f"Interval_order_{i}_{i2}"

                    # 9. Interval resource availability
                    self._problem += pulp.lpSum([
                        p[j, i]
                        for j in range(self._njobs)
                    ]) - resource * (t[i2] - t[i]) - bigM * a[i2, i] <= 0, \
                        f"Interval_capacity_{i}_{i2}"

                    # 11. Successor variable backward
                    self._problem += pulp.lpSum([
                        (a[i, i3] - a[i2, i3])
                        for i3 in range(self._nevents)
                    ]) - 1 - (1 - b[i, i2]) * bigM <= 0, \
                        f"Successor_variable_backward_b_{i},{i2}"

                    # 12. Successor variable forward
                    self._problem += pulp.lpSum([
                        (a[i, i3] - a[i2, i3])
                        for i3 in range(self._nevents)
                    ]) - 1 + (1 - b[i, i2]) * bigM >= 0, \
                        f"Successor_variable_forward_b_{i},{i2}"

                if i2 > i:
                    # 10. Mutual exclusivity of order variables
                    self._problem += a[i, i2] + a[i2, i] == 1, \
                        f"Mutual_exclusive_order_{i}_{i2}"

        for j in range(self._njobs):
            # 2. Enforce deadline
            self._problem += t[2 * j + 1] - instance['jobs'][j, 4] <= 0, \
                f"Deadline_job_{j}"

            # 3. Enforce release time
            self._problem += t[2 * j] - instance['jobs'][j, 3] >= 0, \
                f"Release_time_{j}"

            # 4. Meet resource requirement
            self._problem += pulp.lpSum([
                p[j, i]
                for i in range(self._nevents)
            ]) - instance['jobs'][j, 0] == 0, \
                f"Resource_requirement_job_{j}"

            for i in range(self._nevents):
                # 7. Upper bound after completion
                self._problem += p[j, i] - bigM * a[i, 2 * j + 1] <= 0, \
                    f"Zero_after_completion_p{j},{i}"

                # 8. Upper bound before start
                self._problem += p[j, i] - bigM * (1 - a[i, 2 * j]) <= 0, \
                    f"Zero_before_start_p{j},{i}"

                for i2 in range(self._nevents):
                    if i != i2:
                        # 5. Lower bound
                        self._problem += p[j, i] \
                            - instance['jobs'][j, 1] * (t[i2] - t[i]) \
                            + (1 - b[i, i2]) * bigM \
                            + (1 - a[2 * j, i2]) * bigM \
                            + (1 - a[i, 2 * j + 1]) * bigM >= 0, \
                            f"Lower_bound_{j}_for_{i}_{i2}"

                        # 6. Upper bound
                        self._problem += p[j, i] \
                            - instance['jobs'][j, 2] * (t[i2] - t[i]) \
                            - a[i2, i] * bigM <= 0, \
                            f"Upper_bound_{j}_for_{i}_{i2}"

        # 13. Fix amount of successors
        self._problem += pulp.lpSum([
            b[i, i2]
            for i in range(self._nevents)
            for i2 in range(self._nevents) if i != i2
        ]) == self._nevents - 1, \
        "Amount_of_successors"

    def print_solution(self):
        """Print a human readable version of the (current) solution.

        Notes
        -----
        The current implementation is very minimal, only returning the
        objective value.
        """
        print("Total profit = ", pulp.value(self._problem.objective))
