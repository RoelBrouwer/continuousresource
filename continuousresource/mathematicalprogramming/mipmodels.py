from abc import ABC
from abc import abstractmethod
import docplex.mp.model
from docplex.mp.relax_linear import LinearRelaxer
import numpy as np

from continuousresource.mathematicalprogramming.utils \
    import time_and_resource_vars_to_human_readable_solution_cplex, \
    solution_to_csv_string


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
    def __init__(self, label):
        # Initialize problem
        self._problem = docplex.mp.model.Model(name=label)

    @property
    def problem(self):
        return self._problem

    def solve(self, timelimit=None, threads=1):
        """Solve the MIP.

        Parameters
        ----------
        timelimit : int
            Optional value indicating the timelimit set on solving the
            problem. By default, no timelimit is enforced.
        threads : int
            Optional value indicating the number of threads that the
            solver is allowed to use. Any value below 1 is considered to
            mean no limit is imposed, any positive value will be passed
            as an upper bound on the number of global threads to the
            solver.
        """
        self._problem.set_time_limit(timelimit)
        if threads < 1:
            threads = 0
        self._problem.context.cplex_parameters.threads = threads
        return self._problem.solve()

    def relax_problem(self):
        """Relax all decision variables in the MIP. All integer variables
        (including binary) will be turned into continuous variables.
        """
        self._problem = LinearRelaxer.make_relaxed_model(self._problem)

    def find_conflicts(self):
        """For an infeasible model, find conflicts.
        """
        pass

    @abstractmethod
    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        pass


class JobPropertiesContinuousMIP(MIP):
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
        super().__init__(label)

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
                p[j, i] = self._problem.continuous_var(
                    name=f"p_{j},{i}",
                    lb=0
                )

            # No job can recieve resources in the interval it completes
            p[j, 2 * j + 1].ub = 0

        for i in range(self._nevents):
            t[i] = self._problem.continuous_var(
                name=f"t_{i}",
                lb=0
            )
            for i2 in range(self._nevents):
                if i != i2:
                    a[i, i2] = self._problem.binary_var(
                        name=f"a_{i},{i2}"
                    )
                    b[i, i2] = self._problem.binary_var(
                        name=f"b_{i},{i2}"
                    )

        # Add objective
        self._cost = self._problem.sum(
            instance['jobs'][j, 5] * t[j * 2 + 1] + instance['jobs'][j, 6]
            for j in range(self._njobs)
        )
        self._problem.minimize(self._cost)

        # Add constraints
        for i in range(self._nevents):
            for i2 in range(self._nevents):
                if i != i2:
                    # 1. Define the order amongst the events
                    self._problem.add_constraint(
                        ct=t[i] - t[i2] - bigM * a[i2, i] <= 0,
                        ctname=f"Interval_order_{i}_{i2}"
                    )

                    # 9. Interval resource availability
                    self._problem.add_constraint(
                        ct=self._problem.sum(
                            p[j, i]
                            for j in range(self._njobs)
                        ) - resource * (t[i2] - t[i]) - bigM * a[i2, i] <= 0,
                        ctname=f"Interval_capacity_{i}_{i2}"
                    )

                    # 11. Successor variable backward
                    self._problem.add_constraint(
                        ct=self._problem.sum(
                            (a[i, i3] - a[i2, i3])
                            for i3 in range(self._nevents)
                        ) - 1 - (1 - b[i, i2]) * bigM <= 0,
                        ctname=f"Successor_variable_backward_b_{i},{i2}"
                    )

                    # 12. Successor variable forward
                    self._problem.add_constraint(
                        ct=self._problem.sum(
                            (a[i, i3] - a[i2, i3])
                            for i3 in range(self._nevents)
                        ) - 1 + (1 - b[i, i2]) * bigM >= 0,
                        ctname=f"Successor_variable_forward_b_{i},{i2}"
                    )

                if i2 > i:
                    # 10. Mutual exclusivity of order variables
                    self._problem.add_constraint(
                        ct=a[i, i2] + a[i2, i] == 1,
                        ctname=f"Mutual_exclusive_order_{i}_{i2}"
                    )

        for j in range(self._njobs):
            # 2. Enforce deadline
            self._problem.add_constraint(
                ct=t[2 * j + 1] - instance['jobs'][j, 4] <= 0,
                ctname=f"Deadline_job_{j}"
            )

            # 3. Enforce release time
            self._problem.add_constraint(
                ct=t[2 * j] - instance['jobs'][j, 3] >= 0,
                ctname=f"Release_time_{j}"
            )

            # 4. Meet resource requirement
            self._problem.add_constraint(
                ct=self._problem.sum(
                    p[j, i]
                    for i in range(self._nevents)
                ) - instance['jobs'][j, 0] == 0,
                ctname=f"Resource_requirement_job_{j}"
            )

            for i in range(self._nevents):
                # 7. Upper bound after completion
                self._problem.add_constraint(
                    ct=p[j, i] - bigM * a[i, 2 * j + 1] <= 0,
                    ctname=f"Zero_after_completion_p{j},{i}"
                )

                # 8. Upper bound before start
                self._problem.add_constraint(
                    ct=p[j, i] - bigM * (1 - a[i, 2 * j]) <= 0,
                    ctname=f"Zero_before_start_p{j},{i}"
                )

                for i2 in range(self._nevents):
                    if i != i2:
                        # 5. Lower bound
                        self._problem.add_constraint(
                            ct=p[j, i]
                            - instance['jobs'][j, 1] * (t[i2] - t[i])
                            + (1 - b[i, i2]) * bigM
                            + (1 - a[2 * j, i2]) * bigM
                            + (1 - a[i, 2 * j + 1]) * bigM >= 0,
                            ctname=f"Lower_bound_{j}_for_{i}_{i2}"
                        )

                        # 6. Upper bound
                        self._problem.add_constraint(
                            ct=p[j, i]
                            - instance['jobs'][j, 2] * (t[i2] - t[i])
                            - a[i2, i] * bigM <= 0,
                            ctname=f"Upper_bound_{j}_for_{i}_{i2}"
                        )

        # 13. Fix amount of successors
        self._problem.add_constraint(
            ct=self._problem.sum(
                b[i, i2]
                for i in range(self._nevents)
                for i2 in range(self._nevents) if i != i2
            ) == self._nevents - 1,
            ctname="Amount_of_successors"
        )

        # Store for reference
        self._pvar = p
        self._tvar = t
        self._avar = a
        self._bvar = b

    def add_warmstart(self, t_vars, p_vars, event_order):
        """We assume indices to correspond.
        """
        # TODO: document and sanity checks
        warmstart = self._problem.new_solution()

        for i in range(len(self._tvar)):
            warmstart.add_var_value(self._tvar[i], t_vars[i].solution_value)

        for j in range(self._pvar.shape[0]):
            for i in range(self._pvar.shape[1]):
                if isinstance(p_vars[j, i], docplex.mp.dvar.Var):
                    warmstart.add_var_value(self._pvar[j, i],
                                            p_vars[j, i].solution_value)

        for i in range(len(event_order)):
            for i2 in range(len(event_order)):
                if i == i2:
                    continue
                e1 = 2 * event_order[i, 1] + event_order[i, 0]
                e2 = 2 * event_order[i2, 1] + event_order[i2, 0]
                if i2 == i + 1:
                    warmstart.add_var_value(self._bvar[e1, e2], 1)
                else:
                    warmstart.add_var_value(self._bvar[e1, e2], 0)
                if i2 > i:
                    warmstart.add_var_value(self._avar[e1, e2], 1)
                else:
                    warmstart.add_var_value(self._avar[e1, e2], 0)

        self._problem.add_mip_start(warmstart)

    def get_solution_csv(self):
        (event_labels, event_idx, event_timing, resource_consumption) = \
            time_and_resource_vars_to_human_readable_solution_cplex(
                self._tvar, self._pvar
            )
        return solution_to_csv_string(event_labels, event_idx, event_timing,
                                      resource_consumption)

    def print_solution(self):
        """Print a human readable version of the (current) solution.

        Notes
        -----
        The current implementation is very minimal, only returning the
        objective value.
        """
        print("Total profit = ", self._problem.objective_value)


class JobPropertiesContinuousMIPPlus(JobPropertiesContinuousMIP):
    """Class implementing a Mixed Integer Linear Programming model for a
    resource scheduling problem with continuous time and resource.
    Extends its parent by adding a number of (redundant) constraints on
    the problem, to strengthen the solver's performance.

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
        for j in range(self._njobs):
            # A1. Restrict processing time (upper limit) by lower bound
            if instance['jobs'][j, 1] > 0:
                self._problem.add_constraint(
                    ct=self._tvar[2 * j + 1] - self._tvar[2 * j]
                    - (instance['jobs'][j, 0] / instance['jobs'][j, 1]) <= 0,
                    ctname=f"Processing_time_upper_limit_job_{j}"
                )

            # A2. Restrict processing time (lower limit) by upper bound
            if instance['jobs'][j, 2] > 0:
                self._problem.add_constraint(
                    ct=self._tvar[2 * j + 1] - self._tvar[2 * j]
                    - (instance['jobs'][j, 0] / instance['jobs'][j, 2]) >= 0,
                    ctname=f"Processing_time_lower_limit_job_{j}"
                )


class JumpPointContinuousMIP(MIP):
    def __init__(self, instance, label):
        raise NotImplementedError


class JumpPointContinuousMIPPlus(JumpPointContinuousMIP):
    def __init__(self, instance, label):
        raise NotImplementedError
