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
    def _initialize_model(self, instance):
        """Initialize the MIP model.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        raise NotImplementedError


class EventOrderBasedMIP(MIP):
    """Class implementing the shared elements of Mixed Integer Linear
    Programming models for a resource scheduling problem with continuous
    time and resource, using an event-based approach.

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
        self._initialize_model(instance)

    def _initialize_model(self, instance):
        """Initialize the MIP model.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        # For the numbering of jobs and events we will maintain the
        # following relation: a job j is represented by two events in the
        # order, at index i (start event) and i + 1 (completion event),
        # where i = j * 2. Any other events start from index 2n - 1.
        self._define_variable_arrays()

        # Add objective
        self._set_objective()

        # Event-based constraints
        for i in range(self._nevents):
            for i2 in range(self._nevents):
                if i != i2:
                    # 1. Define the order amongst the events
                    self._add_order_constraint(i, i2)

                    # 11. Successor variable backward
                    self._add_successor_backward_constraint(i, i2)

                    # 12. Successor variable forward
                    self._add_successor_forward_constraint(i, i2)

                if i2 > i:
                    # 10. Mutual exclusivity of order variables
                    self._add_order_exclusivity_constraint(i, i2)

        # Plannable event-based constraints
        for i in range(self._nplannable):
            for i2 in range(self._nplannable):
                if i != i2:
                    # 9. Interval resource availability
                    self._add_interval_capacity_constraint(i, i2)

        # Job-based constraints
        for j in range(self._njobs):
            # 2. Enforce deadline
            deadline = self._get_deadline(j, instance)
            self._add_deadline_constraint(j, deadline)

            # 3. Enforce release time
            release_time = self._get_release_time(j, instance)
            self._add_release_time_constraint(j, release_time)

            # 4. Meet resource requirement
            resource_requirement = self._get_resource_requirement(j, instance)
            self._add_resource_requirement_constraint(j, resource_requirement)

            for i in range(self._nplannable):
                # 7. Upper bound after completion
                self._add_inactive_after_completion_constraint(j, i)

                # 8. Upper bound before start
                self._add_inactive_before_start_constraint(j, i)

                for i2 in range(self._nplannable):
                    if i != i2:
                        # 5. Lower bound
                        lower_bound = self._get_lower_bound(j, instance)
                        self._add_lower_bound_constraint(j, i, i2,
                                                         lower_bound)

                        # 6. Upper bound
                        upper_bound = self._get_upper_bound(j, instance)
                        self._add_upper_bound_constraint(j, i, i2,
                                                         upper_bound)

        # 13. Fix amount of successors
        self._add_successor_limit_constraint()

    @abstractmethod
    def _define_variable_arrays(self):
        """Defines the arrays organizing all decision variables in the
        model. Expected to fill the following four private variables:
        `_pvar` (resources variables), `_tvar` (time variables), `_avar`
        (order variables), `_bvar` (successor variables).
        """
        raise NotImplementedError

    @abstractmethod
    def _set_objective(self, instance):
        """Sets the objective of the MILP model. Expected to store the
        objective in the private `_cost` variable.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_deadline(self, j, instance):
        """Get the deadline of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the deadline for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_release_time(self, j, instance):
        """Get the release time of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the release time for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_resource_requirement(self, j, instance):
        """Get the resource requirement of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the resource requirement for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_lower_bound(self, j, instance):
        """Get the lower bound of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the lower bound for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_upper_bound(self, j, instance):
        """Get the upper bound of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the upper bound for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    def _add_order_constraint(self, i, i2):
        """Adds a constraint to the model that defines the order amongst
        the events with the provided indices in the variable arrays
        `_tvar` and `_avar`.

        Parameters
        ----------
        i : int
            Index of the first event in the variable arrays `_tvar` and
            `_avar`.
        i2 : int
            Index of the second event in the variable arrays `_tvar` and
            `_avar`.
        """
        self._problem.add_constraint(
            ct=self._tvar[i] - self._tvar[i2]
            - self._bigM * self._avar[i2, i] <= 0,
            ctname=f"Interval_order_{i}_{i2}"
        )

    def _add_successor_backward_constraint(self, i, i2):
        """Adds a constraint to the model that regulates the value of the
        successor variable (`_bvar`) for the events with the provided
        indices in the variable arrays `_avar` and `_bvar` (part 1).

        Parameters
        ----------
        i : int
            Index of the first event in the variable arrays `_avar` and
            `_bvar`.
        i2 : int
            Index of the second event in the variable arrays `_avar` and
            `_bvar`.
        """
        self._problem.add_constraint(
            ct=self._problem.sum(
                (self._avar[i, i3] - self._avar[i2, i3])
                for i3 in range(self._nevents)
            ) - 1 - (1 - self._bvar[i, i2]) * self._bigM <= 0,
            ctname=f"Successor_variable_backward_b_{i},{i2}"
        )

    def _add_successor_forward_constraint(self, i, i2):
        """Adds a constraint to the model that regulates the value of the
        successor variable (`_bvar`) for the events with the provided
        indices in the variable arrays `_avar` and `_bvar` (part 2).

        Parameters
        ----------
        i : int
            Index of the first event in the variable arrays `_avar` and
            `_bvar`.
        i2 : int
            Index of the second event in the variable arrays `_avar` and
            `_bvar`.
        """
        self._problem.add_constraint(
            ct=self._problem.sum(
                (self._avar[i, i3] - self._avar[i2, i3])
                for i3 in range(self._nevents)
            ) - 1 + (1 - self._bvar[i, i2]) * self._bigM >= 0,
            ctname=f"Successor_variable_backward_b_{i},{i2}"
        )

    def _add_order_exclusivity_constraint(self, i, i2):
        """Adds a constraint to the model that ensures only one of the
        two order variables (`_avar`) for the events with the provided
        indices in the variable array `_avar` can be 1.

        Parameters
        ----------
        i : int
            Index of the first event in the variable array `_avar`.
        i2 : int
            Index of the second event in the variable array `_avar`.
        """
        self._problem.add_constraint(
            ct=self._avar[i, i2] + self._avar[i2, i] == 1,
            ctname=f"Mutual_exclusive_order_{i}_{i2}"
        )

    def _add_interval_capacity_constraint(self, i, i2):
        """Adds a constraint to the model that ensures the resource limit
        is respected within the interval.

        Parameters
        ----------
        i : int
            Index of the first event in the variable arrays `_tvar` and
            `_avar`.
        i2 : int
            Index of the second event in the variable arrays `_tvar` and
            `_avar`.
        """
        self._problem.add_constraint(
            ct=self._problem.sum(
                self._pvar[j, i]
                for j in range(self._njobs)
            ) - self._resource * (self._tvar[i2] - self._tvar[i])
            - self._bigM * self._avar[i2, i] <= 0,
            ctname=f"Interval_capacity_{i}_{i2}"
        )

    def _add_deadline_constraint(self, j, deadline):
        """Adds a constraint to the model that sets the deadline for job
        `j`.

        Parameters
        ----------
        j : int
            Index of the job to set the deadline for.
        deadline : float
            Deadline of the job.
        """
        self._problem.add_constraint(
            ct=self._tvar[2 * j + 1] - deadline <= 0,
            ctname=f"Deadline_job_{j}"
        )

    def _add_release_time_constraint(self, j, release_time):
        """Adds a constraint to the model that sets the release time for
        job `j`.

        Parameters
        ----------
        j : int
            Index of the job to set the release time for.
        release_time : float
            Release time of the job.
        """
        self._problem.add_constraint(
            ct=self._tvar[2 * j] - release_time >= 0,
            ctname=f"Release_time_{j}"
        )

    def _add_resource_requirement_constraint(self, j, resource_requirement):
        """Adds a constraint to the model that sets the resource requirement
        for job `j`.

        Parameters
        ----------
        j : int
            Index of the job to set the resource requirement for.
        resource_requirement : float
            Resource requirement of the job.
        """
        self._problem.add_constraint(
            ct=self._problem.sum(
                self._pvar[j, i]
                for i in range(self._nplannable)
            ) - resource_requirement == 0,
            ctname=f"Resource_requirement_job_{j}"
        )

    def _add_inactive_after_completion_constraint(self, j, i):
        """Adds a constraint to the model that ensures job `j` does not
        consume any resources within the interval if it is after its
        completion.

        Parameters
        ----------
        j : int
            Index of the job.
        i : int
            Index of the interval.
        """
        self._problem.add_constraint(
            ct=self._pvar[j, i] - self._bigM * self._avar[i, 2 * j + 1] <= 0,
            ctname=f"Zero_after_completion_p{j},{i}"
        )

    def _add_inactive_before_start_constraint(self, j, i):
        """Adds a constraint to the model that ensures job `j` does not
        consume any resources within the interval if it is before its
        start.

        Parameters
        ----------
        j : int
            Index of the job.
        i : int
            Index of the interval.
        """
        self._problem.add_constraint(
            ct=self._pvar[j, i]
            - self._bigM * (1 - self._avar[i, 2 * j]) <= 0,
            ctname=f"Zero_before_start_p{j},{i}"
        )

    def _add_lower_bound_constraint(self, j, i, i2, lower_bound):
        """Adds a constraint to the model that enforces the lower bound
        of job `j` in the interval between the two events, if they are
        consecutive and between the start and completion of job `j`.

        Parameters
        ----------
        j : int
            Index of the job.
        i : int
            Index of the first event.
        i2 : int
            Index of the second event.
        lower_bound : float
            Lower bound of the job.
        """
        self._problem.add_constraint(
            ct=self._pvar[j, i]
            - lower_bound * (self._tvar[i2] - self._tvar[i])
            + (1 - self._bvar[i, i2]) * self._bigM
            + (1 - self._avar[2 * j, i2]) * self._bigM
            + (1 - self._avar[i, 2 * j + 1]) * self._bigM >= 0,
            ctname=f"Lower_bound_{j}_for_{i}_{i2}"
        )

    def _add_upper_bound_constraint(self, j, i, i2, upper_bound):
        """Adds a constraint to the model that enforces the upper bound
        of job `j` in the interval between the two events, if `i2` occurs
        after `i`.

        Parameters
        ----------
        j : int
            Index of the job.
        i : int
            Index of the first event.
        i2 : int
            Index of the second event.
        upper_bound : float
            Upper bound of the job.
        """
        self._problem.add_constraint(
            ct=self._pvar[j, i]
            - upper_bound * (self._tvar[i2] - self._tvar[i])
            - self._avar[i2, i] * self._bigM <= 0,
            ctname=f"Upper_bound_{j}_for_{i}_{i2}"
        )

    def _add_successor_limit_constraint(self):
        """Adds a constraint to the model that sets the amount of
        successor variables (`_bvar`) that can be 1.
        """
        self._problem.add_constraint(
            ct=self._problem.sum(
                self._bvar[i, i2]
                for i in range(self._nevents)
                for i2 in range(self._nevents) if i != i2
            ) == self._nevents - 1,
            ctname="Amount_of_successors"
        )


class JobPropertiesContinuousMIP(EventOrderBasedMIP):
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
        # Set some constants
        self._njobs = len(instance['jobs'])
        self._nevents = self._njobs * 2
        self._nplannable = self._nevents
        self._resource = instance['constants']['resource_availability']
        # TODO: fix to some sensible value
        self._bigM = 100000

        super().__init__(instance, label)

    def _define_variable_arrays(self):
        """Defines the arrays organizing all decision variables in the
        model. Fills the following four private variables: `_pvar`
        (resources variables), `_tvar` (time variables), `_avar` (order
        variables), `_bvar` (successor variables).
        """
        self._pvar = np.zeros(shape=(self._njobs, self._nplannable),
                              dtype=object)
        self._tvar = np.zeros(shape=(self._nevents),
                              dtype=object)
        self._avar = np.zeros(shape=(self._nevents, self._nevents),
                              dtype=object)
        self._bvar = np.zeros(shape=(self._nevents, self._nevents),
                              dtype=object)

        for j in range(self._njobs):
            for i in range(self._nplannable):
                self._pvar[j, i] = self._problem.continuous_var(
                    name=f"p_{j},{i}",
                    lb=0
                )

            # No job can recieve resources in the interval it completes
            self._pvar[j, 2 * j + 1].ub = 0

        for i in range(self._nevents):
            self._tvar[i] = self._problem.continuous_var(
                name=f"t_{i}",
                lb=0
            )
            for i2 in range(self._nevents):
                if i != i2:
                    self._avar[i, i2] = self._problem.binary_var(
                        name=f"a_{i},{i2}"
                    )
                    self._bvar[i, i2] = self._problem.binary_var(
                        name=f"b_{i},{i2}"
                    )

    def _set_objective(self, instance):
        """Sets the objective of the MILP model. Stores the objective in
        the private `_cost` variable.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        self._cost = self._problem.sum(
            instance['jobs'][j, 5] * self._tvar[j * 2 + 1]
            + instance['jobs'][j, 6]
            for j in range(self._nplannable)
        )
        self._problem.minimize(self._cost)

    def _get_deadline(self, j, instance):
        """Get the deadline of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the deadline for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['jobs'][j, 4]

    def _get_release_time(self, j, instance):
        """Get the release time of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the release time for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['jobs'][j, 3]

    def _get_resource_requirement(self, j, instance):
        """Get the resource requirement of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the resource requirement for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['jobs'][j, 0]

    def _get_lower_bound(self, j, instance):
        """Get the lower bound of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the lower bound for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['jobs'][j, 1]

    def _get_upper_bound(self, j, instance):
        """Get the upper bound of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the upper bound for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['jobs'][j, 2]

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


class JumpPointContinuousMIP(EventOrderBasedMIP):
    def __init__(self, instance, label):
        raise NotImplementedError


class JumpPointContinuousMIPPlus(JumpPointContinuousMIP):
    def __init__(self, instance, label):
        raise NotImplementedError
