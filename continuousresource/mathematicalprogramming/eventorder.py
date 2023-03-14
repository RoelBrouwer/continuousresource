from abc import abstractmethod
import docplex.mp.model
import math
import numpy as np
import os

from continuousresource.mathematicalprogramming.abstract \
    import LP, LPWithSlack, MIP
from continuousresource.mathematicalprogramming.utils \
    import time_and_resource_vars_to_human_readable_solution_cplex, \
    solution_to_csv_string


class EventOrderLinearModel(LP):
    """Class implementing the shared elements of Linear Programming
    models for a resource scheduling problem with continuous time and
    resource, using an event-based approach.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    **kwargs
    """
    def __init__(self, instance, label):
        super().__init__(label)
        self._kextra = int((self._nevents - self._nplannable) / self._njobs)
        self._initialize_model(instance)

    def _initialize_model(self, instance):
        """Initialize the LP model.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        # Create variables
        self._define_variable_arrays(instance)

        # Define objective
        self._set_objective(instance)

        # Initialize constraints
        self._define_constraint_arrays(instance)

        for i in range(self._nevents - 1):
            e = instance['eventlist'][i, 1] * 2 + instance['eventlist'][i, 0]
            if instance['eventlist'][i, 0] > 1:
                e += self._nplannable - 2 + \
                    instance['eventlist'][i, 1] * (self._kextra - 2)
            e1 = instance['eventlist'][i + 1, 1] * 2 + \
                instance['eventlist'][i + 1, 0]
            if instance['eventlist'][i + 1, 0] > 1:
                e1 += self._nplannable - 2 + \
                    instance['eventlist'][i + 1, 1] * (self._kextra - 2)
            # 1. Event Order
            self._c_order[e] = self._add_order_constraint(e, e1)

        e = -1
        e1 = -1
        for i in range(self._nevents):
            # We ignore non-plannable events
            if instance['eventlist'][i, 0] < 2:
                e = e1
                e1 = instance['eventlist'][i, 1] * 2 + \
                    instance['eventlist'][i, 0]
                # 7. Resource availability
                if e > -1:
                    self._c_availability[e] = \
                        self._add_interval_capacity_constraint(e, e1)

        for j in range(self._njobs):
            # 2. Enforce deadline
            deadline = self._get_deadline(j, instance)
            self._c_deadline[j] = self._add_deadline_constraint(j, deadline)

            # 3. Enforce release time
            release_time = self._get_release_time(j, instance)
            self._c_release[j] = \
                self._add_release_time_constraint(j, release_time)

            # 4. Resource requirement
            resource_requirement = self._get_resource_requirement(j, instance)
            self._c_resource[j] = self._add_resource_requirement_constraint(
                j, resource_requirement, instance
            )

            e = -1
            e1 = 2 * j + 0
            for i in range(instance['eventmap'][j, 0] + 1,
                           instance['eventmap'][j, 1] + 1):
                # We ignore non-plannable events
                if instance['eventlist'][i, 0] < 2:
                    e = e1
                    e1 = instance['eventlist'][i, 1] * 2 \
                        + instance['eventlist'][i, 0]
                    # 5. Lower bound
                    lower_bound = self._get_lower_bound(j, instance)
                    self._c_lower[j, e] = self._add_lower_bound_constraint(
                        j, e, e1, lower_bound
                    )
                    # 6. Upper bound
                    upper_bound = self._get_upper_bound(j, instance)
                    self._c_upper[j, e] = self._add_upper_bound_constraint(
                        j, e, e1, upper_bound
                    )

    def update_swap_neighbors(self, instance, first_idx):
        """Update the existing model by swapping two neighboring events
        in the eventlist.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        first_idx : int
            Position of the first event in the event list that will
            switch positions with its successor.
        """
        # We assume the events have already been switched in `instance`
        # by the calling method.
        job1 = instance['eventlist'][first_idx + 1, 1]
        type1 = instance['eventlist'][first_idx + 1, 0]
        job2 = instance['eventlist'][first_idx, 1]
        type2 = instance['eventlist'][first_idx, 0]
        e1 = job1 * 2 + type1
        if type1 > 1:
            e1 += self._nplannable - 2 + job1 * (self._kextra - 2)
        e2 = job2 * 2 + type2
        if type2 > 1:
            e2 += self._nplannable - 2 + job2 * (self._kextra - 2)

        # Intervals only need updates if we swap two plannable events.
        e0 = -1
        e3 = -1
        if type1 < 2 and type2 < 2:
            pos = first_idx
            while e0 < 0 and pos > 0:
                pos -= 1
                if instance['eventlist'][pos, 0] < 2:
                    e0 = instance['eventlist'][pos, 1] * 2 + \
                        instance['eventlist'][pos, 0]
            pos = first_idx + 2
            while e3 < 0 and pos < self._nevents:
                if instance['eventlist'][pos, 0] < 2:
                    e3 = instance['eventlist'][pos, 1] * 2 + \
                        instance['eventlist'][pos, 0]
                pos += 1

        if job1 == job2 and type2 == 1 and type1 == 0:
            raise RuntimeError("Cannot put a job's completion before its"
                               " start.")

        # Eventlist and eventmap updates used to be done here, now
        # expected to be handled by the calling method.

        # update self._resource variables
        if type1 == 1 and type2 < 2 and \
           not isinstance(self._pvar[job1, e2], docplex.mp.dvar.Var):
            # The completion of job1 happens one interval later, a new
            # variable must be added.
            self._pvar[job1, e2] = \
                self._problem.continuous_var(
                    name=f"p_{job1},{e2}",
                    lb=0
                )
        if type2 == 0 and type1 < 2 and \
           not isinstance(self._pvar[job2, e1], docplex.mp.dvar.Var):
            # The start of job2 happens one interval earlier, a new
            # variable must be added.
            self._pvar[job2, e1] = \
                self._problem.continuous_var(
                    name=f"p_{job2},{e1}",
                    lb=0
                )

        # update appropriate self._c_order
        if first_idx > 0:
            e = instance['eventlist'][first_idx - 1, 1] * 2 \
                + instance['eventlist'][first_idx - 1, 0]
            if instance['eventlist'][first_idx - 1, 0] > 1:
                e += self._nplannable - 2 + \
                    instance['eventlist'][first_idx - 1, 1] * \
                    (self._kextra - 2)
            self._c_order[e].lhs = \
                self._tvar[e] - self._tvar[e2]

        if isinstance(self._c_order[e2], docplex.mp.constr.LinearConstraint):
            self._c_order[e2].lhs = \
                self._tvar[e2] - self._tvar[e1]
        else:
            self._c_order[e2] = self._add_order_constraint(e2, e1)

        if first_idx + 2 < self._nevents:
            e = instance['eventlist'][first_idx + 2, 1] * 2 \
                + instance['eventlist'][first_idx + 2, 0]
            if instance['eventlist'][first_idx + 2, 0] > 1:
                e += self._nplannable - 2 + \
                    instance['eventlist'][first_idx + 2, 1] * \
                    (self._kextra - 2)
            self._c_order[e1].lhs = \
                self._tvar[e1] - self._tvar[e]
        else:
            self._problem.remove_constraints([
                self._c_order[e1]
            ])
            self._c_order[e1] = None

        # update appropriate self._c_availability
        if e0 > -1:
            self._c_availability[e0].lhs = \
                self._problem.sum(
                    self._pvar[j, e0] for j in range(self._njobs)
                ) - self._resource * (self._tvar[e2] - self._tvar[e0])

        if type1 < 2 and type2 < 2:
            if isinstance(self._c_availability[e2],
                          docplex.mp.constr.LinearConstraint):
                self._c_availability[e2].lhs = \
                    self._problem.sum(
                        self._pvar[j, e2] for j in range(self._njobs)
                    ) - self._resource * (self._tvar[e1] - self._tvar[e2])
            else:
                self._c_availability[e2] = \
                    self._add_interval_capacity_constraint(e2, e1)

        if e3 > -1:
            self._c_availability[e1].lhs = \
                self._problem.sum(
                    self._pvar[j, e1] for j in range(self._njobs)
                ) - self._resource * (self._tvar[e3] - self._tvar[e1])
        elif type1 < 2 and type2 < 2:
            self._problem.remove_constraints([
                self._c_availability[e1]
            ])
            self._c_availability[e1] = None

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1 and type2 < 2:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval (now completes at the start of
            # e2, i.e. has to be enforced during the preceding interval
            # 5. Lower bound
            lower_bound = self._get_lower_bound(job1, instance)
            self._c_lower[job1, e2] = self._add_lower_bound_constraint(
                job1, e2, e1, lower_bound
            )
            # 6. Upper bound
            upper_bound = self._get_upper_bound(job1, instance)
            self._c_upper[job1, e2] = self._add_upper_bound_constraint(
                job1, e2, e1, lower_bound
            )
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for e0 > -1: this is always true
            assert e0 > -1
            self._c_lower[job1, e0].lhs = self._pvar[job1, e0] - \
                lower_bound * (self._tvar[e2] - self._tvar[e0])
            self._c_upper[job1, e0].lhs = self._pvar[job1, e0] - \
                upper_bound * (self._tvar[e2] - self._tvar[e0])
        elif type1 == 0 and type2 < 2:
            # Delayed start, so bounds need to be enforced for one fewer
            # interval
            self._problem.remove_constraints([
                self._c_lower[job1, e2],
                self._c_upper[job1, e2]
            ])
            self._c_lower[job1, e2] = None
            self._c_upper[job1, e2] = None
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check boundary: this always holds
            assert e3 > -1
            lower_bound = self._get_lower_bound(job1, instance)
            self._c_lower[job1, e1].lhs = self._pvar[job1, e1] - \
                lower_bound * (self._tvar[e3] - self._tvar[e1])
            upper_bound = self._get_upper_bound(job1, instance)
            self._c_upper[job1, e1].lhs = self._pvar[job1, e1] - \
                upper_bound * (self._tvar[e3] - self._tvar[e1])

        if type2 == 0 and type1 < 2:
            # Earlier start, so bounds need to be enforced for an
            # additional interval
            # No need to check boundary: this always holds
            assert e3 > -1
            # 5. Lower bound
            lower_bound = self._get_lower_bound(job2, instance)
            self._c_lower[job2, e1] = self._add_lower_bound_constraint(
                job2, e1, e3, lower_bound
            )
            # 6. Upper bound
            upper_bound = self._get_upper_bound(job2, instance)
            self._c_upper[job2, e1] = self._add_upper_bound_constraint(
                job2, e1, e3, upper_bound
            )
            # In addition, the time variables need to be updated in two
            # more constraints.
            self._c_lower[job2, e2].lhs = self._pvar[job2, e2] - \
                lower_bound * (self._tvar[e1] - self._tvar[e2])
            self._c_upper[job2, e2].lhs = self._pvar[job2, e2] - \
                upper_bound * (self._tvar[e1] - self._tvar[e2])
        elif type2 == 1 and type1 < 2:
            # Earlier completion, so bounds need to be enforced for one
            # fewer interval (and it is not enforced on the last
            # interval anyway, because it completes at the start of this
            # one).
            self._problem.remove_constraints([
                self._c_lower[job2, e1],
                self._c_upper[job2, e1]
            ])
            self._c_lower[job2, e1] = None
            self._c_upper[job2, e1] = None
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for e0 > -1: this is always true
            assert e0 > -1
            lower_bound = self._get_lower_bound(job2, instance)
            self._c_lower[job2, e0].lhs = self._pvar[job2, e0] - \
                lower_bound * (self._tvar[e2] - self._tvar[e0])
            upper_bound = self._get_upper_bound(job2, instance)
            self._c_upper[job2, e0].lhs = \
                self._pvar[job2, e0] - \
                upper_bound * (self._tvar[e2] - self._tvar[e0])

        # Update bounds for all other jobs that are active
        # For any other job to be active, both e0 and e3 need to exist.
        if e0 > -1 and e3 > -1:
            for j in range(self._njobs):
                if instance['eventmap'][j, 0] < first_idx \
                   and instance['eventmap'][j, 1] > first_idx + 1:
                    lower_bound = self._get_lower_bound(j, instance)
                    self._c_lower[j, e0].lhs = self._pvar[j, e0] - \
                        lower_bound * (self._tvar[e2] - self._tvar[e0])
                    self._c_lower[j, e2].lhs = self._pvar[j, e2] - \
                        lower_bound * (self._tvar[e1] - self._tvar[e2])
                    self._c_lower[j, e1].lhs = self._pvar[j, e1] - \
                        lower_bound * (self._tvar[e3] - self._tvar[e1])
                    upper_bound = self._get_upper_bound(j, instance)
                    self._c_upper[j, e0].lhs = self._pvar[j, e0] - \
                        upper_bound * (self._tvar[e2] - self._tvar[e0])
                    self._c_upper[j, e2].lhs = self._pvar[j, e2] - \
                        upper_bound * (self._tvar[e1] - self._tvar[e2])
                    self._c_upper[j, e1].lhs = self._pvar[j, e1] - \
                        upper_bound * (self._tvar[e3] - self._tvar[e1])

        # update appropriate self._c_resource
        for j in [job1, job2]:
            resource_requirement = self._get_resource_requirement(j, instance)
            self._c_resource[j].lhs = self._problem.sum(
                self._pvar[
                    j,
                    instance['eventlist'][i, 1] * 2 +
                    instance['eventlist'][i, 0]
                ] for i in range(
                    instance['eventmap'][j, 0], instance['eventmap'][j, 1]
                ) if instance['eventlist'][i, 0] < 2
            ) - resource_requirement

    @abstractmethod
    def _set_objective(self, instance):
        """Sets the objective of the LP model. Expected to store the
        objective in the private `_cost` variable.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _define_variable_arrays(self):
        """Defines the arrays organizing all decision variables in the
        model. Expected to fill the following four private variables:
        `_pvar` (resources variables), `_tvar` (time variables).
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
        the events with the provided indices in the variable array
        `_tvar`.

        Parameters
        ----------
        i : int
            Index of the first event in the variable array `_tvar`.
        i2 : int
            Index of the second event in the variable array `_tvar`.
        """
        return self._problem.add_constraint(
            ct=self._tvar[i] - self._tvar[i2] <= 0,
            ctname=f"Interval_order_{i}_{i2}"
        )

    def _add_interval_capacity_constraint(self, i, i2):
        """Adds a constraint to the model that ensures the resource limit
        is respected within the interval.

        Parameters
        ----------
        i : int
            Index of the first event in the variable array `_tvar`.
        i2 : int
            Index of the second event in the variable array `_tvar`.
        """
        return self._problem.add_constraint(
            ct=self._problem.sum(
                self._pvar[j, i]
                for j in range(self._njobs)
            ) - self._resource * (self._tvar[i2] - self._tvar[i]) <= 0,
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
        return self._problem.add_constraint(
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
        return self._problem.add_constraint(
            ct=self._tvar[2 * j] - release_time >= 0,
            ctname=f"Release_time_{j}"
        )

    def _add_resource_requirement_constraint(self, j, resource_requirement,
                                             instance):
        """Adds a constraint to the model that sets the resource requirement
        for job `j`.

        Parameters
        ----------
        j : int
            Index of the job to set the resource requirement for.
        resource_requirement : float
            Resource requirement of the job.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return self._problem.add_constraint(
            ct=self._problem.sum(
                self._pvar[
                    j,
                    instance['eventlist'][i, 1] * 2
                    + instance['eventlist'][i, 0]
                ] for i in range(
                    instance['eventmap'][j, 0], instance['eventmap'][j, 1]
                ) if instance['eventlist'][i, 0] < 2
            ) - resource_requirement == 0,
            ctname=f"Resource_requirement_job_{j}"
        )

    def _add_lower_bound_constraint(self, j, i, i2, lower_bound):
        """Adds a constraint to the model that enforces the lower bound
        of job `j` in the interval between the two events.

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
        return self._problem.add_constraint(
            ct=self._pvar[j, i]
            - lower_bound * (self._tvar[i2] - self._tvar[i]) >= 0,
            ctname=f"Lower_bound_{j}_for_{i}_{i2}"
        )

    def _add_upper_bound_constraint(self, j, i, i2, upper_bound):
        """Adds a constraint to the model that enforces the upper bound
        of job `j` in the interval between the two events.

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
        return self._problem.add_constraint(
            ct=self._pvar[j, i]
            - upper_bound * (self._tvar[i2] - self._tvar[i]) <= 0,
            ctname=f"Upper_bound_{j}_for_{i}_{i2}"
        )

    def get_schedule(self):
        """Return the schedule that corresponds to the current solution.

        Returns
        -------
        ndarray
            One-dimensional (|E|) array containing the assigned time
            in the schedule for all events, in the order of the
            event indices.
        """
        # The schedule as intended below would mix integer and floats
        # schedule = np.zeros(shape=(len(self._event_list), 3)
        # schedule[:,:2] = event_list
        # time_assignment = np.zeros(shape=len(self._event_list))
        return np.array([t.solution_value for t in self._tvar])

    def get_solution_csv(self):
        (event_labels, event_idx, event_timing, resource_consumption) = \
            time_and_resource_vars_to_human_readable_solution_cplex(
                self._tvar, self._pvar,
            )
        return solution_to_csv_string(event_labels, event_idx, event_timing,
                                      resource_consumption)


class EventOrderMixedModel(MIP, EventOrderLinearModel):
    """Class implementing the shared elements of Mixed Integer Linear
    Programming models for a resource scheduling problem with continuous
    time and resource, using an event-based approach.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    **kwargs
    """
    def __init__(self, instance, label):
        super().__init__(instance, label)

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
        self._define_variable_arrays(instance)

        # Add objective
        self._set_objective(instance)

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
                self._tvar[:self._nplannable], self._pvar
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


class JobPropertiesContinuousMIP(EventOrderMixedModel):
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

    def _define_variable_arrays(self, instance):
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
            for j in range(self._njobs)
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


class JumpPointContinuousMIP(EventOrderMixedModel):
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
        self._njobs = len(instance['properties'])
        self._kjumppoints = instance['jumppoints'].shape[1] - 1
        self._nevents = self._njobs * (self._kjumppoints + 1)
        self._nplannable = self._njobs * 2
        self._resource = instance['constants']['resource_availability']
        # TODO: fix to some sensible value
        self._bigM = 100000

        super().__init__(instance, label)

    def _define_variable_arrays(self, instance):
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
            if i < self._nplannable:
                self._tvar[i] = self._problem.continuous_var(
                    name=f"t_{i}",
                    lb=0
                )
            else:
                j = math.floor((i - self._nplannable) /
                               (self._kjumppoints - 1))
                k = (i - self._nplannable) % (self._kjumppoints - 1)
                self._tvar[i] = instance['jumppoints'][j, k + 1]
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
            instance['weights'][j, 0] +
            self._problem.sum(
                instance['weights'][j, k] *
                self._avar[
                    self._nplannable + (self._kjumppoints - 1) * j + k - 1,
                    2 * j + 1
                ]
                for k in range(1, self._kjumppoints)
            )
            for j in range(self._njobs)
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
        return instance['jumppoints'][j, -1]

    def _get_release_time(self, j, instance):
        """Get the release time of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the release time for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['jumppoints'][j, 0]

    def _get_resource_requirement(self, j, instance):
        """Get the resource requirement of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the resource requirement for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['properties'][j, 0]

    def _get_lower_bound(self, j, instance):
        """Get the lower bound of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the lower bound for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['properties'][j, 1]

    def _get_upper_bound(self, j, instance):
        """Get the upper bound of job `j`.

        Parameters
        ----------
        j : int
            Index of the job to get the upper bound for.
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        return instance['properties'][j, 2]


class JumpPointContinuousMIPPlus(JumpPointContinuousMIP):
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
            if instance['properties'][j, 1] > 0:
                self._problem.add_constraint(
                    ct=self._tvar[2 * j + 1] - self._tvar[2 * j]
                    - (instance['properties'][j, 0] /
                       instance['properties'][j, 1]) <= 0,
                    ctname=f"Processing_time_upper_limit_job_{j}"
                )

            # A2. Restrict processing time (lower limit) by upper bound
            if instance['properties'][j, 2] > 0:
                self._problem.add_constraint(
                    ct=self._tvar[2 * j + 1] - self._tvar[2 * j]
                    - (instance['properties'][j, 0] /
                       instance['properties'][j, 2]) >= 0,
                    ctname=f"Processing_time_lower_limit_job_{j}"
                )


class JobPropertiesContinuousLP(EventOrderLinearModel):
    """Class implementing a Linear Programming model for a resource
    scheduling problem with continuous time and resource, that determines
    a schedule for a given event order.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data, with the following keys:
            - `jobs`: two-dimensional (n x 7) array containing job
              properties:
                - 0: resource requirement (E_j);
                - 1: resource lower bound (P^-_j);
                - 2: resource upper bound (P^+_j);
                - 3: release date (r_j);
                - 4: deadline (d_j);
                - 5: weight (W_j);
                - 6: objective constant (B_j).
            - `resource`: Amount of resource available per time unit
              (float).
            - `eventlist`: two-dimensional (|E| x 2) array representing
              the events in the problem, where the first column contains
              an integer indicating the event type (0 for start, 1 for
              completion) and the second column the associated job ID.
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

        super().__init__(instance, label)

    def _define_variable_arrays(self, instance):
        """Defines the arrays organizing all decision variables in the
        model. Fills the following two private variables: `_pvar`
        (resources variables), `_tvar` (time variables).
        """
        self._pvar = np.zeros(shape=(self._njobs, self._nplannable),
                              dtype=object)
        self._tvar = np.zeros(shape=(self._nevents),
                              dtype=object)

        for j in range(self._njobs):
            for i in range(instance['eventmap'][j, 0],
                           instance['eventmap'][j, 1]):
                e = instance['eventlist'][i, 1] * 2 + \
                    instance['eventlist'][i, 0]
                self._pvar[j, e] = self._problem.continuous_var(
                    name=f"p_{j},{e}",
                    lb=0
                )

        for i in range(self._nevents):
            self._tvar[i] = self._problem.continuous_var(
                name=f"t_{i}",
                lb=0
            )

    def _define_constraint_arrays(self, instance):
        """Defines the arrays organizing all constraints in the model.
        Fills the following seven private variables: `_c_order`,
        `_c_availability`, `_c_deadline`, `_c_release`, `_c_resource`,
        `_c_lower`, `_c_upper`.
        """
        self._c_order = np.zeros(shape=self._nevents,
                                 dtype=object)
        self._c_availability = np.zeros(shape=self._nplannable,
                                        dtype=object)
        self._c_deadline = np.zeros(shape=self._njobs,
                                    dtype=object)
        self._c_release = np.zeros(shape=self._njobs,
                                   dtype=object)
        self._c_resource = np.zeros(shape=self._njobs,
                                    dtype=object)
        self._c_lower = np.zeros(shape=(self._njobs, self._nplannable),
                                 dtype=object)
        self._c_upper = np.zeros(shape=(self._njobs, self._nplannable),
                                 dtype=object)

    def _set_objective(self, instance):
        """Sets the objective of the LP model. Stores the objective in
        the private `_cost` variable.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        self._cost = self._problem.sum(
            instance['jobs'][j, 5] * self._tvar[j * 2 + 1]
            + instance['jobs'][j, 6]
            for j in range(self._njobs)
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


class JobPropertiesContinuousLPWithSlack(LPWithSlack,
                                         JobPropertiesContinuousLP):
    def __init__(self, instance, label):
        super().__init__(instance, label)

    def update_swap_neighbors(self, instance, first_idx):
        """Update the existing model by swapping two neighboring events
        in the eventlist.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        first_idx : int
            Position of the first event in the event list that will
            switch positions with its successor.
        """
        super().update_swap_neighbors(instance, first_idx)

        # Swap has already been done
        job1 = instance['eventlist'][first_idx + 1, 1]
        type1 = instance['eventlist'][first_idx + 1, 0]
        job2 = instance['eventlist'][first_idx, 1]
        type2 = instance['eventlist'][first_idx, 0]
        e1 = job1 * 2 + type1
        if type1 > 1:
            e1 += self._nplannable - 2 + job1 * (self._kextra - 2)
        e2 = job2 * 2 + type2
        if type2 > 1:
            e2 += self._nplannable - 2 + job2 * (self._kextra - 2)

        # Intervals only need updates if we swap two plannable events.
        e0 = -1
        e3 = -1
        if type1 < 2 and type2 < 2:
            pos = first_idx
            while e0 < 0 and pos > 0:
                pos -= 1
                if instance['eventlist'][pos, 0] < 2:
                    e0 = instance['eventlist'][pos, 1] * 2 + \
                        instance['eventlist'][pos, 0]
            pos = first_idx + 2
            while e3 < 0 and pos < self._nevents:
                if instance['eventlist'][pos, 0] < 2:
                    e3 = instance['eventlist'][pos, 1] * 2 + \
                        instance['eventlist'][pos, 0]
                pos += 1

        # update appropriate self._c_availability
        if e0 > -1:
            self._c_availability[e0].lhs -= self._stvar[e0]
        if type1 < 2 and type2 < 2:
            self._c_availability[e2].lhs -= self._stvar[e2]
        if e3 > -1:
            self._c_availability[e1].lhs -= self._stvar[e1]

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1 and type2 < 2:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval (now completes at the start of
            # e2, i.e. has to be enforced during the preceding interval
            if not isinstance(self._slvar[job1, e2], docplex.mp.dvar.Var):
                self._slvar[job1, e2] = self._problem.continuous_var(
                        name=f"s-_{job1},{e2}",
                        lb=0
                    )
                self._suvar[job1, e2] = self._problem.continuous_var(
                        name=f"s+_{job1},{e2}",
                        lb=0
                    )
                self._cost += instance['constants']['slackpenalties'][1] * \
                    (self._slvar[job1, e2] + self._suvar[job1, e2])
            # 5. Lower bound
            self._c_lower[job1, e2].lhs += self._slvar[job1, e2]
            # 6. Upper bound
            self._c_upper[job1, e2].lhs -= self._suvar[job1, e2]
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for e0 > -1: this is always true
            assert e0 > -1
            self._c_lower[job1, e0].lhs += self._slvar[job1, e0]
            self._c_upper[job1, e0].lhs -= self._suvar[job1, e0]
        elif type1 == 0 and type2 < 2:
            # In addition, the time variables need to be updated in two
            # more constraints.
            assert e3 > -1
            self._c_lower[job1, e1].lhs += self._slvar[job1, e1]
            self._c_upper[job1, e1].lhs -= self._suvar[job1, e1]

        if type2 == 0 and type1 < 2:
            # Earlier start, so bounds need to be enforced for an
            # additional interval
            # No need to check boundary: this always holds
            assert e3 > -1
            if not isinstance(self._slvar[job2, e1], docplex.mp.dvar.Var):
                self._slvar[job2, e1] = self._problem.continuous_var(
                        name=f"s-_{job2},{e1}",
                        lb=0
                    )
                self._suvar[job2, e1] = self._problem.continuous_var(
                        name=f"s+_{job2},{e1}",
                        lb=0
                    )
                self._cost += instance['constants']['slackpenalties'][1] * \
                    (self._slvar[job2, e1] + self._suvar[job2, e1])
            # 5. Lower bound
            self._c_lower[job2, e1].lhs += self._slvar[job2, e1]
            # 6. Upper bound
            self._c_upper[job2, e1].lhs -= self._suvar[job2, e1]
            # In addition, the time variables need to be updated in two
            # more constraints.
            self._c_lower[job2, e2].lhs += self._slvar[job2, e2]
            self._c_upper[job2, e2].lhs -= self._suvar[job2, e2]
        elif type2 == 1 and type1 < 2:
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for e0 > -1: this is always true
            assert e0 > -1
            self._c_lower[job2, e0].lhs += self._slvar[job2, e0]
            self._c_upper[job2, e0].lhs -= self._suvar[job2, e0]

    def _define_variable_arrays(self, instance):
        """Defines the arrays organizing all decision variables in the
        model. Fills the following five private variables: `_pvar`
        (resources variables), `_tvar` (time variables), _`stvar`
        (total resource slack variables), `_suvar` (upperbound slack
        variables) and `_slvar` (lowerbound slack variables.
        """
        super()._define_variable_arrays(instance)
        self._stvar = np.zeros(shape=self._nplannable, dtype=object)
        self._suvar = np.zeros(shape=(self._njobs, self._nplannable),
                               dtype=object)
        self._slvar = np.zeros(shape=(self._njobs, self._nplannable),
                               dtype=object)

        for e in range(self._nplannable):
            self._stvar[e] = self._problem.continuous_var(
                name=f"st_{e}",
                lb=0
            )

        for j in range(self._njobs):
            for i in range(instance['eventmap'][j, 0],
                           instance['eventmap'][j, 1]):
                e = instance['eventlist'][i, 1] * 2 + \
                    instance['eventlist'][i, 0]
                self._suvar[j, e] = self._problem.continuous_var(
                    name=f"s+_{j},{e}",
                    lb=0
                )
                self._slvar[j, e] = self._problem.continuous_var(
                    name=f"s-_{j},{e}",
                    lb=0
                )

    def _set_objective(self, instance):
        """Sets the objective of the LP model. Stores the objective in
        the private `_cost` variable.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        super()._set_objective(instance)

        self._cost += self._problem.sum(
            instance['constants']['slackpenalties'][0] * self._stvar[i] +
            self._problem.sum(
                instance['constants']['slackpenalties'][1] *
                (self._slvar[j, i] + self._suvar[j, i])
                for j in range(self._njobs)
            )
            for i in range(self._nplannable)
        )

    def _add_interval_capacity_constraint(self, i, i2):
        """Adds a constraint to the model that ensures the resource limit
        is respected within the interval.

        Parameters
        ----------
        i : int
            Index of the first event in the variable array `_tvar`.
        i2 : int
            Index of the second event in the variable array `_tvar`.
        """
        cap_c = super()._add_interval_capacity_constraint(i, i2)
        cap_c.lhs -= self._stvar[i]
        return cap_c

    def _add_lower_bound_constraint(self, j, i, i2, lower_bound):
        """Adds a constraint to the model that enforces the lower bound
        of job `j` in the interval between the two events.

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
        lb_c = super()._add_lower_bound_constraint(j, i, i2, lower_bound)
        lb_c.lhs += self._slvar[j, i]
        return lb_c

    def _add_upper_bound_constraint(self, j, i, i2, upper_bound):
        """Adds a constraint to the model that enforces the upper bound
        of job `j` in the interval between the two events.

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
        ub_c = super()._add_upper_bound_constraint(j, i, i2, upper_bound)
        ub_c.lhs -= self._suvar[j, i]
        return ub_c

    def compute_slack(self, slackpenalties):
        """Compute the (summed) value of the slack variables in the
        model.

        Parameters
        ----------
        slackpenalties : list of float
            List of penalty coefficients for slack variables, The first
            position is taken to be the penalty on the resource slack,
            the second on the bound violations.

        Returns
        -------
        list of tuple
            List of tuples with in the first position a string
            identifying the type of slack variable, in second position
            the summed value of these variables (float) and in third
            position the unit weight of these variables in the objective.
        """
        resource_slack = 0
        upper_slack = 0
        lower_slack = 0

        for stvar in self._stvar:
            if isinstance(stvar, docplex.mp.dvar.Var):
                resource_slack += stvar.solution_value

        for uppers in self._suvar:
            for suvar in uppers:
                if isinstance(suvar, docplex.mp.dvar.Var):
                    upper_slack += suvar.solution_value

        for lowers in self._slvar:
            for slvar in lowers:
                if isinstance(slvar, docplex.mp.dvar.Var):
                    lower_slack += slvar.solution_value

        return [("resource", resource_slack, slackpenalties[0]),
                ("upperbound", upper_slack, slackpenalties[1]),
                ("lowerbound", lower_slack, slackpenalties[1])]
