from abc import ABC
from abc import abstractmethod
import bisect
import docplex.mp.model
import math
import numpy as np

from continuousresource.mathematicalprogramming.mipmodels \
    import MIP
from continuousresource.mathematicalprogramming.utils \
    import time_and_resource_vars_to_human_readable_solution_cplex, \
    solution_to_csv_string


class LP(MIP):
    """Super class for all Linear Programming models.

    Parameters
    ----------
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, label):
        super().__init__(label)


class LPWithSlack(LP):
    """Super class for all Linear Programming models that have slack
    variables.

    Parameters
    ----------
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, label):
        super().__init__(label)

    @abstractmethod
    def compute_slack(self):
        """Compute the (summed) value of the slack variables in the
        model.

        Returns
        -------
        list of tuple
            List of tuples with in the first position a string
            identifying the type of slack variable, in second position
            the summed value of these variables (float) and in third
            position the unit weight of these variables in the objective.
        """
        raise NotImplementedError


class OrderBasedSubProblem(LP):
    """Class implementing the subproblem for a decomposition approach
    that determines a schedule, given an event order.

    Parameters
    ----------
    eventlist : ndarray
        Two-dimensional (|E| x 2) array representing the events in the
        problem, where the first column contains an integer indicating
        the event type (0 for start, 1 for completion) and the second
        column the associated job ID.
    jobs : ndarray
        Two-dimensional (n x 7) array containing job properties:
            - 0: resource requirement (E_j);
            - 1: resource lower bound (P^-_j);
            - 2: resource upper bound (P^+_j);
            - 3: release date (r_j);
            - 4: deadline (d_j);
            - 5: weight (W_j);
            - 6: objective constant (B_j).
    resource : float
        Amount of resource available per time unit. Required to be
        constant for now.
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, eventlist, jobs, resource, label):
        super().__init__(label)

        self._event_list = eventlist
        self._event_map = self._construct_event_mapping(eventlist,
                                                        jobs.shape[0])
        self._job_properties = jobs
        self._capacity = resource

    @property
    def event_list(self):
        return self._event_list

    @property
    def event_map(self):
        return self._event_map

    @property
    def job_properties(self):
        return self._job_properties

    @property
    def capacity(self):
        return self._capacity

    @staticmethod
    def _construct_event_mapping(eventlist, no_jobs):
        """Construct a mapping from jobs to events based on the event
        list.

        Parameters
        ----------
        eventlist : ndarray
            Two-dimensional (|E| x 2) array representing the events in
            the problem, where the first column contains an integer
            indicating the event type (0 for start, 1 for completion) and
            the second column the associated job ID.
        no_jobs : int
            Number of jobs in the problem.

        Returns
        -------
        ndarray
            Two-dimensional (n x 2) array containing for every job the
            position of its start and completion time in the eventlist.
        """
        event_map = np.empty(shape=(no_jobs, 2), dtype=int)
        # Not necessary if all jobs have exactly one start and completion
        # time.
        # event_map.fill(-1)
        for i in range(len(eventlist)):
            event_map[eventlist[i][1], eventlist[i][0]] = i
        return event_map

    def generate_initial_solution(self):
        """...
        """
        # Initialize eventlist
        eventlist = []

        # Construct intervals
        # `boundaries` is a list of time points at which the list of jobs
        # available for processing changes. These are defined by triples
        # (t, e, j), where t is the time, e the event type (0: release
        # time passed, i.e. a job is added; 1: deadline passed) and j the
        # job ID.
        boundaries = np.concatenate((
            np.array([
                [self._job_properties[j, 3], 0, j]
                for j in range(len(self._job_properties))
            ]),
            np.array([
                [self._job_properties[j, 4], 1, j]
                for j in range(len(self._job_properties))
            ])
        ))
        boundaries = boundaries[boundaries[:, 0].argsort()]

        # Create the reference table we will be working with.
        # This will be a sorted list (sorted on deadline) of jobs that
        # are currently available for processing. For every job it will
        # contain the following information, in order:
        # - 0: deadline;
        # - 1: upper bound (P^+_j);
        # - 2: residual resource need (E_j);
        # - 3: total resource need (E_j);
        # - 4: job ID;
        curr_jobs = []

        # Loop over intervals
        for i in range(len(boundaries) - 1):
            if boundaries[i, 1] == 0:
                # We passed a release time, so we add a job to our list
                bisect.insort(curr_jobs, [
                    self._job_properties[int(boundaries[i, 2]), 4],  # d_j
                    self._job_properties[int(boundaries[i, 2]), 2],  # P^+_j
                    self._job_properties[int(boundaries[i, 2]), 0],  # E_j
                    self._job_properties[int(boundaries[i, 2]), 0],  # E_j
                    boundaries[i, 2]                                 # ID
                ])

            # Compute time and resource in this interval
            interval_time = boundaries[i + 1, 0] - boundaries[i, 0]
            avail_resource = interval_time * self._capacity

            # Compute lower and upper bounds
            lower_bounds = np.array([
                max(
                    0,
                    job[2] - (job[0] - boundaries[i + 1, 0]) * job[1]
                )
                for job in curr_jobs
            ])
            upper_bounds = np.array([
                min(
                    job[2],
                    # self._capacity * interval_time,
                    job[1] * interval_time
                )
                for job in curr_jobs
            ])

            # First assign all lower bounds
            avail_resource -= np.sum(lower_bounds)
            remove = []
            for j in range(len(curr_jobs)):
                if lower_bounds[j] > 0:
                    if curr_jobs[j][2] == curr_jobs[j][3]:
                        eventlist.append([0, curr_jobs[j][4]])
                    curr_jobs[j][2] -= lower_bounds[j]
                    if curr_jobs[j][2] <= 0:
                        eventlist.append([1, curr_jobs[j][4]])
                        remove.append(j)

            # Now distribute any resources that remain
            if avail_resource > 0:
                upper_bounds -= lower_bounds
                for j in range(len(curr_jobs)):
                    if curr_jobs[j][2] <= 0:
                        continue
                    amount = min(avail_resource, upper_bounds[j])
                    if amount > 0:
                        if curr_jobs[j][2] == curr_jobs[j][3]:
                            eventlist.append([0, curr_jobs[j][4]])
                        curr_jobs[j][2] -= amount
                        if curr_jobs[j][2] <= 0:
                            eventlist.append([1, curr_jobs[j][4]])
                            remove.append(j)
                        avail_resource -= amount
                        if avail_resource <= 0:
                            break

            for j in range(len(remove) - 1, -1, -1):
                del curr_jobs[remove[j]]

        eventlist = np.array(eventlist, dtype=int)
        self._event_list = eventlist
        self._event_map = \
            self._construct_event_mapping(eventlist,
                                          len(self._job_properties))

    def generate_random_solution(self, precs):
        """Generate a random initial solution that respects the
        precedence constraints in `precs`.

        Parameters
        ----------
        precs : ndarray
            Two dimensional (|E| x |E|) array listing (inferred)
            precedence relations between events. If the entry at position
            [i, j] is True, this means that i has to come before j.
        """
        # Initialize eventlist
        eventlist = []

        events = np.random.permutation(len(self._event_list))

        while len(events) > 0:
            for i in range(len(events)):
                if np.sum(precs[events, events[i]]) > 0:
                    continue
                eventlist.append([events[i] % 2, math.floor(events[i] / 2)])
                events = np.delete(events, i)
                break

        eventlist = np.array(eventlist, dtype=int)
        self._event_list = eventlist
        self._event_map = \
            self._construct_event_mapping(eventlist,
                                          len(self._job_properties))

    def initialize_problem(self):
        """...
        """

        # Create variables
        self._times = np.zeros(shape=len(self._event_list), dtype=object)
        self._resource = np.zeros(
            shape=(len(self._job_properties), len(self._event_list)),
            dtype=object
        )

        for e in range(len(self._event_list)):
            self._times[e] = self._problem.continuous_var(
                name=f"t_{e}",
                lb=0
            )

        for j in range(len(self._job_properties)):
            for i in range(self._event_map[j, 0], self._event_map[j, 1]):
                e = self._event_list[i, 1] * 2 + self._event_list[i, 0]
                self._resource[j, e] = self._problem.continuous_var(
                    name=f"p_{j},{e}",
                    lb=0
                )

        # Define objective
        self._cost = self._problem.sum(
            self._job_properties[j, 5] * self._times[2 * j + 1] +
            self._job_properties[j, 6]
            for j in range(len(self._job_properties))
        )
        self._problem.minimize(self._cost)

        # Initialize constraints
        self._c_order = np.zeros(shape=len(self._event_list),
                                 dtype=object)
        self._c_availability = np.zeros(shape=len(self._event_list),
                                        dtype=object)
        self._c_deadline = np.zeros(shape=len(self._job_properties),
                                    dtype=object)
        self._c_release = np.zeros(shape=len(self._job_properties),
                                   dtype=object)
        self._c_resource = np.zeros(shape=len(self._job_properties),
                                    dtype=object)
        self._c_lower = np.zeros(shape=(len(self._job_properties),
                                        len(self._event_list)),
                                 dtype=object)
        self._c_upper = np.zeros(shape=(len(self._job_properties),
                                        len(self._event_list)),
                                 dtype=object)

        for i in range(len(self._event_list) - 1):
            e = self._event_list[i, 1] * 2 + self._event_list[i, 0]
            e1 = self._event_list[i + 1, 1] * 2 + self._event_list[i + 1, 0]
            # 1. Event Order
            self._c_order[e] = self._problem.add_constraint(
                ct=self._times[e] - self._times[e1] <= 0,
                ctname=f"event_order_{e}"
            )
            # 7. Resource availability
            self._c_availability[e] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._resource[j, e] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[e1]
                                      - self._times[e]) <= 0,
                ctname=f"resource_availability_{e}"
            )

        for j in range(len(self._job_properties)):
            # 2. Deadline
            self._c_deadline[j] = self._problem.add_constraint(
                ct=self._times[2 * j + 1]
                - self._job_properties[j, 4] <= 0,
                ctname=f"deadline_{j}"
            )
            # 3. Release date
            self._c_release[j] = self._problem.add_constraint(
                ct=self._times[2 * j]
                - self._job_properties[j, 3] >= 0,
                ctname=f"release_{j}"
            )
            # 4. Resource requirement
            self._c_resource[j] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._resource[
                        j,
                        self._event_list[i, 1] * 2 + self._event_list[i, 0]
                    ] for i in range(
                        self._event_map[j, 0], self._event_map[j, 1]
                    )
                ) - self._job_properties[j, 0] == 0,
                ctname=f"resource_{j}"
            )

            for i in range(self._event_map[j, 0], self._event_map[j, 1]):
                e = self._event_list[i, 1] * 2 + self._event_list[i, 0]
                e1 = self._event_list[i + 1, 1] * 2 \
                    + self._event_list[i + 1, 0]
                # 5. Lower bound
                self._c_lower[j, e] = self._problem.add_constraint(
                    ct=self._resource[j, e] -
                    self._job_properties[j, 1] * (self._times[e1]
                                                  - self._times[e]) >= 0,
                    ctname=f"lower_bound_{j},{e}"
                )
                # 6. Upper bound
                self._c_upper[j, e] = self._problem.add_constraint(
                    ct=self._resource[j, e] -
                    self._job_properties[j, 2] * (self._times[e1]
                                                  - self._times[e]) <= 0,
                    ctname=f"upper_bound_{j},{e}"
                )

    def update_swap_neighbors(self, first_idx):
        """Update the existing model by swapping two neighboring events
        in the eventlist.

        Parameters
        ----------
        first_idx : int
            Position of the first event in the event list that will
            switch positions with its successor.
        """
        job1 = self._event_list[first_idx, 1]
        type1 = self._event_list[first_idx, 0]
        job2 = self._event_list[first_idx + 1, 1]
        type2 = self._event_list[first_idx + 1, 0]
        e1 = job1 * 2 + type1
        e2 = job2 * 2 + type2

        if job1 == job2 and type2 == 1:
            raise RuntimeError("Cannot put a job's completion before its"
                               " start.")

        # update self._event_list
        self._event_list[[first_idx, first_idx + 1], :] = \
            self._event_list[[first_idx + 1, first_idx], :]
        # self._event_list[first_idx] = [type2, job2]
        # self._event_list[first_idx + 1] = [type1, job1]

        # update self._event_map
        self._event_map[job1, type1] += 1
        self._event_map[job2, type2] -= 1

        # update self._resource variables
        if type1 == 1 and not isinstance(self._resource[job1, e2],
                                         docplex.mp.dvar.Var):
            # The completion of job1 happens one interval later, a new
            # variable must be added.
            self._resource[job1, e2] = \
                self._problem.continuous_var(
                    name=f"p_{job1},{e2}",
                    lb=0
                )
        if type2 == 0 and not isinstance(self._resource[job2, e1],
                                         docplex.mp.dvar.Var):
            # The start of job2 happens one interval earlier, a new
            # variable must be added.
            self._resource[job2, e1] = \
                self._problem.continuous_var(
                    name=f"p_{job2},{e1}",
                    lb=0
                )

        # update appropriate self._c_order and self._c_availability
        if first_idx > 0:
            e0 = self._event_list[first_idx - 1, 1] * 2 \
                + self._event_list[first_idx - 1, 0]
            self._c_order[e0].lhs = \
                self._times[e0] - self._times[e2]
            self._c_availability[e0].lhs = \
                self._problem.sum(
                    self._resource[j, e0] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[e2]
                                      - self._times[e0])

        if isinstance(self._c_order[e2], docplex.mp.constr.LinearConstraint):
            self._c_order[e2].lhs = \
                self._times[e2] - self._times[e1]
        else:
            self._c_order[e2] = self._problem.add_constraint(
                ct=self._times[e2] - self._times[e1] <= 0,
                ctname=f"event_order_{e2}"
            )
        if isinstance(self._c_availability[e2],
                      docplex.mp.constr.LinearConstraint):
            self._c_availability[e2].lhs = \
                self._problem.sum(
                    self._resource[j, e2] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[e1]
                                      - self._times[e2])
        else:
            self._c_availability[e2] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._resource[j, e2] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[e1]
                                      - self._times[e2]) <= 0,
                ctname=f"resource_availability_{e2}"
            )

        if first_idx + 2 < len(self._event_list):
            e3 = self._event_list[first_idx + 2, 1] * 2 \
                + self._event_list[first_idx + 2, 0]
            self._c_order[e1].lhs = \
                self._times[e1] - self._times[e3]
            self._c_availability[e1].lhs = \
                self._problem.sum(
                    self._resource[j, e1] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[e3]
                                      - self._times[e1])
        else:
            self._problem.remove_constraints([
                self._c_order[e1],
                self._c_availability[e1]
            ])
            self._c_order[e1] = None
            self._c_availability[e1] = None

        # update self._times positions
        # self._times[[first_idx, first_idx + 1]] = \
        #     self._times[[first_idx + 1, first_idx]]

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval (now completes at the start of
            # first_index + 1, i.e. has to be enforced during the
            # preceding interval
            # 5. Lower bound
            self._c_lower[job1, e2] = self._problem.add_constraint(
                ct=self._resource[job1, e2] -
                self._job_properties[job1, 1] * (self._times[e1]
                                                 - self._times[e2])
                >= 0,
                ctname=f"lower_bound_{job1},{e2}"
            )
            # 6. Upper bound
            self._c_upper[job1, e2] = self._problem.add_constraint(
                ct=self._resource[job1, e2] -
                self._job_properties[job1, 2] * (self._times[e1]
                                                 - self._times[e2])
                <= 0,
                ctname=f"upper_bound_{job1},{e2}"
            )
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            e0 = self._event_list[first_idx - 1, 1] * 2 \
                + self._event_list[first_idx - 1, 0]
            self._c_lower[job1, e0].lhs = \
                self._resource[job1, e0] - \
                self._job_properties[job1, 1] * (self._times[e2]
                                                 - self._times[e0])
            self._c_upper[job1, e0].lhs = \
                self._resource[job1, e0] - \
                self._job_properties[job1, 2] * (self._times[e2]
                                                 - self._times[e0])
        elif type1 == 0:
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
            if first_idx + 2 < len(self._event_list):
                e3 = self._event_list[first_idx + 2, 1] * 2 \
                    + self._event_list[first_idx + 2, 0]
                self._c_lower[job1, e1].lhs = \
                    self._resource[job1, e1] - \
                    self._job_properties[job1, 1] * \
                    (self._times[e3] - self._times[e1])
                self._c_upper[job1, e1].lhs = \
                    self._resource[job1, e1] - \
                    self._job_properties[job1, 2] * \
                    (self._times[e3] - self._times[e1])
        else:
            raise ValueError(f"Type code not recognized: {type1}")

        if type2 == 0:
            # Earlier start, so bounds need to be enforced for an
            # additional interval
            # No need to check boundary: this always holds
            assert first_idx + 2 < len(self._event_list)
            e3 = self._event_list[first_idx + 2, 1] * 2 \
                + self._event_list[first_idx + 2, 0]
            # 5. Lower bound
            self._c_lower[job2, e1] = self._problem.add_constraint(
                ct=self._resource[job2, e1] -
                self._job_properties[job2, 1] * (self._times[e3]
                                                 - self._times[e1])
                >= 0,
                ctname=f"lower_bound_{job2},{e1}"
            )
            # 6. Upper bound
            self._c_upper[job2, e1] = self._problem.add_constraint(
                ct=self._resource[job2, e1] -
                self._job_properties[job2, 2] * (self._times[e3]
                                                 - self._times[e1])
                <= 0,
                ctname=f"upper_bound_{job2},{e1}"
            )
            # In addition, the time variables need to be updated in two
            # more constraints.
            self._c_lower[job2, e2].lhs = \
                self._resource[job2, e2] - \
                self._job_properties[job2, 1] * \
                (self._times[e1] - self._times[e2])
            self._c_upper[job2, e2].lhs = \
                self._resource[job2, e2] - \
                self._job_properties[job2, 2] * \
                (self._times[e1] - self._times[e2])
        elif type2 == 1:
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
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            e0 = self._event_list[first_idx - 1, 1] * 2 \
                + self._event_list[first_idx - 1, 0]
            self._c_lower[job2, e0].lhs = \
                self._resource[job2, e0] - \
                self._job_properties[job2, 1] * (self._times[e2]
                                                 - self._times[e0])
            self._c_upper[job2, e0].lhs = \
                self._resource[job2, e0] - \
                self._job_properties[job2, 2] * (self._times[e2]
                                                 - self._times[e0])
        else:
            raise ValueError(f"Type code not recognized: {type2}")

        # Update bounds for all other jobs that are active
        for j in range(len(self._job_properties)):
            if self._event_map[j, 0] < first_idx \
               and self._event_map[j, 1] > first_idx + 1:
                e0 = self._event_list[first_idx - 1, 1] * 2 \
                    + self._event_list[first_idx - 1, 0]
                e3 = self._event_list[first_idx + 2, 1] * 2 \
                    + self._event_list[first_idx + 2, 0]
                self._c_lower[j, e0].lhs = \
                    self._resource[j, e0] - \
                    self._job_properties[j, 1] * (self._times[e2]
                                                  - self._times[e0])
                self._c_lower[j, e2].lhs = \
                    self._resource[j, e2] - \
                    self._job_properties[j, 1] * (self._times[e1]
                                                  - self._times[e2])
                self._c_lower[j, e1].lhs = \
                    self._resource[j, e1] - \
                    self._job_properties[j, 1] * (self._times[e3]
                                                  - self._times[e1])
                self._c_upper[j, e0].lhs = \
                    self._resource[j, e0] - \
                    self._job_properties[j, 2] * (self._times[e2]
                                                  - self._times[e0])
                self._c_upper[j, e2].lhs = \
                    self._resource[j, e2] - \
                    self._job_properties[j, 2] * (self._times[e1]
                                                  - self._times[e2])
                self._c_upper[j, e1].lhs = \
                    self._resource[j, e1] - \
                    self._job_properties[j, 2] * (self._times[e3]
                                                  - self._times[e1])

        # update appropriate self._c_resource
        for j in [job1, job2]:
            self._c_resource[j].lhs = self._problem.sum(
                self._resource[
                    j,
                    self._event_list[i, 1] * 2 + self._event_list[i, 0]
                ] for i in range(
                    self._event_map[j, 0], self._event_map[j, 1]
                )
            ) - self._job_properties[j, 0]

    def update_move_event(self, orig_idx, new_idx):
        """Update the existing model by moving an event to a different
        position in the event list.

        Parameters
        ----------
        orig_idx : int
            Original position of the event in the event list.
        new_idx : int
            Position that the event will be moved to.
        """
        # Interface method
        self.update_move_event_stepwise(orig_idx, new_idx)

    def update_move_event_stepwise(self, orig_idx, new_idx):
        """Update the existing model by moving an event to a different
        position in the event list.

        Parameters
        ----------
        orig_idx : int
            Original position of the event in the event list.
        new_idx : int
            Position that the event will be moved to.
        """
        # Implemented as a sequence of swaps

        # Determine direction of movement
        if orig_idx < new_idx:
            while orig_idx < new_idx:
                self.update_swap_neighbors(orig_idx)
                orig_idx += 1
        elif orig_idx > new_idx:
            while orig_idx > new_idx:
                self.update_swap_neighbors(orig_idx - 1)
                orig_idx -= 1

    def update_move_event_once(self, orig_idx, new_idx):
        """Update the existing model by moving an event to a different
        position in the event list.

        Parameters
        ----------
        orig_idx : int
            Original position of the event in the event list.
        new_idx : int
            Position that the event will be moved to.
        """
        raise NotImplementedError

    def update_move_pair(self, idx1, idx2, offset):
        """Update the existing model by shifting a pair of events through
        the event list.

        Parameters
        ----------
        idx1 : int
            Original position of the first event in the event list.
        idx2 : int
            Original position of the second event in the event list.
        offset : int
            Number that the position of the events will be offset by.
        """
        # Interface method
        self.update_move_pair_stepwise(idx1, idx2, offset)

    def update_move_pair_stepwise(self, idx1, idx2, offset):
        """Update the existing model by shifting a pair of events through
        the event list.

        Parameters
        ----------
        idx1 : int
            Original position of the first event in the event list.
        idx2 : int
            Original position of the second event in the event list.
        offset : int
            Number that the position of the events will be offset by.
        """
        # Implemented as two moves

        # We do a sign check to determine the direction of movement.
        # This makes sure we avoid the situation where we move the first
        # job over the second and than moving the second over the first,
        # messing up the offset.
        if offset == 0:
            return
        if idx1 < idx2:
            if offset < 0:
                first = idx1
                second = idx2
            elif offset > 0:
                first = idx2
                second = idx1
        elif idx1 > idx2:
            if offset < 0:
                first = idx2
                second = idx1
            elif offset > 0:
                first = idx1
                second = idx2

        # Move the events to offset position
        self.update_move_event(first, first + offset)
        self.update_move_event(second, second + offset)

    def update_move_pair_once(self, idx1, idx2, offset):
        """Update the existing model by shifting a pair of events through
        the event list.

        Parameters
        ----------
        idx1 : int
            Original position of the first event in the event list.
        idx2 : int
            Original position of the second event in the event list.
        offset : int
            Number that the position of the events will be offset by.
        """
        raise NotImplementedError

    def get_schedule(self):
        """Return the schedule that corresponds to the current solution.

        Returns
        -------
        ndarray
            One-dimensional (|E| x 3) array containing the assigned time
            in the schedule for all events, in the order of the
            eventlist.
        """
        # The schedule as intended below would mix integer and floats
        # schedule = np.zeros(shape=(len(self._event_list), 3)
        # schedule[:,:2] = event_list
        # time_assignment = np.zeros(shape=len(self._event_list))
        return np.array([t.solution_value for t in self._times])

    def get_solution_csv(self):
        (event_labels, event_idx, event_timing, resource_consumption) = \
            time_and_resource_vars_to_human_readable_solution_cplex(
                self._times, self._resource,
            )
        return solution_to_csv_string(event_labels, event_idx, event_timing,
                                      resource_consumption)

    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        eventorder_str = ''
        timings_str = ''
        resource_str = ''

        for event in self._event_list:
            if event[0] <= 0:
                eventorder_str += 'S_'
            else:
                eventorder_str += 'C_'
            eventorder_str += f'{event[1]}; '

        for timing in self._times:
            if isinstance(timing, docplex.mp.dvar.Var):
                timings_str += f'{timing.name}: {timing.solution_value}; '

        for resource_row in self._resource:
            for resource in resource_row:
                if isinstance(resource, docplex.mp.dvar.Var):
                    timings_str += (f'{resource.name}:'
                                    f' {resource.solution_value}; ')
        print(eventorder_str)
        print(timings_str)
        print(resource_str)

    def find_precedences(self, infer_precedence):
        """Construct an array indicating precedence relations between
        events.

        Parameters
        ----------
        infer_precedence : bool
            Flag indicating whether to infer and continuously check
            (implicit) precedence relations.

        Returns
        -------
        ndarray
            Two dimensional (|E| x |E|) array listing (inferred)
            precedence relations between events. If the entry at position
            [i, j] is True, this means that i has to come before j.
        """
        if not infer_precedence:
            # Start out with an array filled with only the precedence
            # relations that exist between start/completion events.
            return np.array([
                [
                    (i % 2 == 0 and j - i == 1)
                    for j in range(len(self._event_list))
                ]
                for i in range(len(self._event_list))
            ], dtype=bool)

        # For all events we can infer an earliest and latest time
        inferred_limits = np.array([
            e
            for j in range(len(self._job_properties))
            for e in (
                [self._job_properties[j, 3],
                 self._job_properties[j, 4] - (self._job_properties[j, 0] /
                 self._job_properties[j, 2])],
                [self._job_properties[j, 3] + (self._job_properties[j, 0] /
                 self._job_properties[j, 2]),
                 self._job_properties[j, 4]]
            )
        ], dtype=float)

        # Now, any event whose latest possible time is smaller than the
        # earliest possible time of another event, has to come before it.
        return np.array([
            [
                ((i % 2 == 0 and j - i == 1) or (inferred_limits[i, 1] <=
                                                 inferred_limits[j, 0])) and
                i != j  # If this happens, the instance is infeasible
                        # anyway, but we don't want it to crash. Figure
                        # out a way to deal with this better... TODO
                for j in range(len(self._event_list))
            ]
            for i in range(len(self._event_list))
        ], dtype=bool)


class OrderBasedSubProblemWithSlack(OrderBasedSubProblem, LPWithSlack):
    """Class implementing the subproblem for a decomposition approach
    that determines a schedule, given an event order.

    Parameters
    ----------
    eventlist : ndarray
        Two-dimensional (|E| x 2) array representing the events in the
        problem, where the first column contains an integer indicating
        the event type (0 for start, 1 for completion) and the second
        column the associated job ID.
    jobs : ndarray
        Two-dimensional (n x 7) array containing job properties:
            - 0: resource requirement (E_j);
            - 1: resource lower bound (P^-_j);
            - 2: resource upper bound (P^+_j);
            - 3: release date (r_j);
            - 4: deadline (d_j);
            - 5: weight (W_j);
            - 6: objective constant (B_j).
    resource : float
        Amount of resource available per time unit. Required to be
        constant for now.
    slackpenalties : list of float
        List of penalty terms for the different forms of slack in the
        model. In order:
            - 0: penalty for every unit of resource above capacity during
              an interval;
            - 1: penalty for every unit of resource above or below the
              upper or lower bound of a job during an interval.
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, eventlist, jobs, resource, slackpenalties, label):
        self._penalty_capacity = slackpenalties[0]
        self._penalty_bounds = slackpenalties[1]

        super().__init__(eventlist, jobs, resource, label)

    def initialize_problem(self):
        super().initialize_problem()

        # Create slack variables
        self._slack_resource = np.zeros(shape=len(self._event_list),
                                        dtype=object)
        self._slack_upperbound = np.zeros(
            shape=(len(self._job_properties), len(self._event_list)),
            dtype=object
        )
        self._slack_lowerbound = np.zeros(
            shape=(len(self._job_properties), len(self._event_list)),
            dtype=object
        )

        for e in range(len(self._event_list)):
            self._slack_resource[e] = self._problem.continuous_var(
                name=f"st_{e}",
                lb=0
            )

        for j in range(len(self._job_properties)):
            for i in range(self._event_map[j, 0], self._event_map[j, 1]):
                e = self._event_list[i, 1] * 2 + self._event_list[i, 0]
                self._slack_upperbound[j, e] = self._problem.continuous_var(
                    name=f"s+_{j},{e}",
                    lb=0
                )
                self._slack_lowerbound[j, e] = self._problem.continuous_var(
                    name=f"s-_{j},{e}",
                    lb=0
                )

        # Add slack term to objective
        self._cost += self._problem.sum(
            self._penalty_capacity * self._slack_resource[i] +
            self._problem.sum(
                self._penalty_bounds *
                (self._slack_lowerbound[j, i] + self._slack_upperbound[j, i])
                for j in range(len(self._job_properties))
            )
            for i in range(len(self._event_list))
        )
        # self._problem.objective_expr = self._cost

        # Add slack variables to appropriate constraints
        for i in range(len(self._event_list) - 1):
            e = self._event_list[i, 1] * 2 + self._event_list[i, 0]
            # 7. Resource availability
            self._c_availability[e].lhs -= self._slack_resource[e]

        for j in range(len(self._job_properties)):
            for i in range(self._event_map[j, 0], self._event_map[j, 1]):
                e = self._event_list[i, 1] * 2 + self._event_list[i, 0]
                # 5. Lower bound
                self._c_lower[j, e].lhs += self._slack_lowerbound[j, e]
                # 6. Upper bound
                self._c_upper[j, e].lhs -= self._slack_upperbound[j, e]

    def update_swap_neighbors(self, first_idx):
        super().update_swap_neighbors(first_idx)
        job2 = self._event_list[first_idx, 1]
        type2 = self._event_list[first_idx, 0]
        job1 = self._event_list[first_idx + 1, 1]
        type1 = self._event_list[first_idx + 1, 0]
        e1 = job1 * 2 + type1
        e2 = job2 * 2 + type2

        # update appropriate self._c_availability
        if first_idx > 0:
            e0 = self._event_list[first_idx - 1, 1] * 2 \
                + self._event_list[first_idx - 1, 0]
            self._c_availability[e0].lhs -= \
                self._slack_resource[e0]
        self._c_availability[e2].lhs -= self._slack_resource[e2]
        if first_idx + 2 < len(self._event_list):
            self._c_availability[e1].lhs -= \
                self._slack_resource[e1]

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval (now completes at the start of
            # first_index + 1, i.e. has to be enforced during the
            # preceding interval
            if not isinstance(self._slack_lowerbound[job1, e2],
                              docplex.mp.dvar.Var):
                self._slack_lowerbound[job1, e2] = \
                    self._problem.continuous_var(
                        name=f"s-_{job1},{e2}",
                        lb=0
                    )
                self._slack_upperbound[job1, e2] = \
                    self._problem.continuous_var(
                        name=f"s+_{job1},{e2}",
                        lb=0
                    )
                self._cost += self._penalty_bounds * \
                    (self._slack_lowerbound[job1, e2] +
                     self._slack_upperbound[job1, e2])
            # 5. Lower bound
            self._c_lower[job1, e2].lhs += \
                self._slack_lowerbound[job1, e2]
            # 6. Upper bound
            self._c_upper[job1, e2].lhs -= \
                self._slack_upperbound[job1, e2]
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            e0 = self._event_list[first_idx - 1, 1] * 2 \
                + self._event_list[first_idx - 1, 0]
            self._c_lower[job1, e0].lhs += \
                self._slack_lowerbound[job1, e0]
            self._c_upper[job1, e0].lhs -= \
                self._slack_upperbound[job1, e0]
        elif type1 == 0:
            # In addition, the time variables need to be updated in two
            # more constraints.
            if first_idx + 2 < len(self._event_list):
                self._c_lower[job1, e1].lhs += \
                    self._slack_lowerbound[job1, e1]
                self._c_upper[job1, e1].lhs -= \
                    self._slack_upperbound[job1, e1]
        else:
            raise ValueError(f"Type code not recognized: {type1}")

        if type2 == 0:
            # Earlier start, so bounds need to be enforced for an
            # additional interval
            if not isinstance(self._slack_lowerbound[job2, e1],
                              docplex.mp.dvar.Var):
                self._slack_lowerbound[job2, e1] = \
                    self._problem.continuous_var(
                        name=f"s-_{job2},{e1}",
                        lb=0
                    )
                self._slack_upperbound[job2, e1] = \
                    self._problem.continuous_var(
                        name=f"s+_{job2},{e1}",
                        lb=0
                    )
                self._cost += self._penalty_bounds * \
                    (self._slack_lowerbound[job2, e1] +
                     self._slack_upperbound[job2, e1])
            # 5. Lower bound
            self._c_lower[job2, e1].lhs += \
                self._slack_lowerbound[job2, e1]
            # 6. Upper bound
            self._c_upper[job2, e1].lhs -= \
                self._slack_upperbound[job2, e1]
            # In addition, the time variables need to be updated in two
            # more constraints.
            self._c_lower[job2, e2].lhs += \
                self._slack_lowerbound[job2, e2]
            self._c_upper[job2, e2].lhs -= \
                self._slack_upperbound[job2, e2]
        elif type2 == 1:
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            e0 = self._event_list[first_idx - 1, 1] * 2 \
                + self._event_list[first_idx - 1, 0]
            self._c_lower[job2, e0].lhs += \
                self._slack_lowerbound[job2, e0]
            self._c_upper[job2, e0].lhs -= \
                self._slack_upperbound[job2, e0]
        else:
            raise ValueError(f"Type code not recognized: {type2}")

    def compute_slack(self):
        """Compute the (summed) value of the slack variables in the
        model.

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

        for r_var in self._slack_resource:
            if isinstance(r_var, docplex.mp.dvar.Var):
                resource_slack += r_var.solution_value

        for uppers in self._slack_upperbound:
            for u_var in uppers:
                if isinstance(u_var, docplex.mp.dvar.Var):
                    upper_slack += u_var.solution_value

        for lowers in self._slack_lowerbound:
            for l_var in lowers:
                if isinstance(l_var, docplex.mp.dvar.Var):
                    lower_slack += l_var.solution_value

        return [("resource", resource_slack, self._penalty_capacity),
                ("upperbound", upper_slack, self._penalty_bounds),
                ("lowerbound", lower_slack, self._penalty_bounds)]

    def print_solution(self):
        super().print_solution()
        slack_str = ''
        for r_var in self._slack_resource:
            if isinstance(r_var, docplex.mp.dvar.Var):
                slack_str += f'{r_var.name}: {r_var.solution_value}; '

        for uppers in self._slack_upperbound:
            for u_var in uppers:
                if isinstance(u_var, docplex.mp.dvar.Var):
                    slack_str += f'{u_var.name}: {u_var.solution_value}; '

        for lowers in self._slack_lowerbound:
            for l_var in lowers:
                if isinstance(l_var, docplex.mp.dvar.Var):
                    slack_str += f'{l_var.name}: {l_var.solution_value}; '
        print(slack_str)
