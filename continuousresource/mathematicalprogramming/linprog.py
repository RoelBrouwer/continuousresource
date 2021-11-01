from abc import ABC
from abc import abstractmethod
import bisect
import docplex.mp.model
import numpy as np


class LP(ABC):
    """Super class for all Linear Programming models.

    Parameters
    ----------
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    solver : {'cplex'}
        The solver that will be used to solve the LP constructed from the
        provided instance. Currently only CPLEX is supported.
    """
    def __init__(self, label, solver='cplex'):
        self._solver = solver
        self._with_slack = False

        if solver == 'cplex':
            self._problem = docplex.mp.model.Model(name=label)
        else:
            raise ValueError(f"Solver type {solver} is not supported.")
            # self._problem = pulp.LpProblem(label, pulp.LpMaximize)

    @property
    def problem(self):
        return self._problem

    @property
    def solver(self):
        return self._solver

    @property
    def with_slack(self):
        return self._with_slack

    def solve(self):
        """Solve the LP."""
        # self._problem.export_as_lp(os.getcwd())
        if self._solver == 'cplex':
            # print(self._problem.lp_string)
            return self._problem.solve()  # log_output=True)
            # self._problem.print_solution()
        else:
            raise NotImplementedError

    @abstractmethod
    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        pass

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
        raise NotImplementedError("The model does not contain slack"
                                  " variables")


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
    solver : {'cplex'}
        The solver that will be used to solve the LP constructed from the
        provided instance. Currently only CPLEX is supported.
    """
    def __init__(self, eventlist, jobs, resource, label, solver='cplex'):
        super().__init__(label, solver)

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
            for e in range(self._event_map[j, 0], self._event_map[j, 1]):
                self._resource[j, e] = self._problem.continuous_var(
                    name=f"p_{j},{e}",
                    lb=0
                )

        # Define objective
        self._cost = self._problem.sum(
            self._job_properties[j, 5] * self._times[self._event_map[j, 1]] +
            self._job_properties[j, 6]
            for j in range(len(self._job_properties))
        )
        self._problem.minimize(self._cost)

        # Initialize constraints
        self._c_order = np.zeros(shape=len(self._event_list) - 1,
                                 dtype=object)
        self._c_availability = np.zeros(shape=len(self._event_list) - 1,
                                        dtype=object)
        self._c_deadline = np.zeros(shape=len(self._job_properties),
                                    dtype=object)
        self._c_release = np.zeros(shape=len(self._job_properties),
                                   dtype=object)
        self._c_resource = np.zeros(shape=len(self._job_properties),
                                    dtype=object)
        self._c_lower = np.zeros(shape=(len(self._job_properties),
                                        len(self._event_list) - 1),
                                 dtype=object)
        self._c_upper = np.zeros(shape=(len(self._job_properties),
                                        len(self._event_list) - 1),
                                 dtype=object)

        for e in range(len(self._event_list) - 1):
            # 1. Event Order
            self._c_order[e] = self._problem.add_constraint(
                ct=self._times[e] - self._times[e + 1] <= 0,
                ctname=f"event_order_{e}"
            )
            # 7. Resource availability
            self._c_availability[e] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._resource[j, e] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[e + 1]
                                      - self._times[e]) <= 0,
                ctname=f"resource_availability_{e}"
            )

        for j in range(len(self._job_properties)):
            # 2. Deadline
            self._c_deadline[j] = self._problem.add_constraint(
                ct=self._times[self._event_map[j, 1]]
                - self._job_properties[j, 4] <= 0,
                ctname=f"deadline_{j}"
            )
            # 3. Release date
            self._c_release[j] = self._problem.add_constraint(
                ct=self._times[self._event_map[j, 0]]
                - self._job_properties[j, 3] >= 0,
                ctname=f"release_{j}"
            )
            # 4. Resource requirement
            self._c_resource[j] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._resource[j, e] for e in range(
                        self._event_map[j, 0], self._event_map[j, 1]
                    )
                ) - self._job_properties[j, 0] == 0,
                ctname=f"resource_{j}"
            )

            for e in range(self._event_map[j, 0], self._event_map[j, 1]):
                # 5. Lower bound
                self._c_lower[j, e] = self._problem.add_constraint(
                    ct=self._resource[j, e] -
                    self._job_properties[j, 1] * (self._times[e + 1]
                                                  - self._times[e]) >= 0,
                    ctname=f"lower_bound_{j},{e}"
                )
                # 6. Upper bound
                self._c_upper[j, e] = self._problem.add_constraint(
                    ct=self._resource[j, e] -
                    self._job_properties[j, 2] * (self._times[e + 1]
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

        if job1 == job2 and type1 == 1:
            raise RuntimeError("Cannot put a job's completion before its"
                               " start.")

        # update self._event_list
        self._event_list[[first_idx, first_idx + 1], :] = \
            self._event_list[[first_idx + 1, first_idx], :]

        # update self._event_map
        self._event_map[job1, type1] += 1
        self._event_map[job2, type2] -= 1

        # update self._resource variables
        if type1 == 1 and not isinstance(self._resource[job1, first_idx],
                                         docplex.mp.dvar.Var):
            # The completion of job1 happens one interval later, a new
            # variable must be added.
            self._resource[job1, first_idx] = \
                self._problem.continuous_var(
                    name=f"p_{job1},{first_idx}",
                    lb=0
                )
        if type2 == 0 and not isinstance(self._resource[job2, first_idx],
                                         docplex.mp.dvar.Var):
            # The start of job2 happens one interval earlier, a new
            # variable must be added.
            self._resource[job2, first_idx] = \
                self._problem.continuous_var(
                    name=f"p_{job2},{first_idx}",
                    lb=0
                )

        # update appropriate self._c_order and self._c_availability
        if first_idx > 0:
            self._c_order[first_idx - 1].lhs = \
                self._times[first_idx - 1] - self._times[first_idx + 1]
            self._c_availability[first_idx - 1].lhs = \
                self._problem.sum(
                    self._resource[j, first_idx - 1] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[first_idx + 1]
                                      - self._times[first_idx - 1])
        self._c_order[first_idx].lhs = \
            self._times[first_idx + 1] - self._times[first_idx]
        self._c_availability[first_idx].lhs = \
            self._problem.sum(
                self._resource[j, first_idx] for j in range(
                    len(self._job_properties)
                )
            ) - self._capacity * (self._times[first_idx]
                                  - self._times[first_idx + 1])
        if first_idx + 2 < len(self._event_list):
            self._c_order[first_idx + 1].lhs = \
                self._times[first_idx] - self._times[first_idx + 2]
            self._c_availability[first_idx + 1].lhs = \
                self._problem.sum(
                    self._resource[j, first_idx + 1] for j in range(
                        len(self._job_properties)
                    )
                ) - self._capacity * (self._times[first_idx + 2]
                                      - self._times[first_idx])

        # update self._times positions
        self._times[[first_idx, first_idx + 1]] = \
            self._times[[first_idx + 1, first_idx]]

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval (now completes at the start of
            # first_index + 1, i.e. has to be enforced during the
            # preceding interval
            # 5. Lower bound
            self._c_lower[job1, first_idx] = self._problem.add_constraint(
                ct=self._resource[job1, first_idx] -
                self._job_properties[job1, 1] * (self._times[first_idx + 1]
                                                 - self._times[first_idx])
                >= 0,
                ctname=f"lower_bound_{job1},{first_idx}"
            )
            # 6. Upper bound
            self._c_upper[job1, first_idx] = self._problem.add_constraint(
                ct=self._resource[job1, first_idx] -
                self._job_properties[job1, 2] * (self._times[first_idx + 1]
                                                 - self._times[first_idx])
                <= 0,
                ctname=f"upper_bound_{job1},{first_idx}"
            )
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            self._c_lower[job1, first_idx - 1].lhs = \
                self._resource[job1, first_idx - 1] - \
                self._job_properties[job1, 1] * (self._times[first_idx]
                                                 - self._times[first_idx - 1])
            self._c_upper[job1, first_idx - 1].lhs = \
                self._resource[job1, first_idx - 1] - \
                self._job_properties[job1, 2] * (self._times[first_idx]
                                                 - self._times[first_idx - 1])
        elif type1 == 0:
            # Delayed start, so bounds need to be enforced for one fewer
            # interval
            self._problem.remove_constraints([
                self._c_lower[job1, first_idx],
                self._c_upper[job1, first_idx]
            ])
            self._c_lower[job1, first_idx] = None
            self._c_upper[job1, first_idx] = None
            # In addition, the time variables need to be updated in two
            # more constraints.
            if first_idx + 2 < len(self._event_list):
                self._c_lower[job1, first_idx + 1].lhs = \
                    self._resource[job1, first_idx + 1] - \
                    self._job_properties[job1, 1] * \
                    (self._times[first_idx + 2] - self._times[first_idx + 1])
                self._c_upper[job1, first_idx + 1].lhs = \
                    self._resource[job1, first_idx + 1] - \
                    self._job_properties[job1, 2] * \
                    (self._times[first_idx + 2] - self._times[first_idx + 1])
        else:
            raise ValueError(f"Type code not recognized: {type1}")

        if type2 == 0:
            # Earlier start, so bounds need to be enforced for an
            # additional interval
            # 5. Lower bound
            self._c_lower[job2, first_idx] = self._problem.add_constraint(
                ct=self._resource[job2, first_idx] -
                self._job_properties[job2, 1] * (self._times[first_idx + 1]
                                                 - self._times[first_idx])
                >= 0,
                ctname=f"lower_bound_{job2},{first_idx}"
            )
            # 6. Upper bound
            self._c_upper[job2, first_idx] = self._problem.add_constraint(
                ct=self._resource[job2, first_idx] -
                self._job_properties[job2, 2] * (self._times[first_idx + 1]
                                                 - self._times[first_idx])
                <= 0,
                ctname=f"upper_bound_{job2},{first_idx}"
            )
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check boundary: this always holds
            assert first_idx + 2 < len(self._event_list)
            self._c_lower[job2, first_idx + 1].lhs = \
                self._resource[job2, first_idx + 1] - \
                self._job_properties[job2, 1] * \
                (self._times[first_idx + 2] - self._times[first_idx + 1])
            self._c_upper[job2, first_idx + 1].lhs = \
                self._resource[job2, first_idx + 1] - \
                self._job_properties[job2, 2] * \
                (self._times[first_idx + 2] - self._times[first_idx + 1])
        elif type2 == 1:
            # Earlier completion, so bounds need to be enforced for one
            # fewer interval (and it is not enforced on the last
            # interval anyway, because it completes at the start of this
            # one).
            if job1 != job2:
                self._problem.remove_constraints([
                    self._c_lower[job2, first_idx],
                    self._c_upper[job2, first_idx]
                ])
                self._c_lower[job2, first_idx] = None
                self._c_upper[job2, first_idx] = None
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            self._c_lower[job2, first_idx - 1].lhs = \
                self._resource[job2, first_idx - 1] - \
                self._job_properties[job2, 1] * (self._times[first_idx]
                                                 - self._times[first_idx - 1])
            self._c_upper[job2, first_idx - 1].lhs = \
                self._resource[job2, first_idx - 1] - \
                self._job_properties[job2, 2] * (self._times[first_idx]
                                                 - self._times[first_idx - 1])
        else:
            raise ValueError(f"Type code not recognized: {type2}")

        # update appropriate self._c_resource
        for j in [job1, job2]:
            self._c_resource[j].lhs = self._problem.sum(
                self._resource[j, e] for e in range(
                    self._event_map[j, 0], self._event_map[j, 1]
                )
            ) - self._job_properties[j, 0]

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

    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        pass


class OrderBasedSubProblemWithSlack(OrderBasedSubProblem):
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
    solver : {'cplex'}
        The solver that will be used to solve the LP constructed from the
        provided instance. Currently only CPLEX is supported.
    """
    def __init__(self, eventlist, jobs, resource, slackpenalties, label,
                 solver='cplex'):
        self._penalty_capacity = slackpenalties[0]
        self._penalty_bounds = slackpenalties[1]

        super().__init__(eventlist, jobs, resource, label, solver)
        self._with_slack = True

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
            for e in range(self._event_map[j, 0], self._event_map[j, 1]):
                self._slack_upperbound[j, e] = self._problem.continuous_var(
                    name=f"s+_{j},{e}",
                    lb=0
                )
                self._slack_lowerbound[j, e] = self._problem.continuous_var(
                    name=f"s-_{j},{e}",
                    lb=0
                )

        # Add slack term to objective
        # TODO: test if this actually works...
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
        # TODO: test if this actually works...
        for e in range(len(self._event_list) - 1):
            # 7. Resource availability
            self._c_availability[e].lhs -= self._slack_resource[e]

        for j in range(len(self._job_properties)):
            for e in range(self._event_map[j, 0], self._event_map[j, 1]):
                # 5. Lower bound
                self._c_lower[j, e].lhs += self._slack_lowerbound[j, e]
                # 6. Upper bound
                self._c_upper[j, e].lhs -= self._slack_upperbound[j, e]

    def update_swap_neighbors(self, first_idx):
        # TODO: test if this actually works...
        super().update_swap_neighbors(first_idx)
        job2 = self._event_list[first_idx, 1]
        type2 = self._event_list[first_idx, 0]
        job1 = self._event_list[first_idx + 1, 1]
        type1 = self._event_list[first_idx + 1, 0]

        # update appropriate self._c_order and self._c_availability
        if first_idx > 0:
            self._c_availability[first_idx - 1].lhs -= \
                self._slack_resource[first_idx - 1]
        self._c_availability[first_idx].lhs -= self._slack_resource[first_idx]
        if first_idx + 2 < len(self._event_list):
            self._c_availability[first_idx + 1].lhs -= \
                self._slack_resource[first_idx + 1]

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval (now completes at the start of
            # first_index + 1, i.e. has to be enforced during the
            # preceding interval
            if not isinstance(self._slack_lowerbound[job1, first_idx],
                              docplex.mp.dvar.Var):
                self._slack_lowerbound[job1, first_idx] = \
                    self._problem.continuous_var(
                        name=f"s-_{job1},{first_idx}",
                        lb=0
                    )
                self._slack_upperbound[job1, first_idx] = \
                    self._problem.continuous_var(
                        name=f"s+_{job1},{first_idx}",
                        lb=0
                    )
                self._cost += self._penalty_bounds * \
                    (self._slack_lowerbound[job1, first_idx] +
                     self._slack_upperbound[job1, first_idx])
            # 5. Lower bound
            self._c_lower[job1, first_idx].lhs += \
                self._slack_lowerbound[job1, first_idx]
            # 6. Upper bound
            self._c_upper[job1, first_idx].lhs -= \
                self._slack_upperbound[job1, first_idx]
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            self._c_lower[job1, first_idx - 1].lhs += \
                self._slack_lowerbound[job1, first_idx - 1]
            self._c_upper[job1, first_idx - 1].lhs -= \
                self._slack_upperbound[job1, first_idx - 1]
        elif type1 == 0:
            # In addition, the time variables need to be updated in two
            # more constraints.
            if first_idx + 2 < len(self._event_list):
                self._c_lower[job1, first_idx + 1].lhs += \
                    self._slack_lowerbound[job1, first_idx + 1]
                self._c_upper[job1, first_idx + 1].lhs -= \
                    self._slack_upperbound[job1, first_idx + 1]
        else:
            raise ValueError(f"Type code not recognized: {type1}")

        if type2 == 0:
            # Earlier start, so bounds need to be enforced for an
            # additional interval
            if not isinstance(self._slack_lowerbound[job2, first_idx],
                              docplex.mp.dvar.Var):
                self._slack_lowerbound[job2, first_idx] = \
                    self._problem.continuous_var(
                        name=f"s-_{job2},{first_idx}",
                        lb=0
                    )
                self._slack_upperbound[job2, first_idx] = \
                    self._problem.continuous_var(
                        name=f"s+_{job2},{first_idx}",
                        lb=0
                    )
                self._cost += self._penalty_bounds * \
                    (self._slack_lowerbound[job2, first_idx] +
                     self._slack_upperbound[job2, first_idx])
            # 5. Lower bound
            self._c_lower[job2, first_idx].lhs += \
                self._slack_lowerbound[job2, first_idx]
            # 6. Upper bound
            self._c_upper[job2, first_idx].lhs -= \
                self._slack_upperbound[job2, first_idx]
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check boundary: this always holds
            assert first_idx + 2 < len(self._event_list)
            self._c_lower[job2, first_idx + 1].lhs += \
                self._slack_lowerbound[job2, first_idx + 1]
            self._c_upper[job2, first_idx + 1].lhs -= \
                self._slack_upperbound[job2, first_idx + 1]
        elif type2 == 1:
            # In addition, the time variables need to be updated in two
            # more constraints.
            # No need to check for first_idx > 0: this is always true
            assert first_idx > 0
            self._c_lower[job2, first_idx - 1].lhs += \
                self._slack_lowerbound[job2, first_idx - 1]
            self._c_upper[job2, first_idx - 1].lhs -= \
                self._slack_lowerbound[job2, first_idx - 1]
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

        for u_var in self._slack_upperbound:
            if isinstance(u_var, docplex.mp.dvar.Var):
                upper_slack += u_var.solution_value

        for l_var in self._slack_lowerbound:
            if isinstance(l_var, docplex.mp.dvar.Var):
                lower_slack += l_var.solution_value

        return [("resource", resource_slack, self._penalty_capacity),
                ("upperbound", upper_slack, self._penalty_bounds), 
                ("lowerbound", lower_slack, self._penalty_bounds)]
