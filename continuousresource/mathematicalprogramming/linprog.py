from abc import ABC
from abc import abstractmethod
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

    def solve(self):
        """Solve the LP."""
        # self._problem.export_as_lp(os.getcwd())
        if self._solver == 'cplex':
            self._problem.solve()  # log_output=True)
            # self._problem.print_solution()
        else:
            raise NotImplementedError

    @abstractmethod
    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        pass


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
        self.initialize_problem()

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
        self._obj = self._problem.maximize(self._cost)

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
            self._c_availability = self._problem.add_constraint(
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
                ct=self._times[self._event_map[j, 1]]
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
        # We could update the event list & map as well, but keeping them
        # in their original order, allows us to maintain the relation of
        # the variable names to the position in the (original) event list

        # update self._event_list
        self._event_list[[first_idx, first_idx + 1], :] = \
            self._event_list[[first_idx + 1, first_idx], :]

        # update self._event_map
        self._event_map[job1, type1] += 1
        self._event_map[job2, type2] -= 1

        # update appropriate self._c_order
        self._c_order[first_idx - 1].lhs = \
            self._times[first_idx - 1] - self._times[first_idx + 1]
        self._c_order[first_idx].lhs = \
            self._times[first_idx + 1] - self._times[first_idx]
        self._c_order[first_idx + 1].lhs = \
            self._times[first_idx] - self._times[first_idx + 2]

        # update self._resource variables
        if type1 == 1 and self._resource[job1, first_idx + 1] is None:
            # The completion of job1 happens one interval later, a new
            # variable must be added.
            self._resource[job1, first_idx + 1] = \
                self._problem.continuous_var(
                    name=f"p_{job1},{first_idx + 1}",
                    lb=0
                )
        if type2 == 0 and self._resource[job2, first_idx] is None:
            # The start of job2 happens one interval earlier, a new
            # variable must be added.
            self._resource[job2, first_idx] = \
                self._problem.continuous_var(
                    name=f"p_{job2},{first_idx}",
                    lb=0
                )

        # update appropriate self._c_lower & self._c_upper
        if type1 == 1:
            # Delayed completion, so bounds need to be enforced for an
            # additional interval
            # 5. Lower bound
            self._c_lower[job1, first_idx + 1] = self._problem.add_constraint(
                ct=self._resource[job1, first_idx + 1] -
                self._job_properties[job1, 1] * (self._times[first_idx + 2]
                                                 - self._times[first_idx + 1])
                >= 0,
                ctname=f"lower_bound_{job1},{first_idx + 1}"
            )
            # 6. Upper bound
            self._c_upper[job1, first_idx + 1] = self._problem.add_constraint(
                ct=self._resource[job1, first_idx + 1] -
                self._job_properties[job1, 2] * (self._times[first_idx + 2]
                                                 - self._times[first_idx + 1])
                <= 0,
                ctname=f"upper_bound_{job1},{first_idx + 1}"
            )
        elif type1 == 0:
            # Delayed start, so bounds need to be enforced for one fewer
            # interval
            self._problem.remove_constraints([
                self._c_lower[job1, first_idx],
                self._c_upper[job1, first_idx]
            ])
            self._c_lower[job1, first_idx] = None
            self._c_upper[job1, first_idx] = None
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
        elif type2 == 1:
            # Earlier completion, so bounds need to be enforced for one
            # fewer interval
            self._problem.remove_constraints([
                self._c_lower[job2, first_idx + 1],
                self._c_upper[job2, first_idx + 1]
            ])
            self._c_lower[job2, first_idx + 1] = None
            self._c_upper[job2, first_idx + 1] = None
        else:
            raise ValueError(f"Type code not recognized: {type2}")

        # update appropriate self._c_resource
        for j in [job1, job2]:
            self._c_resource[j].lhs = self._problem.sum(
                self._resource[j, e] for e in range(
                    self._event_map[j, 0], self._event_map[j, 1]
                )
            ) - self._job_properties[j, 0]

        # update self._times positions
        self._times[[first_idx, first_idx + 1]] = \
            self._times[[first_idx + 1, first_idx]]

    def print_solution(self):
        """Print a human readable version of the (current) solution."""
        pass
