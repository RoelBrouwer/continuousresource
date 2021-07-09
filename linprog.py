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
        if self._solver == 'cplex':
            self._problem.solve()
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
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    solver : {'cplex'}
        The solver that will be used to solve the LP constructed from the
        provided instance. Currently only CPLEX is supported.
    """
    def __init__(self, eventlist, jobs, label, solver='cplex'):
        super().__init__(label, solver)

        self._event_list = eventlist
        self._event_map = _construct_event_mapping(eventlist, jobs.shape[0])
        self._job_properties = jobs

    @property
    def event_list(self):
        return self._event_list

    @property
    def event_map(self):
        return self._event_map

    @property
    def job_properties(self):
        return self._job_properties

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
        for i in eventlist.shape[0]:
            eventmap[eventlist[i, 1], eventlist[i, 0]] = i
        return event_map
