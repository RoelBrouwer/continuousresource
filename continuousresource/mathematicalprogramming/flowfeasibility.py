from abc import ABC
from abc import abstractmethod
import bisect
import docplex.mp.model
import math
import numpy as np

from continuousresource.mathematicalprogramming.eventorder \
    import EventOrderLinearModel


class FeasibilityWithoutLowerbound(EventOrderLinearModel):
    """Class implementing an LP that checks instances for feasibility,
    ignoring constraints on the lower bounds.

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
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, jobs, resource, label):
        raise NotImplementedError
        # TODO: update it to the new structure
        # First, we construct an eventlist
        events = self._construct_event_list(jobs)
        instance['eventlist'] = events[:, :2].astype(int)
        self._times = events[:, 2]
        super().__init__(eventlist, instance, label)

    def _initialize_model(self, instance):
        """...
        """
        # Create variables
        self._resource = np.zeros(
            shape=(len(self._job_properties), len(self._event_list)),
            dtype=object
        )

        for j in range(len(self._job_properties)):
            for e in range(self._event_map[j, 0], self._event_map[j, 1]):
                self._resource[j, e] = self._problem.continuous_var(
                    name=f"p_{j},{e}",
                    lb=0
                )

        # Define objective
        self._cost = self._problem.sum(
            self._resource[j, e]
            for j in range(len(self._job_properties))
            for e in range(self._event_map[j, 0], self._event_map[j, 1])
        )
        self._problem.maximize(self._cost)

        # Initialize constraints
        self._c_totalwork = np.zeros(shape=len(self._job_properties),
                                     dtype=object)
        self._c_upperbound = np.zeros(shape=(len(self._job_properties),
                                             len(self._event_list) - 1),
                                      dtype=object)
        self._c_powercapacity = np.zeros(shape=len(self._event_list) - 1,
                                         dtype=object)

        for j in range(len(self._job_properties)):
            # 1. Total work
            self._c_totalwork[j] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._resource[j, e] for e in range(
                        self._event_map[j, 0], self._event_map[j, 1]
                    )
                ) - self._job_properties[j, 0] == 0,
                ctname=f"totalwork_{j}"
            )

            for e in range(self._event_map[j, 0], self._event_map[j, 1]):
                # 2. Upper bound
                self._c_upperbound[j, e] = self._problem.add_constraint(
                    ct=self._resource[j, e] -
                    self._job_properties[j, 2] * (self._times[e + 1]
                                                  - self._times[e]) <= 0,
                    ctname=f"upper_bound_{j},{e}"
                )

        for e in range(len(self._event_list) - 1):
            # 3. Power capacity
            cstr = self._problem.sum(
                self._resource[j, e] for j in range(
                    len(self._job_properties)
                )
            ) - self._capacity * (self._times[e + 1]
                                  - self._times[e]) <= 0

            # We test if the contraint is not trivial (this may occur if
            # an interval exists, during which no jobs are available for
            # processing.
            if isinstance(cstr, docplex.mp.constr.LinearConstraint):
                self._c_powercapacity[e] = self._problem.add_constraint(
                    ct=cstr,
                    ctname=f"resource_availability_{e}"
                )

    def _construct_event_list(self, jobs):
        """Constructs an eventlist based on release dates and deadlines.

        Parameters
        ----------
        jobs : ndarray
            Two-dimensional (n x 7) array containing job properties:
                - 0: resource requirement (E_j);
                - 1: resource lower bound (P^-_j);
                - 2: resource upper bound (P^+_j);
                - 3: release date (r_j);
                - 4: deadline (d_j);
                - 5: weight (W_j);
                - 6: objective constant (B_j).

        Returns
        -------
        ndarray
            Two-dimensional (|E| x 3) array representing the events in
            the problem, where the first column contains an integer
            indicating the event type (0 for start, 1 for completion), the
            second column the associated job ID, and the third its time. The
            array is sorted by the third column.
        """
        eventlist = np.empty(shape=(len(jobs) * 2, 3))
        for j in range(len(jobs)):
            eventlist[2 * j] = [0, j, jobs[j, 3]]
            eventlist[2 * j + 1] = [1, j, jobs[j, 4]]
        return eventlist[eventlist[:, 2].argsort()]

    def compute_slack(self):
        pass
