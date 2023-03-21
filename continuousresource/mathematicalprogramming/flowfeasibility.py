from abc import ABC
from abc import abstractmethod
import bisect
import docplex.mp.model
import math
import numpy as np

from continuousresource.mathematicalprogramming.abstract \
    import LP
from continuousresource.localsearch.eventorder_utils \
    import construct_event_mapping

class FeasibilityWithoutLowerbound(LP):
    """Class implementing an LP that checks instances for feasibility,
    ignoring constraints on the lower bounds.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data, with the following keys:
            - `resource-info`: two-dimensional (n x 2) array containing
              job properties related to resource consumption:
                - 0: resource requirement (E_j);
                - 1: resource upper bound (P^+_j);
            - `time-info`: two-dimensional (n x 2) array containing job
              properties related to event timing:
                - 0: release date (r_j);
                - 1: deadline (d_j);
            - `resource`: Amount of resource available per time unit
              (float).
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, instance, label):
        self._njobs = len(instance['resource-info'])
        self._nevents = self._njobs * 2
        events = self._construct_event_list(instance['time-info'])
        instance['eventlist'] = events[:, :2].astype(int)
        instance['eventmap'] = construct_event_mapping(instance['eventlist'],
                                                       shape=(self._njobs, 2))
        self._tvar = events[:, 2]
        super().__init__(label)
        self._initialize_model(instance)

    def _initialize_model(self, instance):
        """...
        """
        # Create variables
        self._pvar = np.zeros(
            shape=(self._njobs, self._nevents),
            dtype=object
        )

        for j in range(self._njobs):
            for e in range(instance['eventmap'][j, 0],
                           instance['eventmap'][j, 1]):
                self._pvar[j, e] = self._problem.continuous_var(
                    name=f"p_{j},{e}",
                    lb=0
                )

        # Define objective
        self._cost = self._problem.sum(
            self._pvar[j, e]
            for j in range(self._njobs)
            for e in range(instance['eventmap'][j, 0],
                           instance['eventmap'][j, 1])
        )
        self._problem.maximize(self._cost)

        # Initialize constraints
        self._c_totalwork = np.zeros(shape=self._njobs, dtype=object)
        self._c_upperbound = np.zeros(shape=(self._njobs, self._nevents - 1),
                                      dtype=object)
        self._c_powercapacity = np.zeros(shape=self._nevents - 1,
                                         dtype=object)

        for j in range(self._njobs):
            # 1. Total work
            self._c_totalwork[j] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._pvar[j, e] for e in range(
                        instance['eventmap'][j, 0], instance['eventmap'][j, 1]
                    )
                ) - instance['resource-info'][j, 0] == 0,
                ctname=f"totalwork_{j}"
            )

            for e in range(instance['eventmap'][j, 0],
                           instance['eventmap'][j, 1]):
                # 2. Upper bound
                self._c_upperbound[j, e] = self._problem.add_constraint(
                    ct=self._pvar[j, e] - instance['resource-info'][j, 1] *
                    (self._tvar[e + 1] - self._tvar[e]) <= 0,
                    ctname=f"upper_bound_{j},{e}"
                )

        for e in range(self._nevents - 1):
            # 3. Power capacity
            cstr = self._problem.sum(
                self._pvar[j, e] for j in range(self._njobs)
            ) - instance['resource'] * (self._tvar[e + 1]
                                        - self._tvar[e]) <= 0

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
            Two-dimensional (n x 2) array containing for each job:
                - 0: release date (r_j);
                - 1: deadline (d_j);

        Returns
        -------
        ndarray
            Two-dimensional (|E| x 3) array representing the events in
            the problem, where the first column contains an integer
            indicating the event type (0 for start, 1 for completion), the
            second column the associated job ID, and the third its time. The
            array is sorted by the third column.
        """
        eventlist = np.empty(shape=(self._njobs * 2, 3))
        for j in range(self._njobs):
            eventlist[2 * j] = [0, j, jobs[j, 0]]
            eventlist[2 * j + 1] = [1, j, jobs[j, 1]]
        return eventlist[eventlist[:, 2].argsort()]

    def compute_slack(self):
        raise NotImplementedError
