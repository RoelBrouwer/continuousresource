import docplex.mp.model
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


class EstimatingPenaltyFlow(LP):
    """Class implementing an LP that checks instances for feasibility,
    ignoring constraints on the lower bounds.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data, with the following keys:
            - `properties`: two-dimensional (n x 2) array containing
              job properties related to resource consumption:
                - 0: resource requirement (E_j);
                - 1: resource lower bound (P^-_j);
                - 2: resource upper bound (P^+_j);
            - `jumppoints`, containing a two-dimensional (n x k+1) array
              of k+1 increasing time points where the cost function of
              each job changes. The first element is the release date,
              the last element is the deadline.
            - `eventlist`: Two-dimensional (|E| x 2) array representing
              the events in the problem, where the first column contains
              an integer indicating the event type and the second column
              the associated job ID.
            - `constants` > `resource_availability`: Amount of resource
              available per time unit (float).
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.

    Returns
    -------
    float
        Total amount of resource units that lower bounds are already
        violated by in the reduced flow.
    float
        Total amount of resource units that interval capacities are
        already violated by in the reduced flow.
    """
    def __init__(self, instance, label):
        self._njobs = len(instance['properties'])
        self._nevents = self._njobs * instance['jumppoints'].shape[1]
        self._nplannable = self._njobs * 2
        super().__init__(label)
        self._lbpen, self._respen, self._target = \
            self._initialize_model(instance)

    def _initialize_model(self, instance):
        """...
        """
        job_mapping, redresreq, intervals = self._construct_intervals(
            instance['properties'], instance['jumppoints'],
            instance['eventlist'],
            instance['constants']['resource_availability']
        )

        # Create variables
        self._pvar = np.zeros(
            shape=(self._njobs, self._nevents),
            dtype=object
        )

        for j in range(self._njobs):
            for i in range(job_mapping[j, 0],
                           job_mapping[j, 1] + 1):
                self._pvar[j, i] = self._problem.continuous_var(
                    name=f"p_{j},{i}",
                    lb=0
                )

        # Define objective
        self._cost = self._problem.sum(
            self._pvar[j, i]
            for j in range(self._njobs)
            for i in range(job_mapping[j, 0],
                           job_mapping[j, 1] + 1)
        )
        self._problem.maximize(self._cost)

        # Initialize constraints
        self._c_totalwork = np.zeros(shape=self._njobs, dtype=object)
        self._c_upperbound = np.zeros(shape=(self._njobs, intervals.shape[0]),
                                      dtype=object)
        self._c_powercapacity = np.zeros(shape=intervals.shape[0],
                                         dtype=object)

        # Start with the job-to-interval constraints
        for j in range(self._njobs):
            for i in range(job_mapping[j, 0], job_mapping[j, 1] + 1):
                duration = intervals[i, 1] - intervals[i, 0]
                if i > job_mapping[j, 0] and i < job_mapping[j, 1]:
                    # For all non start/completion intervals, we
                    # substract the lowerbound
                    red_upper = instance['properties'][j, 2] - \
                        instance['properties'][j, 1]
                    redresreq[j] -= instance['properties'][j, 1] * duration
                    intervals[i, 2] -= instance['properties'][j, 1] * duration
                else:
                    red_upper = instance['properties'][j, 2]
                # 2. Upper bound
                self._c_upperbound[j, i] = self._problem.add_constraint(
                    ct=self._pvar[j, i] - red_upper * duration <= 0,
                    ctname=f"upper_bound_{j},{i}"
                )

        # Correct potential violations
        lb_pen = sum(redresreq[redresreq < 0]) * -1
        redresreq[redresreq < 0] = 0

        res_pen = sum(intervals[intervals[:, 2] < 0, 2]) * -1
        intervals[intervals[:, 2] < 0, 2] = 0

        for j in range(self._njobs):
            # 1. Total work
            self._c_totalwork[j] = self._problem.add_constraint(
                ct=self._problem.sum(
                    self._pvar[j, i] for i in range(
                        job_mapping[j, 0], job_mapping[j, 1] + 1
                    )
                ) - redresreq[j] <= 0,
                ctname=f"totalwork_{j}"
            )

        for i in range(intervals.shape[0]):
            # 3. Power capacity
            cstr = self._problem.sum(
                self._pvar[j, i] for j in range(self._njobs)
            ) - intervals[i, 2] <= 0

            # We test if the contraint is not trivial (this may occur if
            # an interval exists, during which no jobs are available for
            # processing.
            if isinstance(cstr, docplex.mp.constr.LinearConstraint):
                self._c_powercapacity[i] = self._problem.add_constraint(
                    ct=cstr,
                    ctname=f"resource_availability_{i}"
                )
        return lb_pen, res_pen, sum(redresreq)

    def _construct_intervals(self, resource_info, jumppoints, eventlist,
                             resource):
        """Constructs lists mapping jobs to intervals and determines
        (reduced) upper bounds for these intervals.

        Parameters
        ----------
        resource_info : ndarray
            Two-dimensional (n x 3) array containing job properties
            related to resource consumption:
                - 0: resource requirement (E_j);
                - 1: resource lower bound (P^-_j);
                - 2: resource upper bound (P^+_j);
        jumppoints : ndarray
            Two-dimensional (n x k+1) array of k+1 increasing time points
            where the cost function of each job changes. The first
            element is the release date, the last element is the
            deadline.
        eventlist : ndarray
            Two-dimensional (|E| x 2) array representing the events in
            the problem, where the first column contains an integer
            indicating the event type and the second column the
            associated job ID.
        resource : float
            Amount of resource available per time unit

        Returns
        -------
        ndarray
            Two-dimensional (n x 2) array representing the jobs in the
            problem, where each column contains an integer indicating
            the interval in which a job starts and completes,
            respectively.
        ndarray
            One-dimensional (n) array containing for each job the
            reduced resource requirement after subtraction of all
            pre-enforced lower bounds.
        ndarray
            Two-dimensional (n x 3) array representing the fixed time
            points in the problem, where the first two columns contain a
            float indicating the interval's start and end time,
            respectively. The final column contains the reduced resource
            availability after subtraction of all pre-enforced lower
            bounds.
        """
        job_mapping = np.zeros(shape=(self._njobs, 2), dtype=int)
        redresreq = resource_info[:, 0]
        intervals = np.zeros(shape=(self._nevents - self._nplannable + 1, 3),
                             dtype=float)
        intervals[0, 0] = 0.0
        intervals[-1, 1] = jumppoints[:, -1].max()

        curr_interval = 0
        for [etype, job] in eventlist:
            if etype == 0:
                # Job starts in current interval
                job_mapping[job, 0] = curr_interval
            elif etype == 1:
                # Job ends in current interval
                job_mapping[job, 1] = curr_interval
            else:
                # fixed-time event
                intervals[curr_interval, 1] = jumppoints[job, etype - 1]
                intervals[curr_interval, 2] = \
                    resource * (intervals[curr_interval, 1]
                                - intervals[curr_interval, 0])
                curr_interval += 1
                intervals[curr_interval, 0] = jumppoints[job, etype - 1]

        # Complete the computation of the last interval
        intervals[-1, 2] = resource * (intervals[-1, 1] - intervals[-1, 0])

        return job_mapping, redresreq, intervals

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
        self._respen += self._target - self._problem.objective_value
        return [("resource", self._respen, slackpenalties[0]),
                ("upperbound", 0, slackpenalties[1]),
                ("lowerbound", self._lbpen, slackpenalties[1])]
