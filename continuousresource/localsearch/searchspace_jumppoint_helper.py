import copy
import math
import numpy as np
import time
import os
import os.path
import warnings

from continuousresource.mathematicalprogramming.eventorder \
    import JumpPointContinuousLPWithSlack
from continuousresource.mathematicalprogramming.flowfeasibility \
    import EstimatingPenaltyFlow


class JumpPointSearchSpaceData():
    """Object to store and easily interface datastructures used in the
    process of exploring the search space.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    """
    def __init__(self, instance):
        # Store instance data
        self._instance = instance
        self._njobs = instance['properties'].shape[0]
        self._nevents = instance['jumppoints'].shape[1] * self._njobs
        self._nplannable = self._njobs * 2
        self._kextra = int((self._nevents - self._nplannable) / self._njobs)

        # Set validation flags
        self._simple_valid = False
        self._flow_valid = False
        self._lp_valid = False
        pass

    @property
    def instance(self):
        return self._instance

    @property
    def eventlist(self):
        return self._instance['eventlist']

    @eventlist.setter
    def eventlist(self, e):
        # TODO: improve
        self._instance['eventlist'] = e

    @property
    def eventmap(self):
        return self._instance['eventmap']

    @eventmap.setter
    def eventmap(self, e):
        # TODO: improve
        self._instance['eventmap'] = e

    @property
    def job_properties(self):
        return self._instance['properties']

    @property
    def jumppoints(self):
        return self._instance['jumppoints']

    @property
    def resource(self):
        return self._instance['constants']['resource_availability']

    @property
    def slackpenalties(self):
        return self._instance['constants']['slackpenalties']

    @property
    def weights(self):
        return self._instance['weights']

    @property
    def nevents(self):
        return self._nevents

    @property
    def njobs(self):
        return self._njobs

    @property
    def nplannable(self):
        return self._nplannable

    @property
    def kextra(self):
        return self._kextra

    @property
    def flow_model(self):
        return self._flow_model

    @property
    def lp_model(self):
        return self._lp_model

    @property
    def fixed_successor_map(self):
        """Stores a table that matches each (plannable) event to the
        closest fixed-time event following it in the event order.

        One dimensional array of length 2n, linking the index of the
        plannable event (2 * job_id + event_type) to the index of the
        fixed event (2n + kextra * job_id + etype - 2). If an event
        does not have a fixed-time successor, the array stores a -1.
        """
        return self._fixed_successor_map

    @property
    def fixed_predecessor_map(self):
        """Stores a table that matches each (plannable) event to the
        closest fixed-time event preceding it in the event order.

        One dimensional array of length 2n, linking the index of the
        plannable event (2 * job_id + event_type) to the index of the
        fixed event (2n + kextra * job_id + etype - 2). If an event
        does not have a fixed-time predecessor, the array stores a -1.
        """
        return self._fixed_predecessor_map

    @property
    def resource_availability_table(self):
        """A table that stores the cumulative resource totals and
        shortage for computating a lower bound on the slack penalty.

        Two-dimensional array |E| * 2: for every event, the total is
        stored in the first column, and the shortage in the second.
        """
        return self._resource_availability_table

    def base_initiate(self):
        """Compute the score of a feasible solution respecting the
        current event order.

        Returns
        -------
        float
            The (base) cost of a feasible solution that respects the
            current event order.
        """
        self._jumppoint_map = np.zeros(shape=self._njobs, dtype=int)
        self._base_cost = 0

        for [etype, job] in self._instance['eventlist']:
            if etype > 1:
                self._jumppoint_map[job] = etype - 1
            elif etype == 1:
                self._base_cost += sum(self._instance['weights'][
                    job, :self._jumppoint_map[job] + 1
                ])
        return self._base_cost

    def base_update_compute(self, event_idx, offset):
        """Compute the difference in score for moving a completion event
        in a given direction.

        Parameters
        ----------
        event_idx : int
            Index of the completion event in the current eventlist.
        offset : int
            Amount of positions the event is moved.

        Returns
        -------
        float
            Cost difference resulting from the given displacement.
        tuple of int
            Job ID and corresponding index of the relevant jumpppoint.
        """
        job_id = self._instance['eventlist'][event_idx, 1]
        if self._instance['eventlist'][event_idx, 0] != 1:
            warnings.warn(
                "Trying to update the base score for moving an event that is"
                " not a completion event. No change."
            )
            return 0.0, (job_id, self._jumppoint_map[job_id])

        if offset < 0:
            moved_part = self._instance['eventlist'][
                event_idx + offset:event_idx
            ]
            passed_jp = moved_part[
                (moved_part[:, 1] == job_id) & (moved_part[:, 0] > 1)
            ]
            if len(passed_jp) > 0:
                return sum(self._instance['weights'][
                    job_id, passed_jp[0][0] - 1:passed_jp[-1][0]
                ]) * -1, (job_id, passed_jp[0][0] - 2)
        elif offset > 0:
            moved_part = self._instance['eventlist'][
                event_idx + 1:event_idx + offset + 1
            ]
            passed_jp = moved_part[
                (moved_part[:, 1] == job_id) & (moved_part[:, 0] > 1)
            ]
            if len(passed_jp) > 0:
                return sum(self._instance['weights'][
                    job_id, passed_jp[0][0] - 1:passed_jp[-1][0]
                ]), (job_id, passed_jp[-1][0] - 1)
        else:
            warnings.warn(
                "Trying to update the base score for moving a completion"
                " event by zero positions."
            )

        return 0.0, (job_id, self._jumppoint_map[job_id])

    def base_update_apply(self, cost_diff, map_update):
        """Update the internal administration for a precomputed update
        after an update is accepted.

        Parameters
        ----------
        cost_diff : float
            Cost difference resulting from the given displacement.
        map_update : tuple of int
            Job ID and corresponding index of the relevant jumpppoint.
        """
        self._base_cost += cost_diff
        self._jumppoint_map[map_update[0]] = map_update[1]

    def simple_initiate(self):
        """Compute the simple estimation of the penalty term for the
        current event order.

        This consists of three parts:
            - Estimation of the lower bound violations;
            - Estimation of the upper bound violations;
            - Estimation of the resource availability violations.
        """
        self._simple_valid = True
        self._fixed_predecessor_map = np.zeros(shape=self._nplannable,
                                               dtype=int)
        self._fixed_successor_map = np.zeros(shape=self._nplannable,
                                             dtype=int)

        # First compute a mapping for the closest fixed time events
        prev_fix = -1
        for [etype, job] in self._instance['eventlist']:
            if etype > 1:
                prev_fix = self._nplannable + job * self._kextra + etype - 2
            else:
                self._fixed_predecessor_map[2 * job + etype] = prev_fix
        next_fix = -1
        for [etype, job] in self._instance['eventlist'][::-1]:
            if etype > 1:
                next_fix = self._nplannable + job * self._kextra + etype - 2
            else:
                self._fixed_successor_map[2 * job + etype] = next_fix

        # 1 & 2. Compute the lower/upper bound estimation
        min_lb = 0
        min_ub = 0

        for job in range(self._njobs):
            pre_start = self._fixed_predecessor_map[2 * job] \
                - self._nplannable
            pre_compl = self._fixed_predecessor_map[2 * job + 1] \
                - self._nplannable
            suc_start = self._fixed_successor_map[2 * job] \
                - self._nplannable
            suc_compl = self._fixed_successor_map[2 * job + 1] \
                - self._nplannable
            # etype -> x % self._kextra + 1
            # job -> math.floor(x / self._kextra)
            if pre_compl > -1 and suc_start > -1:
                min_lb += max(
                    0,
                    (
                        self._instance['jumppoints'][
                            math.floor(pre_compl / self._kextra),
                            pre_compl % self._kextra + 1] -
                        self._instance['jumppoints'][
                            math.floor(suc_start / self._kextra),
                            suc_start % self._kextra + 1]
                    ) * self._instance['properties'][job, 1] -
                    self._instance['properties'][job, 0]
                )
            if pre_start < 0:
                t_s = self._instance['jumppoints'][job, 0]
            else:
                t_s = self._instance['jumppoints'][
                    math.floor(pre_start / self._kextra),
                    pre_start % self._kextra + 1]
            if suc_compl < 0:
                t_e = self._instance['jumppoints'][job, -1]
            else:
                t_e = self._instance['jumppoints'][
                    math.floor(suc_compl / self._kextra),
                    suc_compl % self._kextra + 1]
            min_ub += max(
                0,
                self._instance['properties'][job, 0] -
                self._instance['properties'][job, 2] * (t_e - t_s)
            )

        # 3. Compute the resource estimation
        self._resource_availability_table = np.zeros(shape=(self._nevents, 2),
                                                     dtype=float)
        for i, [etype, job] in enumerate(self._instance['eventlist']):
            if etype > 1 and i > 0:
                curr_res = self._instance['jumppoints'][job, etype - 1] * \
                    self._instance['constants']['resource_availability']
                self._resource_availability_table[i, 1] = \
                    self._resource_availability_table[i - 1, 1] + max(
                        0,
                        self._resource_availability_table[i - 1, 0] - curr_res
                    )
                self._resource_availability_table[i, 0] = \
                    min(self._resource_availability_table[i - 1, 0], curr_res)
            elif etype == 1:
                self._resource_availability_table[i, 0] = \
                    self._resource_availability_table[i - 1, 0] + \
                    self._instance['properties'][job, 0]
                self._resource_availability_table[i, 1] = \
                    self._resource_availability_table[i - 1, 1]
            elif i > 0:
                self._resource_availability_table[i] = \
                    self._resource_availability_table[i - 1]

        return (min_lb + min_ub) * \
            self._instance['constants']['slackpenalties'][1] + \
            self._resource_availability_table[-1, 1] * \
            self._instance['constants']['slackpenalties'][0]

    def simple_update_compute(self):
        if not self._simple_valid:
            warnings.warn(
                "Trying to update a non-valid state for the simple penalty"
                " term estimation. Performing a full recompute instead."
            )
            self.simple_initiate()
            return
        pass

    def simple_update_apply(self):
        pass

    def flow_initiate(self):
        """Compute the flow approximation of the penalty term for the
        current event order.
        """
        self._flow_valid = True
        self._flow_model = EstimatingPenaltyFlow(self._instance, 'flow-test')
        sol = self._flow_model.solve()
        if sol is None:
            return np.inf
        self._flow_slack = self._flow_model.compute_slack(
            self._instance['constants']['slackpenalties']
        )
        return self._flow_slack[0][1] * self._flow_slack[0][2] + \
            self._flow_slack[1][1] * self._flow_slack[1][2] + \
            self._flow_slack[2][1] * self._flow_slack[2][2]

    def flow_update_compute(self):
        if not self._flow_valid:
            warnings.warn(
                "Trying to update a non-valid state for the flow-based"
                " penalty term estimation. Performing a full recompute"
                " instead."
            )
            self.flow_initiate()
            return
        raise NotImplementedError

    def flow_update_apply(self):
        raise NotImplementedError

    def lp_initiate(self):
        """Compute the penalty term in the LP for the current event
        order.
        """
        self._lp_valid = True
        self._lp_model = JumpPointContinuousLPWithSlack(self._instance,
                                                        'lp-test')
        sol = self._lp_model.solve()
        if sol is None:
            return np.inf

        return sol.get_objective_value()
        # self._lp_slack = self._lp_model.compute_slack(
        #     self._instance['constants']['slackpenalties']
        # )
        # return self._lp_slack[0][1] * self._lp_slack[0][2] + \
        #     self._lp_slack[1][1] * self._lp_slack[1][2] + \
        #     self._lp_slack[2][1] * self._lp_slack[2][2]

    def lp_update_compute(self, event_idx, offset):
        if not self._lp_valid:
            warnings.warn(
                "Trying to update a non-valid state of the LP for the"
                " penalty term computation. Performing a full recompute"
                " instead."
            )
            self.lp_initiate()
            return
        raise NotImplementedError
        pass
