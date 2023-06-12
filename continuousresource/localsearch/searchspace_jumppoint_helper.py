import copy
import math
import numpy as np
import time
import os
import os.path
import warnings


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

    @property
    def eventmap(self):
        return self._instance['eventmap']

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
    def fixed_successor_map(self):
        """Stores a table that matches each (plannable) event to the
        closest fixed-time event following it in the event order.
        """
        return self._fixed_successor_map

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

    def simple_initiate(self, instance):
        self._simple_valid = True
        pass

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
