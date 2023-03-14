import copy
import math
import numpy as np
import time

from .eventorder_utils import construct_event_mapping, find_precedences, \
    generate_initial_solution, generate_random_solution
from .searchspace_abstract import SearchSpace, SearchSpaceState
from continuousresource.mathematicalprogramming.abstract \
    import LPWithSlack
from continuousresource.mathematicalprogramming.eventorder \
    import JobPropertiesContinuousLP, JobPropertiesContinuousLPWithSlack


class JobArraySearchSpace(SearchSpace):
    """Wrapper object for information about the search space.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations;
            - `fracs` (dict of float): Dictionary indicating the
              probability of selecting each neighborhood operator
              (`swap`, `move`, `movepair`).
            - `start_solution` (str): String indicating the method of
              generating a starting solution. Either `random` or
              `greedy`.
    """
    def __init__(self, instance, params=None):
        super().__init__(instance, params=params)
        self._label = "jobarray-superclass"
        self._operator_data = {
            "swap": {
                "performed": 0,
                "succeeded": 0
            },
            "move": {
                "performed": 0,
                "succeeded": 0
            },
            "movepair": {
                "performed": 0,
                "succeeded": 0
            },
        }

    @property
    def operator_data(self):
        """Dictionary containing data on the neighborhood operators that
        have been applied.
        """
        return self._operator_data

    def _find_precedences(self, instance, infer_precedence):
        """Construct an array indicating precedence relations between
        events.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
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
        return find_precedences(
            instance['jobs'][:, [0, 1, 2]], instance['jobs'][:, [3, 4]],
            infer_precedence=self._params['infer_precedence']
        )

    def _generate_initial_solution(self, instance):
        """Generates an initial solution within the search space and sets
        the value of `_lp_model`, `_best_solution`, `_current_solution`
        and `_initial_solution`  accordingly.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        # Generate initial solution
        t_start = time.perf_counter()
        if self._params['start_solution'] == "greedy":
            instance['eventlist'] = generate_initial_solution(
                instance['jobs'][:, [0, 2]], instance['jobs'][:, [3, 4]],
                instance['constants']['resource_availability']
            )
        elif self._params['start_solution'] == "random":
            instance['eventlist'] = generate_random_solution(self._precedences)
        t_end = time.perf_counter()
        self._timings["initial_solution"] = t_end - t_start

        # Construct eventmap
        njobs = instance['jobs'].shape[0]
        instance['eventmap'] = construct_event_mapping(instance['eventlist'],
                                                       (njobs, 2))

        # Initialize states
        self._current_solution = SearchSpaceState(instance)
        self._best_solution = self._current_solution.__copy__()
        self._initial_solution = self._current_solution.__copy__()

        # Initialize LP model
        if instance['constants']['slackpenalties'] is None:
            # No slack
            self._lp_model = JobPropertiesContinuousLP(instance, self._label)
        else:
            # With slack
            self._lp_model = JobPropertiesContinuousLPWithSlack(instance,
                                                                self._label)

        # Compute and register score
        initial_score, initial_slack = self._compute_score()
        self._current_solution.score = initial_score
        self._current_solution.slack = initial_slack
        self._best_solution.score = initial_score
        self._best_solution.slack = initial_slack
        self._initial_solution.score = initial_score
        self._initial_solution.slack = initial_slack

    def _compute_score(self):
        """Compute the (exact) score of the current solution

        Returns
        -------
        float
            Score of the current solution
        """
        assert self._lp_model is not None

        t_start = time.perf_counter()
        sol = self._lp_model.solve()
        t_end = time.perf_counter()
        self.timings["lp_solve"] += t_end - t_start

        if sol is not None:
            score = sol.get_objective_value()
            # self._schedule = self._lp_model.get_schedule()
            if isinstance(self._lp_model, LPWithSlack):
                slack = self._lp_model.compute_slack(
                    self._current_solution.instance['constants'][
                        'slackpenalties'
                    ]
                )
            else:
                slack = []
            return score, slack
        else:
            return np.inf, []

    def get_random_order(self):
        """Returns a random event order that respects (precomputed)
        precedence relations.

        Returns
        -------
        ndarray
            Two-dimensional (|E| x 2) array representing the events in the
            problem, where the first column contains an integer indicating
            the event type and the second column the associated job ID.
        """
        return generate_random_solution(self._precedences, nplannable=0)

    def get_neighbor_swap(self, swap_id=None):
        """Finds candidate solutions by swapping adjacent event pairs.

        Parameters
        ----------
        swap_id : int
            Position of the first event in the event list that will
            switch positions with its successor.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        int
            ID of the first swapped event
        """
        self._operator_data["swap"]["performed"] += 1
        self._operator_data["swap"]["succeeded"] += 1

        # Find two events with no precedence relation to swap. The matrix
        # of precedence relations contains at minimum the precedence
        # between start and completion of the same job.
        if swap_id is None:
            swap_id = np.random.randint(
                len(self._current_solution.instance['eventlist'] - 1)
            )
            while self._precedences[
                self._current_solution.instance['eventlist'][swap_id, 1] * 2 +
                self._current_solution.instance['eventlist'][swap_id, 0],
                self._current_solution.instance['eventlist'][swap_id + 1,
                                                             1] * 2 +
                self._current_solution.instance['eventlist'][swap_id + 1, 0]
            ]:
                swap_id = np.random.randint(
                    len(self._current_solution.instance['eventlist'] - 1)
                )
        else:
            if self._precedences[
                self._current_solution.instance['eventlist'][swap_id, 1] * 2 +
                self._current_solution.instance['eventlist'][swap_id, 0],
                self._current_solution.instance['eventlist'][swap_id + 1,
                                                             1] * 2 +
                self._current_solution.instance['eventlist'][swap_id + 1, 0]
            ]:
                return None, swap_id

        new_state = self._current_solution.__copy__()
        new_state.instance['eventmap'][
            new_state.instance['eventlist'][swap_id, 1],
            new_state.instance['eventlist'][swap_id, 0]
        ] += 1
        new_state.instance['eventmap'][
            new_state.instance['eventlist'][swap_id + 1, 1],
            new_state.instance['eventlist'][swap_id + 1, 0]
        ] -= 1
        new_state.instance['eventlist'][[swap_id, swap_id + 1], :] = \
            new_state.instance['eventlist'][[swap_id + 1, swap_id], :]

        t_start = time.perf_counter()
        self._lp_model.update_swap_neighbors(new_state.instance, swap_id)
        t_end = time.perf_counter()
        self._timings["model_update"] += t_end - t_start

        score, slack = self._compute_score()
        new_state.score = score
        new_state.slack = slack

        return new_state, swap_id

    def get_neighbor_swap_revert(self, new_state, swap_id):
        """Reverts the model to its previous state.

        Parameters
        ----------
        new_state : SearchSpaceState
            Reference to the state containing a link to the model that
            should be reverted.
        swap_id : int
            ID of the first event to be swapped back.
        """
        self._operator_data["swap"]["succeeded"] -= 1
        t_start = time.perf_counter()
        self._lp_model.update_swap_neighbors(self._current_solution.instance,
                                             swap_id)
        t_end = time.perf_counter()
        self._timings["model_update"] += t_end - t_start

    def get_neighbor_move(self, orig_idx=None, dist='uniform'):
        """Finds candidate solutions by moving an event a random number
        of places in the event order, respecting precomputed precedence
        constraints.

        Parameters
        ----------
        orig_idx : int
            Position of the event in the event list that will be moved.
        dist : {'uniform', 'linear'}
            Distribution used to define the probability for a relative
            displacement to be selected.
                - 'uniform' selects any displacement with equal
                  probability.
                - 'linear' selects a displacement with a probability that
                  decreases linearly with increasing size.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        tuple of int
            Indices of the two swapped events
        """
        self._operator_data["move"]["performed"] += 1
        self._operator_data["move"]["succeeded"] += 1
        if orig_idx is None:
            orig_idx = np.random.randint(
                len(self._current_solution.eventlist)
            )
        new_idx = 0

        job = self._current_solution.eventlist[orig_idx, 1]
        etype = self._current_solution.eventlist[orig_idx, 0]

        # Determine closest predecessor with a precedence relation
        llimit = orig_idx
        while not (self._precedences[
                   self._current_solution.eventlist[llimit, 1] * 2
                   + self._current_solution.eventlist[llimit, 0],
                   job * 2 + etype]):
            llimit -= 1
            if llimit == -1:
                break

        # Determine closest successor with a precedence relation
        rlimit = orig_idx
        while not (self._precedences[job * 2 + etype,
                   self._current_solution.eventlist[rlimit, 1] * 2
                   + self._current_solution.eventlist[rlimit, 0]]):
            rlimit += 1
            if rlimit == len(self._current_solution.eventlist):
                break

        # If the range of possibilities is limited to the current
        # position, we select another job.
        if rlimit - llimit <= 2:
            return None, (orig_idx, orig_idx)
        else:
            new_idx = orig_idx
            while new_idx == orig_idx:
                if dist == 'uniform':
                    new_idx = np.random.randint(
                        llimit + 1,
                        rlimit
                    )
                elif dist == 'linear':
                    new_idx = self._get_idx_linear_displacement(
                        orig_idx, llimit, rlimit
                    )

        new_state = self._current_solution.__copy__()

        curr = orig_idx
        inv = 0
        # Invert movement direction
        if orig_idx > new_idx:
            inv = -1
        while curr - new_idx != 0:
            new_state.instance['eventmap'][
                new_state.instance['eventlist'][curr + inv, 1],
                new_state.instance['eventlist'][curr + inv, 0]
            ] += 1
            new_state.instance['eventmap'][
                new_state.instance['eventlist'][curr + inv + 1, 1],
                new_state.instance['eventlist'][curr + inv + 1, 0]
            ] -= 1
            new_state.instance['eventlist'][[curr + inv,
                                             curr + inv + 1], :] = \
                new_state.instance['eventlist'][[curr + inv + 1,
                                                 curr + inv], :]

            t_start = time.perf_counter()
            self._lp_model.update_swap_neighbors(new_state.instance,
                                                 curr + inv)
            t_end = time.perf_counter()
            self._timings["model_update"] += t_end - t_start

            if inv < 0:
                curr -= 1
            else:
                curr += 1

        score, slack = self._compute_score()
        new_state.score = score
        new_state.slack = slack

        return new_state, (orig_idx, new_idx)

    def _get_idx_linear_displacement(self, orig_idx, llimit, rlimit):
        """Randomly selects a new index based on the provided limits,
        where the probability of a displacement being selected decreases
        linearly with increasing size.

        Parameters
        ----------
        orig_idx : int
            Current position of the event.
        llimit : int
            Exclusive lower boundary for its new position.
        rlimit : int
            Exclusive upper boundary for its new position.

        Returns
        -------
        int
            New position.

        Notes
        -----
        TODO The current implementation leaves room for improvement.
        """
        displacements = [
            i
            for j in (range(orig_idx - llimit - 1, 0, -1),
                      range(1, rlimit - orig_idx, 1))
            for i in j
        ]
        max_dis = max(displacements[0], displacements[-1])
        inv_displacements = [max_dis - i + 1 for i in displacements]
        total = sum(inv_displacements)
        probabilities = [i / total for i in inv_displacements]
        selected = np.random.random()
        cum_prob = probabilities[0]
        curr_idx = 0
        while cum_prob < selected:
            curr_idx += 1
            cum_prob += probabilities[curr_idx]
        if curr_idx >= orig_idx - llimit - 1:
            curr_idx += 1
        return llimit + curr_idx + 1

    def get_neighbor_move_revert(self, new_state, idcs):
        """Reverts the model to its previous state.

        Parameters
        ----------
        new_state : SearchSpaceState
            Reference to the state containing a link to the model that
            should be reverted.
        idcs : Tuple
            Tuple containing the original and new index of the moved
            event.
        """
        self._operator_data["move"]["succeeded"] -= 1

        curr = idcs[1]
        inv = 0
        # Invert movement direction
        if idcs[1] > idcs[0]:
            inv = -1
        while curr - idcs[0] != 0:
            new_state.instance['eventmap'][
                new_state.instance['eventlist'][curr + inv, 1],
                new_state.instance['eventlist'][curr + inv, 0]
            ] += 1
            new_state.instance['eventmap'][
                new_state.instance['eventlist'][curr + inv + 1, 1],
                new_state.instance['eventlist'][curr + inv + 1, 0]
            ] -= 1
            new_state.instance['eventlist'][[curr + inv,
                                             curr + inv + 1], :] = \
                new_state.instance['eventlist'][[curr + inv + 1,
                                                 curr + inv], :]

            t_start = time.perf_counter()
            self._lp_model.update_swap_neighbors(new_state.instance,
                                                 curr + inv)
            t_end = time.perf_counter()
            self._timings["model_update"] += t_end - t_start

            if inv < 0:
                curr -= 1
            else:
                curr += 1

    def get_neighbor_move_pair(self, job=None, dist='uniform'):
        """Finds candidate solutions by moving two events, belonging to
        the same job, a random number of places in the event order,
        respecting precomputed precedence constraints.

        Parameters
        ----------
        job : int
            Index of the job of which both associated events (start and
            completion) will be moved.
        dist : {'uniform', 'linear'}
            Distribution used to define the probability for a relative
            displacement to be selected.
                - 'uniform' selects any displacement with equal
                  probability.
                - 'linear' selects a displacement with a probability that
                  decreases linearly with increasing size.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        tuple of int
            Original indices of the job's events and the offset used to
            move them.

        Notes
        -----
        The maximum offset is currently limited by the relative distance
        of the two events associated with a job. This restriction can be
        lifted in the future, but the order of swaps then has to be
        carefully chosen.
        """
        self._operator_data["movepair"]["performed"] += 1
        self._operator_data["movepair"]["succeeded"] += 1

        if job is None:
            job = np.random.randint(
                len(self._current_solution.instance['jobs'])
            )
        offset = 0

        idx1 = self._current_solution.eventmap[job, 0]
        idx2 = self._current_solution.eventmap[job, 1]

        # Determine closest predecessor with a precedence relation
        llimit1 = idx1
        while not (self._precedences[
                   self._current_solution.eventlist[llimit1, 1] * 2
                   + self._current_solution.eventlist[llimit1, 0],
                   job * 2]):
            llimit1 -= 1
            if llimit1 == -1:
                break
        llimit2 = idx2
        while not (self._precedences[
                   self._current_solution.eventlist[llimit2, 1] * 2
                   + self._current_solution.eventlist[llimit2, 0],
                   job * 2 + 1]):
            llimit2 -= 1
            if llimit2 == -1:
                break

        # Determine closest successor with a precedence relation
        rlimit1 = idx1
        while not (self._precedences[job * 2,
                   self._current_solution.eventlist[rlimit1, 1] * 2
                   + self._current_solution.eventlist[rlimit1, 0]]):
            rlimit1 += 1
            if rlimit1 == len(self._current_solution.eventlist):
                break
        rlimit2 = idx2
        while not (self._precedences[job * 2 + 1,
                   self._current_solution.eventlist[rlimit2, 1] * 2
                   + self._current_solution.eventlist[rlimit2, 0]]):
            rlimit2 += 1
            if rlimit2 == len(self._current_solution.eventlist):
                break

        llimit = max(llimit1 - idx1, llimit2 - idx2)
        rlimit = min(rlimit1 - idx1, rlimit2 - idx2)

        # If the range of possibilities is limited to the current
        # position, we select another job.
        if rlimit - llimit <= 2:
            return None, (idx1, idx2, 0)
        else:
            while offset == 0:
                if dist == 'uniform':
                    offset = np.random.randint(
                        llimit + 1,
                        rlimit
                    )
                elif dist == 'linear':
                    offset = self._get_idx_linear_displacement(
                        0, llimit, rlimit
                    )

        new_state = self._current_solution.__copy__()

        # Determine direction of movement
        for idx in [idx1, idx2]:
            curr = idx
            if idx > idx + offset:
                inv = -1
            else:
                inv = 0
            while curr != idx + offset:
                new_state.instance['eventmap'][
                    new_state.instance['eventlist'][curr + inv, 1],
                    new_state.instance['eventlist'][curr + inv, 0]
                ] += 1
                new_state.instance['eventmap'][
                    new_state.instance['eventlist'][curr + inv + 1, 1],
                    new_state.instance['eventlist'][curr + inv + 1, 0]
                ] -= 1
                new_state.instance['eventlist'][[curr + inv,
                                                 curr + inv + 1], :] = \
                    new_state.instance['eventlist'][[curr + inv + 1,
                                                     curr + inv], :]

                t_start = time.perf_counter()
                self._lp_model.update_swap_neighbors(new_state.instance,
                                                     curr + inv)
                t_end = time.perf_counter()
                self._timings["model_update"] += t_end - t_start

                if inv < 0:
                    curr -= 1
                else:
                    curr += 1

        score, slack = self._compute_score()
        new_state.score = score
        new_state.slack = slack

        return new_state, (idx1, idx2, offset)

    def get_neighbor_move_pair_revert(self, new_state, idcs):
        """Reverts the model to its previous state.

        Parameters
        ----------
        new_state : SearchSpaceState
            Reference to the state containing a link to the model that
            should be reverted.
        idcs : Tuple
            Tuple containing the original indices of the moved events and
            the offset used to determine their new positions.
        """
        self._operator_data["movepair"]["succeeded"] -= 1

        # Determine direction of movement
        for idx in [idcs[0], idcs[1]]:
            curr = idx + idcs[2]
            if idx + idcs[2] > idx:
                inv = -1
            else:
                inv = 0
            while curr != idx:
                new_state.instance['eventmap'][
                    new_state.instance['eventlist'][curr + inv, 1],
                    new_state.instance['eventlist'][curr + inv, 0]
                ] += 1
                new_state.instance['eventmap'][
                    new_state.instance['eventlist'][curr + inv + 1, 1],
                    new_state.instance['eventlist'][curr + inv + 1, 0]
                ] -= 1
                new_state.instance['eventlist'][[curr + inv,
                                                 curr + inv + 1], :] = \
                    new_state.instance['eventlist'][[curr + inv + 1,
                                                     curr + inv], :]

                t_start = time.perf_counter()
                self._lp_model.update_swap_neighbors(new_state.instance,
                                                     curr + inv)
                t_end = time.perf_counter()
                self._timings["model_update"] += t_end - t_start

                if inv < 0:
                    curr -= 1
                else:
                    curr += 1

    def random_walk(self, no_steps=100):
        """Performs a random walk from the current solution by swapping
        `no_steps` random pairs of adjacent events. These swaps are not
        executed simultaneously, but one after another. So, the second
        swap is performed on the event order that results after the first
        swap, and so on.

        Parameters
        ----------
        no_steps : int
            Length of the random walk, i.e. the number of swaps
            performed.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        """
        raise NotImplementedError
        # TODO
        new_state = self._current_solution.__copy__()

        for i in range(no_steps):
            swap_id = np.random.randint(
                len(new_state.model.event_list) - 1
            )
            while (new_state.model.event_list[swap_id, 1] ==
                   new_state.model.event_list[swap_id + 1, 1]):
                swap_id = np.random.randint(
                    len(new_state.model.event_list) - 1
                )
            new_state.model.update_swap_neighbors(swap_id)

        new_state.eventorder = copy.copy(
            new_state.model.event_list
        )
        new_state.compute_score()

        if new_state.score < self._best_solution.score:
            self._best_solution = copy.copy(new_state)
            self._best_solution.score = new_state.score
            self._best_solution.slack = new_state.slack
            # new_state.model.print_solution()
        self._current_solution = new_state

        return new_state


class JobArraySearchSpaceCombined(JobArraySearchSpace):
    """Search space that only uses the job-event pair move operator.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
    """
    def __init__(self, instance, params=None):
        super().__init__(instance, params=params)
        self._fracs = params["fracs"]
        self._label = (f"combined_so{self._fracs['swap']*100:.1f}_ms"
                       f"{self._fracs['move']*100:.1f}_mp"
                       f"{self._fracs['movepair']*100:.1f}")

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        int
            Number of candidate solutions that were considered, but
            rejected.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Awful programming, but it works
        frac = np.random.random()
        if frac < self._fracs["swap"]:
            fail_count, new_state = \
                self._get_neighbor_swap_aggregate(temperature)
            if new_state is None:
                fail_count1, new_state = \
                    self._get_neighbor_movelinear_aggregate(temperature)
                fail_count += fail_count1
                if new_state is None:
                    fail_count2, new_state = \
                        self._get_neighbor_move_pair_aggregate(temperature)
                    fail_count += fail_count2
        elif frac < self._fracs["swap"] + self._fracs["move"]:
            fail_count, new_state = \
                self._get_neighbor_movelinear_aggregate(temperature)
            if new_state is None:
                fail_count1, new_state = \
                    self._get_neighbor_swap_aggregate(temperature)
                fail_count += fail_count1
                if new_state is None:
                    fail_count2, new_state = \
                        self._get_neighbor_move_pair_aggregate(temperature)
                    fail_count += fail_count2
        else:
            fail_count, new_state = \
                self._get_neighbor_move_pair_aggregate(temperature)
            if new_state is None:
                fail_count1, new_state = \
                    self._get_neighbor_swap_aggregate(temperature)
                fail_count += fail_count1
                if new_state is None:
                    fail_count2, new_state = \
                        self._get_neighbor_movelinear_aggregate(temperature)
                    fail_count += fail_count2

        return fail_count, new_state

    def _get_neighbor_move_pair_aggregate(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        int
            Number of candidate solutions that were considered, but
            rejected.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order
        att_ord = np.random.permutation(
            np.arange(len(self._current_solution.instance['jobs']))
        )
        fail_count = 0

        for idx in att_ord:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_move_pair(job=idx)
            if new_state is None:
                fail_count += 1
                continue
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = new_state.__copy__()
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                break
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_move_pair_revert(new_state, revert_info)
                fail_count += 1
                new_state = None

        return fail_count, new_state

    def _get_neighbor_moveuniform_aggregate(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        int
            Number of candidate solutions that were considered, but
            rejected.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order
        att_ord = np.random.permutation(
            np.arange(len(self._current_solution.eventlist))
        )
        fail_count = 0

        for idx in att_ord:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_move(orig_idx=idx)
            if new_state is None:
                fail_count += 1
                continue
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = new_state.__copy__()
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                break
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_move_revert(new_state, revert_info)
                fail_count += 1
                new_state = None

        return fail_count, new_state

    def _get_neighbor_movelinear_aggregate(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        int
            Number of candidate solutions that were considered, but
            rejected.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order
        att_ord = np.random.permutation(
            np.arange(len(self._current_solution.eventlist))
        )
        fail_count = 0

        for idx in att_ord:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_move(orig_idx=idx,
                                                            dist='linear')
            if new_state is None:
                fail_count += 1
                continue
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = new_state.__copy__()
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                break
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_move_revert(new_state, revert_info)
                fail_count += 1
                new_state = None

        return fail_count, new_state

    def _get_neighbor_swap_aggregate(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        int
            Number of candidate solutions that were considered, but
            rejected.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order
        att_ord = np.random.permutation(
            # We cannot try the last one.
            np.arange(len(self._current_solution.eventlist) - 1)
        )
        fail_count = 0

        for idx in att_ord:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_swap(swap_id=idx)
            if new_state is None:
                fail_count += 1
                continue
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = new_state.__copy__()
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                break
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_swap_revert(new_state, revert_info)
                fail_count += 1
                new_state = None

        return fail_count, new_state


class JobArraySearchSpaceHillClimb(JobArraySearchSpace):
    """Search space that only uses the job-event pair move operator.

    Parameters
    ----------
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
    """
    def __init__(self, params=None):
        raise NotImplementedError
        # TODO
        super().__init__(params=params)
        self._label = ("hillclimb")

    def get_neighbor_swap_hc(self):
        """Finds candidate solutions by swapping adjacent event pairs.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        int
            ID of the first swapped event
        """
        # Determine order
        att_ord = np.random.permutation(
            # We cannot try the last one.
            np.arange(len(self._current_solution.model.event_list) - 1)
        )

        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model

        for swap_id in att_ord:
            if self._current_solution.model.event_list[swap_id, 1] == \
               self._current_solution.model.event_list[swap_id + 1, 1]:
                continue
            new_state.model.update_swap_neighbors(swap_id)
            new_state.eventorder = copy.copy(
                self._current_solution.model.event_list
            )
            new_state.compute_score()
            if new_state.score < self._current_solution.score:
                self._current_solution = new_state
                self._best_solution = new_state
                return new_state
            else:
                new_state.model.update_swap_neighbors(swap_id)
        return None

    def get_neighbor_move_hc(self):
        """Finds candidate solutions by moving an event a random number
        of places in the event order, respecting precomputed precedence
        constraints.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        ?
            ?
        """
        # Determine order
        att_ord = np.random.permutation(
            np.arange(len(self._current_solution.model.event_list))
        )

        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model

        for orig_idx in att_ord:
            job = self._current_solution.model.event_list[orig_idx, 1]
            etype = self._current_solution.model.event_list[orig_idx, 0]

            # Determine closest predecessor with a precedence relation
            llimit = orig_idx
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit, 1] * 2
                       + self._current_solution.model.event_list[llimit, 0],
                       job * 2 + etype]):
                llimit -= 1
                if llimit == -1:
                    break

            # Determine closest successor with a precedence relation
            rlimit = orig_idx
            while not (self._precedences[job * 2 + etype,
                       self._current_solution.model.event_list[rlimit, 1] * 2
                       + self._current_solution.model.event_list[rlimit, 0]]):
                rlimit += 1
                if rlimit == len(self._current_solution.model.event_list):
                    break

            # If the range of possibilities is limited to the current
            # position, we select another job.
            if rlimit - llimit == 2:
                continue
            else:
                ran_ord = np.random.permutation(np.arange(llimit + 1, rlimit))
                for new_idx in ran_ord:
                    if new_idx == orig_idx:
                        continue
                    new_state.model.update_move_event(orig_idx, new_idx)
                    new_state.eventorder = copy.copy(
                        self._current_solution.model.event_list
                    )
                    new_state.compute_score()
                    if new_state.score < self._current_solution.score:
                        self._current_solution = new_state
                        self._best_solution = new_state
                        return new_state
                    else:
                        new_state.model.update_move_event(new_idx, orig_idx)

        return None

    def get_neighbor_move_pair_hc(self):
        """Finds candidate solutions by moving two events, belonging to
        the same job, a random number of places in the event order,
        respecting precomputed precedence constraints.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        ?
            ?
        """
        # TODO: DOES NOT WORK, fix indices (etype1 etc)
        # Determine order
        att_ord = np.random.permutation(
            # We cannot try the last one.
            np.arange(len(self._current_solution.model.job_properties))
        )

        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model

        for job_idx in att_ord:
            idx1 = self._current_solution.model.event_map[job_idx, 0]
            idx2 = self._current_solution.model.event_map[job_idx, 1]

            # Determine closest predecessor with a precedence relation
            llimit1 = idx1
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit1, 1] * 2
                       + self._current_solution.model.event_list[llimit1, 0],
                       job_idx * 2 + etype1]):
                llimit1 -= 1
                if llimit1 == -1:
                    break
            llimit2 = idx2
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit2, 1] * 2
                       + self._current_solution.model.event_list[llimit2, 0],
                       job_idx * 2 + 1 - etype1]):
                llimit2 -= 1
                if llimit2 == -1:
                    break

            # Determine closest successor with a precedence relation
            rlimit1 = idx1
            while not (self._precedences[job_idx * 2 + etype1,
                       self._current_solution.model.event_list[rlimit1, 1] * 2
                       + self._current_solution.model.event_list[rlimit1,
                                                                 0]]):
                rlimit1 += 1
                if rlimit1 == len(self._current_solution.model.event_list):
                    break
            rlimit2 = idx2
            while not (self._precedences[job_idx * 2 + 1 - etype1,
                       self._current_solution.model.event_list[rlimit2, 1] * 2
                       + self._current_solution.model.event_list[rlimit2,
                                                                 0]]):
                rlimit2 += 1
                if rlimit2 == len(self._current_solution.model.event_list):
                    break

            llimit = max(llimit1 - idx1, llimit2 - idx2)
            rlimit = min(rlimit1 - idx1, rlimit2 - idx2)

            # If the range of possibilities is limited to the current
            # position, we select another job.
            if rlimit - llimit == 2:
                continue
            else:
                ran_ord = np.random.permutation(np.arange(llimit + 1, rlimit))
                for offset in ran_ord:
                    if offset == 0:
                        continue
                    new_state.model.update_move_pair(idx1, idx2, offset)
                    new_state.eventorder = copy.copy(
                        self._current_solution.model.event_list
                    )
                    new_state.compute_score()
                    if new_state.score < self._current_solution.score:
                        self._current_solution = new_state
                        self._best_solution = new_state
                        return new_state
                    else:
                        new_state.model.update_move_event(
                            idx1 + offset, idx2 + offset, -1 * offset
                        )

        return None
