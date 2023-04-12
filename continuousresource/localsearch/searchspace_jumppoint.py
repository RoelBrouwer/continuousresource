import copy
import math
import numpy as np
import time
import os
import os.path

from .eventorder_utils import construct_event_mapping, find_precedences, \
    generate_initial_solution, generate_random_solution
from .searchspace_abstract import SearchSpace, SearchSpaceState
from continuousresource.mathematicalprogramming.abstract \
    import LPWithSlack
from continuousresource.mathematicalprogramming.eventorder \
    import JumpPointContinuousLP, JumpPointContinuousLPWithSlack


class JumpPointSearchSpace(SearchSpace):
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
        self._njobs = instance['properties'].shape[0]
        self._nevents = instance['jumppoints'].shape[1] * self._njobs
        self._nplannable = self._njobs * 2
        self._kextra = int((self._nevents - self._nplannable) / self._njobs)
        with open(os.path.join(os.getcwd(), "compare_score.csv"), "w") as f:
            f.write(
                'score;score_estimate;planned_part;slack_part;slack_estimate;'
                'frac_dif\n'
            )
        super().__init__(instance, params=params)
        self._label = "jumppoint-superclass"
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
            instance['properties'], instance['jumppoints'][:, [0, -1]],
            fixed_events=instance['jumppoints'][:, 1:-1].flatten(),
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
            dummy_instance = copy.deepcopy(instance)
            dummy_instance['jumppoints'] = instance['jumppoints'][:, [0, -1]]
            dummy_instance['weights'] = instance['weights'][:, [0]]
            dummy_instance['eventlist'] = generate_initial_solution(
                instance['properties'][:, [0, 2]],
                instance['jumppoints'][:, [0, -1]],
                instance['constants']['resource_availability']
            )
            dummy_instance['eventmap'] = construct_event_mapping(
                dummy_instance['eventlist'], (self._njobs, 2)
            )
            if instance['constants']['slackpenalties'] is None:
                init_lp = \
                    JumpPointContinuousLP(dummy_instance, "construct_initial")
            else:
                init_lp = \
                    JumpPointContinuousLPWithSlack(dummy_instance,
                                                   "construct_initial")
            sol = init_lp.solve()
            initial_slack = []
            if sol is not None:
                initial_score = sol.get_objective_value()
                if isinstance(init_lp, LPWithSlack):
                    initial_slack = init_lp.compute_slack(
                        instance['constants']['slackpenalties']
                    )
            else:
                initial_score = np.inf
            instance['eventlist'] = self._construct_eventlist(
                init_lp.get_schedule(),
                instance['jumppoints'][:, 1:-1].flatten()
            )
        elif self._params['start_solution'] == "random":
            instance['eventlist'] = generate_random_solution(
                self._precedences,
                nplannable=self._nplannable
            )
        t_end = time.perf_counter()
        self._timings["initial_solution"] = t_end - t_start

        # Construct eventmap
        njobs = instance['properties'].shape[0]
        kjumps = instance['jumppoints'].shape[1]
        instance['eventmap'] = construct_event_mapping(instance['eventlist'],
                                                       (njobs, kjumps))

        # Initialize states
        self._current_solution = SearchSpaceState(instance)
        self._best_solution = self._current_solution.__copy__()
        self._initial_solution = self._current_solution.__copy__()

        # Initialize LP model
        if instance['constants']['slackpenalties'] is None:
            # No slack
            self._lp_model = JumpPointContinuousLP(instance, self._label)
        else:
            # With slack
            self._lp_model = JumpPointContinuousLPWithSlack(instance,
                                                                self._label)

        # Compute and register score
        if self._params['start_solution'] != "greedy":
            initial_score, initial_slack = self._compute_score()
        if initial_score != np.inf:
            initial_score += self._compute_score_plannable_part(instance)
        self._current_solution.score = initial_score
        self._current_solution.slack = initial_slack
        self._best_solution.score = initial_score
        self._best_solution.slack = initial_slack
        self._initial_solution.score = initial_score
        self._initial_solution.slack = initial_slack

    def _construct_eventlist(self, plannable_times, fixed_times):
        """Construct an eventlist based on the times that each event
        occurs in an initial solution.
        """
        times = np.array(
            [[plannable_times[i], i % 2, math.floor(i / 2)]
            for i in range(self._nplannable)] +
            [[fixed_times[i], i % self._kextra + 2, math.floor(i / self._kextra)]
            for i in range(self._nevents - self._nplannable)]
        )
        times = times[times[:, 0].argsort()]
        return np.array(times[:, [1, 2]], dtype=int)

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

    def _compute_score_plannable_part(self, instance):
        """Compute the score of a feasible solution respecting the
        current event order.
        """
        cost = 0
        jp = np.zeros(shape=self._njobs, dtype=int)
        for [etype, job] in instance['eventlist']:
            if etype > 1:
                jp[job] = etype - 1
            elif etype == 1:
                cost += sum(instance['weights'][job, :jp[job] + 1])
        return cost

    def _estimate_score_slack_part(self, instance):
        """Get a lower bound on the contribution of slack penalties to
        the objective function for any schedule using the current event
        order.
        """
        total_res = 0
        short_res = 0
        for [etype, job] in instance['eventlist']:
            if etype > 1:
                curr_res = instance['jumppoints'][job, etype - 1] * \
                    instance['constants']['resource_availability']
                short_res += max(0, total_res - curr_res)
                total_res = min(total_res, curr_res)
            elif etype == 1:
                total_res += instance['properties'][job, 0]
        return short_res * instance['constants']['slackpenalties'][0]

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
        return generate_random_solution(self._precedences,
                                        nplannable=self._nplannable)

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
            job1 = self._current_solution.eventlist[swap_id, 1]
            type1 = self._current_solution.eventlist[swap_id, 0]
            job2 = self._current_solution.eventlist[swap_id + 1, 1]
            type2 = self._current_solution.eventlist[swap_id + 1, 0]
            e1 = job1 * 2 + type1
            if type1 > 1:
                e1 += self._nplannable - 2 + job1 * (self._kextra - 2)
            e2 = job2 * 2 + type2
            if type2 > 1:
                e2 += self._nplannable - 2 + job2 * (self._kextra - 2)
            while self._precedences[e1, e2]:
                swap_id = np.random.randint(
                    len(self._current_solution.instance['eventlist'] - 1)
                )
                job1 = self._current_solution.eventlist[swap_id, 1]
                type1 = self._current_solution.eventlist[swap_id, 0]
                job2 = self._current_solution.eventlist[swap_id + 1, 1]
                type2 = self._current_solution.eventlist[swap_id + 1, 0]
                e1 = job1 * 2 + type1
                if type1 > 1:
                    e1 += self._nplannable - 2 + job1 * (self._kextra - 2)
                e2 = job2 * 2 + type2
        else:
            job1 = self._current_solution.eventlist[swap_id, 1]
            type1 = self._current_solution.eventlist[swap_id, 0]
            job2 = self._current_solution.eventlist[swap_id + 1, 1]
            type2 = self._current_solution.eventlist[swap_id + 1, 0]
            e1 = job1 * 2 + type1
            if type1 > 1:
                e1 += self._nplannable - 2 + job1 * (self._kextra - 2)
            e2 = job2 * 2 + type2
            if type2 > 1:
                e2 += self._nplannable - 2 + job2 * (self._kextra - 2)
            if self._precedences[e1, e2]:
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

        plan_part = self._compute_score_plannable_part(new_state.instance)
        slack_part, slack = self._compute_score()
        new_state.score = plan_part + slack_part
        new_state.slack = slack
        
        slack_estimate = self._estimate_score_slack_part(new_state.instance)
        
        with open(os.path.join(os.getcwd(), "compare_score.csv"), "a") as f:
            f.write(
                f'{plan_part + slack_part};{plan_part + slack_estimate};'
                f'{plan_part};{slack_part};{slack_estimate};'
                f'{(slack_part - slack_estimate) / plan_part + slack_part}\n'
            )

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
        jobl = self._current_solution.eventlist[llimit, 1]
        typel = self._current_solution.eventlist[llimit, 0]
        ell = jobl * 2 + typel
        if typel > 1:
            ell += self._nplannable - 2 + jobl * (self._kextra - 2)
        while not (self._precedences[ell, job * 2 + etype]):
            llimit -= 1
            if llimit == -1:
                break
            jobl = self._current_solution.eventlist[llimit, 1]
            typel = self._current_solution.eventlist[llimit, 0]
            ell = jobl * 2 + typel
            if typel > 1:
                ell += self._nplannable - 2 + jobl * (self._kextra - 2)

        # Determine closest successor with a precedence relation
        rlimit = orig_idx
        jobr = self._current_solution.eventlist[rlimit, 1]
        typer = self._current_solution.eventlist[rlimit, 0]
        erl = jobr * 2 + typer
        if typer > 1:
            erl += self._nplannable - 2 + jobr * (self._kextra - 2)
        while not (self._precedences[job * 2 + etype, erl]):
            rlimit += 1
            if rlimit == len(self._current_solution.eventlist):
                break
            jobr = self._current_solution.eventlist[rlimit, 1]
            typer = self._current_solution.eventlist[rlimit, 0]
            erl = jobr * 2 + typer
            if typer > 1:
                erl += self._nplannable - 2 + jobr * (self._kextra - 2)

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

        plan_part = self._compute_score_plannable_part(new_state.instance)
        slack_part, slack = self._compute_score()
        new_state.score = plan_part + slack_part
        new_state.slack = slack
        
        slack_estimate = self._estimate_score_slack_part(new_state.instance)
        
        with open(os.path.join(os.getcwd(), "compare_score.csv"), "a") as f:
            f.write(
                f'{plan_part + slack_part};{plan_part + slack_estimate};'
                f'{plan_part};{slack_part};{slack_estimate};'
                f'{(slack_part - slack_estimate) / plan_part + slack_part}\n'
            )

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
        jobl1 = self._current_solution.eventlist[llimit1, 1]
        typel1 = self._current_solution.eventlist[llimit1, 0]
        ell1 = jobl1 * 2 + typel1
        if typel1 > 1:
            ell1 += self._nplannable - 2 + jobl1 * (self._kextra - 2)
        while not (self._precedences[ell1, job * 2]):
            llimit1 -= 1
            if llimit1 == -1:
                break
            jobl1 = self._current_solution.eventlist[llimit1, 1]
            typel1 = self._current_solution.eventlist[llimit1, 0]
            ell1 = jobl1 * 2 + typel1
            if typel1 > 1:
                ell1 += self._nplannable - 2 + jobl1 * (self._kextra - 2)
        llimit2 = idx2
        jobl2 = self._current_solution.eventlist[llimit2, 1]
        typel2 = self._current_solution.eventlist[llimit2, 0]
        ell2 = jobl2 * 2 + typel2
        if typel2 > 1:
            ell2 += self._nplannable - 2 + jobl2 * (self._kextra - 2)
        while not (self._precedences[ell2, job * 2 + 1]):
            llimit2 -= 1
            if llimit2 == -1:
                break
            jobl2 = self._current_solution.eventlist[llimit2, 1]
            typel2 = self._current_solution.eventlist[llimit2, 0]
            ell2 = jobl2 * 2 + typel2
            if typel2 > 1:
                ell2 += self._nplannable - 2 + jobl2 * (self._kextra - 2)

        # Determine closest successor with a precedence relation
        rlimit1 = idx1
        jobr1 = self._current_solution.eventlist[rlimit1, 1]
        typer1 = self._current_solution.eventlist[rlimit1, 0]
        erl1 = jobr1 * 2 + typer1
        if typer1 > 1:
            erl1 += self._nplannable - 2 + jobr1 * (self._kextra - 2)
        while not (self._precedences[job * 2, erl1]):
            rlimit1 += 1
            if rlimit1 == len(self._current_solution.eventlist):
                break
            jobr1 = self._current_solution.eventlist[rlimit1, 1]
            typer1 = self._current_solution.eventlist[rlimit1, 0]
            erl1 = jobr1 * 2 + typer1
            if typer1 > 1:
                erl1 += self._nplannable - 2 + jobr1 * (self._kextra - 2)
        rlimit2 = idx2
        jobr2 = self._current_solution.eventlist[rlimit2, 1]
        typer2 = self._current_solution.eventlist[rlimit2, 0]
        erl2 = jobr2 * 2 + typer2
        if typer2 > 1:
            erl2 += self._nplannable - 2 + jobr2 * (self._kextra - 2)
        while not (self._precedences[job * 2 + 1, erl2]):
            rlimit2 += 1
            if rlimit2 == len(self._current_solution.eventlist):
                break
            jobr2 = self._current_solution.eventlist[rlimit2, 1]
            typer2 = self._current_solution.eventlist[rlimit2, 0]
            erl2 = jobr2 * 2 + typer2
            if typer2 > 1:
                erl2 += self._nplannable - 2 + jobr2 * (self._kextra - 2)

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

        plan_part = self._compute_score_plannable_part(new_state.instance)
        slack_part, slack = self._compute_score()
        new_state.score = plan_part + slack_part
        new_state.slack = slack
        
        slack_estimate = self._estimate_score_slack_part(new_state.instance)
        
        with open(os.path.join(os.getcwd(), "compare_score.csv"), "a") as f:
            f.write(
                f'{plan_part + slack_part};{plan_part + slack_estimate};'
                f'{plan_part};{slack_part};{slack_estimate};'
                f'{(slack_part - slack_estimate) / plan_part + slack_part}\n'
            )

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
        self._current_solution = new_state

        return new_state


class JumpPointSearchSpaceCombined(JumpPointSearchSpace):
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
            np.arange(len(self._current_solution.instance['properties']))
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
