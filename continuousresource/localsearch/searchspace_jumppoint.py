import copy
import math
import numpy as np
import time
import os
import os.path

from .eventorder_utils import construct_event_mapping, find_precedences, \
    generate_initial_solution, generate_random_solution
from .searchspace_abstract import SearchSpace, SearchSpaceState
from .searchspace_jumppoint_helper import JumpPointSearchSpaceData
from . import distributions as dists
from continuousresource.mathematicalprogramming.abstract \
    import LPWithSlack
from continuousresource.mathematicalprogramming.eventorder \
    import JumpPointContinuousLP, JumpPointContinuousLPWithSlack
from continuousresource.mathematicalprogramming.flowfeasibility \
    import EstimatingPenaltyFlow


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
            - `logdir` (str): location to write outputfiles to.
    """
    def __init__(self, instance, params=None):
        self._logdir = params['logdir']
        self._data = JumpPointSearchSpaceData(instance)
        with open(os.path.join(self._logdir,
                               "compare_score.csv"), "w") as f:
            f.write(
                'score;flow_estimate;linear_estimate;planned_part;lp_time;'
                'flow_time;simple_time;det_time\n'
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

    @property
    def nevents(self):
        return self._data.nevents

    @property
    def njobs(self):
        return self._data.njobs

    @property
    def nplannable(self):
        return self._data.nplannable

    @property
    def kextra(self):
        return self._data.kextra

    @property
    def lp(self):
        return self._data.lp_model

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
                dummy_instance['eventlist'], (self.njobs, 2)
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
            self._data.eventlist = self._construct_eventlist(
                init_lp.get_schedule(),
                instance['jumppoints'][:, 1:-1].flatten()
            )
        elif self._params['start_solution'] == "random":
            self._data.eventlist = generate_random_solution(
                self._precedences,
                nplannable=self.nplannable
            )
        t_end = time.perf_counter()
        self._timings["initial_solution"] = t_end - t_start

        # Construct eventmap
        njobs = instance['properties'].shape[0]
        kjumps = instance['jumppoints'].shape[1]
        self._data.eventmap = construct_event_mapping(self._data.eventlist,
                                                      (njobs, kjumps))

        # Initialize states
        self._current_solution = SearchSpaceStateJumpPoint(self._data.eventlist)
        self._best_solution = self._current_solution.__copy__()
        self._initial_solution = self._current_solution.__copy__()

        # Compute and register score
        initial_score, initial_slack = self._compute_current()
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
             for i in range(self.nplannable)] +
            [[fixed_times[i], i % self.kextra + 2,
              math.floor(i / self.kextra)]
             for i in range(self.nevents - self.nplannable)]
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

    def _compute_current(self):
        """Compute the scores associated with the current state of _data.

        Returns
        -------
        float
            Score of the current solution
        """
        t_1 = time.perf_counter()
        base_score = self._data.base_initiate()
        t_2 = time.perf_counter()
        simple_slack = self._data.simple_initiate()
        t_3 = time.perf_counter()
        flow_slack = self._data.flow_initiate()
        t_4 = time.perf_counter()
        lp_slack = self._data.lp_initiate()
        t_5 = time.perf_counter()

        with open(os.path.join(self._logdir,
                               "compare_score.csv"), "a") as f:
            f.write(
                f'{lp_slack};{flow_slack};{simple_slack};{base_score};'
                f'{t_5-t_4};{t_4 - t_3};{t_3 - t_2};{t_2 - t_1}\n'
            )
        return base_score, lp_slack
        

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
                                        nplannable=self.nplannable)
        

    def get_neighbor_move(self, orig_idx=None, dist=uniform):
        """Finds candidate solutions by moving an event a random number
        of places in the event order, respecting precomputed precedence
        constraints.

        Parameters
        ----------
        orig_idx : int
            Position of the event in the event list that will be moved.
        dist : {uniform, linear, plus_one}
            Distribution used to define the probability for a relative
            displacement to be selected.
                - `uniform` selects any displacement with equal
                  probability.
                - `linear` selects a displacement with a probability that
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

        # Determine id of current event
        orig_id = job * 2 + etype
        if etype > 1:
            orig_id += self.nplannable - 2 + job * (self.kextra - 2)

        # Determine closest predecessor with a precedence relation
        llimit = orig_idx
        ell = orig_id
        while not (self._precedences[ell, orig_id]):
            llimit -= 1
            if llimit == -1:
                break
            jobl = self._current_solution.eventlist[llimit, 1]
            typel = self._current_solution.eventlist[llimit, 0]
            ell = jobl * 2 + typel
            if typel > 1:
                ell += self.nplannable - 2 + jobl * (self.kextra - 2)

        # Determine closest successor with a precedence relation
        rlimit = orig_idx
        erl = orig_id
        while not (self._precedences[orig_id, erl]):
            rlimit += 1
            if rlimit == len(self._current_solution.eventlist):
                break
            jobr = self._current_solution.eventlist[rlimit, 1]
            typer = self._current_solution.eventlist[rlimit, 0]
            erl = jobr * 2 + typer
            if typer > 1:
                erl += self.nplannable - 2 + jobr * (self.kextra - 2)

        # If the range of possibilities is limited to the current
        # position, we select another job.
        new_idx = dists.dist(orig_idx, llimit, rlimit)

        if new_idx < 0:
            return None, (orig_idx, orig_idx)

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
        flow = EstimatingPenaltyFlow(new_state.instance, 'flow-test')
        flow.solve()
        sf = flow.compute_slack(new_state.instance['constants'][
                            'slackpenalties'
                        ])
        slack_flow = sf[0][1] * sf[0][2] + sf[1][1] * sf[1][2] + \
            sf[2][1] * sf[2][2]

        with open(os.path.join(self._logdir, "compare_score.csv"), "a") as f:
            f.write(
                f'{plan_part + slack_part};{plan_part + slack_flow};'
                f'{plan_part + slack_estimate};{plan_part}\n'
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

    def _get_neighbor_move_aggregate(self, temperature, dist='uniform'):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.
        dist : {'uniform', 'linear'}
            Distribution used to select the distance moved.

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
                                                            dist=dist)
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
        return self._get_neighbor_movelinear_aggregate(temperature,
                                                       dist="linear")


class SearchSpaceStateJumpPoint(SearchSpaceState):
    """Class object describing a state in the search space for a local
    search approach.

    Parameters
    ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
    """
    def __init__(self, eventlist):
        self._eventlist = eventlist
        self._score = np.inf
        self._slack = []

    def __copy__(self):
        """Override copy method to make copies of some attributes and
        reset others.
        """
        return self.__class__(copy.deepcopy(self._eventlist))

    @property
    def instance(self):
        raise NotImplementedError

    @property
    def score(self):
        """float : Score of the represented solution (objective value of
        the LP referenced in `model`).
        """
        return self._score

    @score.setter
    def score(self, score):
        """Manually set the value of the score attribute for this state.
        Use with caution.
        """
        self._score = score

    @property
    def slack(self):
        """list of tuple : List of tuples with in the first position a
        string identifying the type of slack variable, in second position
        the summed value of these variables (float) and in third position
        the unit weight of these variables in the objective.
        """
        return self._slack

    @slack.setter
    def slack(self, slack):
        """Manually set the value of the slack attribute for this state.
        Use with caution."""
        self._slack = slack

    @property
    def eventlist(self):
        return self._eventlist

    @eventlist.setter
    def eventlist(self, eventlist):
        """Manually set the value of the eventorder attribute for this
        state. Use with caution."""
        self._eventlist = eventlist

    @property
    def eventmap(self):
        raise NotImplementedError

