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
from .utils import get_slack_value
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
        super().__init__(instance, params=params)
        self._tabulist = np.full(params['tabu_length'], -1, dtype=int)
        self._label = "jumppoint-superclass"

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

    @property
    def operator_data(self):
        """Dictionary containing data on the neighborhood operators that
        have been applied.
        """
        return {}

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

    def _compute_score(self):
        """Compute the (exact) score of the current solution

        Returns
        -------
        float
            Score of the current solution
        """
        raise NotImplementedError

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
            if sol is None:
                raise RuntimeError("No initial solution found.")
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
        self._data.simple_initiate()

        # Initialize states
        self._current_solution = SearchSpaceStateJumpPoint(None)
        self._best_solution = SearchSpaceStateJumpPoint(
            self._data.eventlist.copy()
        )
        self._initial_solution = SearchSpaceStateJumpPoint(
            self._data.eventlist.copy()
        )

        # Compute and register score
        initial_slack = self._data.lp_initiate()
        initial_score = self._data.base_initiate() + \
            get_slack_value(initial_slack)
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

    def _find_limits(self, orig_id, orig_idx):
        """Find the index of the first event right and left of the event
        with ID `orig_id` and/or index `orig_idx` has a precedence
        relation.

        Parameters
        ----------
        orig_id : int
            ID of the job for which we find the limit
        orig_idx : int
            Index of the job for which we find the limit

        Returns
        -------
        int
            Index of the left limit
        int
            Index of the right limit

        Notes
        -----
        It is assumed, but not checked, that `orig_id` and `orig_idx`
        belong to the same event.
        """
        rlimit = orig_idx
        erl = orig_id
        while not (self._precedences[orig_id, erl]):
            rlimit += 1
            if rlimit == self.nevents:
                break
            jobr = self._data.eventlist[rlimit, 1]
            typer = self._data.eventlist[rlimit, 0]
            erl = jobr * 2 + typer
            if typer > 1:
                erl += self.nplannable - 2 + jobr * (self.kextra - 2)

        llimit = orig_idx
        ell = orig_id
        while not (self._precedences[ell, orig_id]):
            llimit -= 1
            if llimit == -1:
                break
            jobl = self._data.eventlist[llimit, 1]
            typel = self._data.eventlist[llimit, 0]
            ell = jobl * 2 + typel
            if typel > 1:
                ell += self.nplannable - 2 + jobl * (self.kextra - 2)

        return llimit, rlimit

    def _add_to_tabu_list(self, event_id):
        """Add new ID to tabu list, and remove the last element.

        event_id : int
            ID of the event to add to the list
        """
        if len(self._tabulist) > 0:
            self._tabulist = np.roll(self._tabulist, 1)
            self._tabulist[0] = event_id

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
        for i in range(no_steps):
            new_idx = -1
            while new_idx < 0:
                orig_job = np.random.randint(self.njobs)
                orig_etype = np.random.randint(2)
                orig_id = 2 * orig_job + orig_etype
                orig_idx = self._data.eventmap[orig_job, orig_etype]
                llimit, rlimit = self._find_limits(orig_id, orig_idx)
                new_idx = dists.plus_one(orig_idx, llimit, rlimit)
            self._data.instance_update(orig_idx, new_idx)

        base_score = self._data.base_initiate()
        new_slack = self._data.lp_initiate()
        self._data.simple_initiate()
        new_score = base_score + get_slack_value(new_slack)

        self._current_solution.score = new_score
        self._current_solution.slack = new_slack

        if new_score < self._best_solution.score:
            self._best_solution = SearchSpaceStateJumpPoint(
                self._data.eventlist.copy()
            )
            self._best_solution.score = new_score
            self._best_solution.slack = new_slack

        return self._current_solution


    def solve_and_write_best(self):
        """Solve the LP again to get the corresponding schedule"""
        self._data.eventlist = self._best_solution.eventlist
        self._data.eventmap = construct_event_mapping(
            self._best_solution.eventlist, (self.njobs, self.kextra + 2)
        )
        if not np.array_equal(self._data._instance['eventlist'],
                              self._best_solution.eventlist):
            print("Setting up the eventlist for solution checking failed.")
        slack = self._data.lp_initiate()
        base = self._data.base_initiate()
        # Print solution to file
        with open(os.path.join(self._logdir, "solution.csv"), "w") as sol:
            sol.write(self._data._lp_model.get_solution_csv())
        if not np.isclose(
            get_slack_value(slack), self._best_solution.slack_value
        ):
            print(slack)
            print(self._best_solution.slack)
            print("The re-solving of the best LP did not yield the same"
            f" result - recomputed: {get_slack_value(slack)}, stored:",
            f"{self._best_solution.slack_value}")
        if not np.isclose(
            base, self._best_solution.score - self._best_solution.slack_value
        ):
            print("The re-solving of the base score did not yield the same"
            f" result - recomputed: {base}, stored:",
            f"{self._best_solution.score - self._best_solution.slack_value}")

    def write_log(self, filename, array, fmt='%.0f'):
        """Write any array-like object to a log file"""
        np.savetxt(os.path.join(self._logdir, f"{filename}.txt"), array, fmt=fmt)


class JumpPointSearchSpaceLP(JumpPointSearchSpace):
    """Search space implementing the search strategy using only the LP
    for evaluating candidate solutions.

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
        self._label = f"jumppoint-lp-{params['dist'].__qualname__}"

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_score = self._current_solution.score
        current_base = self._current_solution.score - \
            self._current_solution.slack_value

        for event_id in att_ord:
            if event_id in self._tabulist:
                continue
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Precompute acceptance threshold
            max_acc_score = -1 * math.log(np.random.random()) * \
                temperature + self._current_solution.score

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > max_acc_score:
                continue

            new_slack = self._data.lp_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_slack)

            if np.isinf(get_slack_value(new_slack)) or \
               new_score > max_acc_score:
                self._data.lp_update_apply(idx, new_idx, success=False)
                continue

            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.lp_update_apply(idx, new_idx, success=True)
            self._current_solution.score = new_score
            self._current_solution.slack = new_slack
            self._add_to_tabu_list(event_id)

            if new_score < self._best_solution.score:
                self._best_solution.eventlist = self._data.eventlist.copy()
                self._best_solution.score = new_score
                self._best_solution.slack = new_slack
            accepted = True
            break

        improved = accepted and new_score < current_score

        return improved, (self._current_solution if accepted else None)


class JumpPointSearchSpaceMix(JumpPointSearchSpace):
    """Search space implementing the search strategy using the LP and the
    simple bound estimations. The LP will only be computed if a candidate
    is feasible, according to the simple bound estimation.

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

        # State administration
        initial_score = self._initial_solution.score
        initial_slack = self._initial_solution.slack
        initial_list = self._initial_solution.eventlist
        simple_slack = self._data.simple_initiate()
        simple_score = self._initial_solution.score - \
            get_slack_value(initial_slack) + get_slack_value(simple_slack)
        self._current_solution = SearchSpaceStateJumpPointMultiScore()
        self._best_solution = \
            SearchSpaceStateJumpPointMultiScore(initial_list.copy())
        self._initial_solution = \
            SearchSpaceStateJumpPointMultiScore(initial_list)
        self._current_solution.score = initial_score
        self._current_solution.slack = initial_slack
        self._current_solution.simple_slack = simple_slack
        self._current_solution.simple_score = simple_score
        self._best_solution.score = initial_score
        self._best_solution.slack = initial_slack
        self._best_solution.simple_slack = simple_slack
        self._best_solution.simple_score = simple_score
        self._initial_solution.score = initial_score
        self._initial_solution.slack = initial_slack
        self._initial_solution.simple_slack = simple_slack
        self._initial_solution.simple_score = simple_score
        self._label = f"jumppoint-mix-{params['dist'].__qualname__}"

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_lp_score = self._current_solution.lp_score
        current_simple_score = self._current_solution.simple_score
        current_base = self._current_solution.simple_score - \
            self._current_solution.simple_slack_value

        for event_id in att_ord:
            if event_id in self._tabulist:
                continue
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Precompute acceptance threshold
            max_acc_diff = -1 * math.log(np.random.random()) * temperature

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > current_simple_score + max_acc_diff:
                continue

            new_simple_slack = self._data.simple_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_simple_slack)

            if np.isinf(get_slack_value(new_simple_slack)) or \
               new_score > current_simple_score + max_acc_diff:
                self._data.simple_update_apply(
                    idx, new_idx, success=False
                )
                continue

            # Compute LP both for the found-feasible and found-best case
            # Instance is up-to-date at this point
            new_lp_score = - 1
            if np.isclose(get_slack_value(new_simple_slack), 0.0) or \
               new_score < self._best_solution.simple_score:
                lp_slack = self._data.lp_initiate()
                new_lp_score = new_base + get_slack_value(lp_slack)

            # Reconsider accepting based on LP score
            if np.isclose(get_slack_value(new_simple_slack), 0.0) and \
               (np.isinf(get_slack_value(lp_slack)) or
                new_lp_score > max(current_lp_score,
                current_simple_score) + max_acc_diff
               ):
                self._data.simple_update_apply(
                    idx, new_idx, success=False
                )
                continue

            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.simple_update_apply(idx, new_idx, success=True)
            self._current_solution.simple_score = new_score
            self._current_solution.simple_slack = new_simple_slack
            self._current_solution.lp_score = new_lp_score
            if new_lp_score > 0:
                self._current_solution.lp_slack = lp_slack
            self._add_to_tabu_list(event_id)

            if new_lp_score > 0 and \
               new_lp_score < self._best_solution.lp_score:
                self._best_solution.eventlist = \
                    self._data.eventlist.copy()
                self._best_solution.simple_score = new_score
                self._best_solution.simple_slack = new_simple_slack
                self._best_solution.lp_score = new_lp_score
                self._best_solution.lp_slack = lp_slack
            accepted = True
            break

        improved = accepted and (new_score < current_simple_score or
                                 (new_lp_score > 0 and
                                  new_lp_score < current_lp_score))

        return improved, (self._current_solution if accepted else None)


class JumpPointSearchSpaceMixMinimal(JumpPointSearchSpaceMix):
    """Search space implementing the search strategy using the LP and the
    simple bound estimations. The LP will only be computed if a potential
    new best solution is found.

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
        self._label = f"jumppoint-mix-minimal-{params['dist'].__qualname__}"

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_lp_score = self._current_solution.lp_score
        current_simple_score = self._current_solution.simple_score
        current_base = self._current_solution.simple_score - \
            self._current_solution.simple_slack_value

        for event_id in att_ord:
            if event_id in self._tabulist:
                continue
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Precompute acceptance threshold
            max_acc_diff = -1 * math.log(np.random.random()) * temperature

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > current_simple_score + max_acc_diff:
                continue

            new_simple_slack = self._data.simple_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_simple_slack)

            if np.isinf(get_slack_value(new_simple_slack)) or \
               new_score > current_simple_score + max_acc_diff:
                self._data.simple_update_apply(
                    idx, new_idx, success=False
                )
                continue

            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.simple_update_apply(idx, new_idx, success=True)
            self._current_solution.simple_score = new_score
            self._current_solution.simple_slack = new_simple_slack
            self._add_to_tabu_list(event_id)

            if new_score < self._best_solution.simple_score:
                # Compute LP
                lp_slack = self._data.lp_initiate()
                new_lp_score = new_base + get_slack_value(lp_slack)
                self._current_solution.lp_slack = lp_slack
                self._current_solution.lp_score = new_lp_score
                if new_lp_score < self._best_solution.lp_score:
                    self._best_solution.eventlist = \
                        self._data.eventlist.copy()
                    self._best_solution.simple_score = new_score
                    self._best_solution.simple_slack = new_simple_slack
                    self._best_solution.lp_score = new_lp_score
                    self._best_solution.lp_slack = lp_slack
            accepted = True
            break

        improved = accepted and new_score < current_simple_score

        return improved, (self._current_solution if accepted else None)


class JumpPointSearchSpaceSimple(JumpPointSearchSpaceMixMinimal):
    """Search space implementing the search strategy using the simple
    bound estimations. The LP will only be computed for the best found
    simple solution, which will be improved with an LP-based hill
    climber.

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
        self._label = f"jumppoint-simple-{params['dist'].__qualname__}"

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_lp_score = self._current_solution.lp_score
        current_simple_score = self._current_solution.simple_score
        current_base = self._current_solution.simple_score - \
            self._current_solution.simple_slack_value

        for event_id in att_ord:
            if event_id in self._tabulist:
                continue
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Precompute acceptance threshold
            max_acc_diff = -1 * math.log(np.random.random()) * temperature

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > current_simple_score + max_acc_diff:
                continue

            new_simple_slack = self._data.simple_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_simple_slack)

            if np.isinf(get_slack_value(new_simple_slack)) or \
               new_score > current_simple_score + max_acc_diff:
                self._data.simple_update_apply(
                    idx, new_idx, success=False
                )
                continue

            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.simple_update_apply(idx, new_idx, success=True)
            self._current_solution.simple_score = new_score
            self._current_solution.simple_slack = new_simple_slack
            self._add_to_tabu_list(event_id)

            if new_score < self._best_solution.simple_score:
                self._best_solution.eventlist = \
                    self._data.eventlist.copy()
                self._best_solution.simple_score = new_score
                self._best_solution.simple_slack = new_simple_slack
            accepted = True
            break

        improved = accepted and new_score < current_simple_score

        return improved, (self._current_solution if accepted else None)

    def solve_and_write_best(self):
        """Solve the LP for the best found solution, and if it is not
        feasible: climb the hill.
        """
        self._data.eventlist = self._best_solution.eventlist
        self._data.eventmap = construct_event_mapping(
            self._best_solution.eventlist, (self.njobs, self.kextra + 2)
        )
        slack = self._data.lp_initiate()
        base = self._data.base_initiate()
        orig_base = base
        orig_slack = get_slack_value(slack)
        if get_slack_value(slack) > 0.0:
            # We need to get to a feasible solution.
            # Initiate a SA-LP search space with temperature 0.
            improved = True
            self._current_solution.score = base + get_slack_value(slack)
            self._current_solution.slack = slack
            while improved:
                improved = self._get_neighbor_hill_climb()

            slack = self._data.lp_initiate()
            base = self._data.base_initiate()
            
            with open(os.path.join(self._logdir, "fix-signaler.txt"),
                      "w") as sol:
                sol.write(f"Improved best from base: {orig_base:.4f} + slack:"
                          f" {orig_slack:.4f} to base: {base:.4f} + slack:"
                          f"{get_slack_value(slack)}")
        # Print solution to file
        with open(os.path.join(self._logdir, "solution.csv"), "w") as sol:
            sol.write(self._data._lp_model.get_solution_csv())

    def _get_neighbor_hill_climb(self):
        """Template method for finding candidate solutions.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_score = self._current_solution.score
        current_base = self._current_solution.score - \
            self._current_solution.slack_value

        for event_id in att_ord:
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > current_score:
                continue

            new_slack = self._data.lp_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_slack)

            if np.isinf(get_slack_value(new_slack)) or \
               new_score >= current_score:
                self._data.lp_update_apply(idx, new_idx, success=False)
                continue

            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.lp_update_apply(idx, new_idx, success=True)
            self._current_solution.score = new_score
            self._current_solution.slack = new_slack
            accepted = True
            break

        return accepted


class JumpPointSearchSpaceSwitch(JumpPointSearchSpaceMix):
    """Search space implementing the search strategy using the LP and the
    simple bound estimations. Will use the simple estimation, until it
    reports a penalty term of 0.0. Then, it switches over to an LP-based
    search.

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
        self._label = f"jumppoint-switch-{params['dist'].__qualname__}"
        self._use_simple = True

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        if self._use_simple:
            return self._get_neighbor_simple(temperature)
        else:
            return self._get_neighbor_lp(temperature)

    def _get_neighbor_simple(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_simple_score = self._current_solution.simple_score
        current_base = self._current_solution.simple_score - \
            self._current_solution.simple_slack_value

        for event_id in att_ord:
            if event_id in self._tabulist:
                continue
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Precompute acceptance threshold
            max_acc_diff = -1 * math.log(np.random.random()) * temperature

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > current_simple_score + max_acc_diff:
                continue

            new_simple_slack = self._data.simple_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_simple_slack)

            if np.isinf(get_slack_value(new_simple_slack)) or \
               new_score > current_simple_score + max_acc_diff:
                self._data.simple_update_apply(
                    idx, new_idx, success=False
                )
                continue
            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.simple_update_apply(idx, new_idx, success=True)
            self._current_solution.simple_score = new_score
            self._current_solution.simple_slack = new_simple_slack
            self._add_to_tabu_list(event_id)

            if new_score < self._best_solution.simple_score:
                self._best_solution.eventlist = \
                    self._data.eventlist.copy()
                self._best_solution.simple_score = new_score
                self._best_solution.simple_slack = new_simple_slack
            accepted = True

            # Compute LP both for the found-feasible case
            if np.isclose(get_slack_value(new_simple_slack), 0.0):
                # Compute lp score for best solution
                curr_list = self._data.eventlist
                curr_map = self._data.eventmap
                self._data.eventlist = self._best_solution.eventlist
                self._data.eventmap = construct_event_mapping(
                    self._best_solution.eventlist,
                    (self.njobs, self.kextra + 2)
                )
                self._best_solution.lp_slack  = self._data.lp_initiate()
                self._best_solution.lp_score = self._data.base_initiate() + \
                    self._best_solution.lp_slack_value
                self._data.eventlist = curr_list
                self._data.eventmap = curr_map
                self._current_solution.lp_slack = self._data.lp_initiate()
                self._current_solution.lp_score = new_base + \
                    self._current_solution.lp_slack_value
                self._use_simple = False
            break

        improved = accepted and (new_score < current_simple_score)

        return improved, (self._current_solution if accepted else None)

    def _get_neighbor_lp(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # Determine order. Only plannable events need to be considered.
        att_ord = np.random.permutation(
            np.arange(self.nplannable)
        )
        accepted = False
        current_score = self._current_solution.score
        current_base = self._current_solution.score - \
            self._current_solution.slack_value

        for event_id in att_ord:
            if event_id in self._tabulist:
                continue
            job = math.floor(event_id / 2)
            etype = event_id % 2
            idx = self._data.eventmap[job, etype]
            llimit, rlimit = self._find_limits(event_id, idx)

            new_idx = self._params['dist'](idx, llimit, rlimit)

            if new_idx < 0:
                continue

            # Precompute acceptance threshold
            max_acc_score = -1 * math.log(np.random.random()) * \
                temperature + self._current_solution.score

            # Compute new base score & LP slack
            cost_diff = 0
            if etype == 1:
                cost_diff, apply_base = \
                    self._data.base_update_compute(idx, new_idx - idx)
            new_base = current_base + cost_diff

            # We do not have to compute the slack if the base is enough
            # to reject.
            if new_base > max_acc_score:
                continue

            new_slack = self._data.lp_update_compute(idx, new_idx)
            new_score = new_base + get_slack_value(new_slack)

            if np.isinf(get_slack_value(new_slack)) or \
               new_score > max_acc_score:
                self._data.lp_update_apply(idx, new_idx, success=False)
                continue

            # If we accept
            if etype == 1:
                self._data.base_update_apply(cost_diff, apply_base)
            self._data.lp_update_apply(idx, new_idx, success=True)
            self._current_solution.score = new_score
            self._current_solution.slack = new_slack
            self._add_to_tabu_list(event_id)

            if new_score < self._best_solution.score:
                self._best_solution.eventlist = self._data.eventlist.copy()
                self._best_solution.score = new_score
                self._best_solution.slack = new_slack
            accepted = True
            break

        improved = accepted and new_score < current_score

        return improved, (self._current_solution if accepted else None)


class JumpPointSearchSpaceTest(JumpPointSearchSpace):
    """Search space used to check the correctness of the implementation
    of a number of helper functions.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.

    Notes
    -----
    This class abuses the implementation of the search space, for the
    purpose of checking the speed and correctness of parts of the
    implementation. DO NOT use this as a template for another search
    space subclass.
    """
    def __init__(self, instance, params=None):
        super().__init__(instance, params=params)
        self._label = "test"

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """
        # We will only do one call to get_neighbor.
        orig_eventlist = self._data.eventlist.copy()
        orig_eventmap = self._data.eventmap.copy()
        curr_score = self.initial.score - self.initial.slack_value
        curr_simple_slack = self._data.simple_initiate()
        with open(os.path.join(self._logdir,
                               "compare_score.csv"), "w") as f:
            f.write(
                'base_score;lp_score;flow_score;simple_score;base_time;'
                'base_update_time;lp_time;lp_update_time;flow_time;'
                'simple_time;simple_update_time\n'
            )
            # For each iteration we will
            for i in range(1000):
                # Generate a new candidate solution.
                new_idx = -1
                while new_idx < 0:
                    orig_job = np.random.randint(self.njobs)
                    orig_etype = np.random.randint(2)
                    orig_id = 2 * orig_job + orig_etype
                    orig_idx = self._data.eventmap[orig_job, orig_etype]
                    llimit, rlimit = self._find_limits(orig_id, orig_idx)
                    new_idx = dists.linear(orig_idx, llimit, rlimit)

                # Compute the updated base score
                t_0 = time.perf_counter()
                if orig_etype == 1:
                    base_diff, _ = \
                        self._data.base_update_compute(orig_idx,
                                                       new_idx - orig_idx)
                else:
                    base_diff = 0
                t_1 = time.perf_counter()
                t_base_update = t_1 - t_0
                new_score = curr_score + base_diff

                # Compute the updated LP score, apply: false
                t_2 = time.perf_counter()
                slack_lp_update = get_slack_value(
                    self._data.lp_update_compute(orig_idx, new_idx)
                )
                t_3 = time.perf_counter()
                t_lp_update = t_3 - t_2
                # Extract and keep eventlist and eventmap
                lp_eventlist = self._data.eventlist.copy()
                lp_eventmap = self._data.eventmap.copy()

                self._data.lp_update_apply(orig_idx, new_idx, success=False)
                lp_rev_eventlist = self._data.eventlist.copy()
                lp_rev_eventmap = self._data.eventmap.copy()

                # Compute the updated simple score, apply: false
                t_4 = time.perf_counter()
                slack_simple_update = get_slack_value(curr_simple_slack) + \
                    get_slack_value(
                        self._data.simple_update_compute(orig_idx, new_idx)
                    )
                t_5 = time.perf_counter()
                t_simple_update = t_5 - t_4
                # Extract and keep eventlist and eventmap
                simple_eventlist = self._data.eventlist.copy()
                simple_eventmap = self._data.eventmap.copy()

                self._data.simple_update_apply(orig_idx, new_idx,
                                               success=False)
                simple_rev_eventlist = self._data.eventlist.copy()
                simple_rev_eventmap = self._data.eventmap.copy()

                # Compute the updated instance
                self._data.instance_update(orig_idx, new_idx)
                # Extract and keep eventlist and eventmap
                inst_eventlist = self._data.eventlist.copy()
                inst_eventmap = self._data.eventmap.copy()

                # Compute initial base score
                t_6 = time.perf_counter()
                base_init = self._data.base_initiate()
                t_7 = time.perf_counter()
                t_base_init = t_7 - t_6

                # Compute initial LP score
                t_8 = time.perf_counter()
                new_slack = self._data.lp_initiate()
                lp_slack_init = get_slack_value(new_slack)
                t_9 = time.perf_counter()
                t_lp_init = t_9 - t_8

                # Compute initial flow score
                t_10 = time.perf_counter()
                flow_slack_init = get_slack_value(self._data.flow_initiate())
                t_11 = time.perf_counter()
                t_flow_init = t_11 - t_10

                # Compute initial simple score
                t_12 = time.perf_counter()
                new_simple_slack = self._data.simple_initiate()
                simple_slack_init = get_slack_value(
                    new_simple_slack
                )
                t_13 = time.perf_counter()
                t_simple_init = t_13 - t_12

                # Assert equalities of solution
                assert np.array_equal(lp_eventlist, simple_eventlist), \
                    "LP and simple update result in different eventlist"
                assert np.array_equal(lp_eventlist, inst_eventlist), \
                    "LP and instance update result in different eventlist"
                assert np.array_equal(simple_eventlist, inst_eventlist), \
                    "Simple and instance update result in different eventlist"

                assert np.array_equal(lp_eventmap, simple_eventmap), \
                    "LP and simple update result in different eventmap"
                assert np.array_equal(lp_eventmap, inst_eventmap), \
                    "LP and instance update result in different eventmap"
                assert np.array_equal(simple_eventmap, inst_eventmap), \
                    "Simple and instance update result in different eventmap"

                assert np.array_equal(lp_rev_eventlist, orig_eventlist), \
                    ("Reverting LP update does not result in original"
                     " eventlist")
                assert np.array_equal(simple_rev_eventlist, orig_eventlist), \
                    ("Reverting simple update does not result in original"
                     " eventlist")

                assert np.array_equal(lp_rev_eventmap, orig_eventmap), \
                    "Reverting LP update does not result in original eventmap"
                assert np.array_equal(simple_rev_eventmap, orig_eventmap), \
                    ("Reverting simple update does not result in original"
                     " eventmap")

                assert not np.array_equal(inst_eventlist, orig_eventlist), \
                    "Eventlist is not updated at all."
                assert not np.array_equal(inst_eventmap, orig_eventmap), \
                    "Eventmap is not updated at all."

                # Assert equalities of scores
                assert np.isclose(new_score, base_init), \
                    ("Recompute and update of base score give different"
                     " results.")
                assert np.isclose(slack_lp_update, lp_slack_init), \
                    "Recompute and update of LP slack give different results."
                assert np.isclose(slack_simple_update, simple_slack_init), \
                    ("Recompute and update of simple slack give different"
                     " results.")

                # Collect timing and scores in log file
                f.write(
                    f'{new_score:.2f};{slack_lp_update:.2f};'
                    f'{flow_slack_init:.2f};{slack_simple_update:.2f};'
                    f'{t_base_init:.6f}; {t_base_update:.6f};{t_lp_init:.6f};'
                    f'{t_lp_update:.6f};{t_flow_init:.6f};'
                    f'{t_simple_init:.6f};{t_simple_update:.6f}\n'
                )

                # Decide on keeping or reverting
                if not np.isinf(lp_slack_init):
                    curr_score = new_score
                    curr_simple_slack = new_simple_slack
                    orig_eventlist = inst_eventlist
                    orig_eventmap = inst_eventmap
                else:
                    self._data.instance_update(new_idx, orig_idx)
                    self._data.lp_initiate()
                    self._data.simple_initiate()

        return False, None


class JumpPointSearchSpaceTestLP(JumpPointSearchSpace):
    """Search space used to check the correctness of the implementation
    of the LP

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data.
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.

    Notes
    -----
    This class abuses the implementation of the search space, for the
    purpose of checking the speed and correctness of parts of the
    implementation. DO NOT use this as a template for another search
    space subclass.
    """
    def __init__(self, instance, params=None):
        super().__init__(instance, params=params)
        # Create two more data instances
        self._data2 = JumpPointSearchSpaceData(copy.deepcopy(instance))
        self._data3 = JumpPointSearchSpaceData(copy.deepcopy(instance))
        self._inst_store = copy.deepcopy(instance)
        self._label = "test2"

    def get_neighbor(self, temperature):
        """Template method for finding candidate solutions.

        Parameters
        ----------
        temperature : float
            Current annealing temperature, used in determining if a
            candidate with a lower objective should be accepted.

        Returns
        -------
        bool
            Whether an improvement was found.
        SearchSpaceState
            New state for the search to continue with.
        """    
        # For each iteration we will
        prev_lp = 0
        while True:
            # Generate a new candidate solution.
            new_idx = -1
            while new_idx < 0:
                orig_job = np.random.randint(self.njobs)
                orig_etype = np.random.randint(2)
                orig_id = 2 * orig_job + orig_etype
                orig_idx = self._data.eventmap[orig_job, orig_etype]
                llimit, rlimit = self._find_limits(orig_id, orig_idx)
                new_idx = dists.linear(orig_idx, llimit, rlimit)

            # Compute the updated LP score, apply: true
            slack_lp_update = get_slack_value(
                self._data.lp_update_compute(orig_idx, new_idx)
            )
            self._data.lp_update_apply(orig_idx, new_idx, success=True)
            update_lp_eventlist = self._data.eventlist.copy()
            update_lp_eventmap = self._data.eventmap.copy()

            # Compute the updated instance
            self._data2.instance_update(orig_idx, new_idx)
            slack_inst = get_slack_value(self._data2.lp_initiate())
            # Extract and keep eventlist and eventmap
            inst_eventlist = self._data2.eventlist.copy()
            inst_eventmap = self._data2.eventmap.copy()

            # Compute initial LP score
            self._data3.eventlist = update_lp_eventlist
            self._data3.eventmap = construct_event_mapping(
                update_lp_eventlist, (self.njobs, self.kextra + 2)
            )
            slack_lp = get_slack_value(self._data3.lp_initiate())

            # Assert equalities of solution
            assert np.array_equal(update_lp_eventlist, inst_eventlist), \
                (f"Update of eventlist went wrong: {update_lp_eventlist} -"
                 f" {inst_eventlist}")
            assert np.array_equal(update_lp_eventmap, inst_eventmap), \
                (f"Update of eventmap went wrong: {update_lp_eventmap} -"
                 f" {inst_eventmap}")
            assert np.array_equal(update_lp_eventmap, self._data3.eventmap), \
                (f"Generating new eventmap went wrong: {update_lp_eventmap} -"
                 f" {self._data3.eventmap}")

            if not np.isclose(slack_lp_update, slack_inst) or \
               not np.isclose(slack_inst, slack_lp):
                print(f"Slack data1: {slack_lp_update}\n"
                      f"Slack data2: {slack_inst}\n"
                      f"Slack data3: {slack_lp}\n"
                      f"Slack data2 previous: {prev_lp}")
                self._data._lp_model._problem.export_as_lp(
                    path=os.path.join(self._logdir, "lp_data1.lp")
                )
                self._data2._lp_model._problem.export_as_lp(
                    path=os.path.join(self._logdir, "lp_data2.lp")
                )
                self._data3._lp_model._problem.export_as_lp(
                    path=os.path.join(self._logdir, "lp_data3.lp")
                )
                with open(os.path.join(self._logdir, "solution1.csv"), "w") as sol:
                    sol.write(self._data._lp_model.get_solution_csv())
                with open(os.path.join(self._logdir, "solution2.csv"), "w") as sol:
                    sol.write(self._data2._lp_model.get_solution_csv())
                with open(os.path.join(self._logdir, "solution3.csv"), "w") as sol:
                    sol.write(self._data3._lp_model.get_solution_csv())
                np.savetxt(os.path.join(self._logdir, "eventlist1.txt"), update_lp_eventlist, fmt='%.0f')
                np.savetxt(os.path.join(self._logdir, "eventlist2.txt"), inst_eventlist, fmt='%.0f')
                raise RuntimeError()

            # Decide on keeping or reverting
            if np.isinf(slack_lp):
                # Restart
                self._data = JumpPointSearchSpaceData(
                    copy.deepcopy(self._inst_store)
                )
                self._data2 = JumpPointSearchSpaceData(
                    copy.deepcopy(self._inst_store)
                )
                self._data3 = JumpPointSearchSpaceData(
                    copy.deepcopy(self._inst_store)
                )
            prev_lp = slack_inst

        return False, None


class SearchSpaceStateJumpPoint(SearchSpaceState):
    """Class object describing a state in the search space for a local
    search approach.

    Parameters
    ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
    """
    def __init__(self, eventlist=None):
        self._eventlist = eventlist
        self._score = np.inf
        self._slack = []
        self._slack_value = np.inf

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
        self._slack_value = get_slack_value(slack)

    @property
    def slack_value(self):
        """float : slack value"""
        return self._slack_value

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


class SearchSpaceStateJumpPointMultiScore(SearchSpaceStateJumpPoint):
    """Class object describing a state in the search space for a local
    search approach. Keeping track of multiple scoring approaches.

    Parameters
    ----------
    """
    def __init__(self, eventlist=None):
        super().__init__(eventlist)
        self._simple_score = np.inf
        self._simple_slack = []
        self._simple_slack_value = np.inf

    @property
    def simple_score(self):
        """float : Score of the represented solution using lower bound
        estimations on the penalty term.
        """
        return self._simple_score

    @simple_score.setter
    def simple_score(self, score):
        """Manually set the value of the simple score attribute for this
        state. Use with caution.
        """
        self._simple_score = score

    @property
    def simple_slack(self):
        """list of tuple : List of tuples with in the first position a
        string identifying the type of slack variable, in second position
        the summed value of these variables (float) and in third position
        the unit weight of these variables in the objective. As estimated
        by the lower bound computations.
        """
        return self._simple_slack

    @simple_slack.setter
    def simple_slack(self, slack):
        """Manually set the value of the simple slack attribute for this
        state. Use with caution."""
        self._simple_slack = slack
        self._simple_slack_value = get_slack_value(slack)

    @property
    def simple_slack_value(self):
        """float : slack value"""
        return self._simple_slack_value

    @property
    def lp_score(self):
        """float : Score of the represented solution (objective value of
        the LP referenced in `model`).
        """
        return self._score

    @lp_score.setter
    def lp_score(self, score):
        """Manually set the value of the score attribute for this state.
        Use with caution.
        """
        self._score = score

    @property
    def lp_slack(self):
        """list of tuple : List of tuples with in the first position a
        string identifying the type of slack variable, in second position
        the summed value of these variables (float) and in third position
        the unit weight of these variables in the objective.
        """
        return self._slack

    @lp_slack.setter
    def lp_slack(self, slack):
        """Manually set the value of the slack attribute for this state.
        Use with caution."""
        self._slack = slack
        self._slack_value = get_slack_value(slack)

    @property
    def lp_slack_value(self):
        """float : slack value"""
        return self._slack_value
