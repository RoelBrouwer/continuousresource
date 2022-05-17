import copy
import math
import numpy as np
import time

from continuousresource.simulatedannealing.utils \
    import sanitize_search_space_params


class SearchSpace():
    """Wrapper object for information about the search space.

    Parameters
    ----------
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
    """
    def __init__(self, params=None):
        self._params = sanitize_search_space_params(params)
        self._current_solution = None
        self._best_solution = None
        self._label = "superclass"

    @property
    def best(self):
        """The current best solution does not maintain a link to its
        associated model, but its event order and score are accessible
        through this state object.
        """
        return self._best_solution

    @property
    def current(self):
        return self._current_solution

    @property
    def name(self):
        """String used to identify the search space in reporting."""
        return self._label

    @property
    def precedences(self):
        """Two dimensional (|E| x |E|) array listing (inferred)
        precedence relations between events. If the entry at position
        [i, j] is True, this means that i has to come before j.
        """
        return self._precedences

    # @precedences.setter
    # def precedences(self, p)
    #     self._precedences = p

    def generate_initial_solution(self, model_class, eventlist,
                                  *args, **kwargs):
        """Generates an initial solution within the search space and sets
        the value of self._current_solution accordingly.

        Parameters
        ----------
        model_class : str
            Class name of the class implementing the state model.
        eventlist : ndarray
            Two-dimensional (|E| x 2) array representing the events in the
            problem, where the first column contains an integer indicating
            the event type (0 for start, 1 for completion) and the second
            column the associated job ID.
        *args :
            Should contain exactly the (non-keyword) arguments required
            by the constructor of `model_class`, other than `eventlist`.
        **kwargs :
            Should contain exactly the keyword arguments required by the
            constructor of `model_class`, other than `eventlist`.

        Returns
        -------
        float
            Timing (in seconds) of the actual initial solution generation,
        """
        # For now the initial solution is just the eventlist exactly as
        # presented.
        initial = SearchSpaceState(self, eventlist)
        initial.create_model(model_class, eventlist, *args, **kwargs)
        self._precedences = \
            initial.model.find_precedences(self._params['infer_precedence'])
        t_start = time.perf_counter()
        if self._params['start_solution'] == "greedy":
            initial.model.generate_initial_solution()
        elif self._params['start_solution'] == "random":
            initial.model.generate_random_solution(self._precedences)
        t_end = time.perf_counter()
        initial.model.initialize_problem()
        initial.eventorder = initial.model.event_list
        # print(initial.model.problem.lp_string)
        self._current_solution = initial
        initial.compute_score()
        self._best_solution = copy.copy(initial)
        self._best_solution.score = initial.score
        self._best_solution.slack = initial.slack
        return t_end - t_start, initial.score

    def compute_search_space_reductions(self):
        """Eliminate parts of the search space by looking at implicit
        precedence constraints in the data.
        """
        pass

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
        # TODO: Currently only implements the swap-neighborhood operator.
        accepted = False
        fail_count = 0

        while not accepted:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_swap()
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                # print(f"accepted score: {new_state.score}")
                if new_state.score < self._best_solution.score:
                    # print(f"best score: {new_state.score}")
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                    # new_state.model.print_solution()
                self._current_solution = new_state
                accepted = True
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_swap_revert(new_state, revert_info)

                if fail_count > 200:
                    return fail_count, None
                fail_count += 1

        return fail_count, new_state

    def get_neighbor_swap(self):
        """Finds candidate solutions by swapping adjacent event pairs.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        int
            ID of the first swapped event
        """
        swap_id = np.random.randint(
            len(self._current_solution.model.event_list) - 1
        )
        while (self._current_solution.model.event_list[swap_id, 1] ==
               self._current_solution.model.event_list[swap_id + 1, 1]):
            swap_id = np.random.randint(
                len(self._current_solution.model.event_list) - 1
            )

        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model
        new_state.model.update_swap_neighbors(swap_id)
        new_state.eventorder = copy.copy(
            self._current_solution.model.event_list
        )
        new_state.compute_score()
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
        new_state.model.update_swap_neighbors(swap_id)

    def get_neighbor_move(self):
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
        retry = True
        count = 0
        orig_idx = 0
        new_idx = 0

        while retry and count < len(self._current_solution.model.event_list):
            retry = False
            orig_idx = np.random.randint(
                len(self._current_solution.model.event_list)
            )
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
                retry = True
                count += 1
            else:
                new_idx = orig_idx
                while new_idx == orig_idx:
                    new_idx = np.random.randint(
                        llimit + 1,
                        rlimit
                    )

        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model
        new_state.model.update_move_event(orig_idx, new_idx)
        new_state.eventorder = copy.copy(
            self._current_solution.model.event_list
        )
        new_state.compute_score()
        return new_state, (orig_idx, new_idx)

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
        new_state.model.update_move_event(idcs[1], idcs[0])

    def get_neighbor_move_pair(self):
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
        # TODO
        retry = True
        count = 0
        idx1 = 0
        idx2 = 0
        offset = 0

        while retry and count < len(self._current_solution.model.event_list):
            retry = False
            idx1 = np.random.randint(
                len(self._current_solution.model.event_list)
            )
            job = self._current_solution.model.event_list[idx1, 1]
            etype1 = self._current_solution.model.event_list[idx1, 0]
            idx2 = self._current_solution.model.event_map[job, 1 - etype1]

            # Determine closest predecessor with a precedence relation
            llimit1 = idx1
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit1, 1] * 2
                       + self._current_solution.model.event_list[llimit1, 0],
                       job * 2 + etype1]):
                llimit1 -= 1
                if llimit1 == -1:
                    break
            llimit2 = idx2
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit2, 1] * 2
                       + self._current_solution.model.event_list[llimit2, 0],
                       job * 2 + 1 - etype1]):
                llimit2 -= 1
                if llimit2 == -1:
                    break

            # Determine closest successor with a precedence relation
            rlimit1 = idx1
            while not (self._precedences[job * 2 + etype1,
                       self._current_solution.model.event_list[rlimit1, 1] * 2
                       + self._current_solution.model.event_list[rlimit1,
                                                                 0]]):
                rlimit1 += 1
                if rlimit1 == len(self._current_solution.model.event_list):
                    break
            rlimit2 = idx2
            while not (self._precedences[job * 2 + 1 - etype1,
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
                retry = True
                count += 1
            else:
                while offset == 0:
                    offset = np.random.randint(
                        llimit + 1,
                        rlimit
                    )

        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model
        new_state.model.update_move_pair(idx1, idx2, offset)
        new_state.eventorder = copy.copy(
            self._current_solution.model.event_list
        )
        new_state.compute_score()
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
        new_state.model.update_move_pair(idcs[0] + idcs[2], idcs[1] + idcs[2],
                                         -1 * idcs[2])

    def get_neighbor_simultaneous(self):
        # Find candidates by looking at the adjacent pairs that were
        # scheduled simultaneously.
        sched = self._current_solution.model.get_schedule()
        pairs = np.array([
            i for i in (range(len(sched - 1)))
            if np.isclose(sched[i], sched[i + 1])
        ])

        if len(pairs) < 1:
            raise NotImplementedError("Neighbors can only be obtained by"
                                      " swapping simultaneous events")
        else:
            # Consider pairs in random order
            np.random.shuffle(pairs)
            for first_idx in pairs:
                new_state = copy.copy(self._current_solution)
                new_state.model = self._current_solution.model
                new_state.model.update_swap_neighbors(first_idx)
                new_state.compute_score()
                if new_state.score < self._best_solution.score:
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                    # new_state.model.print_solution()
                # if new_state.score > -1:
                return new_state

    def random_walk(self, no_swaps=100):
        """Performs a random walk from the current solution by swapping
        `no_swaps` random pairs of adjacent events. These swaps are not
        executed simultaneously, but one after another. So, the second
        swap is performed on the event order that results after the first
        swap, and so on.

        Parameters
        ----------
        no_swaps : int
            Length of the random walk, i.e. the number of swaps
            performed.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with
        """
        new_state = copy.copy(self._current_solution)
        new_state.model = self._current_solution.model

        for i in range(no_swaps):
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


class SearchSpaceSwap(SearchSpace):
    """Search space that only uses the neighbor swap operator.

    Parameters
    ----------
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
    """
    def __init__(self, params=None):
        super().__init__(params=params)
        self._label = "onlyswap"

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
        accepted = False
        fail_count = 0

        while not accepted:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_swap()
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                accepted = True
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_swap_revert(new_state, revert_info)

                if fail_count > 200:
                    return fail_count, None
                fail_count += 1

        return fail_count, new_state


class SearchSpaceMove(SearchSpace):
    """Search space that only uses the single event move operator.

    Parameters
    ----------
    params : Dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
    """
    def __init__(self, params=None):
        super().__init__(params=params)
        self._label = "onlymovesingle"

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
        accepted = False
        fail_count = 0

        while not accepted:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_move()
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                accepted = True
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_move_revert(new_state, revert_info)

                if fail_count > 200:
                    return fail_count, None
                fail_count += 1

        return fail_count, new_state


class SearchSpaceMovePair(SearchSpace):
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
        super().__init__(params=params)
        self._label = "onlymovepair"

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
        accepted = False
        fail_count = 0

        while not accepted:
            # Obtain a new state
            new_state, revert_info = self.get_neighbor_move_pair()
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                accepted = True
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                self.get_neighbor_move_pair_revert(new_state, revert_info)

                if fail_count > 200:
                    return fail_count, None
                fail_count += 1

        return fail_count, new_state


class SearchSpaceCombined(SearchSpace):
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
        super().__init__(params=params)
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
        accepted = False
        fail_count = 0

        while not accepted:
            # Obtain a new state
            frac = np.random.random()
            if frac < self._fracs["swap"]:
                new_state, revert_info = self.get_neighbor_swap()
            elif frac < self._fracs["swap"] + self._fracs["move"]:
                new_state, revert_info = self.get_neighbor_move()
            else:
                new_state, revert_info = self.get_neighbor_move_pair()
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                if new_state.score < self._best_solution.score:
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                    self._best_solution.slack = new_state.slack
                self._current_solution = new_state
                accepted = True
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                if frac < self._fracs["swap"]:
                    self.get_neighbor_swap_revert(new_state, revert_info)
                elif frac < self._fracs["swap"] + self._fracs["move"]:
                    self.get_neighbor_move_revert(new_state, revert_info)
                else:
                    self.get_neighbor_move_pair_revert(new_state, revert_info)

                if fail_count > 200:
                    return fail_count, None
                fail_count += 1

        return fail_count, new_state


class SearchSpaceHillClimb(SearchSpace):
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
            # We cannot try the last one.
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
        # Determine order
        att_ord = np.random.permutation(
            # We cannot try the last one.
            np.arange(len(self._current_solution.model.job_properties))
        )

        for job_idx in att_ord:
            idx1 = self._current_solution.model.event_map[job_idx, 0]
            idx2 = self._current_solution.model.event_map[job_idx, 1]

            # Determine closest predecessor with a precedence relation
            llimit1 = idx1
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit1, 1] * 2
                       + self._current_solution.model.event_list[llimit1, 0],
                       job * 2 + etype1]):
                llimit1 -= 1
                if llimit1 == -1:
                    break
            llimit2 = idx2
            while not (self._precedences[
                       self._current_solution.model.event_list[llimit2, 1] * 2
                       + self._current_solution.model.event_list[llimit2, 0],
                       job * 2 + 1 - etype1]):
                llimit2 -= 1
                if llimit2 == -1:
                    break

            # Determine closest successor with a precedence relation
            rlimit1 = idx1
            while not (self._precedences[job * 2 + etype1,
                       self._current_solution.model.event_list[rlimit1, 1] * 2
                       + self._current_solution.model.event_list[rlimit1,
                                                                 0]]):
                rlimit1 += 1
                if rlimit1 == len(self._current_solution.model.event_list):
                    break
            rlimit2 = idx2
            while not (self._precedences[job * 2 + 1 - etype1,
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


class SearchSpaceState():
    """Class object describing a state in the search space for a local
    search approach.

    Parameters
    ----------
    belongs_to : SearchSpace
        Search space wherein this object represents a state.
    """
    def __init__(self, belongs_to, eventorder):
        # TODO: we may not need to include the eventorder here at all:
        # access via self._lp_model.event_list
        self._searchspace = belongs_to
        self._eventorder = eventorder
        self._score = np.inf
        self._slack = []
        self._lp_model = None
        self._schedule = None

    def __copy__(self):
        """Override copy method to make copies of some attributes and
        reset others.
        """
        new_state = SearchSpaceState(self._searchspace,
                                     copy.copy(self._eventorder))
        new_state.model = self._lp_model
        new_state.schedule = copy.copy(self._schedule)
        return new_state

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        """Manually set the value of the score attribute for this state.
        Use with caution."""
        self._score = score

    @property
    def slack(self):
        return self._slack

    @slack.setter
    def slack(self, slack):
        """Manually set the value of the slack attribute for this state.
        Use with caution."""
        self._slack = slack

    @property
    def model(self):
        return self._lp_model

    @model.setter
    def model(self, model):
        self._lp_model = model

    @property
    def eventorder(self):
        return self._eventorder

    @eventorder.setter
    def eventorder(self, eventorder):
        self._eventorder = eventorder

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, schedule):
        self._schedule = schedule

    def create_model(self, model_class, *args, **kwargs):
        """Create an (LP) model for the current state.

        Parameters
        ----------
        model_class : str
            Class name of the class implementing the state model.
        *args :
            Should contain exactly the (non-keyword) arguments required
            by the constructor of `model_class`.
        **kwargs :
            Should contain exactly the keyword arguments required by the
            constructor of `model_class`.
        """
        self._lp_model = model_class(*args, **kwargs)

    def compute_score(self):
        """Solve the underlying LP and set the score equal to its
        objective value, if a feasible solution exists.
        """
        if self._lp_model is None:
            raise RuntimeError("A score can only be computed if a model has"
                               " been specified for this state.")
        sol = self._lp_model.solve()
        if sol is not None:
            self._score = sol.get_objective_value()
            self._schedule = self._lp_model.get_schedule()
            if self._lp_model.with_slack:
                self._slack = self._lp_model.compute_slack()
            else:
                self._slack = []
        else:
            self._score = np.inf
            self._slack = []
