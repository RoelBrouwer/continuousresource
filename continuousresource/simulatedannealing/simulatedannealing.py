import copy
import math
import numpy as np
import os
import time


def simulated_annealing(search_space, initial_temperature, alfa, alfa_period,
                        cutoff=100000):
    """Routine performing Simulated Annealing local search.
    The search stops when one of the following three conditions is met:
        - No candidate solution was accepted after trying 200 options in
          a single iteration, or;
        - The percentage of accepted solutions was under 2% for over 10%
          of the iterations of a single alfa-period, or;
        - The total number of iterations exceeds the `cutoff` parameter.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that this simulated annealing run will be
        performed on.
    initial_temperature : float
        Initial value for the temperature of the annealing process.
    alfa : float
        Multiplication factor for updating the temperature.
    alfa_period : int
        Number of iterations to go through before updating the
        temperature.
    cutoff : int
        Maximum number of iterations to run the simulated annealing
        process.

    Returns
    -------
    int
        Number of iterations performed
    SearchSpaceState
        Best found solution.
    """
    # Temperature
    temperature = initial_temperature
    stop_crit = 0
    iters = cutoff

    # Main loop
    for i in range(cutoff):
        # Select a random neighbor
        fails, new_state = search_space.get_neighbor(temperature)

        if new_state is None:
            # If we were unable to accept a candidate solution from 200
            # options, we give up.
            iters = i + 1
            break

        # Update temperature for next iteration block
        if i % alfa_period:
            temperature = temperature * alfa

    # Return solution
    return iters, search_space.best


def simulated_annealing_verbose(search_space, initial_temperature, alfa,
                                alfa_period, cutoff=100000, output_dir=None):
    """Routine performing Simulated Annealing local search.
    The search stops when one of the following three conditions is met:
        - No candidate solution was accepted after trying 200 options in
          a single iteration, or;
        - The percentage of accepted solutions was under 2% for over 10%
          of the iterations of a single alfa-period, or;
        - The total number of iterations exceeds the `cutoff` parameter.

    Parameters
    ----------
    search_space : SearchSpace
        Search space object that this simulated annealing run will be
        performed on.
    initial_temperature : float
        Initial value for the temperature of the annealing process.
    alfa : float
        Multiplication factor for updating the temperature.
    alfa_period : int
        Number of iterations to go through before updating the
        temperature.
    cutoff : int
        Maximum number of iterations to run the simulated annealing
        process.
    output_dir : string
        Directory to put the verbose progression report.

    Returns
    -------
    int
        Number of iterations performed
    SearchSpaceState
        Best found solution.
    """
    if output_dir is None:
        output_dir = os.getcwd()

    with open(os.path.join(output_dir,
                           f"iterations.csv"), "w") as csv:
        added_header = ""
        if search_space.current.model.with_slack:
            added_header = ";slack"

        csv.write(
            f'#;time;best_score;curr_score;rejected{added_header}\n'
        )

        # Temperature
        temperature = initial_temperature
        stop_crit = 0
        iters = cutoff
        start_time = time.perf_counter()

        # Main loop
        for i in range(cutoff):
            # Select a random neighbor
            fails, new_state = search_space.get_neighbor(temperature)

            if new_state is None:
                # If we were unable to accept a candidate solution from 200
                # options, we stop.
                iters = i + 1

                slack_string = ""
                if search_space.current.model.with_slack:
                    total_slack = 0
                    for (label, value, weight) in search_space.current.slack:
                        total_slack += value * weight
                    slack_string = f";{total_slack}"

                csv.write(
                    f'{i};{time.perf_counter() - start_time:0.2f};'
                    f'{search_space.best.score};{search_space.current.score};'
                    f'{fails}{slack_string}\n'
                )
                break
                    

            # Update temperature for next iteration block
            if i % alfa_period:
                temperature = temperature * alfa

            slack_string = ""
            if search_space.current.model.with_slack:
                total_slack = 0
                for (label, value, weight) in search_space.current.slack:
                    total_slack += value * weight
                slack_string = f";{total_slack}"

            csv.write(
                f'{i};{time.perf_counter() - start_time:0.2f};'
                f'{search_space.best.score};{search_space.current.score};'
                f'{fails}{slack_string}\n'
            )

    # Return solution
    return iters, search_space.best


class SearchSpace():
    """Wrapper object for information about the search space.
    """
    def __init__(self):
        self._current_solution = None
        self._best_solution = None

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
        t_start = time.perf_counter()
        initial.model.generate_initial_solution()
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
        pass

    def get_neighbor_move_revert(self, new_state, data):
        """Reverts the model to its previous state.

        Parameters
        ----------
        new_state : SearchSpaceState
            Reference to the state containing a link to the model that
            should be reverted.
        ? : ?
            ?
        """
        pass

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
