# params
# T = controlparameter
# T_init -> accept 50% verslechteringen
# alfa (T_new = T * alfa)
# same T for #iterations: constant (16) * size(neighborhood)
# ends when accepted verslechteringen < 2%
import copy
import math
import numpy as np


def simulated_annealing(search_space, initial_temperature, alfa, alfa_period):
    """
    """
    # Initial solution
    # search_space.generate_initial_solution()

    # Temperature
    temperature = initial_temperature

    # Main loop
    # TODO: propper stopping condition
    # while True:
    for i in range(100):
        # Select a random neighbor
        new_state = search_space.get_neighbor(temperature)
        if new_state is None:
            break
        # print(new_state.eventorder)

        # Update temperature for next iteration block
        if i % alfa_period:
            temperature = temperature * alfa

    # Return solution
    return search_space.best


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
        """
        # For now the initial solution is just the eventlist exactly as
        # presented.
        initial = SearchSpaceState(self, eventlist)
        initial.create_model(model_class, eventlist, *args, **kwargs)
        initial.model.generate_initial_solution()
        initial.model.initialize_problem()
        initial.eventorder = initial.model.event_list
        print(initial.model.event_list)
        # print(initial.model.problem.lp_string)
        self._current_solution = initial
        self._best_solution = copy.copy(initial)

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
                # if new_state.score > -1:
                return new_state

    def get_neighbor(self, temperature):
        # Find candidates by looking at all adjacent pairs
        accepted = False
        fail_count = 0

        # Select a random pair to swap
        while not accepted:
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
            if new_state.score <= self._current_solution.score or \
               np.random.random() <= math.exp((self._current_solution.score
                                               - new_state.score)
                                              / temperature):
                # print(f"accepted score: {new_state.score}")
                if new_state.score < self._best_solution.score:
                    # print(f"best score: {new_state.score}")
                    self._best_solution = copy.copy(new_state)
                    self._best_solution.score = new_state.score
                self._current_solution = new_state
                accepted = True
            else:
                # Change the model back (the same model instance is used
                # for both the current and candidate solutions).
                new_state.model.update_swap_neighbors(swap_id)

                if fail_count > 100:
                    return None
                fail_count += 1

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
        self._score = 100000  # TODO: !!!!!
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
        else:
            self._score = 100000  # TODO: !!!!!!!!!!!
