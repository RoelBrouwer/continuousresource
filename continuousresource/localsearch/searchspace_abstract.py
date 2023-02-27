from abc import ABC
from abc import abstractmethod
import copy
import numpy as np
import time

from .utils import sanitize_search_space_params
from continuousresource.mathematicalprogramming.linprog \
    import LPWithSlack


class SearchSpace(ABC):
    """Wrapper object for information about the search space.

    Parameters
    ----------
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
    def __init__(self, params=None):
        self._params = sanitize_search_space_params(params)
        self._current_solution = None
        self._best_solution = None
        self._label = "superclass"
        self._timings = {
            "initial_solution": 0,
            "model_update": 0,
            "lp_solve": 0
        }

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

    @property
    def timings(self):
        """Dictionary containing cumulative time spent on several parts
        of the process.
        """
        return self._timings

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
        self._current_solution = initial
        initial.compute_score()
        self._best_solution = copy.copy(initial)
        self._best_solution.score = initial.score
        self._best_solution.slack = initial.slack
        self._timings["initial_solution"] = t_end - t_start
        return t_end - t_start, initial.score

    def compute_search_space_reductions(self):
        """Eliminate parts of the search space by looking at implicit
        precedence constraints in the data.
        """
        raise NotImplementedError

    def get_random_order(self, prec_matrix=None):
        """Returns a random event order that respects (precomputed)
        precedence relations.
        Parameters
        ----------
        eventlist : list of list of int
            List of lists of length two, representing the events in the
            eventlist being built. First element in each lists is the
            event type, second element the job ID.
        prec_matrix : ndarray
            Two-dimensional array (|E| x |E|) representing the precedence
            relations between events. A 1 on position [i, j] means that i
            comes before j. I.e., only if the sum of column i is 0, the
            event can occur freely.
        Returns
        -------
        list of list of int
            List of lists of length two, representing the events in the
            eventlist. First element in each lists is the event type,
            second element the job ID.
        """
        if self._current_solution is None:
            raise RuntimeError("Please initialize a model first")

        if prec_matrix is None:
            prec_matrix = self._precedences

        eventlist = self._current_solution.model.event_list

        random_list = [
            [0, 0] for i in range(len(eventlist))
        ]

        for i in range(len(eventlist)):
            # Indices of events that are "precedent-free"
            opt = np.where(np.all(~prec_matrix, axis=0))[0]
            if (len(opt) > 0):
                selected = np.random.choice(opt)
                random_list[i] = eventlist[selected]
                prec_matrix = np.delete(prec_matrix, selected, 0)
                prec_matrix = np.delete(prec_matrix, selected, 1)
                del eventlist[selected]
            else:
                return None

        return random_list

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def random_walk(self, no_steps=100):
        """Performs a random walk from the current solution of length
        `no_steps`. These steps are not executed simultaneously, but one
        after another. So, the second step is performed on the event
        order that results after the first, and so on.

        Parameters
        ----------
        no_steps : int
            Length of the random walk, i.e. the number of steps
            performed.

        Returns
        -------
        SearchSpaceState
            New state for the search to continue with.
        """
        raise NotImplementedError


class SearchSpaceState():
    """Class object describing a state in the search space for a local
    search approach.

    Parameters
    ----------
    belongs_to : SearchSpace
        Search space wherein this object represents a state.
    eventorder : ndarray
        Two-dimensional (|E| x 2) array representing the events in
        the problem, where the first column contains an integer
        indicating the event type (0 for start, 1 for completion) and
        the second column the associated job ID.
    """
    def __init__(self, belongs_to, eventorder):
        self._searchspace = belongs_to
        self._eventorder = eventorder
        self._score = np.inf
        self._slack = []
        self._lp_model = None

    def __copy__(self):
        """Override copy method to make copies of some attributes and
        reset others.
        """
        new_state = SearchSpaceState(self._searchspace,
                                     copy.copy(self._eventorder))
        new_state.model = self._lp_model
        return new_state

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
    def model(self):
        """OrderBasedSubProblem : Object containing the LP model and
        associated functions."""
        return self._lp_model

    @model.setter
    def model(self, model):
        """Manually set the value of the model attribute for this state.
        Note that this does not make a copy, it just stores a reference.
        Use with caution."""
        self._lp_model = model

    @property
    def eventorder(self):
        return self._eventorder

    @eventorder.setter
    def eventorder(self, eventorder):
        """Manually set the value of the eventorder attribute for this
        state. Use with caution."""
        self._eventorder = eventorder

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
        t_start = time.perf_counter()
        sol = self._lp_model.solve()
        t_end = time.perf_counter()
        self._searchspace.timings["lp_solve"] += t_end - t_start
        if sol is not None:
            self._score = sol.get_objective_value()
            self._schedule = self._lp_model.get_schedule()
            if isinstance(self._lp_model, LPWithSlack):
                self._slack = self._lp_model.compute_slack()
            else:
                self._slack = []
        else:
            self._score = np.inf
            self._slack = []
