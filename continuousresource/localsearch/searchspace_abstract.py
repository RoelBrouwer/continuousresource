from abc import ABC
from abc import abstractmethod
import copy
import numpy as np

from .utils import sanitize_search_space_params


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
    def __init__(self, instance, params=None):
        self._label = "superclass"
        self._timings = {
            "initial_solution": 0,
            "model_update": 0,
            "lp_solve": 0
        }

        # Set parameters
        self._params = sanitize_search_space_params(params)

        # Find and set precedences
        self._precedences = self._find_precedences(instance,
                                                   params['infer_precedence'])

        # Generate an initial solution
        self._generate_initial_solution(instance)

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
    def initial(self):
        return self._initial_solution

    @property
    def lp(self):
        return self._lp_model

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

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def _generate_initial_solution(self, instance):
        """Generates an initial solution within the search space and sets
        the value of `_lp_model`, `_best_solution`, `_current_solution`
        and `_initial_solution`  accordingly.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_score(self):
        """Compute the (exact) score of the current solution

        Returns
        -------
        float
            Score of the current solution
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

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
        instance : Dict of ndarray
            Dictionary containing the instance data.
    """
    def __init__(self, instance):
        self._instance = instance
        self._score = np.inf
        self._slack = []

    def __copy__(self):
        """Override copy method to make copies of some attributes and
        reset others.
        """
        return self.__class__(copy.deepcopy(self._instance))

    @property
    def instance(self):
        """Dict of ndarray : Dictionary containing the instance data."""
        return self._instance

    @instance.setter
    def instance(self, instance):
        """Manually set the value of the instance attribute for this
        state. Use with caution.
        """
        self._instance = instance

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
        return self._instance['eventlist']

    @eventlist.setter
    def eventlist(self, eventlist):
        """Manually set the value of the eventorder attribute for this
        state. Use with caution."""
        self._instance['eventlist'] = eventlist

    @property
    def eventmap(self):
        return self._instance['eventmap']

    @eventmap.setter
    def eventmap(self, eventmap):
        """Manually set the value of the eventmap attribute for this
        state. Use with caution."""
        self._instance['eventmap'] = eventmap
