from abc import ABC
from abc import abstractmethod
import datetime
import docplex.mp.model
from docplex.mp.relax_linear import LinearRelaxer


class LP(ABC):
    """Super class for all Linear Programming models.

    Parameters
    ----------
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    def __init__(self, label):
        super().__init__()

        # Initialize problem
        label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + \
            f'_{label}'
        self._problem = docplex.mp.model.Model(name=label)

    @property
    def problem(self):
        return self._problem

    def solve(self, timelimit=1e+75, threads=1):
        """Solve the model.

        Parameters
        ----------
        timelimit : int
            Optional value indicating the timelimit set on solving the
            problem. By default, no timelimit is enforced.
        threads : int
            Optional value indicating the number of threads that the
            solver is allowed to use. Any value below 1 is considered to
            mean no limit is imposed, any positive value will be passed
            as an upper bound on the number of global threads to the
            solver.
        """
        self._problem.set_time_limit(timelimit)
        if threads < 1:
            threads = 0
        self._problem.context.cplex_parameters.threads = threads
        return self._problem.solve()

    @abstractmethod
    def _initialize_model(self, instance):
        """Initialize the model.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data.
        """
        raise NotImplementedError


class LPWithSlack(ABC):
    """Mixin class for all Linear Programming models that have slack
    variables.

    Parameters
    ----------
    *args
        Pass along any positional arguments to the next constructor
    **kwargs
        Pass along any keyword arguments to the next constructor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def compute_slack(self, slackpenalties):
        """Compute the (summed) value of the slack variables in the
        model.

        Parameters
        ----------
        slackpenalties : list of float
            List of penalty coefficients for slack variables.

        Returns
        -------
        list of tuple
            List of tuples with in the first position a string
            identifying the type of slack variable, in second position
            the summed value of these variables (float) and in third
            position the unit weight of these variables in the objective.
        """
        raise NotImplementedError


class MIP(object):
    """Mixin class for all Mixed Integer Linear Programming models.

    Parameters
    ----------
    *args
        Pass along any positional arguments to the next constructor
    **kwargs
        Pass along any keyword arguments to the next constructor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def relax_problem(self):
        """Relax all decision variables in the MIP. All integer variables
        (including binary) will be turned into continuous variables.
        """
        self._problem = LinearRelaxer.make_relaxed_model(self._problem)
