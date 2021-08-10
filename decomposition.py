from abc import ABC
from abc import abstractmethod
import numpy as np

from linprog import OrderBasedSubProblem


class EventOrderMasterProblem(ABC):
    """Super class for all Event Order-based decomposition approaches,
    implementing the part that generates the event orders to be
    investigated in the subproblem.

    Parameters
    ----------
    jobs : ndarray
        Two-dimensional (n x 7) array containing job properties:
            - 0: resource requirement (E_j);
            - 1: resource lower bound (P^-_j);
            - 2: resource upper bound (P^+_j);
            - 3: release date (r_j);
            - 4: deadline (d_j);
            - 5: weight (W_j);
            - 6: objective constant (B_j).
    """
    def __init__(self, jobs):
        self._job_properties = jobs

    @property
    def job_properties(self):
        return self._job_properties

    def solve_subproblem(self, eventlist):
        """Calls the subproblem on the event order, regardless of how it
        was arrived at.

        Parameters
        ----------
        eventlist : ndarray
            Two-dimensional (|E| x 2) array representing the events in the
            problem, where the first column contains an integer indicating
            the event type (0 for start, 1 for completion) and the second
            column the associated job ID.
        """
        # TODO: implement this properly,
        raise NotImplementedError

    def compute_precedence_matrix(self, eventlist, bool_expr=None):
        """Computes a precedence matrix on the events based on the job
        properties.

        Parameters
        ----------
        eventlist : list of list of int
            List of lists of length two, representing the events in the
            eventlist being built. First element in each lists is the
            event type, second element the job ID.
        bool_expr : function
            Function that, given two events, returns True if the second
            event should preceed the first, and False otherwise.
            Two rows of ``eventlist'' are its input, as is a reference to
            a list of job_properties.
            Default function that is used only defines precedence
            relations between events belonging to the same job, assuming
            they have to be done in order of their event type ID.

        Returns
        -------
        ndarray
            Two-dimensional array (|E| x |E|) representing the precedence
            relations between events. A 1 on position [i, j] means that j
            comes before i. I.e., only if the sum of row i is 0, the
            event can occur freely.
        """
        def default_test(i, j, job_properties):
            """Default precedence test"""
            if (i[1] == j[1] and i[0] > j[0]):
                return True
            return False

        if bool_expr is None:
            bool_expr = default_test

        prec_matrix = np.array(
            [
                [
                    bool_expr(i, j, self.job_properties)
                    for j in eventlist
                ]
                for i in eventlist
            ],
            dtype=bool
        )
        return prec_matrix

    def get_random_order(self, eventlist, prec_matrix=None):
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
            relations between events. A 1 on position [i, j] means that j
            comes before i. I.e., only if the sum of row i is 0, the
            event can occur freely.

        Returns
        -------
        list of list of int
            List of lists of length two, representing the events in the
            eventlist. First element in each lists is the event type,
            second element the job ID.
        """
        if prec_matrix is None:
            # Compute a basic precedence matrix
            prec_matrix = self.compute_precedence_matrix(eventlist)

        random_list = [
            [0, 0] for i in range(len(eventlist))
        ]

        for i in range(len(eventlist)):
            # Indices of events that are "precedent-free"
            opt = np.where(np.all(~prec_matrix, axis=1))[0]
            if (len(opt) > 0):
                selected = np.random.choice(opt)
                random_list[i] = eventlist[selected]
                prec_matrix = np.delete(prec_matrix, selected, 0)
                prec_matrix = np.delete(prec_matrix, selected, 1)
                del eventlist[selected]
            else:
                return None

        return random_list


class EventOrderEnumeration(EventOrderMasterProblem):
    """Not very efficient implementation, enumerating all possible event
    orders.

    Parameters
    ----------
    jobs : ndarray
        Two-dimensional (n x 7) array containing job properties:
            - 0: resource requirement (E_j);
            - 1: resource lower bound (P^-_j);
            - 2: resource upper bound (P^+_j);
            - 3: release date (r_j);
            - 4: deadline (d_j);
            - 5: weight (W_j);
            - 6: objective constant (B_j).
    """
    def __init__(self, jobs):
        super().__init__(jobs)
        # TODO: generate precedence constraints based on binary relations

    def enumerate_eventlists(self):
        """Enumerate all possible eventlists, containing a start and
        completion event for each job in the job_properties list.
        """
        initial_list = [
            [e, i] for i in range(len(self._job_properties)) for e in range(2)
        ]
        self._recursive_enumerate([], initial_list)

    def _recursive_enumerate(self, current_order, remaining_events):
        """Recursively enumerates the event orders, by recursively
        branching on the next element in the list.

        Parameters
        ----------
        current_order : list of list of int
            List of lists of length two, representing the events in the
            eventlist being built. First element in each lists is the
            event type, second element the job ID.
        remaining_events : list of list of int
            List of lists of length two, representing the events that
            remain to be inserted in the eventlist. First element in each
            lists is the event type, second element the job ID.
        """
        # TODO: keep track of best solution
        # Base case:
        if len(remaining_events) == 1:
            self.solve_subproblem(current_order.append(remaining_events[0]))
            return

        for idx in range(len(remaining_events)):
            # Shallow copy
            new_order = list(current_order).append(remaining_events[idx])
            new_remaining = list(remaining_events)
            del new_remaining[idx]

            # Depth first
            self._recursive_enumerate(new_order, new_remaining)
