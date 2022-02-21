"""Utility functions that *may* be reused across mathematical programming
implementations."""

import docplex.mp.model
import math
import numpy as np
import pulp
import re


def time_and_resource_vars_to_human_readable_solution_cplex(time_vars,
                                                            resource_vars):
    """Constructs a human readble (tabular) version of the solution
    represented by the variables as passed in the parameters.

    Parameters
    ----------
    time_vars : ndarray of docplex.mp.dvar.Var
        Numpy array linking the objects that reference the time
        variables. It is assumed that job and event type information can
        be extracted from the label (t_{e}) as follows:
            - If the event index {e} is an even number it is a start
              event, otherwise it is a completion event;
            - The associated job index can be obtained by dividing the
              event index {e} by two, rounding down.
    resource_vars : ndarray of ndarray of docplex.mp.dvar.Var
        Matrix linking the objects that reference the resource variables.
        It is assumed that job and event information can be extracted
        from the label (p_{j},{e}) as follows:
            - The job index is {j};
            - The event index is {e}.

    Returns
    -------
    Tuple of ndarray
        A tuple of four ndarrays, in order:
            - A one-dimensional array (|E|) of strings representing the
              events in chronological order.
            - A two-dimensional array (|E| x 2) of integers listing the
              Job ID and event type (in that order) in chronological
              order.
            - A one-dimensional array (|E|) of floats indicating the time
              of occurrence of each event, in chronological order.
            - A two-dimensional array (n x |E|) of floats indicating the
              resource consumption for each job during all intervals.
    """
    event_labels = np.zeros(shape=len(time_vars), dtype='U6')
    event_idx = np.zeros(shape=(len(time_vars), 2), dtype=int)
    event_timing = np.zeros(shape=len(time_vars), dtype=float)
    resource_consumption = np.zeros(shape=(math.floor(len(time_vars) / 2),
                                           len(time_vars)), dtype=float)

    # Sort the array of time variables by t
    sorted_time = sorted(time_vars, key=lambda a: a.solution_value)
    event_idx_map = np.zeros(shape=len(time_vars), dtype=int)

    # Extract event related information
    for i in range(len(sorted_time)):
        idx = int(re.match(r't_(\d+)', sorted_time[i].name).group(1))
        event_idx_map[idx] = i

        event_idx[i][0] = math.floor(idx / 2)  # Job ID
        event_idx[i][1] = idx % 2  # Event type (0 = S; 1 = C)

        event_labels[i] = (f"{'S' if idx % 2 == 0 else 'C'}_"
                           f"{math.floor(idx / 2)}")

        event_timing[i] = sorted_time[i].solution_value

    # Construct the resource consumption matrix
    for row in resource_vars:
        for p in row:
            if isinstance(p, docplex.mp.dvar.Var):
                # Parse the label
                idcs = re.match(r'p_(\d+),(\d+)', p.name)
                j = int(idcs.group(1))
                e = int(idcs.group(2))

                # Look up interval label
                i = event_idx_map[e]

                resource_consumption[j, i] = p.solution_value

    return (event_labels, event_idx, event_timing, resource_consumption)


def time_and_resource_vars_to_human_readable_solution_pulp(time_vars,
                                                           resource_vars):
    """Constructs a human readble (tabular) version of the solution
    represented by the variables as passed in the parameters.

    Parameters
    ----------
    time_vars : ndarray of pulp.LpVariable
        Numpy array linking the objects that reference the time
        variables. It is assumed that job and event type information can
        be extracted from the label (t_{e}) as follows:
            - If the event index {e} is an even number it is a start
              event, otherwise it is a completion event;
            - The associated job index can be obtained by dividing the
              event index {e} by two, rounding down.
    resource_vars : ndarray of ndarray of pulp.LpVariable
        Matrix linking the objects that reference the resource variables.
        It is assumed that job and event information can be extracted
        from the label (p_{j},{e}) as follows:
            - The job index is {j};
            - The event index is {e}.

    Returns
    -------
    Tuple of ndarray
        A tuple of four ndarrays, in order:
            - A one-dimensional array (|E|) of strings representing the
              events in chronological order.
            - A two-dimensional array (|E| x 2) of integers listing the
              Job ID and event type (in that order) in chronological
              order.
            - A one-dimensional array (|E|) of floats indicating the time
              of occurrence of each event, in chronological order.
            - A two-dimensional array (n x |E|) of floats indicating the
              resource consumption for each job during all intervals.
    """
    event_labels = np.zeros(shape=len(time_vars), dtype='U6')
    event_idx = np.zeros(shape=(len(time_vars), 2), dtype=int)
    event_timing = np.zeros(shape=len(time_vars), dtype=float)
    resource_consumption = np.zeros(shape=(math.floor(len(time_vars) / 2),
                                           len(time_vars)), dtype=float)

    # Sort the array of time variables by t
    # Note that events that occur at the same time may end up in the
    # wrong order!
    sorted_time = sorted(time_vars, key=lambda a: a.varValue)
    event_idx_map = np.zeros(shape=len(time_vars), dtype=int)

    # Extract event related information
    for i in range(len(sorted_time)):
        idx = int(re.match(r't_(\d+)', sorted_time[i].name).group(1))
        event_idx_map[idx] = i

        event_idx[i][0] = math.floor(idx / 2)  # Job ID
        event_idx[i][1] = idx % 2  # Event type (0 = S; 1 = C)

        event_labels[i] = (f"{'S' if idx % 2 == 0 else 'C'}_"
                           f"{math.floor(idx / 2)}")

        event_timing[i] = sorted_time[i].varValue

    # Construct the resource consumption matrix
    for row in resource_vars:
        for p in row:
            if isinstance(p, pulp.LpVariable):
                # Parse the label
                idcs = re.match(r'p_(\d+),(\d+)', p.name)
                j = int(idcs.group(1))
                e = int(idcs.group(2))

                # Look up interval label
                i = event_idx_map[e]

                resource_consumption[j, i] = p.varValue

    return (event_labels, event_idx, event_timing, resource_consumption)


def solution_to_csv_string(event_labels, event_idx, event_timing,
                           resource_consumption):  # , separator=';'):
    csv_string = 'LABELS;' + ';'.join(event_labels) + '\n'
    csv_string += 'JOB ID;' + ';'.join(map(str, event_idx[:, 0])) + '\n'
    csv_string += 'EVENT TYPE;' + ';'.join(map(str, event_idx[:, 1])) + '\n'
    csv_string += 'TIME;' + ';'.join(map(str, event_timing)) + '\n'

    for j in range(len(resource_consumption)):
        csv_string += f'RESOURCE JOB {j};'
        csv_string += ';'.join(map(str, resource_consumption[j])) + '\n'

    return csv_string
