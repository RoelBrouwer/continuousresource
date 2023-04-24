"""Functions that *may* be reused across any eventorder based
approach."""


import bisect
import math
import numpy as np


def construct_event_mapping(eventlist, shape):
    """Construct a mapping from jobs to events based on the event
    list.

    Parameters
    ----------
    eventlist : ndarray
        Two-dimensional (|E| x 2) array representing the events in
        the problem, where the first column contains an integer
        indicating the event type and the second column the
        associated job ID.
    shape : tuple of int
        Tuple of length two, defining the shape of the event map.

    Returns
    -------
    ndarray
        Two-dimensional array containing for every job the
        position of its associated events in the eventlist.
    """
    event_map = np.empty(shape=shape, dtype=int)
    for i in range(len(eventlist)):
        event_map[eventlist[i][1], eventlist[i][0]] = i
    return event_map


def generate_initial_solution(resource_info, time_info, resource,
                              fixed_events=None):
    """Generate an eventlist for the given instance in a greedy way.

    The solution generates an event list, using the resource requirement
    and upper bound of a job. In general, jobs with an early deadline are
    processed first (if the release time has passed), while trying to
    ensure any job can still be finished before its deadline. It assumes
    a constant resource availability. Lower bounds and costs are not
    considered at all.

    Parameters
    ----------
    resource_info : ndarray
        Two-dimensional (n x 2) array containing job properties related
        to the resource:
            - 0: resource requirement (E_j);
            - 1: resource upper bound (P^+_j).
    time_info : ndarray
        Two-dimensional (n x 2) array containing job properties related
        to the time:
            - 0: release time (r_j);
            - 1: deadline (d_j).
    fixed_events : ndarray
        One-dimensional array containing the time of all fixed time
        events, in order of their index (offset by all plannable events).
    resource : float
        Amount of resource available per time unit.

    Returns
    -------
    ndarray
        Two-dimensional (|E| x 2) array representing the events in the
        problem, where the first column contains an integer indicating
        the event type and the second column the associated job ID.
    """
    assert resource_info.shape[0] == time_info.shape[0]
    # TODO: what with fixed events? How to insert time points?
    # Solve LP with only plannable and then base if on times in solution?

    # Initialize eventlist
    eventlist = []

    # Useful constants
    njobs = resource_info.shape[0]
    nfixed = 0
    if fixed_events is not None:
        nfixed = len(fixed_events)
    nplannable = njobs * 2
    nevents = nplannable + nfixed
    kextra = int(nfixed / njobs)

    # Construct intervals
    # `boundaries` is a list of time points at which the list of jobs
    # available for processing changes. These are defined by triples
    # (t, e, j), where t is the time, e the event type (0: release
    # time passed, i.e. a job is added; 1: deadline passed) and j the
    # job ID.
    boundaries = np.concatenate((
        np.array([
            [time_info[j, 0], 0, j]
            for j in range(njobs)
        ]),
        np.array([
            [time_info[j, 1], 1, j]
            for j in range(njobs)
        ])
    ))
    boundaries = boundaries[boundaries[:, 0].argsort()]

    # Create a reference table for the fixed events
    if nfixed > 0:
        fixed = np.array([(fixed_events[i], i) for i in range(nfixed)])
        fixed = fixed[fixed[:, 0].argsort()]

    # Create the reference table we will be working with.
    # This will be a sorted list (sorted on deadline) of jobs that
    # are currently available for processing. For every job it will
    # contain the following information, in order:
    # - 0: deadline;
    # - 1: upper bound (P^+_j);
    # - 2: residual resource need (E_j);
    # - 3: total resource need (E_j);
    # - 4: job ID;
    curr_jobs = []
    f = 0

    # Loop over intervals
    for i in range(len(boundaries) - 1):
        if boundaries[i, 1] == 0:
            # We passed a release time, so we add a job to our list
            bisect.insort(curr_jobs, [
                time_info[int(boundaries[i, 2]), 1],      # d_j
                resource_info[int(boundaries[i, 2]), 1],  # P^+_j
                resource_info[int(boundaries[i, 2]), 0],  # E_j
                resource_info[int(boundaries[i, 2]), 0],  # E_j
                boundaries[i, 2]                          # ID
            ])

        # Compute time and resource in this interval
        interval_time = boundaries[i + 1, 0] - boundaries[i, 0]
        avail_resource = interval_time * resource

        # Compute lower and upper bounds
        lower_bounds = np.array([
            max(
                0,
                job[2] - (job[0] - boundaries[i + 1, 0]) * job[1]
            )
            for job in curr_jobs
        ])
        upper_bounds = np.array([
            min(
                job[2],
                # instance['resource'] * interval_time,
                job[1] * interval_time
            )
            for job in curr_jobs
        ])

        # First assign all lower bounds
        avail_resource -= np.sum(lower_bounds)
        remove = []
        for j in range(len(curr_jobs)):
            if lower_bounds[j] > 0:
                if curr_jobs[j][2] == curr_jobs[j][3]:
                    eventlist.append([0, curr_jobs[j][4]])
                curr_jobs[j][2] -= lower_bounds[j]
                if curr_jobs[j][2] <= 0:
                    eventlist.append([1, curr_jobs[j][4]])
                    remove.append(j)

        # Now distribute any resources that remain
        if avail_resource > 0:
            upper_bounds -= lower_bounds
            for j in range(len(curr_jobs)):
                if curr_jobs[j][2] <= 0:
                    continue
                amount = min(avail_resource, upper_bounds[j])
                if amount > 0:
                    if curr_jobs[j][2] == curr_jobs[j][3]:
                        eventlist.append([0, curr_jobs[j][4]])
                    curr_jobs[j][2] -= amount
                    if curr_jobs[j][2] <= 0:
                        eventlist.append([1, curr_jobs[j][4]])
                        remove.append(j)
                    avail_resource -= amount
                    if avail_resource <= 0:
                        break

        for j in range(len(remove) - 1, -1, -1):
            del curr_jobs[remove[j]]

    return np.array(eventlist, dtype=int)


def generate_random_solution(precs, nplannable=-1):
    """Generate a random initial solution that respects the precedence
    constraints in `precs`.

    Parameters
    ----------
    precs : ndarray
        Two dimensional (|E| x |E|) array listing (inferred) precedence
        relations between events. If the entry at position [i, j] is
        True, this means that i has to come before j.
    nplannable : int
        Amount of plannable events, or boundary where the translation of
        the event index e changes from `e = 2 * job_ID + type` to
        `e = nplannable + k * job_ID + type - 2`.

    Returns
    -------
    ndarray
        Two-dimensional (|E| x 2) array representing the events in the
        problem, where the first column contains an integer indicating
        the event type and the second column the associated job ID.
    """
    assert precs.shape[0] == precs.shape[1]

    # Initialize eventlist
    eventlist = []
    nevents = precs.shape[0]
    if nplannable < 1:
        nplannable = nevents
    else:
        njobs = int(nplannable / 2)
        kextra = int((nevents - nplannable) / njobs)

    events = np.random.permutation(nevents)

    while len(events) > 0:
        for i in range(len(events)):
            if np.sum(precs[events, events[i]]) > 0:
                continue
            if events[i] < nplannable:
                eventlist.append([events[i] % 2, math.floor(events[i] / 2)])
            else:
                e = (events[i] - nplannable) % kextra + 2
                j = math.floor((events[i] - nplannable) / kextra)     
                eventlist.append([e, j])
            events = np.delete(events, i)
            break

    return np.array(eventlist, dtype=int)


def find_precedences(resource_info, time_info, fixed_events=None,
                     infer_precedence=True):
    """Construct an array indicating precedence relations between
    events.

    Parameters
    ----------
    resource_info : ndarray
        Two-dimensional (n x 3) array containing job properties related
        to the resource:
            - 0: resource requirement (E_j);
            - 1: resource lower bound (P^-_j);
            - 2: resource upper bound (P^+_j).
    time_info : ndarray
        Two-dimensional (n x 2) array containing job properties related
        to the time:
            - 0: release time (r_j);
            - 1: deadline (d_j).
    fixed_events : ndarray
        One-dimensional containing the time of all fixed time events, in
        order of their index (offset by all plannable events).
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
    assert resource_info.shape[0] == time_info.shape[0]

    if fixed_events is None:
        fixed_events = np.array([])

    njobs = resource_info.shape[0]
    nplannable = 2 * njobs
    nevents = nplannable + len(fixed_events)

    if not infer_precedence:
        # Start out with an array filled with only the precedence
        # relations that exist between start/completion events.
        return np.array([
            [
                (i % 2 == 0 and j - i == 1)
                if j < nplannable
                else False
                for j in range(nevents)
            ]
            for i in range(nevents)
        ], dtype=bool)

    # For all events we can infer an earliest and latest time
    inferred_limits = np.array([
        e
        for j in range(njobs)
        for e in (
            [time_info[j, 0],
             time_info[j, 1] - (resource_info[j, 0] / resource_info[j, 2])],
            [time_info[j, 0] + (resource_info[j, 0] / resource_info[j, 2]),
             time_info[j, 1]]
        )
    ] + [
        [t, t]
        for t in fixed_events
    ], dtype=float)

    # Now, any event whose latest possible time is smaller than the
    # earliest possible time of another event, has to come before it.
    return np.array([
        [
            ((i % 2 == 0 and j - i == 1 and j < nplannable) or
             (inferred_limits[i, 1] <= inferred_limits[j, 0])) and
            i != j
            for j in range(nevents)
        ]
        for i in range(nevents)
    ], dtype=bool)
