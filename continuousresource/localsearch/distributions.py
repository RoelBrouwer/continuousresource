"""Distributions used to define the probability for a relative
displacement to be selected."""


import numpy as np
import warnings


def uniform(idx, left, right):
    """Select any displacement with equal probability.

    Parameters
    ----------
    idx : int
        Position of the event in the event list that will be moved.
    left : int
        Left limit of displacement range (exclusive).
    right : int
        Right limit of displacement range (exclusive).

    Returns
    -------
    int
        New position.
    """
    if right - left <= 2:
        return -1
    new_idx = idx
    while new_idx == orig_idx:
        new_idx = np.random.randint(left + 1, right)
    return new_idx

def linear(idx, left, right):
    """Select a displacement with a probability that decreases linearly
    with increasing size.

    Parameters
    ----------
    idx : int
        Position of the event in the event list that will be moved.
    left : int
        Left limit of displacement range (exclusive).
    right : int
        Right limit of displacement range (exclusive).

    Returns
    -------
    int
        New position.

    Notes
    -----
    TODO The current implementation leaves room for improvement.
    """
    if right - left <= 2:
        return -1

    displacements = [
        i
        for j in (range(idx - left - 1, 0, -1),
                  range(1, right - idx, 1))
        for i in j
    ]
    max_dis = max(displacements[0], displacements[-1])
    inv_displacements = [max_dis - i + 1 for i in displacements]
    total = sum(inv_displacements)
    probabilities = [i / total for i in inv_displacements]
    selected = np.random.random()
    cum_prob = probabilities[0]
    curr_idx = 0
    while cum_prob < selected:
        curr_idx += 1
        cum_prob += probabilities[curr_idx]
    if curr_idx >= idx - left - 1:
        curr_idx += 1
    return left + curr_idx + 1

def plus_one(idx, left, right):
    """Get a displacement of plus one (mirroring a swap).

    Parameters
    ----------
    idx : int
        Position of the event in the event list that will be moved.
    left : int
        Left limit of displacement range (exclusive).
    right : int
        Right limit of displacement range (exclusive).

    Returns
    -------
    int
        New position.
    """
    if right - idx <= 1:
        return -1
    return idx + 1
