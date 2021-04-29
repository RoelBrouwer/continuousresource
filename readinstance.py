import numpy as np
import os
import os.path


def from_binary(path):
    """Read a problem instance from a binary file.

    Parameters
    ----------
    path : str
        Path to the binary file containing the instance.

    Returns
    -------
    Dict of ndarray
        Dictionary containing the instance data. It has three fields:

            - `resource_requirement`, containing a one-dimensional array;
            - `jump_points`, containing a two-dimensional array. The
              length of the first dimension is equal to the length of
              `resource_requirement`;
            - `weights`, containing a two-dimensional array with the
              exact same dimensions as `jump_points`.
            - `bounds`, containing a two-dimensional array with the
              lower and upper bounds.
            - `resource_availability`, containing a one-dimensional
              array.
    """
    file_contents = np.load(path, allow_pickle=True)
    instance = {
        'resource_requirement': file_contents['resource_requirement'],
        'jump_points': file_contents['jump_points'],
        'weights': file_contents['weights'],
        'bounds': file_contents['bounds'],
        'resource_availability': file_contents['resource_availability']
    }
    check_input_dimensions(instance)
    return instance


def from_csv(path):
    """Read a problem instance from three csv files.

    Parameters
    ----------
    path : str
        Path to the folder containing the csv files describing the
        instance.

    Returns
    -------
    Dict of ndarray
        Dictionary containing the instance data. It has three fields:

            - `resource_requirement`, containing a one-dimensional array;
            - `jump_points`, containing a two-dimensional array. The
              length of the first dimension is equal to the length of
              `resource_requirement`;
            - `weights`, containing a two-dimensional array with the
              exact same dimensions as `jump_points`;
            - `bounds`, containing a two-dimensional array with the
              lower and upper bounds;
            - `resource_availability`, containing a one-dimensional
              array.
    """
    np.genfromtxt(os.path.join(path, 'resource_requirement.csv'), delimiter=';')
    instance = {
        'resource_requirement': np.genfromtxt(
            os.path.join(path, 'resource_requirement.csv'),
            delimiter=';',
            dtype=np.float64
        ),
        'jump_points': np.genfromtxt(
            os.path.join(path, 'jump_points.csv'),
            delimiter=';',
            dtype=np.int32
        ),
        'weights': np.genfromtxt(
            os.path.join(path, 'weights.csv'),
            delimiter=';',
            dtype=np.int32
        ),
        'bounds': np.genfromtxt(
            os.path.join(path, 'bounds.csv'),
            delimiter=';',
            dtype=np.float64
        ),
        'resource_availability': np.genfromtxt(
            os.path.join(path, 'resource_availability.csv'),
            delimiter=';',
            dtype=np.float64
        )
    }
    check_input_dimensions(instance)
    return instance


def check_input_dimensions(instance):
    """Performs some superficial checks on the form of the input data, to
    ensure that an instance can be constructed from it. Does not perform
    (m)any meaningful checks on the contents itself.

    Parameters
    ----------
    instance : Dict of ndarray
        Dictionary containing the instance data, should be of the form as
        returned by `from_csv` or `from_binary`.

    Returns
    -------
    boolean
        True if all checks are passed, will raise an error otherwise.
    """
    proc_len = instance['resource_requirement'].shape[0]
    jump_shape = instance['jump_points'].shape
    weight_shape = instance['weights'].shape
    bound_shape = instance['bounds'].shape

    # Test if the length of the resource_requirement array equals that of the
    # jump_points.
    if proc_len != jump_shape[0]:
        raise ValueError(
            "The number of values in the resource_requirement array"
            f" ({proc_len}) does not equal the number of rows in the"
            f" jump_points matrix ({jump_shape[0]}). A valid instance cannot"
            " be constructed."
        )

    # Test if the shape of the jump_points and weights arrays is equal
    if weight_shape != jump_shape:
        raise ValueError(
            f"The shape of the jump_points matrix {jump_shape} does not equal"
            f" the shape of the weights matrix {weight_shape}. A valid"
            " instance cannot be constructed."
        )

    # Test if the shape of the bounds array is as expected:
    # As long as the resource_requirement array...
    if proc_len != bound_shape[0]:
        raise ValueError(
            "The number of values in the resource_requirement array"
            f" ({proc_len}) does not equal the number of rows in the"
            f" bounds matrix ({bound_shape[0]}). A valid instance cannot"
            " be constructed."
        )
    # ... and with two columns
    if bound_shape[1] != 2:
        raise ValueError(
            "The bounds matrix should have two rows: a lower and an upper"
            f" bound, but it has {bound_shape[1]} rows. A valid instance"
            " cannot be constructed."
        )

    # Test if the jump points are increasing
    if not np.all(np.diff(instance['jump_points'], axis=-1) >= 0):
        raise ValueError(
            "The provided jump points are not increasing for every job."
        )

    # Test if the weights are strictly decreasing
    if not np.all(np.diff(instance['weights'], axis=-1) < 0):
        raise ValueError(
            "The provided weights are not strictly decreasing for every job."
        )

    # TODO: Test if the lower bounds are lower than the upper bounds.

    # TODO: Include tests on the resource_availability.

    return True
