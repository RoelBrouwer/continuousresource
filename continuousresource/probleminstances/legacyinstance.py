import datetime
import math
import numpy as np
import os
import os.path
import warnings


from .baseinstance import BaseInstance


class LegacyInstance(BaseInstance):
    """Class for grouping functions related to problem instances that are
    used in the early implementations of MIPs and so on.
    """
    def __init__(self):
        pass

    @staticmethod
    def from_binary(path):
        """Read a problem instance from a binary file.

        Parameters
        ----------
        path : str
            Path to the binary file containing the instance.

        Returns
        -------
        Dict of ndarray
            Dictionary containing the instance data. It has five fields:

                - `resource_requirement`, containing a one-dimensional
                  array;
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
        LegacyInstance.check_input_dimensions(instance)
        return instance

    @staticmethod
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
            Dictionary containing the instance data. It has five fields:

                - `resource_requirement`, containing a one-dimensional
                  array;
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
        LegacyInstance.check_input_dimensions(instance)
        return instance

    @staticmethod
    def to_binary(path, instance):
        """Write instance data to a binary file.

        Parameters
        ----------
        path : str
            Filename of outputfile (IMPORTANT: without extension).
        instance : Dict of ndarray
            Dictionary containing the instance data. It has five fields:

                - `resource_requirement`, containing a one-dimensional
                  array;
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
        # Check for existence of output file
        if os.path.exists(f'{path}.npz'):
            path = f'{path}_{datetime.datetime.now().strftime("%f")}'
            warnings.warn(
                "The provided label was already in use. The output will be"
                f" written to \"{path}.npz\" instead."
            )
        np.savez(
            path,
            resource_requirement=instance['resource_requirement'],
            jump_points=instance['jump_points'],
            weights=instance['weights'],
            bounds=instance['bounds'],
            resource_availability=instance['resource_availability']
        )

    @staticmethod
    def to_csv(path, instance):
        """Write instance data to five csv files.

        Parameters
        ----------
        path : str
            Path to output folder.
        instance : Dict of ndarray
            Dictionary containing the instance data. It has five fields:

                - `resource_requirement`, containing a one-dimensional
                  array;
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
        # Check for existence of output directory
        if os.path.exists(path):
            path = f'{path}_{datetime.datetime.now().strftime("%f")}'
            warnings.warn(
                "The provided label was already in use. The output will be"
                f" written to \"{path}\" instead."
            )
        os.mkdir(path)

        np.savetxt(
            os.path.join(path, 'resource_requirement.csv'),
            instance['resource_requirement'],
            delimiter=';',
            fmt='%1.2f'
        )
        np.savetxt(
            os.path.join(path, 'jump_points.csv'),
            instance['jump_points'],
            delimiter=';',
            fmt='%i'
        )
        np.savetxt(
            os.path.join(path, 'weights.csv'),
            instance['weights'],
            delimiter=';',
            fmt='%i'
        )
        np.savetxt(
            os.path.join(path, 'bounds.csv'),
            instance['bounds'],
            delimiter=';',
            fmt='%1.2f'
        )
        np.savetxt(
            os.path.join(path, 'resource_availability.csv'),
            instance['resource_availability'],
            delimiter=';',
            fmt='%1.2f'
        )

    @staticmethod
    def generate_instance(njobs, avg_resource, std_resource, jumppoints,
                          release_times):
        """Generate a single instance of a specific type.

        Parameters
        ----------
        njobs : int
            Number of jobs in the generated instance.
        avg_resource : float
            Mean of a normal distribution from which the amount of
            available resource at each time step will be drawn.
        std_resource : float
            Standard deviation of a normal distribution from which the
            amount of available resource at each time step will be drawn.
        jumppoints : int
            Number of jump points in the cost functions of each job.
        release_times : bool
            Generate release_times.

        Returns
        -------
        Dict of ndarray
            Dictionary containing the instance data. It has five fields:

                - `resource_requirement`, containing a one-dimensional
                  array;
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
        # Sample njobs random processing times
        resource_requirement = np.random.uniform(
            low=0.0,
            high=100.0,  # *avg_resource,
            size=njobs
        ).round(decimals=2)

        # We take the expected processing time per machine as an upperbound
        # for jumppoint generation.
        total_time = math.ceil(sum(resource_requirement) / avg_resource)

        # Sample a (njobs, jumppoints)-matrix of jump points
        if release_times:
            jump_points = np.array([
                np.sort(
                    np.random.randint(low=0,
                                      high=total_time, size=jumppoints + 1))
                for pt in resource_requirement
            ])
        else:
            jump_points = np.array([
                np.concatenate((
                    [0],
                    np.sort(
                        np.random.randint(low=math.ceil(pt / avg_resource),
                                          high=total_time, size=jumppoints))
                ))
                for pt in resource_requirement
            ])
        # Sample a (njobs, jumppoints)-matrix of cost/reward levels
        weights = np.random.randint(
            low=1,
            high=100,
            size=(njobs, jumppoints)
        )[..., ::-1].cumsum(axis=-1)[..., ::-1]  # reverse sum the rewards

        # Sample bounds on resource consumption
        lower_bounds = np.array([
            np.random.uniform(
                low=0.0,
                high=min(0.25 * avg_resource, 0.25 * pt)
            )
            for pt in resource_requirement
        ]).round(decimals=2)
        upper_bounds = np.array([
            np.random.uniform(
                low=0.25*pt,
                high=pt
            )
            for pt in resource_requirement
        ]).round(decimals=2)
        bounds = np.concatenate(
            [lower_bounds.reshape(-1, 1), upper_bounds.reshape(-1, 1)],
            axis=1
        )

        # Sample resource availability
        resource_availability = np.random.normal(
            loc=avg_resource,
            scale=std_resource,
            size=math.ceil(2 * total_time + 1)
        ).round(decimals=2)
        resource_availability[resource_availability < 0] = 0.0

        return {
            'resource_requirement': resource_requirement,
            'jump_points': jump_points,
            'weights': weights,
            'bounds': bounds,
            'resource_availability': resource_availability
        }

    @staticmethod
    def check_input_dimensions(instance):
        """Performs some superficial checks on the form of the input
        data, to ensure that an instance can be constructed from it. Does
        not perform (m)any meaningful checks on the contents itself.

        Parameters
        ----------
        instance : Dict of ndarray
            Dictionary containing the instance data. It has five fields:

                - `resource_requirement`, containing a one-dimensional
                  array;
                - `jump_points`, containing a two-dimensional array. The
                  length of the first dimension is equal to the length of
                  `resource_requirement`;
                - `weights`, containing a two-dimensional array with the
                  exact same dimensions as `jump_points`.
                - `bounds`, containing a two-dimensional array with the
                  lower and upper bounds.
                - `resource_availability`, containing a one-dimensional
                  array.

        Returns
        -------
        boolean
            True if all checks are passed, will raise an error otherwise.
        """
        proc_len = instance['resource_requirement'].shape[0]
        jump_shape = instance['jump_points'].shape
        weight_shape = instance['weights'].shape
        bound_shape = instance['bounds'].shape

        # Test if the length of the resource_requirement array equals
        # that of the jump_points.
        if proc_len != jump_shape[0]:
            raise ValueError(
                "The number of values in the resource_requirement array"
                f" ({proc_len}) does not equal the number of rows in the"
                f" jump_points matrix ({jump_shape[0]}). A valid instance"
                " cannot be constructed."
            )

        # Test if the shape of the jump_points and weights arrays is equal
        if weight_shape != (jump_shape[0], jump_shape[1] - 1):
            raise ValueError(
                f"The shape of the jump_points matrix {jump_shape} does not"
                f" equal the shape of the weights matrix {weight_shape}. A"
                " valid instance cannot be constructed."
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
                "The provided weights are not strictly decreasing for every"
                " job."
            )

        # TODO: Test if the lower bounds are lower than the upper bounds.

        # TODO: Include tests on the resource_availability.

        return True
