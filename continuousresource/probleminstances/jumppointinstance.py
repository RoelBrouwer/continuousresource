import csv
import datetime
import numpy as np
import os
import os.path
import warnings


from continuousresource.probleminstances.baseinstance import BaseInstance


class JumpPointInstance(BaseInstance):
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
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
        """
        file_contents = np.load(path, allow_pickle=True)
        instance = {
            'properties': file_contents['properties'],
            'jumppoints': file_contents['jumppoints'],
            'weights': file_contents['weights'],
            'constants': file_contents['constants'].item()
        }
        # JumpPointInstance.check_input_dimensions(instance)
        return instance

    @staticmethod
    def from_csv(path):
        """Read a problem instance from two csv files.

        Parameters
        ----------
        path : str
            Path to the folder containing the csv files describing the
            instance.

        Returns
        -------
        Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
        """
        with open(os.path.join(path, 'constants.csv'), "r") as cst:
            rdr = csv.reader(cst, delimiter=';')
            constants = {row[0]: float(row[1]) for row in rdr}
        instance = {
            'properties': np.genfromtxt(
                os.path.join(path, 'properties.csv'),
                delimiter=';',
                dtype=np.float64
            ),
            'jumppoints': np.genfromtxt(
                os.path.join(path, 'jumppoints.csv'),
                delimiter=';',
                dtype=np.float64
            ),
            'weights': np.genfromtxt(
                os.path.join(path, 'weights.csv'),
                delimiter=';',
                dtype=np.float64
            ),
            'constants': constants
        }
        # JumpPointInstance.check_input_dimensions(instance)
        return instance

    @staticmethod
    def to_binary(path, instance):
        """Write instance data to a binary file.

        Parameters
        ----------
        path : str
            Filename of outputfile (IMPORTANT: without extension).
        instance : Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
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
            properties=instance['properties'],
            jumppoints=instance['jumppoints'],
            weights=instance['weights'],
            constants=instance['constants']
        )

    @staticmethod
    def to_csv(path, instance):
        """Write instance data to five csv files.

        Parameters
        ----------
        path : str
            Path to output folder.
        instance : Dict of ndarray
            Dictionary containing the instance data, in the format
            expected for a particular instance type.
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
            os.path.join(path, 'properties.csv'),
            instance['properties'],
            delimiter=';',
            fmt='%1.2f'
        )
        np.savetxt(
            os.path.join(path, 'jumppoints.csv'),
            instance['jumppoints'],
            delimiter=';',
            fmt='%1.2f'
        )
        np.savetxt(
            os.path.join(path, 'weights.csv'),
            instance['weights'],
            delimiter=';',
            fmt='%1.2f'
        )
        with open(os.path.join(path, 'constants.csv'), "w") as csv:
            for key in instance['constants'].keys():
                csv.write(f"{key};{instance['constants'][key]}\n")

    def generate_instance(njobs, resource_availability, kjumppoints,
                          params=None):
        """Generate a single instance of a specific type.

        Parameters
        ----------
        njobs : int
            Number of jobs in the generated instance.
        resource_availability : float
            The continuously available amount of resource in the
            instance.
        kjumppoints : int
            Number of jump points in the cost functions of each job.
        params : Dict
            Dictionary of parameters that can be used in generation. The
            following parameters are recognized:
                - `resource_fraction`: controls the limit on lower bounds
                  based on resource availability;
                - `requirement_fraction`: controls the limit on lower
                  bounds based on a job's resource requirement;
                - `release_date_shift`: shifts the window for generating
                  release dates;
                - `processing_window_size`: scales the upper bound for
                  the size of processing windows.

        Returns
        -------
        Dict
            Dictionary containing the instance data. It has four fields:
                - `properties`, containing a two-dimensional (n x 3)
                  array of job properties:
                    - 0: resource requirement (E_j);
                    - 1: resource lower bound (P^-_j);
                    - 2: resource upper bound (P^+_j);
                - `jumppoints`, containing a two-dimensional (n x k+1)
                  array of k+1 increasing time points where the cost
                  function of each job changes. The first element is the
                  release date, the last element is the deadline.
                - `weights`, containing a two-dimensional (n x k) array of
                  of k increasing weights used in the cost function of
                  each job.
                - `constants`, containing a dictionary of floating points
                  constants for the instance.
        """
        # Define parameters
        a_ml = 0.25
        a_mu = 0.25
        a_rs = 0.125
        a_ps = 2
        if params is not None:
            if 'resource_fraction' in params:
                a_ml = params['resource_fraction']
            if 'requirement_fraction' in params:
                a_mu = params['requirement_fraction']
            if 'release_date_shift' in params:
                a_rs = params['release_date_shift']
            if 'processing_window_size' in params:
                a_ps = params['processing_window_size']

        # Prepare instance containers
        properties = np.empty(shape=(njobs, 3))
        jumppoints = np.empty(shape=(njobs, kjumppoints + 1))
        weights = np.empty(shape=(njobs, kjumppoints))
        constants = {
            'resource_availability': resource_availability
        }

        # Sample njobs random processing times
        properties[:, 0] = np.random.uniform(
            low=10.0,
            high=100.0,
            size=njobs
        ).round(decimals=2)

        # Sample njobs random lower bounds
        properties[:, 1] = np.array([
            np.random.uniform(
                low=0.0,
                high=min(a_ml * resource_availability,
                         a_mu * properties[j, 0])
            )
            for j in range(njobs)
        ]).round(decimals=2)

        # Sample njobs random upper bounds
        properties[:, 2] = np.array([
            np.random.uniform(
                low=a_mu * properties[j, 0],
                high=properties[j, 0]
            )
            for j in range(njobs)
        ]).clip(max=resource_availability).round(decimals=2)

        # We take the minimal processing time as an upperbound for
        # for release date and deadline generation.
        total_time = sum(properties[:, 0]) / resource_availability

        # Sample njobs random release times
        jumppoints[:, 0] = np.random.uniform(
            low=(-1 * a_rs * total_time),
            high=(1 - a_rs) * total_time,
            size=njobs
        ).clip(min=0.0).round(decimals=2)

        # Sample njobs random offsets and deduce deadlines
        jumppoints[:, -1] = np.array([
            jumppoints[j, 0] + np.random.uniform(
                low=(properties[j, 0] /
                     min(resource_availability, properties[j, 2])),
                high=a_ps * total_time
            )
            for j in range(njobs)
        ]).round(decimals=2)

        # Sample k - 1 additional jumppoints
        jumppoints[:, 1:-1] = np.array([
            [
                np.random.uniform(
                    low=(jumppoints[j, 0] + properties[j, 0] /
                         min(resource_availability, properties[j, 2])),
                    high=jumppoints[j, -1]
                )
                for k in range(kjumppoints - 1)
            ]
            for j in range(njobs)
        ]).round(decimals=2)
        jumppoints.sort()

        # Sample njobs random initial weights
        weights[:, 0] = np.random.uniform(
            low=0.0,
            high=5.0,
            size=njobs
        ).round(decimals=2)

        # Sample k - 1 additional weights
        weights[:, 1:] = np.array([
            np.random.uniform(
                low=weights[j, 0],
                high=5.0,
                size=kjumppoints - 1
            )
            for j in range(njobs)
        ]).round(decimals=2)

        # Sort weights and make them cumulative
        weights.sort()
        weights[:, 1:] = np.diff(weights)

        return {
            'properties': properties,
            'jumppoints': jumppoints,
            'weights': weights,
            'constants': constants
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
        return NotImplementedError
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
