import csv
import datetime
import math
import numpy as np
import os
import os.path
import warnings


from continuousresource.probleminstances.baseinstance import BaseInstance


class JobPropertiesInstance(BaseInstance):
    """Class for grouping functions related to problem instances that are
    fully described by an array of job properties and an optional list of
    additional constants.
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
            'jobs': file_contents['jobs'],
            'constants': file_contents['constants'].item()
        }
        # JobPropertiesInstance.check_input_dimensions(instance)
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
            rdr = csv.reader(cst)
            constants = {row[0]: float(row[1]) for row in rdr}
        instance = {
            'jobs': np.genfromtxt(
                os.path.join(path, 'jobs.csv'),
                delimiter=';',
                dtype=np.float64
            ),
            'constants': constants
        }
        # JobPropertiesInstance.check_input_dimensions(instance)
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
            jobs=instance['jobs'],
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
            os.path.join(path, 'jobs.csv'),
            instance['jobs'],
            delimiter=';',
            fmt='%1.2f'
        )
        with open(os.path.join(path, 'constants.csv'), "w") as csv:
            for key in instance['constants'].keys():
                csv.write(f"{key};{instance['constants'][key]}\n")

    @staticmethod
    def generate_instance(njobs, resource_availability, adversarial=False):
        """Generate a single instance of a specific type.

        Parameters
        ----------
        njobs : int
            Number of jobs in the generated instance.
        resource_availability : float
            The continuously available amount of resource in the
            instance.
        adversarial : boolean
            Generate an adversarial instance, where jobs with a high
            deadline also have a high weight.

        Returns
        -------
        Dict
            Dictionary containing the instance data. It has two fields:
                - `jobs`, containing a two-dimensional (n x 7) array of
                  job properties:
                    - 0: resource requirement (E_j);
                    - 1: resource lower bound (P^-_j);
                    - 2: resource upper bound (P^+_j);
                    - 3: release date (r_j);
                    - 4: deadline (d_j);
                    - 5: weight (W_j);
                    - 6: objective constant (B_j).
                - `constants`, containing a dictionary of floating points
                  constants for the instance.
        """
        # Prepare instance containers
        jobs = np.empty(shape=(njobs, 7))
        constants = {
            'resource_availability': resource_availability
        }

        # Sample njobs random processing times
        jobs[:, 0] = np.random.uniform(
            low=0.0,
            high=100.0,
            size=njobs
        ).round(decimals=2)

        # Sample njobs random weights
        jobs[:, 5] = np.random.uniform(
            low=0.0,
            high=5.0,
            size=njobs
        ).round(decimals=2)

        # Sample njobs random objective constants
        jobs[:, 6] = np.random.uniform(
            low=0.0,
            high=10.0,
            size=njobs
        ).round(decimals=2)

        # Sample njobs random lower bounds
        jobs[:, 1] = np.array([
            np.random.uniform(
                low=0.0,
                high=min(0.25 * resource_availability, 0.25 * jobs[j, 0])
            )
            for j in range(njobs)
        ]).round(decimals=2)

        # Sample njobs random upper bounds
        jobs[:, 2] = np.array([
            np.random.uniform(
                low=0.25 * jobs[j, 0],
                high=jobs[j, 0]
            )
            for j in range(njobs)
        ]).round(decimals=2)

        # We take the minimal processing time as an upperbound for
        # for release date and deadline generation.
        total_time = math.ceil(sum(jobs[:, 0]) / resource_availability)

        # Sample njobs random release times
        jobs[:, 3] = np.random.uniform(
            low=0.0,
            high=total_time,
            size=njobs
        ).round(decimals=2)

        # Sample njobs random offsets and deduce deadlines
        jobs[:, 4] = np.array([
            jobs[j, 3] + np.random.uniform(
                low=(jobs[j, 0] / min(resource_availability, jobs[j, 2])),
                high=total_time
            )
            for j in range(njobs)
        ]).round(decimals=2)

        if adversarial:
            # Sort by deadline, and reorder the weights to be
            # non-decreasing.
            arr = jobs[jobs[:, 4].argsort()]
            jobs[:, 5].sort()

        return {
            'jobs': jobs,
            'constants': constants
        }
