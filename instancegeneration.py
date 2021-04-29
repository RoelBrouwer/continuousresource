import click
import datetime
import math
import numpy as np
import os.path
import warnings


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--njobs',
    '-n',
    'njobs',
    required=True,
    type=int,
    help="Number of jobs in the generated instance."
)
@click.option(
    '--resource_avg',
    '-r',
    'avg_resource',
    required=False,
    default=1,
    type=float,
    help="Average amount of available resource."
)
@click.option(
    '--resource_std',
    '-s',
    'std_resource',
    required=False,
    default=0,
    type=float,
    help="Standard deviation for the available resource."
)
@click.option(
    '--jumppoints',
    '-k',
    'jumppoints',
    required=True,
    type=int,
    help="Number of jump points in the cost functions of each job."
)
@click.option(
    '--export-format',
    '-f',
    'exportformat',
    required=True,
    type=click.Choice(['binary', 'csv', 'both'], case_sensitive=False),
    default='both',
    help="Export format for the provided instance."
)
@click.option(
    '--export-path',
    '-p',
    'exportpath',
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        resolve_path=True
    ),
    help="Path to the output folder."
)
@click.option(
    '--label',
    '-l',
    'label',
    required=False,
    type=str,
    default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"),
    help="Label or name for the generated instance."
)
def main(njobs, avg_resource, std_resource, jumppoints, exportformat,
         exportpath, label):
    """Generate a single instance of a single machine scheduling problem
    with a regular step cost function.

    Parameters
    ----------
    njobs : int
        Number of jobs in the generated instance.
    avg_resource : float
        Mean of a normal distribution from which the amount of available
        resource at each time step will be drawn.
    std_resource : float
        Standard deviation of a normal distribution from which the amount
        of available resource at each time step will be drawn.
    jumppoints : int
        Number of jump points in the cost functions of each job.
    exportformat : {'both', 'binary', 'csv'}
        Export format for the provided instance.
    exportpath : str
        Path to the output folder.
    label : str
        Label or name for the generated instance.

    Notes
    -----
    Currently, this function implements the instance generation method as
    it is described in "B. Detienne, S. Dauzere-Peres, and C. Yugma. An
    exact approach for scheduling jobs with regular step cost functions
    on a single machine. Computers and Operations Research, 2012."
    (section 4.1). With two exceptions: the cost function has been
    inverted and instead of processing time, jobs require a total amount
    of resource.
    """
    # Sample njobs random processing times
    resource_requirement = np.random.uniform(
        low=0.0,
        high=100*avg_resource,
        size=njobs
    ).round(decimals=2)
    # We take the expected processing time per machine as an upperbound
    # for jumppoint generation.
    total_time = math.ceil(sum(resource_requirement) / avg_resource)
    # Sample a (njobs, jumppoints)-matrix of jump points
    jump_points = np.array([
        np.sort(np.random.randint(low=math.ceil(pt / avg_resource),
                                  high=total_time, size=jumppoints))
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
        [lower_bounds.reshape(-1,1), upper_bounds.reshape(-1,1)],
        axis=1
    )

    # Sample resource availability
    resource_availability = np.random.normal(
        loc=avg_resource,
        scale=std_resource,
        size=math.ceil(1.5 * total_time / njobs + 1)
    ).round(decimals=2)

    if exportformat in ['both', 'binary']:
        path = os.path.join(exportpath, label)
        to_binary(path, resource_requirement, jump_points, weights, bounds,
                  resource_availability)
    if exportformat in ['both', 'csv']:
        path = os.path.join(exportpath, label)
        to_csv(path, resource_requirement, jump_points, weights, bounds,
               resource_availability)


def to_binary(path, resource_requirement, jump_points, weights, bounds,
              resource_availability):
    """Write instance data to a binary file.

    Parameters
    ----------
    path : str
        Filename of outputfile (IMPORTANT: without extension).
    resource_requirement : ndarray
        One-dimensional array of real resource requirements.
    jump_points : ndarray
        Two-dimensional array of integer jump points.
    weights : ndarray
        Two-dimensional array of integer weights.
    bounds : ndarray
        Two-dimensional array of real lower and upper bounds.
    resource_availability : ndarray
        One-dimensional array of the amount of available resource.
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
        resource_requirement=resource_requirement,
        jump_points=jump_points,
        weights=weights,
        bounds=bounds,
        resource_availability=resource_availability
    )


def to_csv(path, resource_requirement, jump_points, weights, bounds,
           resource_availability):
    """Write instance data to three csv files.

    Parameters
    ----------
    path : str
        Path to output folder.
    resource_requirement : ndarray
        One-dimensional array of real resource requirements.
    jump_points : ndarray
        Two-dimensional array of integer jump points.
    weights : ndarray
        Two-dimensional array of integer weights.
    bounds : ndarray
        Two-dimensional array of real lower and upper bounds.
    resource_availability : ndarray
        One-dimensional array of the amount of available resource.
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
        resource_requirement,
        delimiter=';',
        fmt='%1.2f'
    )
    np.savetxt(
        os.path.join(path, 'jump_points.csv'),
        jump_points,
        delimiter=';',
        fmt='%i'
    )
    np.savetxt(
        os.path.join(path, 'weights.csv'),
        weights,
        delimiter=';',
        fmt='%i'
    )
    np.savetxt(
        os.path.join(path, 'bounds.csv'),
        bounds,
        delimiter=';',
        fmt='%1.2f'
    )
    np.savetxt(
        os.path.join(path, 'resource_availability.csv'),
        resource_availability,
        delimiter=';',
        fmt='%1.2f'
    )


if __name__ == "__main__":
    main()
