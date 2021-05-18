import click
import datetime
import os.path

from instance import to_binary, to_csv, generate_instance

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
    '-m',
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
    '--releasetimes',
    '-r',
    'release_times',
    required=False,
    type=bool,
    default=False,
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
def main(njobs, avg_resource, std_resource, jumppoints, release_times,
         exportformat, exportpath, label):
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
    release_times : bool
        Generate release_times.
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
    (resource_requirement, jump_points, weights, bounds,
    resource_availability) = generate_instance(njobs, avg_resource, std_resource, jumppoints, release_times)

    if exportformat in ['both', 'binary']:
        path = os.path.join(exportpath, label)
        to_binary(path, resource_requirement, jump_points, weights, bounds,
                  resource_availability)
    if exportformat in ['both', 'csv']:
        path = os.path.join(exportpath, label)
        to_csv(path, resource_requirement, jump_points, weights, bounds,
               resource_availability)


if __name__ == "__main__":
    main()
