import click
import datetime
import os.path

from continuousresource.probleminstances.jobarrayinstance \
    import JobPropertiesInstance


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
    '--resource_avail',
    '-r',
    'resource_availability',
    required=False,
    default=1,
    type=float,
    help="Average amount of available resource."
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
def main(njobs, resource_availability, exportformat, exportpath, label):
    """Generate a single instance of ...

    Parameters
    ----------
    njobs : int
        Number of jobs in the generated instance.
    resource_availability : float
        Mean of a normal distribution from which the amount of available
        resource at each time step will be drawn.
    exportformat : {'both', 'binary', 'csv'}
        Export format for the provided instance.
    exportpath : str
        Path to the output folder.
    label : str
        Label or name for the generated instance.
    """
    instance = JobPropertiesInstance.generate_instance(njobs,
                                                       resource_availability)

    if exportformat in ['both', 'binary']:
        path = os.path.join(exportpath, label)
        JobPropertiesInstance.to_binary(path, instance)
    if exportformat in ['both', 'csv']:
        path = os.path.join(exportpath, label)
        JobPropertiesInstance.to_csv(path, instance)


if __name__ == "__main__":
    main()
