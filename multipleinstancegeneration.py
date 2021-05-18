import click
import datetime
import os.path

from instance import to_binary, to_csv, generate_instance


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
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
    '--export-format',
    '-f',
    'exportformat',
    required=True,
    type=click.Choice(['binary', 'csv', 'both'], case_sensitive=False),
    default='both',
    help="Export format for the provided instance."
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
def main(exportpath, exportformat, label):
    # TODO: think of useful parameters
    for n in [2, 3]:  # [10, 20, 50, 100, 200]:
        for k in [2, 3]:
            for m in [1.0, 2.0]:
                for s in [0.5, 1.0]:
                    for r in [True, False]:
                        (resource_requirement, jump_points, weights, bounds,
                         resource_availability) = generate_instance(n, m, s,
                                                                    k, r)

                        if exportformat in ['both', 'binary']:
                            path = os.path.join(exportpath, label)
                            to_binary(path, resource_requirement, jump_points,
                                      weights, bounds, resource_availability)
                        if exportformat in ['both', 'csv']:
                            path = os.path.join(exportpath, label)
                            to_csv(path, resource_requirement, jump_points,
                                   weights, bounds, resource_availability)


if __name__ == "__main__":
    main()
