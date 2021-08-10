import click
import datetime
import os.path

from continuousresource.instance import to_binary, to_csv, generate_instance


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
    for n in [10, 20, 50, 100]:  # [10, 20, 50, 100, 200]:
        for k in [2, 3, 4, 9]:
            for m in [25.0, 50.0, 100.0, 200.0]:
                for s in [0.05 * m, 0.1 * m, 0.2 * m]:
                    for r in [True]:  # , False]:
                        (resource_requirement, jump_points, weights, bounds,
                         resource_availability) = generate_instance(n, m, s,
                                                                    k, r)

                        inst_label = f"n{n}k{k}m{m:.2f}" + \
                                     f"s{(s / m):.2f}r{'1' if r else '0'}"
                        if exportformat in ['both', 'binary']:
                            path = os.path.join(exportpath, inst_label)
                            to_binary(path, resource_requirement, jump_points,
                                      weights, bounds, resource_availability)
                        if exportformat in ['both', 'csv']:
                            path = os.path.join(exportpath, inst_label)
                            to_csv(path, resource_requirement, jump_points,
                                   weights, bounds, resource_availability)


if __name__ == "__main__":
    main()
