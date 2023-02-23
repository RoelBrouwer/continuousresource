import click
import datetime
import numpy as np
import os.path

from continuousresource.probleminstances.jumppointinstance \
    import JumpPointInstance
from continuousresource.mathematicalprogramming.flowfeasibility \
    import FeasibilityWithoutLowerbound


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
    params = {
        '5': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 2
        },
        '10': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 2
        },
        '15': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 1.5
        },
        '20': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 1.5
        },
        '30': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 1.5
        },
        '50': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 1.5
        },
        '100': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 1
        },
        '200': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 1
        },
        '300': {
            'resource_fraction': 0.25,
            'requirement_fraction': 0.25,
            'release_date_shift': 0.125,
            'processing_window_size': 0.75
        }
    }
    with open(os.path.join(exportpath,
                           f"{label}_instances_overview.csv"), "w") as csv:
        # Header
        csv.write(
            'n;k;r;serialID;feasible\n'
        )
        for n in [5, 10, 15, 20, 30, 50, 100, 200, 300]:
            for k in [3, 4, 5]:
                for r in [25.0, 50.0, 100.0, 200.0]:
                    for i in range(4):
                        instance = JumpPointInstance.generate_instance(
                            n, r, k, params=params[str(n)]
                        )

                        # Squeeze instance into a format understood by
                        # the feasibility test
                        jobs = np.empty(shape=(n, 5))
                        jobs[:, :3] = instance['properties']
                        jobs[:, 3] = instance['jumppoints'][:, 0]
                        jobs[:, 4] = instance['jumppoints'][:, -1]

                        # Solve flow problem
                        lp = FeasibilityWithoutLowerbound(
                            jobs,
                            instance['constants']['resource_availability'],
                            f'{n}-{r}-{k}-{i}'
                        )
                        lp.initialize_problem()
                        feasible = lp.solve() is not None

                        csv.write(
                            f'{n};{r:.2f};{k};{i};'
                            f'{"1" if feasible else "0"}\n'
                        )

                        inst_label = (f"{label}_n{n}r{r:.2f}k{k}i{i}")
                        if exportformat in ['both', 'binary']:
                            path = os.path.join(exportpath, inst_label)
                            JumpPointInstance.to_binary(path, instance)
                        if exportformat in ['both', 'csv']:
                            path = os.path.join(exportpath, inst_label)
                            JumpPointInstance.to_csv(path, instance)


if __name__ == "__main__":
    main()
