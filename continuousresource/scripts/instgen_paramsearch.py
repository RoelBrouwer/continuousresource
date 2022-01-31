import click
import datetime
import os.path


from continuousresource.probleminstances.jobarrayinstance \
    import JobPropertiesInstance
from continuousresource.mathematicalprogramming.linprog \
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
    '--label',
    '-l',
    'label',
    required=False,
    type=str,
    default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"),
    help="Prefix/label or name for the report."
)
def main(exportpath, label):
    # For a given n and r, try some parameter combinations and report the
    # percentage that is feasible according to the flow problem (no lb).
    n = 5
    r = 50.0
    repeats = 100

    with open(os.path.join(exportpath,
                           f"{label}_{n}-{r:.2f}-paramsearch.csv"), "w") as csv:
        # Header
        csv.write(
            'a_ml;a_mu;a_rs;a_ps;%\n'
        )
        for a_ml in [0.125, 0.25, 0.5]:  # [10, 20, 50, 100, 200]:
            for a_mu in [0.125, 0.25, 0.5]:
                for a_rs in [0.05, 0.1, 0.125, 0.15, 0.2]:
                    for a_ps in [1, 1.5, 2, 2.5, 3]:
                        succ = 0
                        for i in range(repeats):
                            params = {
                                'resource_fraction': a_ml,
                                'requirement_fraction': a_mu,
                                'release_date_shift': a_rs,
                                'processing_window_size': a_ps
                            }

                            # Generate instance
                            inst = \
                            JobPropertiesInstance.generate_instance(
                                n, r, params=params
                            )

                            # Solve flow problem
                            lp = FeasibilityWithoutLowerbound(
                                inst['jobs'],
                                inst['constants']['resource_availability'],
                                f'{a_ml}-{a_mu}-{a_rs}-{a_ps}-{i}'
                            )
                            lp.initialize_problem()
                            if lp.solve() is not None:
                                succ += 1

                        csv.write(
                            f'{a_ml};{a_mu};{a_rs};{a_ps};{succ / repeats:.2f}\n'
                        )


if __name__ == "__main__":
    main()
