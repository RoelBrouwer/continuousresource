import click
import datetime
import numpy as np
import os
import os.path
import pulp
import shutil


from continuousresource.probleminstances.legacyinstance \
    import LegacyInstance
from continuousresource.mathematicalprogramming.mipmodels \
    import TimeIndexedNoDeadline


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--format',
    '-f',
    'format',
    required=True,
    type=click.Choice(['binary', 'csv'], case_sensitive=False),
    help="Input format of the provided instance."
)
@click.option(
    '--path',
    '-p',
    'path',
    required=True,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to the binary file or the folder containing the csv files."
)
@click.option(
    '--model',
    '-m',
    'model',
    required=True,
    type=click.Choice(['ti'], case_sensitive=False),
    help="Type of model to be used."
)
@click.option(
    '--solver',
    '-s',
    'solver',
    required=True,
    type=click.Choice(['cplex', 'glpk', 'gurobi'], case_sensitive=False),
    help="LP solver to be used for the problem."
)
@click.option(
    '--output-dir',
    '-o',
    'output_dir',
    required=False,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    default=os.getcwd(),
    help="Path to the output directory."
)
@click.option(
    '--label',
    '-l',
    'label',
    required=False,
    type=str,
    default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"),
    help="Sufficiently unique label or name for the instance."
)
def main(format, path, model, solver, output_dir, label):
    """Solve a single instance of a single machine scheduling problem
    with a regular step cost function.

    Parameters
    ----------
    format : {'binary', 'csv'}
        Input format of the provided instance. Currently, binary files
        and csv files are supported as input, provided they conform to
        the expected lay-out (see specialized read functions for more
        details).
    path : str
        Path to the binary file or the folder containing the csv files.
    model : {'dof'}
        The type of model that will be used to model the problem. Either
        Time Indexed (ti), or [tbi].
    solver : {'glpk', 'gurobi', 'cplex'}
        The solver that will be used to solve the LP constructed from the
        provided instance.
    output_dir : str
        Path to the output directory.
    label : str
        Label or name for the (solved) instance. Be sure to use one that
        is sufficiently unique, to avoid conflicts or other unexpected
        behavior.
    """
    # Read instance
    if format == 'binary':
        instance = LegacyInstance.from_binary(path)
    elif format == 'csv':
        instance = LegacyInstance.from_csv(path)
    else:
        raise ValueError("Unsupported input format, must be binary or csv.")

    if model == 'ti':
        mdl = TimeIndexedNoDeadline(instance, label)
    else:
        raise ValueError(f"'{model}' is not recognized as an implemented"
                         " model")

    mdl.solve(solver)

    # Print solution
    mdl.print_solution()


if __name__ == "__main__":
    main()
