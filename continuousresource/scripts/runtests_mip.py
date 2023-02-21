import click
import datetime
import os
import os.path
import re
import shutil
import time

from docplex.mp.utils import DOcplexException

from continuousresource.probleminstances.jobarrayinstance \
    import JobPropertiesInstance
from continuousresource.probleminstances.jumppointinstance \
    import JumpPointInstance
from continuousresource.mathematicalprogramming.mipmodels \
    import JobPropertiesContinuousMIPPlus


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--format',
    '-f',
    'format',
    required=True,
    type=click.Choice(['binary', 'csv'], case_sensitive=False),
    help="Input format of the provided instances."
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
    help="Path to the folder containing the instances."
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
    help="Sufficiently unique label or name for the run."
)
def main(format, path, output_dir, label):
    """Run the MIP model over all instances within the given directory.

    Parameters
    ----------
    format : {'binary', 'csv'}
        Input format of the provided instances.
    path : str
        Path to the folder containing the instances.
    output_dir : str
        Path to the output directory.
    label : DateTime
        Sufficiently unique label or name for the run.
    """
    timelimit = 3600

    with open(os.path.join(output_dir,
                           f"{label}_summary.csv"), "w") as csv:
        csv.write(
            'n;r;a;i;timelimit;time;objective\n'
        )
        for inst in os.listdir(path):
            if format == 'binary':
                if inst.endswith(".npz"):
                    instance = JobPropertiesInstance.from_binary(
                        os.path.join(path, inst)
                    )
                    instance_name = inst[:-4]
                else:
                    continue
            elif format == 'csv':
                if os.path.isdir(os.path.join(path, inst)):
                    instance = JobPropertiesInstance.from_csv(
                        os.path.join(path, inst)
                    )
                    instance_name = inst
                else:
                    continue
            else:
                raise ValueError("Unsupported input format, must be binary or"
                                 " csv.")

            # Make output_dir for this instance
            os.mkdir(os.path.join(output_dir, instance_name))

            partial_label = f"{label}_{instance_name}"
            params = re.match(r'.*n(\d+)r(\d+.\d+)a?([01])?i?(\d+)?',
                              instance_name)

            with open(os.path.join(output_dir, instance_name,
                                   "cplex.log"), "w") as cplexlog:
                t_start = time.perf_counter()
                mip = JobPropertiesContinuousMIPPlus(
                    instance,
                    f"{partial_label}_cont_mip",
                )
                mip._problem.context.solver.log_output = cplexlog
                mip.solve(timelimit)
                t_end = time.perf_counter()

            obj = 0.0
            try:
                obj = mip.problem.objective_value

                # Print solution to file
                with open(os.path.join(output_dir, instance_name,
                                       "solution.csv"), "w") as sol:
                    sol.write(mip.get_solution_csv())

            except DOcplexException:
                obj = -1.0

            # Write timings to text file
            with open(os.path.join(output_dir, instance_name,
                                   f"{partial_label}_timings.txt"),
                      "w") as txt:
                txt.write(
                    f"""
Total measured time (s): {t_end - t_start}
                    """
                )

            # Build-up CSV-file
            # TODO: fix if either a or i is not included in filename
            csv.write(
                f'{params.group(1)};{params.group(2)};{params.group(3)};'
                f'{params.group(4)};{timelimit};{t_end - t_start};{obj}\n'
            )

            # Move all files with names starting with the label
            for file in os.listdir(os.getcwd()):
                if file.startswith(partial_label):
                    shutil.move(file, os.path.join(output_dir, instance_name))


if __name__ == "__main__":
    main()
