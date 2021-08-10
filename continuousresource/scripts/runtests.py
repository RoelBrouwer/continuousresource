import click
import datetime
import os
import os.path
import pulp
import re
import shutil
import time

from continuousresource.instance import from_binary, from_csv
from continuousresource.mipmodels import TimeIndexedNoDeadline


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
    help="Path to the folder containing the instances."
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
    help="Sufficiently unique label or name for the run."
)
@click.option(
    '--skip',
    '-a',
    'skip_approaches',
    required=False,
    multiple=True,
    type=click.Choice(['ti_mip'], case_sensitive=False),
    help="Approaches that should be excluded from the current run."
)
def main(format, path, solver, output_dir, label, skip_approaches):
    # TODO: document
    with open(os.path.join(output_dir,
                           f"{label}_summary.csv"), "w") as csv:
        csv.write(
            'n;k;m;s;r;'
            'ti_mip_t;ti_mip_c;ti_mip_r_t;ti_mip_r_c\n'
        )
        for inst in os.listdir(path):
            if format == 'binary':
                if inst.endswith(".npz"):
                    instance = from_binary(os.path.join(path, inst))
                    instance_name = inst[:-4]
            elif format == 'csv':
                if os.path.isdir(inst):
                    instance = from_csv(os.path.join(path, inst))
                    instance_name = inst
            else:
                raise ValueError("Unsupported input format, must be binary or"
                                 " csv.")

            # Make output_dir for this instance
            os.mkdir(os.path.join(output_dir, instance_name))

            partial_label = f"{label}_{instance_name}"
            params = re.match(r'n(\d+)k(\d+)m(\d+.\d+)s(\d+.\d+)r(\d+)',
                              instance_name)

            if 'ti_mip' not in skip_approaches:
                # TI based MIP
                t_start_ti_mip = time.perf_counter()
                mip = TimeIndexedNoDeadline(instance,
                                            f"{partial_label}_ti_mip")
                mip.solve(solver)
                t_end_ti_mip = time.perf_counter()
                ti_mip_c = pulp.value(mip.problem.objective)
                if ti_mip_c == None:
                    ti_mip_c = -1.0

                # TI based relaxed MIP
                t_start_ti_mip_relax = time.perf_counter()
                mip = TimeIndexedNoDeadline(instance,
                                            f"{partial_label}_ti_mip_relaxed")
                mip.relax_problem()
                mip.solve(solver)
                t_end_ti_mip_relax = time.perf_counter()
                ti_mip_r_c = pulp.value(mip.problem.objective)
                if ti_mip_r_c == None:
                    ti_mip_r_c = -1.0
            else:
                # Skip TI, as it is really slow
                t_start_ti_mip = -1.0
                t_end_ti_mip = -1.0
                ti_mip_c = -1.0
                t_start_ti_mip_relax = -1.0
                t_end_ti_mip_relax = -1.0
                ti_mip_r_c = -1.0

            # Write timings to text file
            with open(os.path.join(output_dir, instance_name,
                                   f"{partial_label}_timings.txt"),
                      "w") as txt:
                txt.write(
                    f"""
TI based MIP (s): {t_end_ti_mip - t_start_ti_mip:0.2f}
TI based MIP relaxation (s): {t_end_ti_mip_relax - t_start_ti_mip_relax:0.2f}
                    """
                )

            # Build-up CSV-file
            csv.write(
                f'{params.group(1)};{params.group(2)};{params.group(3)};'
                f'{params.group(4)};{params.group(5)};'
                f'{t_end_ti_mip - t_start_ti_mip:0.2f};{ti_mip_c:.0f};'
                f'{t_end_ti_mip_relax - t_start_ti_mip_relax:0.2f};'
                f'{ti_mip_r_c:.0f}\n'
            )

            # Move all files with names starting with the label
            for file in os.listdir(os.getcwd()):
                if file.startswith(partial_label):
                    shutil.move(file, os.path.join(output_dir, instance_name))


if __name__ == "__main__":
    main()
