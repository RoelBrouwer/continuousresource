import click
import datetime
import os
import os.path
import re
import shutil
import time

import numpy as np

from continuousresource.probleminstances.jobarrayinstance \
    import JobPropertiesInstance
from continuousresource.mathematicalprogramming.linprog \
    import OrderBasedSubProblemWithSlack
from continuousresource.simulatedannealing.simulatedannealing \
    import variable_neighborhood_descent
from continuousresource.simulatedannealing.searchspace \
    import SearchSpaceHillClimb


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
    required=False,
    type=click.Choice(['cplex', 'glpk', 'gurobi'], case_sensitive=False),
    default='cplex',
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
    '--verbose',
    '-v',
    'verbose',
    is_flag=True,
    help="Log extensive information on the runs."
)
def main(format, path, solver, output_dir, label, verbose):
    # Vary parameters here
    slackpenalties = [5, 5]
    sp = {
    }
    spp = {
        'infer_precedence': False,
        'fracs': {
            "swap": 0.95,
            "move": 0.025,
            "movepair": 0.025
        }
    }
    search_space = SearchSpaceHillClimb(spp)
    model_class = OrderBasedSubProblemWithSlack
    run_on_instances(format, path, solver, output_dir, label, verbose,
                     sp, spp, search_space, model_class, slackpenalties)


def run_on_instances(format, path, solver, output_dir, label, verbose,
                     sp, spp, search_space, model_class, slackpenalties=None):
    # TODO: document
    with open(os.path.join(output_dir,
                           f"{label}_summary.csv"), "w") as csv:
        csv.write(
            'n;r;a;i;#iter;init_time;init_score;'
            'time;best;slack\n'
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

            dummy_eventlist = np.array([
                [e, j]
                for j in range(len(instance['jobs']))
                for e in [0, 1]
            ])

            t_start = time.perf_counter()
            sol_init_time, sol_init_score = \
                search_space.generate_initial_solution(
                    model_class, dummy_eventlist, instance['jobs'],
                    instance['constants']['resource_availability'],
                    slackpenalties, partial_label
                )

            if verbose:
                raise NotImplementedError
                # iters, solution = variable_neighborhood_descent_verbose(
                #     search_space, sp,
                #     output_dir=os.path.join(output_dir, instance_name)
                # )
            else:
                iters, solution = variable_neighborhood_descent(
                    search_space, sp
                )

            t_end = time.perf_counter()

            # Print solution (eventlist) to file
            np.savetxt(os.path.join(output_dir, instance_name,
                                    f"{partial_label}_solution.txt"),
                       solution.eventorder)
            with open(os.path.join(output_dir, instance_name,
                                   f"{partial_label}_solution.txt"),
                      "a") as txt:
                txt.write(
                    f"\nScore: {solution.score}"
                )

            # Write timings to text file
            with open(os.path.join(output_dir, instance_name,
                                   f"{partial_label}_timings.txt"),
                      "w") as txt:
                txt.write(
                    f"""
Start solution (s): {sol_init_time}
Total time (s): {t_end - t_start}
                    """
                )

            total_slack = 0
            if solution.model.with_slack and len(solution.slack) > 0:
                for (slack_label, value, weight) in solution.slack:
                    # print(f'{slack_label}: {value} * {weight}')
                    total_slack += value * weight

            # Build-up CSV-file
            # TODO: fix if either a or i is not included in filename
            csv.write(
                f'{params.group(1)};{params.group(2)};{params.group(3)};'
                f'{params.group(4)};'
                f'{iters};{sol_init_time};'
                f'{sol_init_score};{t_end - t_start};{solution.score};'
                f'{total_slack}\n'
            )

            # Move all files with names starting with the label
            for file in os.listdir(os.getcwd()):
                if file.startswith(partial_label):
                    shutil.move(file, os.path.join(output_dir, instance_name))


if __name__ == "__main__":
    main()