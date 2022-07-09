import click
import datetime
import json
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
    import simulated_annealing, simulated_annealing_verbose
from continuousresource.simulatedannealing.searchspace \
    import SearchSpaceSwap, SearchSpaceMove, SearchSpaceMovePair, \
    SearchSpaceCombined, SearchSpaceMoveLinear


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
    sp_class = SearchSpaceCombined
    temps = [30, 45, 60, 75]
    alfas = [0.95, 0.99]
    period_multiplier = [2, 4, 6, 8, 10]
    fracs = [
        {"swap": 1.00, "move": 0.0, "movepair": 0.0},
        {"swap": 0.9, "move": 0.1, "movepair": 0.0},
        {"swap": 0.9, "move": 0.05, "movepair": 0.05},
        {"swap": 0.75, "move": 0.25, "movepair": 0.0},
        {"swap": 0.75, "move": 0.15, "movepair": 0.1}
    ]
    slackpenalties = [5, 5]

    for temperature in temps:
        for alfa in alfas:
            for multiplier in period_multiplier:
                for idx, frac in enumerate(fracs):
                    # if (temperature == 15 and alfa == 0.95 and multiplier in [2, 4, 6]):
                        # continue
                    if idx == 0 or temperature == 15 or alfa == 0.99 or idx == 2 or multiplier == 2 or multiplier == 6:
                        continue

                    sp = {
                        'initial_temperature': temperature,  # 50% acceptence for diff ~20
                        'alfa': alfa,
                        # neighborhood size = 2n - 1 for adjacent swap
                        'alfa_period_func': (lambda n: (2 * n - 1) * multiplier),
                        'cutoff_func': (lambda n: (2 * n - 1) * 500)
                        # Cool off to 4.34 for 20; 0.22 for 1 to accept only 1%
                    }
                    model_class = OrderBasedSubProblemWithSlack

                    spp = {
                        'infer_precedence': True,
                        'fracs': frac,
                        'start_solution': "greedy"
                    }
                    search_space = sp_class(spp)
                    output_dir2 = os.path.join(output_dir, f'temp{temperature}alfa{alfa}mult{multiplier}frac{idx}')
                    if not os.path.isdir(output_dir2):
                        os.mkdir(output_dir2)
                    run_on_instances(format, path, solver, output_dir2, label, verbose,
                                     sp, spp, search_space, model_class, slackpenalties)

    # for spp in spps:
        # search_space = SearchSpaceCombined(spp)
        # output_dir2 = os.path.join(output_dir, search_space.name)
        # if not os.path.isdir(output_dir2):
            # os.mkdir(output_dir2)
        # run_on_instances(format, path, solver, output_dir2, label, verbose,
                         # sp, spp, search_space, model_class, slackpenalties)


def run_on_instances(format, path, solver, output_dir, label, verbose,
                     sp, spp, search_space, model_class, slackpenalties=None):
    # TODO: document
    with open(os.path.join(output_dir,
                           f"{label}_summary.csv"), "w") as csv:
        csv.write(
            'n;r;a;i;T_init;alfa;alfa_period;stop;#iter;init_time;init_score;'
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

            sp['alfa_period'] = sp['alfa_period_func'](int(params.group(1)))
            sp['cutoff'] = sp['cutoff_func'](int(params.group(1)))

            t_start = time.perf_counter()
            sol_init_time, sol_init_score = \
                search_space.generate_initial_solution(
                    model_class, dummy_eventlist, instance['jobs'],
                    instance['constants']['resource_availability'],
                    slackpenalties, partial_label
                )

            if verbose:
                iters, solution = simulated_annealing_verbose(
                    search_space, sp,
                    output_dir=os.path.join(output_dir, instance_name)
                )
            else:
                iters, solution = simulated_annealing(
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

            # Write diagnostics to json files
            json.dump(
                search_space.timings, 
                open(os.path.join(output_dir, instance_name,
                                  f"{partial_label}_timings.json"), 'w')
            )
            json.dump(
                search_space.operator_data, 
                open(os.path.join(output_dir, instance_name,
                                  f"{partial_label}_operator_data.json"), 'w')
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
                f'{params.group(4)};{sp["initial_temperature"]};{sp["alfa"]};'
                f'{sp["alfa_period"]};{sp["cutoff"]};{iters};{sol_init_time};'
                f'{sol_init_score};{t_end - t_start};{solution.score};'
                f'{total_slack}\n'
            )

            # Move all files with names starting with the label
            for file in os.listdir(os.getcwd()):
                if file.startswith(partial_label):
                    shutil.move(file, os.path.join(output_dir, instance_name))


if __name__ == "__main__":
    main()
