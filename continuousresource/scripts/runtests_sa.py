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
from continuousresource.probleminstances.jumppointinstance \
    import JumpPointInstance
from continuousresource.mathematicalprogramming.linprog \
    import LPWithSlack, OrderBasedSubProblemWithSlack
from continuousresource.localsearch.localsearch \
    import simulated_annealing, simulated_annealing_verbose
from continuousresource.localsearch.searchspace_jobarray \
    import JobArraySearchSpaceCombined


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
@click.option(
    '--verbose',
    '-v',
    'verbose',
    is_flag=True,
    help="Log extensive information on the runs."
)
def main(format, path, output_dir, label, verbose):
    """Perform simulated annealing runs for all instances within the
    given directory.

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
    verbose : bool
        Log extensive information on the runs.
    """
    # Vary parameters here
    model_class = OrderBasedSubProblemWithSlack
    sp_class = JobArraySearchSpaceCombined
    instance_class = JobPropertiesInstance
    slackpenalties = [5, 5]
    sa_params = {
        'initial_temperature_func': (lambda n: 200 * n),
        'alfa': 0.95,
        'alfa_period_func': (lambda n: (2 * n - 1) * 2),
        'cutoff_func': (lambda n: (2 * n - 1) * 8 * 50)
    }
    spp = {
        'infer_precedence': True,
        'fracs': {"swap": 0.75, "move": 0.15, "movepair": 0.1},
        'start_solution': "greedy"
    }
    search_space = sp_class(spp)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    run_on_instances(format, path, output_dir, label, verbose, sa_params,
                     search_space, model_class, instance_class,
                     slackpenalties)


def run_on_instances(format, path, output_dir, label, verbose, sp,
                     search_space, model_class, instance_class,
                     slackpenalties=None):
    """Perform simulated annealing runs for all instances within the
    given directory.

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
    verbose : bool
        Log extensive information on the runs.
    sp : dict
        Dictionary with values for the parameters used to tune the
        simulated annealing approach, with the following keys:
            - `initial_temperature_func`: function (lambda n) that
              computes the initial temperature;
            - `alfa`: float indicating the cooling factor;
            - `alfa_period_func`: function (lambda n) that computes the
              number of iterations between cooling events;
            - `cutoff_func`: function (lambda n) that computes the
              maximum number of iterations.
    search_space : SearchSpace
        Search space used in simulated annealing.
    model_class : type
        Name of the class used to model the LP subproblem in.
    instance_class : type
        Name of the class representing an instance.
    slackpenalties : list of float
        List of penalty coefficients for slack variables.
    """
    with open(os.path.join(output_dir,
                           f"{label}_summary.csv"), "w") as csv:
        if instance_class == JobPropertiesInstance:
            third_param = 'a'
        elif instance_class == JumpPointInstance:
            third_param = 'k'
        csv.write(
            f'n;r;{third_param};i;T_init;alfa;alfa_period;stop;#iter;'
            'init_time;init_score;time;best;slack\n'
        )
        for inst in os.listdir(path):
            if format == 'binary':
                if inst.endswith(".npz"):
                    instance = instance_class.from_binary(
                        os.path.join(path, inst)
                    )
                    instance_name = inst[:-4]
                else:
                    continue
            elif format == 'csv':
                if os.path.isdir(os.path.join(path, inst)):
                    instance = instance_class.from_csv(
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
            if instance_class == JobPropertiesInstance:
                params = re.match(r'.*n(\d+)r(\d+.\d+)a?([01])?i?(\d+)?',
                                  instance_name)
                dummy_eventlist = np.array([
                    [e, j]
                    for j in range(len(instance['jobs']))
                    for e in [0, 1]
                ])
            elif instance_class == JumpPointInstance:
                params = re.match(r'.*n(\d+)r(\d+.\d+)k(\d+)i(\d+)',
                                  instance_name)
                dummy_eventlist = np.array([
                    [e, j]
                    for j in range(len(instance['properties']))
                    for e in range(instance['jumppoints'].shape[1])
                ])

            sp['alfa_period'] = sp['alfa_period_func'](int(params.group(1)))
            sp['cutoff'] = sp['cutoff_func'](int(params.group(1)))
            sp['initial_temperature'] = \
                sp['initial_temperature_func'](int(params.group(1)))

            t_start = time.perf_counter()
            if instance_class == JobPropertiesInstance:
                sol_init_time, sol_init_score = \
                    search_space.generate_initial_solution(
                        model_class, dummy_eventlist, instance['jobs'],
                        instance['constants']['resource_availability'],
                        slackpenalties, partial_label
                    )
            elif instance_class == JumpPointInstance:
                raise NotImplementedError

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
            if isinstance(solution.model, LPWithSlack) \
               and len(solution.slack) > 0:
                for (slack_label, value, weight) in solution.slack:
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
