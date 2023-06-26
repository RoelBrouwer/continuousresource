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
from continuousresource.mathematicalprogramming.abstract \
    import LPWithSlack
from continuousresource.localsearch.localsearch \
    import simulated_annealing, simulated_annealing_verbose
from continuousresource.localsearch.searchspace_jobarray \
    import JobArraySearchSpaceCombined
from continuousresource.localsearch.searchspace_jumppoint \
    import JumpPointSearchSpaceCombined


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--format',
    '-f',
    'input_format',
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
def main(input_format, path, output_dir, label, verbose):
    """Perform simulated annealing runs for all instances within the
    given directory.

    Parameters
    ----------
    input_format : {'binary', 'csv'}
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
    sp_class = JumpPointSearchSpaceCombined
    instance_class = JumpPointInstance
    slackpenalties = [5, 5]
    sa_params = {
        'initial_temperature_func': (lambda n: n),
        'alfa': 0.95,
        'alfa_period_func': (lambda n: (2 * n - 1) * 4),
        'cutoff_func': (lambda n: (2 * n - 1) * 8 * 50)
    }
    sp_params = {
        'infer_precedence': True,
        'fracs': {"swap": 1.0, "move": 0.0, "movepair": 0.0},
        'start_solution': "greedy"
    }

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    run_on_instances(input_format, path, output_dir, label, verbose,
                     sa_params, sp_class, sp_params, instance_class,
                     slackpenalties)


def run_on_instances(input_format, path, output_dir, label, verbose,
                     sa_params, sp_class, sp_params, instance_class,
                     slackpenalties=None):
    """Perform simulated annealing runs for all instances within the
    given directory.

    Parameters
    ----------
    input_format : {'binary', 'csv'}
        Input format of the provided instances.
    path : str
        Path to the folder containing the instances.
    output_dir : str
        Path to the output directory.
    label : DateTime
        Sufficiently unique label or name for the run.
    verbose : bool
        Log extensive information on the runs.
    sa_params : dict
        Dictionary with values for the parameters used to tune the
        simulated annealing approach, with the following keys:
            - `initial_temperature_func`: function (lambda n) that
              computes the initial temperature;
            - `alfa`: float indicating the cooling factor;
            - `alfa_period_func`: function (lambda n) that computes the
              number of iterations between cooling events;
            - `cutoff_func`: function (lambda n) that computes the
              maximum number of iterations.
    sp_class : type
        Search space used in simulated annealing.
    sp_params : dict
        Dictionary containing parameters defining the search space, with
        the following keys:
            - `infer_precedence` (bool): Flag indicating whether to infer
              and continuously check (implicit) precedence relations.
            - `fracs` (dict of float): Dictionary indicating the
              probability of selecting each neighborhood operator.
            - `start_solution` (str): String indicating the method of
              generating a starting solution. Either "random" or
              "greedy".
    instance_class : type
        Name of the class representing an instance.
    slackpenalties : list of float
        List of penalty coefficients for slack variables.
    """
    # TODO: initialize searchspace
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
            if input_format == 'binary':
                if inst.endswith(".npz"):
                    instance = instance_class.from_binary(
                        os.path.join(path, inst)
                    )
                    instance_name = inst[:-4]
                else:
                    continue
            elif input_format == 'csv':
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

            # Add slackpenalties information to the instance
            instance['constants']['slackpenalties'] = slackpenalties

            # Make output_dir for this instance
            os.mkdir(os.path.join(output_dir, instance_name))

            partial_label = f"{label}_{instance_name}"
            if instance_class == JobPropertiesInstance:
                params = re.match(r'.*n(\d+)r(\d+.\d+)a?([01])?i?(\d+)?',
                                  instance_name)
            elif instance_class == JumpPointInstance:
                params = re.match(r'.*n(\d+)r(\d+.\d+)k(\d+)i(\d+)',
                                  instance_name)

            # Get parameters for the SA
            sa_params['alfa_period'] = \
                sa_params['alfa_period_func'](int(params.group(1)))
            sa_params['cutoff'] = \
                sa_params['cutoff_func'](int(params.group(1)))
            sa_params['initial_temperature'] = \
                sa_params['initial_temperature_func'](int(params.group(1)))

            # Set log directory (may be used)
            sp_params['logdir'] = os.path.join(output_dir, instance_name)

            t_start = time.perf_counter()

            # Initialize the search space
            search_space = sp_class(instance, sp_params)

            if verbose:
                iters, solution = simulated_annealing_verbose(
                    search_space, sa_params,
                    output_dir=os.path.join(output_dir, instance_name)
                )
            else:
                iters, solution = simulated_annealing(
                    search_space, sa_params
                )

            t_end = time.perf_counter()

            # Print solution (eventlist) to file
            np.savetxt(os.path.join(output_dir, instance_name,
                                    f"{partial_label}_solution.txt"),
                       solution.eventlist)
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
Start solution (s): {search_space.timings["initial_solution"]}
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

            total_slack = solution.slack
            # total_slack = 0
            # if isinstance(search_space.lp, LPWithSlack) \
            #    and len(solution.slack) > 0:
            #     for (slack_label, value, weight) in solution.slack:
            #         total_slack += value * weight

            # Build-up CSV-file
            csv.write(
                f'{params.group(1)};{params.group(2)};{params.group(3)};'
                f'{params.group(4)};{sa_params["initial_temperature"]};'
                f'{sa_params["alfa"]};{sa_params["alfa_period"]};'
                f'{sa_params["cutoff"]};{iters};'
                f'{search_space.timings["initial_solution"]};'
                f'{search_space.initial.score};{t_end - t_start};'
                f'{solution.score};{total_slack}\n'
            )

            # Move all files with names starting with the label
            for file in os.listdir(os.getcwd()):
                if file.startswith(partial_label):
                    shutil.move(file, os.path.join(output_dir, instance_name))


if __name__ == "__main__":
    main()
