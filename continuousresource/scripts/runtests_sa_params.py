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
    import JumpPointSearchSpaceLP, JumpPointSearchSpaceTest, \
    JumpPointSearchSpaceMix

from continuousresource.localsearch import distributions as dists


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
####
@click.option(
    '--init-temp-mult',
    'init_temp_mult',
    required=False,
    type=float,
    default=1.0,
    help="Initial temperature will be n * init_temp_mult."
)
@click.option(
    '--alfa',
    'alfa',
    required=False,
    type=float,
    default=0.95,
    help="Fraction of temperature after each update."
)
@click.option(
    '--alfa-period-mult',
    'alfa_period_mult',
    required=False,
    type=float,
    default=4.0,
    help=("Multiple of 2n, number of iterations between each temperature" \
          " update.")
)
@click.option(
    '--cutoff-mult',
    'cutoff_mult',
    required=False,
    type=float,
    default=100,
    help="Multiple of alfa_period_mult, maximum number of iterations."
)
@click.option(
    '--start-solution',
    'start_solution',
    required=False,
    type=click.Choice(['greedy', 'random'], case_sensitive=False),
    default='greedy',
    help="Method used to generate an initial solution."
)
@click.option(
    '--slack-value',
    'slack_value',
    required=False,
    type=float,
    default=5,
    help="Penalty term multiplier."
)
@click.option(
    '--tabu-length',
    'tabu_length',
    required=False,
    type=int,
    default=0,
    help="Penalty term multiplier."
)
@click.option(
    '--approach',
    'approach',
    required=False,
    type=click.Choice([
        'jobarray',
        'jumppoint-lp',
        'jumppoint-test',
        'jumppoint-mix'
    ], case_sensitive=False),
    default='jumppoint-mix',
    help="SA approach."
)
def main(input_format, path, output_dir, label, verbose, init_temp_mult, alfa,
         alfa_period_mult, cutoff_mult, start_solution, slack_value,
         tabu_length, approach):
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
    init_temp_mult : float
        Initial temperature will be `n` * `init_temp_mult`.
    alfa : float
        Fraction of temperature after each update. 0 < `alfa` < 1.
    alfa_period_mult : float
        Multiple of 2n, number of iterations between each temperature
        update.
    cutoff_mult : float
        Multiple of alfa_period_mult, maximum number of iterations.
    start_solution : {'greedy', 'random'}
        Method used to generate an initial solution.
    slack_value : float
        Penalty term multiplier.
    tabu_length: int
        Length of a tabulist.
    approach : {'jobarray', 'jumppoint-lp', 'jumppoint-test',
                'jumppoint-mix'}
        SA approach.
    """
    for tabu_length in [0, 1]:
        for slack_value in [0.5, 1.0, 2.0, 5.0]:
            for alfa_period_mult in [2, 4, 6, 8]:
                for init_temp_mult in [0.1, 0.2, 0.5, 1, 2, 5]:
                    # Vary parameters here
                    if approach == 'jobarray':
                        sp_class = JobArraySearchSpaceCombined
                        instance_class = JobPropertiesInstance
                        sp_params = {
                            'infer_precedence': True,
                            'fracs': {"swap": 1.0, "move": 0.0, "movepair": 0.0},
                            'start_solution': start_solution
                        }
                    else:
                        instance_class = JumpPointInstance
                        sp_params = {
                            'infer_precedence': True,
                            'dist': dists.linear,
                            'start_solution': start_solution,
                            'tabu_length': tabu_length
                        }
                        if approach == 'jumppoint-lp':
                            sp_class = JumpPointSearchSpaceLP
                        elif approach == 'jumppoint-test':
                            sp_class = JumpPointSearchSpaceTest
                        elif approach == 'jumppoint-mix':
                            sp_class = JumpPointSearchSpaceMix
                    slackpenalties = [slack_value, slack_value]
                    sa_params = {
                        'initial_temperature_func': (lambda n: init_temp_mult * n),
                        'alfa': alfa,
                        'alfa_period_func': (lambda n: (2 * n) * alfa_period_mult),
                        'cutoff_func': (lambda n: (2 * n) * alfa_period_mult * cutoff_mult)
                    }
                    
                    outputdir = os.path.join(output_dir, f't{init_temp_mult:.1f}-a{alfa_period_mult:.0f}-s{slack_value:.1f}-l{tabu_length:.0f}')

                    if not os.path.isdir(outputdir):
                        os.mkdir(outputdir)

                    run_on_instances(input_format, path, outputdir, label, verbose,
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

            total_slack = solution.slack_value
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
