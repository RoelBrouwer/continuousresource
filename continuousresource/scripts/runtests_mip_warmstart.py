import click
import datetime
import os
import os.path
import re
import shutil
import time

import numpy as np

import cplex
from docplex.mp.utils import DOcplexException

from continuousresource.probleminstances.jobarrayinstance \
    import JobPropertiesInstance
from continuousresource.mathematicalprogramming.mipmodels \
    import ContinuousResourceMIP
from continuousresource.mathematicalprogramming.mipmodels \
    import ContinuousResourceMIPPlus
from continuousresource.mathematicalprogramming.linprog \
    import OrderBasedSubProblemWithSlack
from continuousresource.simulatedannealing.simulatedannealing \
    import simulated_annealing
from continuousresource.simulatedannealing.searchspace \
    import SearchSpaceSwap


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
    help="Log extensive information on the runs. Not used."
)
def main(format, path, solver, output_dir, label, verbose):
    # TODO: document
    timelimit = 3600
    slackpenalties = [5, 5]
    sp = {
        'initial_temperature': 30,  # 50% acceptence for diff ~20
        'alfa': 0.95,
        # neighborhood size = 2n - 1 for adjacent swap
        'alfa_period_func': (lambda n: (2 * n - 1) * 8),
        'cutoff_func': (lambda n: (2 * n - 1) * 8 * 50)
        # Cool off to 4.34 for 20; 0.22 for 1 to accept only 1%
    }
    spp = {
        'infer_precedence': False,
        'fracs': {
            "swap": 0.95,
            "move": 0.025,
            "movepair": 0.025
        },
        'start_solution': "greedy"
    }
    search_space = SearchSpaceSwap(spp)
    model_class = OrderBasedSubProblemWithSlack

    with open(os.path.join(output_dir,
                           f"{label}_summary.csv"), "w") as csv:
        csv.write(
            'n;r;a;i;timelimit;mip_time;mip_objective;sa_time;sa_obj;'
            'sa_slack;mip_time_warm;mip_warm_objective\n'
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

            if solver == 'cplex':
                with cplex.Cplex() as cpx, \
                 open(os.path.join(output_dir, instance_name,
                                   "cplex.log"), "w") as cplexlog:
                    cpx.set_results_stream(cplexlog)
                    cpx.set_warning_stream(cplexlog)
                    cpx.set_error_stream(cplexlog)
                    cpx.set_log_stream(cplexlog)

            # MIP section
            mip_start = time.perf_counter()
            mip = ContinuousResourceMIPPlus(
                instance,
                f"{partial_label}_cont_mip",
                solver
            )
            mip.solve(timelimit)
            mip_end = time.perf_counter()

            mip_obj = 0.0
            try:
                mip_obj = mip.problem.objective_value

                # Print solution to file
                with open(os.path.join(output_dir, instance_name,
                                       "mip_solution.csv"), "w") as sol:
                    sol.write(mip.get_solution_csv())

            except DOcplexException:
                mip_obj = -1.0

            # MIP warmstart section
            # Get SA solution

            dummy_eventlist = np.array([
                [e, j]
                for j in range(len(instance['jobs']))
                for e in [0, 1]
            ])

            sp['alfa_period'] = sp['alfa_period_func'](int(params.group(1)))
            sp['cutoff'] = sp['cutoff_func'](int(params.group(1)))

            sa_start = time.perf_counter()
            sol_init_time, sol_init_score = \
                search_space.generate_initial_solution(
                    model_class, dummy_eventlist, instance['jobs'],
                    instance['constants']['resource_availability'],
                    slackpenalties, partial_label
                )

            iters, solution = simulated_annealing(
                search_space, sp
            )

            sa_end = time.perf_counter()

            # Print solution (eventlist) to file
            np.savetxt(os.path.join(output_dir, instance_name,
                                    f"{partial_label}_sa_solution.txt"),
                       solution.eventorder)
            with open(os.path.join(output_dir, instance_name,
                                   f"{partial_label}_sa_solution.txt"),
                      "a") as txt:
                txt.write(
                    f"\nScore: {solution.score}"
                )

            sa_slack = 0
            if solution.model.with_slack and len(solution.slack) > 0:
                for (slack_label, value, weight) in solution.slack:
                    # print(f'{slack_label}: {value} * {weight}')
                    sa_slack += value * weight

            # We need to revert the model to its best state, rather than
            # the last-seen state.
            best_model = model_class(
                solution.eventorder, instance['jobs'],
                instance['constants']['resource_availability'],
                slackpenalties=slackpenalties, label=f"{partial_label}_best"
            )
            best_model.initialize_problem()
            best_model.solve()

            t_vars = best_model._times
            p_vars = best_model._resource
            event_order = best_model.event_list

            # Warmstart MIP
            wmip_start = time.perf_counter()
            wmip = ContinuousResourceMIPPlus(
                instance,
                f"{partial_label}_warm_cont_mip",
                solver
            )
            wmip.add_warmstart(t_vars, p_vars, event_order)
            wmip.solve(timelimit)
            wmip_end = time.perf_counter()

            wmip_obj = 0.0
            try:
                wmip_obj = wmip.problem.objective_value

                # Print solution to file
                with open(os.path.join(output_dir, instance_name,
                                       "wmip_solution.csv"), "w") as sol:
                    sol.write(wmip.get_solution_csv())

            except DOcplexException:
                wmip_obj = -1.0

            # Write timings to text file
            with open(os.path.join(output_dir, instance_name,
                                   f"{partial_label}_wmip_timings.txt"),
                      "w") as txt:
                txt.write(
                    f"""
Start solution (s): {sol_init_time}
SA time (s): {sa_end - sa_start}
MIP time (s): {wmip_end - wmip_start}
Total measured time (s): {sa_end - sa_start + wmip_end - wmip_start}
                    """
                )

            # Build-up CSV-file
            # TODO: fix if either a or i is not included in filename
            csv.write(
                f'{params.group(1)};{params.group(2)};{params.group(3)};'
                f'{params.group(4)};{timelimit};{mip_end - mip_start};'
                f'{mip_obj};{sa_end - sa_start};{solution.score};{sa_slack};'
                f'{wmip_end - wmip_start};{wmip_obj}\n'
            )


if __name__ == "__main__":
    main()
