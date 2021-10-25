import click
import os
import os.path

import matplotlib.pyplot as plt
import pandas as pd


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--file',
    '-f',
    'file',
    multiple=False,
    required=True,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to the csv file containing data for the plot."
)
@click.option(
    '--output-dir',
    '-o',
    'output_dir',
    default=os.getcwd(),
    required=False,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to the output directory."
)
@click.option(
    '--prefix',
    '-p',
    'prefix',
    default='',
    required=True,
    type=str,
    help="Prefix for the output filenames."
)
def main(file, output_dir, prefix):
    """Plot the evolution of the objective value.

    Parameters
    ----------
    path : str
        Path to the csv file containing data for the plots. The expected
        format is as follows: the first row contains header labels. After
        the first row, each row contains the data for an iteration of the
        SA algorithm. Five columns are expected: "#" (the iteration
        number), "time", "best_score", "curr_score" and "rejected".
    output_dir : str
        Path to the output directory.
    prefix : str
        Prefix for the output filenames.
    """
    if len(prefix) > 0:
        prefix += '_'

    # Collect data
    data = pd.read_csv(
        file,
        sep=';',
        usecols=(['#', 'time', 'best_score', 'curr_score', 'rejected'])
    )
    
    fig = plt.figure()
    kpl = fig.add_subplot()
    kpl.set_xlabel('Number of iterations')
    kpl.set_ylabel('Objective value')
    kpl.set_title(f'Convergence of the objective value')
    kpl.plot(
        data['#'],
        data['best_score'],
        label="Best"
    )
    kpl.plot(
        data['#'],
        data['curr_score'],
        label="Current"
    )
    kpl.legend()
    plt.savefig(
        os.path.join(
            output_dir,
            f'{prefix}plot_convergence_iter.png'
        )
    )
    plt.clf()
    
    fig = plt.figure()
    kpl = fig.add_subplot()
    kpl.set_xlabel('Time (s)')
    kpl.set_ylabel('Objective value')
    kpl.set_title(f'Convergence of the objective value')
    kpl.plot(
        data['time'],
        data['best_score'],
        label="Best"
    )
    kpl.plot(
        data['time'],
        data['curr_score'],
        label="Current"
    )
    kpl.legend()
    plt.savefig(
        os.path.join(
            output_dir,
            f'{prefix}plot_convergence_time.png'
        )
    )
    plt.clf()


if __name__ == "__main__":
    main()