import click
import datetime
import os.path


from continuousresource.probleminstances.jobarrayinstance \
    import JobPropertiesInstance


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--export-path',
    '-p',
    'exportpath',
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
        resolve_path=True
    ),
    help="Path to the output folder."
)
@click.option(
    '--export-format',
    '-f',
    'exportformat',
    required=True,
    type=click.Choice(['binary', 'csv', 'both'], case_sensitive=False),
    default='both',
    help="Export format for the provided instance."
)
@click.option(
    '--label',
    '-l',
    'label',
    required=False,
    type=str,
    default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"),
    help="Label or name for the generated instance."
)
def main(exportpath, exportformat, label):
    # TODO: think of useful parameters
    for n in [5, 10, 20, 50, 100, 200]:  # [10, 20, 50, 100, 200]:
        for r in [25.0, 50.0, 100.0, 200.0]:
            instance = JobPropertiesInstance.generate_instance(n, r)

            inst_label = f"{label}_n{n}r{r:.2f}"
            if exportformat in ['both', 'binary']:
                path = os.path.join(exportpath, inst_label)
                JobPropertiesInstance.to_binary(path, instance)
            if exportformat in ['both', 'csv']:
                path = os.path.join(exportpath, inst_label)
                JobPropertiesInstance.to_csv(path, instance)


if __name__ == "__main__":
    main()
