import os
import sys
import traceback

import click

from pioneer import logutils
from pioneer.launch.pioneer_train import train
from pioneer.tensorboard_util import launch_tensorboard


@click.command(name='pioneer-train')
def cli_pioneer_train():
    train()


@click.command(name='tensorboard')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
def cli_tensorboard(experiment: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    tensorboard_root = config['tracking']['training_root']
    tensorboard_dir = os.path.join(tensorboard_root, experiment)

    launch_tensorboard(tensorboard_dir)

    input("\nPress Enter to exit (this will terminate TensorBoard)\n")


@click.group()
def cli():
    pass


if __name__ == '__main__':
    try:
        cli.add_command(cli_pioneer_train)
        cli.add_command(cli_tensorboard)

        cli()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
