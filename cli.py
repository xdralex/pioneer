import os
import sys
import traceback
import yaml

import click

from pioneer.launch.pioneer_train import train
from pioneer.util import launch_tensorboard, configure_logging, dump


@click.command(name='pioneer-train')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('-c', '--checkpoint-freq', 'checkpoint_freq', default=10, type=int, help='checkpoint frequency (default: 10)')
@click.option('-n', '--num-samples', 'num_samples', default=128, type=int, help='number of search samples (default: 100')
@click.option('-w', '--num-workers', 'num_workers', default=1, type=int, help='number of rollout workers (default: 1')
@click.option('--no-monitor', 'no_monitor', is_flag=True, help='disable monitoring')
def cli_pioneer_train(experiment: str, checkpoint_freq: int, num_samples: int, num_workers: int, no_monitor: bool):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        configure_logging(config['logging'])

    training_root = config['tracking']['training_root']
    experiment_dir = os.path.join(training_root, experiment)

    df = train(results_dir=experiment_dir,
               checkpoint_freq=checkpoint_freq,
               num_samples=num_samples,
               num_workers=num_workers,
               monitor=not no_monitor)

    df_dump = dump(df, cols=['experiment_id',
                             'trial_id',
                             'episode_reward_max',
                             'episode_reward_min',
                             'episode_reward_mean',
                             'episode_len_mean',
                             'episodes_total'])

    print(f'Results: \n\n{df_dump}\n\n\n')


@click.command(name='tensorboard')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
def cli_tensorboard(experiment: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        configure_logging(config['logging'])

    training_root = config['tracking']['training_root']
    experiment_dir = os.path.join(training_root, experiment)

    launch_tensorboard(experiment_dir)

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
