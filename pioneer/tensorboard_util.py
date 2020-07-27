import logging

from tensorboard import program


def launch_tensorboard(tensorboard_root: str, port: int = 6006):
    logger = logging.getLogger(__name__)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}', '--logdir', tensorboard_root])
    url = tb.launch()

    logger.info(f'Launched TensorBoard at {url}')
