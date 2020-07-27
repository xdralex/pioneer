import logging
import math
import os
import pathlib
from logging.config import dictConfig
from typing import List, Optional

import pandas as pd
from tabulate import tabulate
from tensorboard import program


class BetterFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _open(self):
        pathlib.Path(os.path.dirname(self.baseFilename)).mkdir(parents=True, exist_ok=True)
        return logging.FileHandler._open(self)


class BetterRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _open(self):
        pathlib.Path(os.path.dirname(self.baseFilename)).mkdir(parents=True, exist_ok=True)
        return logging.handlers.RotatingFileHandler._open(self)


def configure_logging(config):
    dictConfig(config)


def launch_tensorboard(tensorboard_root: str, port: int = 6006):
    logger = logging.getLogger(__name__)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}', '--logdir', tensorboard_root])
    url = tb.launch()

    logger.info(f'Launched TensorBoard at {url}')


def dump(df: pd.DataFrame, cols: List[str]) -> str:
    def format_float(v: float) -> str:
        if abs(int(v) - v) < 1e-6:
            return f'{v:.1f}'

        if abs(v) < 1e-3 or abs(v) >= 1e+5:
            return f'{v:.2e}'

        zeros = math.ceil(math.log10(math.fabs(v) + 1))
        if zeros < 5:
            return f'{v:.{5 - zeros}f}'
        else:
            return f'{v:.1f}'

    df = df[cols]
    for col in list(df.columns):
        df[col] = df[col].apply(lambda x: format_float(x) if isinstance(x, float) else str(x))

    return tabulate(df, headers="keys", showindex=False, tablefmt='github')
