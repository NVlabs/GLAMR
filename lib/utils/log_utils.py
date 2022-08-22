import os
import logging
import time
from lib.utils.tools import get_eta_str, convert_sec_to_time
from pytorch_lightning.loggers import LightningLoggerBase


def create_logger(file_path, file_handle=True):
    # create logger
    logger = logging.getLogger(file_path)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fh = logging.FileHandler(file_path, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger


class TextLogger(LightningLoggerBase):
    def __init__(self, file_path, cfg=None, write_file=True, training=True, max_epochs=-1):
        super().__init__()
        self.training = training
        self.write_file = write_file
        self.cfg = cfg
        self.max_epochs = max_epochs
        self.cfg_name = cfg.id if cfg is not None else 'Exp'
        self.setup_log(file_path, write_file)
        self.cur_metrics = dict()
        self.metrics_to_ignore = set(['epoch'])
        self.last_epoch_time = time.time()

    def setup_log(self, file_path, write_file):
        self.log = log = logging.getLogger(file_path)
        log.propagate = False
        log.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(stream_formatter)
        log.addHandler(ch)

        if write_file:
            # create file handler which logs even debug messages
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fh = logging.FileHandler(file_path, mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
            log.addHandler(fh)

    def log_metrics(self, metrics, step):
        self.cur_metrics.update(metrics)
        if self.training:
            if 'val_loss' in metrics:
                self.log_train()
        else:
            self.log_eval()

    def log_train(self):
        log = self.log
        epoch = self.cur_metrics["epoch"]
        epoch_secs = time.time() - self.last_epoch_time
        eta_str = get_eta_str(epoch, self.max_epochs, epoch_secs)
        loss_str = ' | '.join([f'{x}: {y:7.3f}' for x, y in self.cur_metrics.items() if x not in self.metrics_to_ignore])
        info_str = f'{self.cfg_name} | {epoch:4d}/{self.max_epochs} | TE: {convert_sec_to_time(epoch_secs)} ETA: {eta_str} | {loss_str}'
        log.info(info_str)
        self.last_epoch_time = time.time()

    def log_eval(self):
        pass

    @property
    def experiment():
      pass

    @property
    def name(self):
        return 'textlogger'

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass
