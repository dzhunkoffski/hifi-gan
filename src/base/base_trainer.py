from abc import abstractmethod

import torch
from numpy import inf

from src.base import BaseModel
from src.logger import get_visualizer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
            self, model_generator: BaseModel, model_mpd: BaseModel, model_msd: BaseModel,
              criterion_generator, criterion_discriminator, metrics, 
              optimizer_generator, optimizer_discriminator, config, device, lr_scheduler_generator=None, lr_scheduler_discriminator=None):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model_generator = model_generator
        self.model_mpd = model_mpd
        self.model_msd = model_msd

        self.criterion_generator = criterion_generator
        self.criterion_discriminator = criterion_discriminator
        self.metrics = metrics
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.lr_scheduler_generator = lr_scheduler_generator
        self.lr_scheduler_discriminator = lr_scheduler_discriminator

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        print(config.resume)
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch_generator = type(self.model_generator).__name__
        arch_mpd = type(self.model_mpd).__name__
        arch_msd = type(self.model_msd).__name__
        state = {
            "arch_generator": arch_generator,
            "arch_mpd": arch_mpd,
            "arch_msd": arch_msd,
            "epoch": epoch,
            "state_dict_generator": self.model_generator.state_dict(),
            "state_dict_mpd": self.model_mpd.state_dict(),
            "state_dict_msd": self.model_msd.state_dict(),
            "optimizer_generator": self.optimizer_generator.state_dict(),
            "optimizer_discriminator": self.optimizer_discriminator.state_dict(),
            "lr_scheduler_generator": self.lr_scheduler_generator.state_dict(),
            "lr_scheduler_discriminator": self.lr_scheduler_discriminator.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        print('go resume')
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch_generator"] != self.config["arch_generator"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        if checkpoint["config"]["arch_mpd"] != self.config["arch_mpd"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        if checkpoint["config"]["arch_msd"] != self.config["arch_msd"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )

        self.model_generator.load_state_dict(checkpoint["state_dict_generator"])
        self.model_mpd.load_state_dict(checkpoint["state_dict_mpd"])
        self.model_msd.load_state_dict(checkpoint["state_dict_msd"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer_generator"] != self.config["optimizer_generator"] or
                checkpoint["config"]["lr_scheduler_generator"] != self.config["lr_scheduler_generator"] or
                checkpoint["config"]["optimizer_discriminator"] != self.config["optimizer_discriminator"] or
                checkpoint["config"]["lr_scheduler_discriminator"] != self.config["lr_scheduler_discriminator"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
            self.optimizer_discriminator.load_state_dict(checkpoint["optimizer_discriminator"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
