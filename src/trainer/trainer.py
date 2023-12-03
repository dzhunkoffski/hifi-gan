import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker

import wandb


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model_generator,
            model_mpd,
            model_msd,
            criterion_generator,
            criterion_discriminator,
            metrics,
            optimizer_generator,
            optimizer_discriminator,
            config,
            device,
            dataloaders,
            lr_scheduler_generator=None,
            lr_scheduler_discriminator=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model_generator, model_mpd, model_msd, criterion_generator, criterion_discriminator, metrics, optimizer_generator, optimizer_discriminator, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_generator = lr_scheduler_generator
        self.lr_scheduler_discriminator = lr_scheduler_discriminator
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss_generator", "loss_discriminator", "grad norm generator", "grad norm mpd", "grad norm msd", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss_generator", "loss_discriminator", "grad norm generator", "grad norm mpd", "grad norm msd", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "wav_real"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, model: str):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            if model == 'generator':
                clip_grad_norm_(
                    self.model_generator.parameters(), self.config["trainer"]["grad_norm_clip"]
                )
            elif model == 'mpd':
                clip_grad_norm_(
                    self.model_mpd.parameters(), self.config['trainer']['grad_norm_clip']
                )
            elif model == 'msd':
                clip_grad_norm_(
                    self.model_msd.parameters(), self.config['trainer']['grad_norm_clip']
                )
            else:
                raise NotImplementedError

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model_generator.train()
        self.model_mpd.train()
        self.model_msd.train()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model_generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.model_mpd.parameters():
                        if p.grad is not None:
                            del p.grad
                    for p in self.model_msd.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm generator", self.get_grad_norm(model='generator'))
            self.train_metrics.update("grad norm mpd", self.get_grad_norm(model='mpd'))
            self.train_metrics.update("grad norm msd", self.get_grad_norm(model='msd'))
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator Loss: {:.6f} Discriminator Loss {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss_generator"].item(), batch['loss_discriminator'].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate generator", self.lr_scheduler_generator.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate discriminator", self.lr_scheduler_discriminator.get_last_lr()[0]
                )
                # TODO: log predictions
                # self._log_predictions(**batch)
                # self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
    
    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        batch['wav_generated'] = self.model_generator(**batch)[:, :, :-1]
        if is_train:
            self.optimizer_discriminator.zero_grad()
        # Discriminator
        batch['mpd_d_out_generated'], _ = self.model_mpd(x=batch['wav_generated'].detach())
        batch['msd_d_out_generated'], _ = self.model_msd(x=batch['wav_generated'].detach())
        batch['loss_discriminator'] = self.criterion_discriminator(**batch)
        if is_train:
            batch['loss_discriminator'].backward()
            self._clip_grad_norm(model='mpd')
            self._clip_grad_norm(model='msd')
            self.optimizer_discriminator.step()
            if self.lr_scheduler_discriminator is not None:
                self.lr_scheduler_discriminator.step()
        
        # Generator
        if is_train:
            self.optimizer_generator.zero_grad()
        batch['mpd_d_out_generated'], batch['mpd_features_generated'] = self.model_mpd(x=batch['wav_generated'])
        batch['mpd_d_out_real'], batch['mpd_features_real'] = self.model_mpd(x=batch['wav_real'])
        batch['msd_d_out_generated'], batch['msd_features_generated'] = self.model_msd(x=batch['wav_generated'])
        batch['msd_d_out_real'], batch['msd_features_real'] = self.model_msd(x=batch['wav_real'])
        batch['loss_generator'] = self.criterion_generator(**batch)
        if is_train:
            batch['loss_generator'].backward()
            self._clip_grad_norm(model='generator')
            self.optimizer_generator.step()
            if self.lr_scheduler_generator is not None:
                self.lr_scheduler_generator.step()
            
        metrics.update("loss_generator", batch['loss_generator'].item())
        metrics.update("loss_discriminator", batch['loss_discriminator'])
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model_generator.eval()
        self.model_mpd.eval()
        self.model_msd.eval()

        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(generator_model=self.model_generator, eval_dataloader=self.evaluation_dataloaders['val'], **batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            generator_model,
            eval_dataloader,
            *args,
            **kwargs,
    ):
        rows = {}
        for i, batch in enumerate(eval_dataloader):
            batch = self.move_batch_to_device(batch=batch, device=self.device)
            output = generator_model(**batch)
            output = output.squeeze(0).squeeze(0)
            gen_audio = wandb.Audio(output.cpu().numpy(), sample_rate=22050)
            raw_audio = wandb.Audio(batch['wav_real'].squeeze(0).squeeze(0).cpu().numpy(), sample_rate=22050)
            rows[i] = {
                "real audio": raw_audio,
                "generated audio": gen_audio
            }

        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, model: str, norm_type=2):
        if model == 'generator':
            parameters = self.model_generator.parameters()
        elif model == 'mpd':
            parameters = self.model_mpd.parameters()
        elif model == 'msd':
            parameters = self.model_msd.parameters()
        else:
            raise NotImplementedError
        
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
