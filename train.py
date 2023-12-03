import argparse
import collections
import warnings
import itertools

import numpy as np
import torch

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model_generator = config.init_obj(config["arch_generator"], module_arch)
    logger.info(model_generator)
    model_mpd = config.init_obj(config['arch_mpd'], module_arch)
    logger.info(model_mpd)
    model_msd = config.init_obj(config['arch_msd'], module_arch)
    logger.info(model_msd)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model_generator = model_generator.to(device)
    model_mpd = model_mpd.to(device)
    model_msd = model_msd.to(device)

    # get function handles of loss and metrics
    loss_module_generator = config.init_obj(config["loss_generator"], module_loss).to(device)
    loss_module_disriminator = config.init_obj(config['loss_discriminator'], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_generator_params = filter(lambda p: p.requires_grad, model_generator.parameters())
    trainable_discriminator_params = itertools.chain(
        filter(lambda p: p.requires_grad, model_mpd.parameters()),
        filter(lambda p: p.requires_grad, model_msd.parameters())
    )
    optimizer_generator = config.init_obj(config["optimizer_generator"], torch.optim, trainable_generator_params)
    lr_scheduler_generator = config.init_obj(config["lr_scheduler_generator"], torch.optim.lr_scheduler, optimizer_generator)
    optimizer_discriminator = config.init_obj(config['optimizer_discriminator'], torch.optim, trainable_discriminator_params)
    lr_scheduler_discriminator = config.init_obj(config['lr_scheduler_discriminator'], torch.optim.lr_scheduler, optimizer_discriminator)

    trainer = Trainer(
        model_generator=model_generator,
        model_mpd=model_mpd,
        model_msd=model_msd,
        criterion_generator=loss_module_generator,
        criterion_discriminator=loss_module_disriminator,
        metrics=metrics,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_generator=lr_scheduler_generator,
        lr_scheduler_discriminator=lr_scheduler_discriminator,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
