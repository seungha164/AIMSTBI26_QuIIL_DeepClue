import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path
import shutil
import os
import math

try:
    import wandb
    # os.environ['nnUNet_wandb_enabled'] = '1'
    # os.environ['nnUNet_wandb_project'] = 'ISLES26_nnUNetv2'
except ImportError:
    wandb = None

WANDB_API_KEY = ""


def get_cluster_job_id():
    job_id = None
    if "LSB_JOBID" in os.environ:
        job_id = os.environ["LSB_JOBID"]
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
    return job_id


class MetaLogger(object):
    """A meta logger that bundles multiple loggers behind a single interface.

    The default configuration includes a local logger used for reading values,
    plotting progress, and checkpointing.
    """

    def __init__(self, output_folder, resume, verbose: bool = False, model_name: str = "model", fold: int = 0, dataset:str="nnunet"):
        """Initialize the meta logger.

        Args:
            output_folder: The output folder.
            resume: Whether to resume training if possible.
            verbose: Whether to enable verbose logging in the local logger.
            model_name: The name of the model.
            fold: The fold number.
        """
        self.output_folder = output_folder
        self.resume = resume
        self.loggers = []
        self.local_logger = LocalLogger(verbose)
        if self._is_logger_enabled("nnUNet_wandb_enabled"):
            self.loggers.append(WandbLogger(output_folder, resume, model_name = model_name, fold = fold, dataset=dataset))

    def update_config(self, config: dict):
        """Add a new or update an existing experiment configuration to the logger.

        Args:
            config: Logger configuration options.
        """
        for logger in self.loggers:
            logger.update_config(config)

    def log(self, key: str, value: Any, step: int):
        """Log a value for a given step.

        Args:
            key: Metric or field name.
            value: Value to log.
            step: Step index (typically epoch).
        """
        self.local_logger.log(key, value, step)
        if isinstance(value, list):
            for i, val in enumerate(value):
                for logger in self.loggers:
                    logger.log(f"{key}/class_{i+1}", val, step)
        else:
            for logger in self.loggers:
                logger.log(key, value, step)
        
        # handle the EMA pseudo Dice special cases! They are automatically logged from the corresponding mean values.
        if key == 'mean_fg_dice':
            self._log_ema('ema_fg_dice', value, step)
        if key == 'mean_fg_dice_abnormal':
            self._log_ema('ema_fg_dice_abnormal', value, step)

    def _log_ema(self, key: str, value: Any, step: int):
        previous_values = self.get_value(key, step=None)
        previous_value = previous_values[step - 1] if step > 0 and len(previous_values) >= step else None
        if previous_value is not None and math.isfinite(float(previous_value)):
            new_ema_pseudo_dice = previous_value * 0.9 + 0.1 * value
        else:
            new_ema_pseudo_dice = value
        self.log(key, new_ema_pseudo_dice, step)

    def log_summary(self, key: str, value: Any):
        """Log a summary value. These are usually values that are not logged every step but only once. 
        This can be for example the final validation Dice.

        Args:
            key: Metric or field name.
            value: Value to summarize.
        """
        for logger in self.loggers:
            logger.log_summary(key, value)

    def get_value(self, key: str, step: Any):
        """Fetch a logged value from the local logger.

        Args:
            key: Metric or field name.
            step: Step index to retrieve, or None to return all values.

        Returns:
            The logged value or list of values from the local logger.
        """
        return self.local_logger.get_value(key, step)

    def plot_progress_png(self, output_folder: str):
        """Write a progress plot PNG using local logger data.

        Args:
            output_folder: Directory where the plot image is saved.
        """
        self.local_logger.plot_progress_png(output_folder)

    def get_checkpoint(self):
        """Return the local logger checkpoint data.

        Returns:
            The checkpoint payload used to restore logging state.
        """
        return self.local_logger.get_checkpoint()

    def load_checkpoint(self, checkpoint: dict):
        """Restore the local logger from a checkpoint payload.

        Args:
            checkpoint: Checkpoint data returned by `get_checkpoint`.
        """
        self.local_logger.load_checkpoint(checkpoint)

    def _is_logger_enabled(self, env_var):
        env_var_result = str(os.getenv(env_var, "0"))
        if env_var_result in ("0", "False", "false"):
            return False
        elif env_var_result in ("1", "True", "true"):
            return True
        else:
            raise RuntimeError("nnU-Net logger environement variable has the wrong value. Must be '0' (disabled) or '1'(enabled).")


class LocalLogger:
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = self._get_default_logging()
        self.verbose = verbose
        # shut up, this logging is great

    @staticmethod
    def _get_default_logging():
        return {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'mean_fg_dice_abnormal': list(),
            'ema_fg_dice_abnormal': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list(),
            # NEW
            'f1_per_class_or_region': list(),
            'absolute_instance_count_difference_aicd': list(),

            'val_loss_deep': list(),
            'val_loss_shallow': list(),
            'val_loss_final': list(),
        }

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

    def get_value(self, key, step):
        if step is not None:
            return self.my_fantastic_logging[key][step]
        else:
            return self.my_fantastic_logging[key]

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice",
                 linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)",
                 linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        logging = self._get_default_logging()
        existing_length = max([len(v) for v in checkpoint.values() if isinstance(v, list)], default=0)
        for key in logging:
            if key in checkpoint:
                logging[key] = checkpoint[key]
            else:
                logging[key] = [float('nan')] * existing_length
        self.my_fantastic_logging = logging


class WandbLogger:
    """Weights & Biases logger for nnU-Net training runs.

    Environment Variables:
        nnUNet_wandb_enabled: Whether W&B logger is enabled (default: 0 -> Disabled)
        nnUNet_wandb_project: W&B project name (default: "nnunet").
        nnUNet_wandb_mode: W&B mode, e.g. "online" or "offline" (default: "online").
    """

    def __init__(self, output_folder, resume, model_name:str, fold:int, dataset:str):
        """Initialize a W&B run and handle resume behavior.

        Args:
            output_folder: Directory where W&B run data is stored.
            resume: Whether to resume a previous W&B run if present.
            verbose: Unused verbosity flag (kept for interface compatibility).
        """
        if wandb is None:
            raise RuntimeError("W&B is not installed. Please install W&B with 'pip install wandb' before using the WandbLogger.")

        self.output_folder = Path(output_folder)
        self.resume = resume
        self.project = os.getenv("WANDB_PROJECT", dataset)
        self.mode = os.getenv("nnUNet_wandb_mode", "online")
        # self.run_name = os.getenv("nnUNet_wandb_run_name", "export") # 추가: 환경변수 또는 기본값 "export"
        self.run_name = f"{model_name}_fold{fold}"

        wandb.login(key=WANDB_API_KEY, relogin=True)

        wandb_id = None
        if (self.output_folder / "wandb").is_dir():
            if self.resume:
                wandb_dir = self.output_folder / "wandb" / "latest-run"
                wandb_filename = next(filename for filename in wandb_dir.iterdir() if filename.suffix == ".wandb")
                wandb_id = wandb_filename.stem[4:]
            else:
                shutil.rmtree(str(self.output_folder / "wandb"))

        _resume = "allow" if self.resume else "never"
        self.run = wandb.init(
            project=self.project, 
            dir=str(self.output_folder), 
            id=wandb_id, mode=self.mode, 
            resume=_resume, 
            name = self.run_name
        )
        self.run.config.update({"JobID": get_cluster_job_id()})
        self.wandb_init_step = self.run.step
    
    def update_config(self, config: dict):
        """Update W&B config with training metadata.

        Args:
            config: Configuration values to merge into the run config.
        """
        self.run.config.update(config)

    def log(self, key, value, step: int):
        """Log a scalar value to W&B.

        Args:
            key: Metric or field name.
            value: Value to log.
            step: Step index (typically epoch).
        """
        self.log_summary("current_epoch", step)
        if self.resume and step < self.wandb_init_step:
            return
        self.run.log({key: value}, step=step)

    def log_summary(self, key, value):
        """Write a summary value to W&B.

        Args:
            key: Metric or field name.
            value: Summary value to store.
        """
        self.run.summary[key] = value
