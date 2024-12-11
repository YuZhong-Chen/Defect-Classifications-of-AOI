from pathlib import Path

import wandb
import torch.utils.tensorboard as tensorboard


class Logger(object):
    def __init__(self, project: str, project_name: str, config: dict, project_dir: Path, enable: bool = True, use_wandb: bool = True):
        self.enable = enable
        self.writer = None

        if self.enable:
            logs_dir = project_dir / "logs"
            logs_dir.mkdir(exist_ok=True)

            if use_wandb:
                wandb.init(
                    project=project,
                    name=project_name,
                    dir=project_dir.as_posix(),
                    sync_tensorboard=True,
                    config=config,
                    save_code=False,
                )

            self.writer = tensorboard.SummaryWriter(logs_dir)
            self.writer.add_text("Hyper Parameters", "|Param|Value|\n|-|-|\n%s" % ("\n".join([f"|{param}|{value}|" for param, value in config.items()])))

            print(f"Saves logs to {logs_dir}")

    def Log(self, epoch: int, loss: float, accuracy: float) -> None:
        if not self.enable:
            return

        self.writer.add_scalar("Train/Loss", loss, epoch)
        self.writer.add_scalar("Train/Accuracy", accuracy, epoch)
        self.writer.flush()