import attr
import torch

from src.utils import registry

registry.register("optimizer", "adam")(torch.optim.Adam)
registry.register("optimizer", "sgd")(torch.optim.SGD)


@registry.register("lr_scheduler", "warmup_polynomial")
@attr.s
class WarmupPolynomialLRScheduler:
    param_groups = attr.ib()
    num_warmup_steps = attr.ib()
    start_lr = attr.ib()
    end_lr = attr.ib()
    decay_steps = attr.ib()
    power = attr.ib()

    def update_lr(self, current_step):
        if current_step < self.num_warmup_steps:  # warmup steps
            warmup_frac_done = current_step / self.num_warmup_steps
            new_lr = self.start_lr * warmup_frac_done
        else:  # after warmup steps
            new_lr = (
                    (self.start_lr - self.end_lr) * (
                        1 - (current_step - self.num_warmup_steps) / self.decay_steps) ** self.power
                    + self.end_lr)

        for param_group in self.param_groups:
            param_group["lr"] = new_lr

