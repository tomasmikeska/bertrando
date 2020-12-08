from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, n_warmup_steps, n_training_steps):
    '''
    During warmup LR increases linearly from 0 to optimizers initial LR.
    Then it decreases linearly to 0 after num_warmup_steps.
    '''

    def lr_lambda(current_step: int):
        if current_step < n_warmup_steps:
            return float(current_step) / float(max(1, n_warmup_steps))
        return max(
            0.0, float(n_training_steps - current_step) / float(max(1, n_training_steps - n_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)
