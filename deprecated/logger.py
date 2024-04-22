import wandb

class Logger:
    def __init__(self, args):
        wandb.init(project='Helldivers2 DQN', entity='elrod-michael95', name=f'Stratagem AI')
        wandb.config.update({
            "num_labels": args.num_labels,
            "image_size": args.image_size,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "eps_decay": args.eps_decay,
            "max_steps": args.max_steps,
        })

    def log_metrics(self, step, eps, avg_accuracy, avg_loss, avg_magnitude, lr, prio_a, prio_b):
        wandb.log({
            "Step": step,
            "Epsilon": eps,
            "Average Accuracy": avg_accuracy,
            "Average Loss": avg_loss,
            "Average Gradient Magnitude": avg_magnitude,
            "Learning Rate": lr,
            "Priority Alpha": prio_a,
            "Priority Beta": prio_b
        })

    def log_test_metrics(self, accuracy):
        wandb.log({
            "Test Accuracy": accuracy
        })

    def close(self):
        wandb.finish()