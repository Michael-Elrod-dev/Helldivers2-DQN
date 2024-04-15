import wandb

class Logger:
    def __init__(self, image_size, num_labels, eps_start, eps_end, eps_decay, max_steps):
        wandb.init(project='Helldivers2 DQN', entity='elrod-michael95', name=f'Stratagem AI')
        wandb.config.update({
            "num_labels": num_labels,
            "image_size": image_size,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay": eps_decay,
            "max_steps": max_steps,
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