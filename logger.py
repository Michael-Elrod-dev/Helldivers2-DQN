import wandb

class Logger:
    def __init__(self, args):
        wandb.init(project='Helldivers2 CNN', entity='elrod-michael95')
        wandb.config.update({
            "num_labels": args.num_labels,
            "image_size": args.image_size,
            "learning_rate": args.LR,
            "batch_size": args.BATCH_SIZE,
            "max_steps": args.max_steps,
            "conv_channels": args.conv_channels,
            "fc_units": args.fc_units,
            "dropout_rate": args.dropout_rate,
            "use_batch_norm": args.use_batch_norm
        })

    def log_metrics(self, step, loss, accuracy, avg_accuracy, learning_rate):
        wandb.log({
            "Step": step,
            "Loss": loss,
            "Accuracy": accuracy,
            "Average Accuracy": avg_accuracy,
            "Learning Rate": learning_rate
        })

    def close(self):
        wandb.finish()