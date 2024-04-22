import wandb


class Logger:
    def __init__(self, args):
        wandb.init(project='Helldivers2 DQN', entity='elrod-michael95')
        wandb.config.update({
            "num_labels": args.num_labels,
            "image_size": args.image_size,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "eps_decay": args.eps_decay,
            "max_steps": args.max_steps,
        })

    def log_metrics(self, step, eps, reward, avg_score):
        wandb.log({
            "Step": step,
            "Epsilon": eps,
            "Reward": reward,
            "Average Score": avg_score
        })

    def close(self):
        wandb.finish()
