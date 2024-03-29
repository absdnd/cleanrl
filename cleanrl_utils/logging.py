import wandb

class WeightsAndBiasesWriter:
    def __init__(self, args, run_name):
        self.args = args
        self.run_name = run_name

        self.run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        

    def add_scalar(self, key, value, global_step):
        wandb.log({key:value} ,step=global_step)

    def close(self):
        self.run.finish()

    def __del__(self):
        self.close()
