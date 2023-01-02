import wandb

wandb.init(project="first-project")
wandb.log({'accuracy': train_acc, 'loss': train_loss})
