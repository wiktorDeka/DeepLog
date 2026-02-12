import optuna
from Main import run_experiment

def make_objective(dataset_name):
    def objective(trial: optuna.Trial):

        params = dict(
            dataset_name=dataset_name,

            window=trial.suggest_categorical(
                "window", [10, 20, 30]
            ),
            hidden_size=trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256]
            ),
            batch_size=trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            ),
            lr=trial.suggest_categorical(
                "lr", [1e-4, 3e-4, 1e-3]
            ),
            top_k=5,
            hpo=True
        )

        top1_acc, top1_te, topk_acc, topk_te, train_losses, val_losses = run_experiment(**params)

        return topk_acc
    return objective