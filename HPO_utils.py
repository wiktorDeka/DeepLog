import optuna
from Main import run_experiment

def make_objective(dataset_name, hpo):
    def objective(trial: optuna.Trial):

        params = dict(
            # --- dane ---
            data_name=dataset_name,
            data_seed=1213,          # STAŁY split danych (ważne!)

            # --- model ---
            hidden_size=trial.suggest_categorical(
                "hidden_size", [32, 64, 128, 256]
            ),

            # --- trening ---
            batch_size=trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            ),
            lr=trial.suggest_categorical(
                "lr", [1e-4, 3e-4, 1e-3, 3e-3]
            ),
            epochs=100,
            patience=3,
            delta=1e-3,

            # --- reproducibility ---
            model_seed=trial.suggest_int("model_seed", 0, 10),
            device="cuda",
            k=5,
            hpo=True
        )

        top1_acc, topk_acc = run_experiment(**params)

        return topk_acc
    return objective