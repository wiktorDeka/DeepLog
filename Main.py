import pandas as pd
import math
import numpy as np

from deeplog              import DeepLog
from deeplog.preprocessor import Preprocessor

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

root_path = r'/home/ubuntu/bsc/BootDet/DeepLog-master'

WINDOW  = 10          # długość kontekstu (h w paperze)
TIMEOUT = float('inf')  # nie obcinamy po czasie
pre = Preprocessor(length=WINDOW, timeout=TIMEOUT)

def run_experiment(dataset_name, hpo):
    _, _, _, mapping_global = pre.csv(f"{root_path}/Data/{dataset_name}/{dataset_name}_deeplog_test.csv")

    # === TRAIN ===
    ctx_tr, ev_tr, labels_tr, mapping_tr = pre.csv(f"{root_path}/Data/{dataset_name}/{dataset_name}_deeplog_train.csv", mapping=mapping_global)
    # === TEST  ===
    ctx_te, ev_te, labels_te, mapping_te = pre.csv(f"{root_path}/Data/{dataset_name}/{dataset_name}_deeplog_test.csv", mapping=mapping_global)
    # === VAL   ===
    ctx_val, ev_val, labels_val, mapping_val = pre.csv(f"{root_path}/Data/{dataset_name}/{dataset_name}_deeplog_val.csv", mapping=mapping_global)
    n_events = len(mapping_tr)


    train_ds = TensorDataset(ctx_tr, ev_tr)
    val_ds   = TensorDataset(ctx_val, ev_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    deeplog = DeepLog(
        input_size  = n_events,
        hidden_size = 64,
        output_size = n_events,
    ).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(deeplog.parameters(), lr=1e-3)

    EPOCHS = 100
    PATIENCE = 3
    DELTA    = 1e-3

    best_val_loss      = math.inf
    epochs_no_improve  = 0

    train_losses = []
    val_losses   = []

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        deeplog.train()
        running_train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to('cuda')
            yb = yb.to('cuda')

            optimizer.zero_grad()
            logits = deeplog(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- VALIDATION ---
        deeplog.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to('cuda')
                yb = yb.to('cuda')

                logits = deeplog(xb)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * xb.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # print(
        #     f"Epoch {epoch+1}/{EPOCHS} "
        #     f"- train_loss={epoch_train_loss:.4f} "
        #     f"- val_loss={epoch_val_loss:.4f}"
        # )

        if epoch_val_loss < best_val_loss - DELTA:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    if hpo == False:
        X_p = ctx_val
        y_p = ev_val
    else:
        X_p = ctx_te
        y_p = ev_te

    y_pred_test, conf = deeplog.predict(
        X = X_p,
        y = y_p,
        k = 5
    )

    top1_te = (y_pred_test[:, 0] == ev_te)
    top1_acc = top1_te.float().mean().item()
    topk_te = (y_pred_test == ev_te.unsqueeze(1)).any(dim=1)
    topk_acc = topk_te.float().mean().item()

    return top1_acc, topk_acc