from logparser.Drain import LogParser
import re
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

root_path = r'/home/ubuntu/bsc/BootDet/DeepLog-master'


def parse_dataset(dataset_name, st, depth):
    input_dir = f'{root_path}/Data/{dataset_name}' # The input directory of log file
    output_dir = f'{root_path}/Data/{dataset_name}'  # The output directory of parsing results
    log_file = f'{dataset_name}.log'  # The input log file name
    log_format = '<GroupId> <DateTime> <Host> <Component>:<Content>' # Define log format to split message fields

    regex = [
        r'https?://(?:[\w\-]+\.)+[a-zA-Z]{2,}(?:/[\w\-./?%&=]*)?',  # URL
        r'\[mem\s+[^\]-]+-[^\]]+\]',                      # zakresy pamięci e820: [mem 0x...-0x...]
        r'\b[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b',      # PCI bus:device.function (BDF)
        r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b',  # UUID/GUID
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b(?::\d{1,5})?',      # IPv4 z opcjonalnym portem
        r'\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b',      # MAC
        r'/org/[A-Za-z0-9_/\.]+',                         # ścieżki obiektów D-Bus
        r'\b[\w\-\.]+\.(?:service|socket|target|mount|slice|timer)\b',  # jednostki systemd

        # --- ścieżki, wersje, hex: ---
        r'\b/(?:[\w\-.]+/)*[\w\-.]*\b',                   # ścieżki plików (dość agresywne)
        r'\b\d+(?:\.\d+){1,3}[-\w]*\b',                   # wersje typu 1.2.3-rc1
        r'0x[0-9a-fA-F]+',                                # liczby heksadecymalne
        r'\[mem\s+[^\]-]+-[^\]]+\]',  # memory address

        # --- wielkości i identyfikatory: ---
        r'\b\d+[KMG]?B\b',                                # rozmiary: 512KB, 4MB, 1GB
        r'\b(PID|TID|UID|GID)?\s*[:=]?\s*\d+\b',          # PID/TID/UID/GID lub gołe ID
    ]

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, maxChild=120)
    parser.parse(log_file)

def split_dataset(dataset_name, data_seed):
    logs = pd.read_csv(f"{root_path}/Data/{dataset_name}/{dataset_name}.log_structured.csv")
    labels = pd.read_csv(f"{root_path}/Data/{dataset_name}/anomaly_label_{dataset_name}.csv")

    logs = logs.merge(labels, on="GroupId")

    df = logs.rename(columns={
        "DateTime": "timestamp",
        "GroupId": "machine",
        "EventId": "event",
        "Label": "label",
    })

    mapping = {'Normal': 0, 'Anomaly': 1}
    df['label'] = df['label'].map(mapping)

    df = df[["timestamp", "machine", "event", "label"]]

    train_df = df[df["label"] == 0]
    test_df  = df.copy()

    train_df.drop(columns=["label"], inplace=True)
    test_df.drop(columns=["label"], inplace=True)

    boot_ids = train_df["machine"].unique()
    train_ids, val_ids = train_test_split(
        boot_ids, test_size=0.2, random_state=data_seed
    )

    print(f"Train IDs: {train_ids}")
    print(f"Val IDs: {val_ids}")

    val_df   = train_df[train_df["machine"].isin(val_ids)]
    train_df = train_df[train_df["machine"].isin(train_ids)]

    train_df.to_csv(f"{root_path}/Data/{dataset_name}/Temp/{dataset_name}_deeplog_train.csv", index=False)
    val_df.to_csv(f"{root_path}/Data/{dataset_name}/Temp/{dataset_name}_deeplog_val.csv", index=False)
    test_df.to_csv(f"{root_path}/Data/{dataset_name}/Temp/{dataset_name}_deeplog_test.csv",  index=False)