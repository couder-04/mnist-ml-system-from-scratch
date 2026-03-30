#  DATA PIPELINE

import os
import numpy as np
import pandas as pd
import gdown


#  CONFIG


DATA_DIR = "data"
FOLDER_ID = "1yB-LBqkj3-SD8gyznsZ4H0tWkfIonk7E"
NUM_CLASSES = 10
SEED = 42

TRAIN_FILE = "mnist_train.csv"
TEST_FILE  = "mnist_test.csv"

#  DOWNLOAD DATA

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    test_path  = os.path.join(DATA_DIR, TEST_FILE)

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(" Dataset already present")
        return train_path, test_path

    print(" Downloading dataset from Google Drive...")

    try:
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        gdown.download_folder(url, output=DATA_DIR, quiet=False)
    except Exception as e:
        raise RuntimeError(f" Download failed: {e}")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(" Download incomplete. Files missing.")

    return train_path, test_path

#  PROCESS FUNCTION

def process_dataframe(df):
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].values.astype(int)

    X /= 255.0  # normalize
    Y = np.eye(NUM_CLASSES)[y]  # one-hot

    return X, Y, y

#  LOAD DATA

def load_data(val_split=0.1, subset=None, shuffle=True):


    train_path, test_path = download_data()

    try:
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    except Exception as e:
        raise RuntimeError(f" Failed to load CSV: {e}")

    #  SHUFFLE


    if shuffle:
        train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)



    if subset is not None:
        subset = min(subset, len(train_df))
        train_df = train_df.iloc[:subset]

    
    #  SPLIT
    

    split_idx = int(len(train_df) * (1 - val_split))

    train_split = train_df.iloc[:split_idx]
    val_split_df = train_df.iloc[split_idx:]

    
    #  PROCESS


    X_train, Y_train, y_train = process_dataframe(train_split)
    X_val,   Y_val,   y_val   = process_dataframe(val_split_df)
    X_test,  Y_test,  y_test  = process_dataframe(test_df)

    #  INFO


    print("\n Data Loaded:")
    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test