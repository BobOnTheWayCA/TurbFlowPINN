import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad
from tqdm import tqdm
import pandas as pd
import sys
import time

# Input and output columns
input_columns = [3,4,5,1]
output_columns = [6,7,8,9,10,11]

def csv_to_hdf5(csv_file, hdf5_file, chunksize=50000):
    # Convert a large CSV file to an HDF5 file for efficient IO
    with h5py.File(hdf5_file, 'w') as hf:
        for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize)):
            if i == 0:
                data_shape = (0, chunk.shape[1])
                max_shape = (None, chunk.shape[1])
                data = hf.create_dataset("data", shape=data_shape, maxshape=max_shape,
                                         dtype='float32', chunks=True)
            data.resize(data.shape[0] + chunk.shape[0], axis=0)
            data[-chunk.shape[0]:] = chunk.values

def compute_mean_std_hdf5(hdf5_file_path, input_idx, output_idx, chunk_size=1_000_000):
    # Compute mean and std for normalization over entire dataset
    with h5py.File(hdf5_file_path, 'r') as hf:
        data = hf['data']
        total_len = data.shape[0]

        sum_in = np.zeros(len(input_idx))
        sum_out = np.zeros(len(output_idx))
        count = total_len

        # Compute mean
        start = 0
        with tqdm(total=total_len, desc="Computing mean", unit="row") as pbar:
            while start < total_len:
                end = min(start + chunk_size, total_len)
                chunk = data[start:end]
                sum_in += chunk[:, input_idx].sum(axis=0)
                sum_out += chunk[:, output_idx].sum(axis=0)
                pbar.update(end - start)
                start = end
        mean_in = sum_in / count
        mean_out = sum_out / count

        # Compute std
        sum_in_sq = np.zeros(len(input_idx))
        sum_out_sq = np.zeros(len(output_idx))
        start = 0
        with tqdm(total=total_len, desc="Computing std", unit="row") as pbar:
            while start < total_len:
                end = min(start + chunk_size, total_len)
                chunk = data[start:end]
                diff_in = chunk[:, input_idx] - mean_in
                diff_out = chunk[:, output_idx] - mean_out
                sum_in_sq += (diff_in**2).sum(axis=0)
                sum_out_sq += (diff_out**2).sum(axis=0)
                pbar.update(end - start)
                start = end

        std_in = np.sqrt(sum_in_sq / count)
        std_out = np.sqrt(sum_out_sq / count)
        std_in[std_in==0] = 1.0
        std_out[std_out==0] = 1.0
        return mean_in, std_in, mean_out, std_out

class FNNModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=6, hidden_units=512, hidden_layers=8):
        super(FNNModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.LeakyReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_units, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_dataset_from_indices(hdf5_file, indices, input_idx, output_idx, mean_in, std_in, mean_out, std_out, chunk_size=1_000_000, sub_chunk_size=50_000):
    # Load a subset of data from HDF5 by given indices
    total_indices = len(indices)
    inputs_list = []
    outputs_list = []

    with h5py.File(hdf5_file, 'r') as hf:
        dset = hf['data']
        start = 0
        with tqdm(total=total_indices, desc="Loading subset from HDF5", unit="row") as pbar:
            while start < total_indices:
                end = min(start+chunk_size, total_indices)
                chunk_indices = indices[start:end]

                sub_start = 0
                inputs_chunk_list = []
                outputs_chunk_list = []
                while sub_start < len(chunk_indices):
                    sub_end = min(sub_start+sub_chunk_size, len(chunk_indices))
                    sub_indices = chunk_indices[sub_start:sub_end]
                    sub_data = dset[sub_indices]
                    inputs_chunk_list.append(sub_data[:, input_idx])
                    outputs_chunk_list.append(sub_data[:, output_idx])
                    pbar.update(sub_end - sub_start)
                    sub_start = sub_end

                inputs_chunk = np.concatenate(inputs_chunk_list, axis=0)
                outputs_chunk = np.concatenate(outputs_chunk_list, axis=0)

                inputs_list.append(inputs_chunk)
                outputs_list.append(outputs_chunk)

                start = end

    inputs_all = np.concatenate(inputs_list, axis=0)
    outputs_all = np.concatenate(outputs_list, axis=0)

    inputs_norm = (inputs_all - mean_in) / std_in
    outputs_norm = (outputs_all - mean_out) / std_out

    return TensorDataset(torch.tensor(inputs_norm, dtype=torch.float32),
                         torch.tensor(outputs_norm, dtype=torch.float32))

def main():
    print("/*------------------------------------------------------------------------------*\\")
    print("|                                                                                |")
    print("|                                                                                |")
    print("|     _______         _     ______ _               _____ _____ _   _ _   _       |")
    print("|    |__   __|       | |   |  ____| |             |  __ \\_   _| \\ | | \\ | |      |")
    print("|       | |_   _ _ __| |__ | |__  | | _____      _| |__) || | |  \\| |  \\| |      |")
    print("|       | | | | | '__| '_ \\|  __| | |/ _ \\ \\ /\\ / /  ___/ | | | . ` | . ` |      |")
    print("|       | | |_| | |  | |_) | |    | | (_) \\ V  V /| |    _| |_| |\\  | |\\  |      |")
    print("|       |_|\\__,_|_|  |_.__/|_|    |_|\\___/ \\_/\\_/ |_|   |_____|_| \\_|_| \\_|      |")
    print("|                                                                                |")
    print("|              Turbulent Flow PINN Training and Validation Framework             |")
    print("|                          A Personal Project by Bob Bu                          |")
    print("|             Department of Computing Science, University of Alberta             |")
    print("|                                                                                |")
    print("\\*------------------------------------------------------------------------------*/")

    # Main training routine
    csv_file_path = 'TrainingDataset/dataset.csv'
    hdf5_file_path = 'TrainingDataset/dataset.h5'
    model_save_path = 'pinn_model.pth'
    best_model_path = 'pinn_best_model.pth'

    if not os.path.exists(hdf5_file_path):
        print("HDF5 file not found, converting CSV to HDF5...")
        csv_to_hdf5(csv_file_path, hdf5_file_path)
    else:
        print(f"HDF5 file already exists at {hdf5_file_path}, skipping conversion.")

    with h5py.File(hdf5_file_path, 'r') as hf:
        total_len = hf['data'].shape[0]

    file_size = os.path.getsize(hdf5_file_path)
    print(f"HDF5 file: {hdf5_file_path}, File size: {file_size/1024/1024:.2f} MB")
    print(f"Total rows in dataset: {total_len}")

    with tqdm(total=3, desc="Preparing mean/std calculation", unit="step") as prep_bar:
        time.sleep(0.5); prep_bar.update(1)
        time.sleep(0.5); prep_bar.update(1)
        time.sleep(0.5); prep_bar.update(1)

    print("Computing mean and std from entire dataset...")
    input_idx = input_columns
    output_idx = output_columns
    mean_in, std_in, mean_out, std_out = compute_mean_std_hdf5(hdf5_file_path, input_idx, output_idx, chunk_size=1_000_000)
    print("Mean_in:", mean_in)
    print("Std_in:", std_in)
    print("Mean_out:", mean_out)
    print("Std_out:", std_out)

    # Define training and validation ranges (using a large subset, e.g. 10 million for training, 10 million for validation)
    train_samples_start = 0
    train_samples_end = 10_000_000
    train_end = int(0.8 * total_len)
    if train_samples_end > train_end:
        train_samples_end = train_end
    train_indices = np.arange(train_samples_start, train_samples_end)

    val_samples = 10_000_000
    val_count = total_len - train_end
    if val_samples > val_count:
        val_samples = val_count
    val_start = total_len - val_samples
    val_indices = np.arange(val_start, total_len)

    print("Loading training data...")
    train_dataset = create_dataset_from_indices(hdf5_file_path, train_indices, input_idx, output_idx,
                                                mean_in, std_in, mean_out, std_out,
                                                chunk_size=1_000_000, sub_chunk_size=50_000)
    print("Loading validation data...")
    val_dataset = create_dataset_from_indices(hdf5_file_path, val_indices, input_idx, output_idx,
                                              mean_in, std_in, mean_out, std_out,
                                              chunk_size=1_000_000, sub_chunk_size=50_000)

    batch_size = 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNNModel(input_dim=4, output_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    data_loss_function = nn.MSELoss()

    best_val_loss = float('inf')
    # Increase PDE subset size and vary PDE weight over epochs
    epochs = 10

    if os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    else:
        print("No saved model found. Training a new model.")

    # Use a scheduler to help with optimization stability
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        current_lr = scheduler.get_last_lr()[0]  # Get current LR
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr}")

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
            for i, (inputs, outputs) in enumerate(t, start=1):
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions = model(inputs)
                data_loss = data_loss_function(predictions, outputs)

                total_batch_loss = data_loss

                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += total_batch_loss.item()
                t.set_postfix(loss=total_loss / i)

        avg_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        val_loss_sum = 0.0
        count = 0
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions = model(inputs)
                val_loss_batch = data_loss_function(predictions, outputs)
                val_loss_sum += val_loss_batch.item() * outputs.size(0)
                count += outputs.size(0)
        val_loss = val_loss_sum / count

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch+1}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Loss: {val_loss:.4f}")

        # Step the scheduler with validation loss
        scheduler.step(val_loss)

if __name__ == "__main__":
    main()