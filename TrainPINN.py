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

class PINNModel(nn.Module):
    def __init__(self, input_dim=4, output_dim=6, hidden_layers=8, hidden_units=512):
        super(PINNModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        # Using LeakyReLU to have some gradient even for negative inputs
        layers.append(nn.LeakyReLU())
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_units, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def physics_loss(model, inputs, rho=1.225, mu=1.8375e-5, beta=0.075, beta_star=0.09,
                 sigma_k=0.85, sigma_omega=0.5, alpha=5/9):
    # Compute PDE residual loss
    inputs.requires_grad = True
    pred = model(inputs)

    u = pred[:,0]
    v = pred[:,1]
    w = pred[:,2]
    k = pred[:,3]
    omega = pred[:,4]
    p = pred[:,5]

    def gradients(y, x, order=1):
        g = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        if order == 1:
            return g
        elif order == 2:
            second_derivs = []
            for i in range(g.shape[1]):
                d2y_dx2 = torch.autograd.grad(g[:, i], x, torch.ones_like(g[:, i]),
                                              create_graph=True, retain_graph=True)[0][:, i]
                second_derivs.append(d2y_dx2)
            return torch.stack(second_derivs, dim=1)
        else:
            raise ValueError("Only first and second derivatives are implemented.")

    def compute_strain_rate_sq(u, v, w, inputs):
        du = gradients(u, inputs, order=1)
        dv = gradients(v, inputs, order=1)
        dw = gradients(w, inputs, order=1)

        Sxx = du[:,0]; Syy = dv[:,1]; Szz = dw[:,2]
        Sxy = 0.5*(du[:,1] + dv[:,0])
        Sxz = 0.5*(du[:,2] + dw[:,0])
        Syz = 0.5*(dv[:,2] + dw[:,1])

        S_sq = (2*(Sxy**2 + Sxz**2 + Syz**2) + Sxx**2 + Syy**2 + Szz**2)
        return S_sq, du, dv, dw

    def f_IDDES():
        return 1.0

    S_sq, du, dv, dw = compute_strain_rate_sq(u, v, w, inputs)
    nu_t = (k / (omega + 1e-12)) * f_IDDES()

    continuity = du[:,0] + dv[:,1] + dw[:,2]

    dp = gradients(p, inputs, order=1)
    d2u = gradients(u, inputs, order=2)
    d2v = gradients(v, inputs, order=2)
    d2w = gradients(w, inputs, order=2)

    lap_u = d2u[:,0] + d2u[:,1] + d2u[:,2]
    lap_v = d2v[:,0] + d2v[:,1] + d2v[:,2]
    lap_w = d2w[:,0] + d2w[:,1] + d2w[:,2]

    conv_u = u*du[:,0] + v*du[:,1] + w*du[:,2]
    conv_v = u*dv[:,0] + v*dv[:,1] + w*dv[:,2]
    conv_w = u*dw[:,0] + v*dw[:,1] + w*dw[:,2]

    dt_u = du[:,3]
    dt_v = dv[:,3]
    dt_w = dw[:,3]

    nu_eff = (mu/rho) + nu_t
    momentum_x = dt_u + conv_u + (1.0/rho)*dp[:,0] - nu_eff*lap_u
    momentum_y = dt_v + conv_v + (1.0/rho)*dp[:,1] - nu_eff*lap_v
    momentum_z = dt_w + conv_w + (1.0/rho)*dp[:,2] - nu_eff*lap_w

    dk = gradients(k, inputs, order=1)
    domega = gradients(omega, inputs, order=1)
    d2k = gradients(k, inputs, order=2)
    d2omega = gradients(omega, inputs, order=2)
    lap_k = d2k[:,0] + d2k[:,1] + d2k[:,2]
    lap_omega = d2omega[:,0] + d2omega[:,1] + d2omega[:,2]

    conv_k = u*dk[:,0] + v*dk[:,1] + w*dk[:,2]
    conv_omega = u*domega[:,0] + v*domega[:,1] + w*domega[:,2]

    dt_k = dk[:,3]
    dt_omega = domega[:,3]

    P_k = rho*nu_t*S_sq

    R_k = dt_k + conv_k - P_k + beta*rho*k*omega - (mu/rho + nu_t*sigma_k)*lap_k
    R_omega = dt_omega + conv_omega - alpha*S_sq + beta*rho*(omega**2) - (mu/rho + nu_t*sigma_omega)*lap_omega

    continuity_loss = torch.mean(continuity**2)
    momentum_loss = torch.mean(momentum_x**2 + momentum_y**2 + momentum_z**2)
    k_loss = torch.mean(R_k**2)
    omega_loss = torch.mean(R_omega**2)

    physics_residual_loss = continuity_loss + momentum_loss + k_loss + omega_loss
    return physics_residual_loss

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
    model = PINNModel(input_dim=4, output_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    data_loss_function = nn.MSELoss()

    best_val_loss = float('inf')
    # Increase PDE subset size and vary PDE weight over epochs
    pde_subset_size = 64
    initial_pde_weight = 1e-4
    final_pde_weight = 1e-2
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
        # Linearly increase PDE weight over epochs
        pde_weight = initial_pde_weight + (final_pde_weight - initial_pde_weight)*(epoch/epochs)

        current_lr = scheduler.get_last_lr()[0]  # Get current LR
        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr}, PDE Weight: {pde_weight}")

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
            for i, (inputs, outputs) in enumerate(t, start=1):
                inputs, outputs = inputs.to(device), outputs.to(device)
                predictions = model(inputs)
                data_loss = data_loss_function(predictions, outputs)

                if inputs.size(0) > pde_subset_size:
                    subset_indices = torch.randperm(inputs.size(0))[:pde_subset_size]
                else:
                    subset_indices = torch.arange(inputs.size(0))
                pde_inputs = inputs[subset_indices].requires_grad_(True)
                pde_loss_val = physics_loss(model, pde_inputs)

                total_batch_loss = data_loss + pde_weight * pde_loss_val

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
