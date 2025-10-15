#Initialize and train the neural network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import importlib
import architecture
import numpy as np
importlib.reload(architecture)
from architecture import ThermalMassDataset,ThermalMassCorrectionNet,ThermalMassLoss


def train_thermal_mass_network(mvp_data, ctd_data, sequence_length=1000, num_epochs=50):
    """
    Train the thermal mass correction neural network
    """
    try:
        # Check if we have enough data
        n_pairs = len(mvp_data['TEMP_down'])
        if n_pairs < 2:
            print(f"Not enough data pairs ({n_pairs}) for training. Need at least 2.")
            return None
        
        print(f"Training neural network with {n_pairs} profile pairs...")
        
        # Create dataset
        dataset = ThermalMassDataset(mvp_data, ctd_data, sequence_length=sequence_length)
        
        # Split into train/validation (80/20 split)
        train_size = max(1, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"Training set: {train_size} pairs, Validation set: {val_size} pairs")
        
        # Create data loaders
        batch_size = min(2, train_size)  # Small batch size for limited data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_size > 0 else None
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = ThermalMassCorrectionNet(
            sequence_length=sequence_length,
            hidden_size=64,  # Smaller network for limited data
            num_layers=1
        ).to(device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = ThermalMassLoss()
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                input_features = batch['input_features'].to(device)
                temp_down = batch['temp_down'].to(device)
                cond_down = batch['cond_down'].to(device)
                pres_down = batch['pres_down'].to(device)
                speed_down = batch['speed_down'].to(device)
                temp_up = batch['temp_up'].to(device)
                cond_up = batch['cond_up'].to(device)
                pres_up = batch['pres_up'].to(device)
                speed_up = batch['speed_up'].to(device)
                salt_ctd_down = batch['salt_ctd_down'].to(device)
                salt_ctd_up = batch['salt_ctd_up'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predicted_params = model(input_features)
                
                # Compute loss
                loss = criterion(predicted_params, temp_down, cond_down, pres_down, speed_down,
                               temp_up, cond_up, pres_up, speed_up, salt_ctd_down, salt_ctd_up)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                model.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_features = batch['input_features'].to(device)
                        temp_down = batch['temp_down'].to(device)
                        cond_down = batch['cond_down'].to(device)
                        pres_down = batch['pres_down'].to(device)
                        speed_down = batch['speed_down'].to(device)
                        temp_up = batch['temp_up'].to(device)
                        cond_up = batch['cond_up'].to(device)
                        pres_up = batch['pres_up'].to(device)
                        speed_up = batch['speed_up'].to(device)
                        salt_ctd_down = batch['salt_ctd_down'].to(device)
                        salt_ctd_up = batch['salt_ctd_up'].to(device)
                        
                        predicted_params = model(input_features)
                        loss = criterion(predicted_params, temp_down, cond_down, pres_down, speed_down,
                                       temp_up, cond_up, pres_up, speed_up, salt_ctd_down, salt_ctd_up)
                        
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                
                print(f'Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            else:
                print(f'Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {avg_train_loss:.6f}')
                scheduler.step(avg_train_loss)
            
            # Early stopping check
            if len(train_losses) > 10 and avg_train_loss > np.mean(train_losses[-10:]) * 1.1:
                print("Early stopping: loss not improving")
                break
        
        # Save trained model
        torch.save(model.state_dict(), 'thermal_mass_model.pth')
        print("Model saved as 'thermal_mass_model.pth'")
        
        return model, train_losses, val_losses
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None
