"""
Neural Network for Thermal Mass Correction based on Garau et al. 2011
Input: 6 time series (T, C, P for downcast and upcast)
Output: 8 parameters (alpha0, alphaS, tau0, tauS for up and down)
"""

from turtle import speed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gsw


class ThermalMassCorrectionNet(nn.Module):
    """
    Neural Network for predicting thermal mass correction parameters
    """
    
    def __init__(self, sequence_length=1000, hidden_size=128, num_layers=2):
        super(ThermalMassCorrectionNet, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input: 6 channels (T_down, C_down, P_down, T_up, C_up, P_up)
        self.input_size = 6
        
        # LSTM encoder for temporal patterns
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layers for 8 parameters
        # alpha0_down, alphaS_down, tau0_down, tauS_down, alpha0_up, alphaS_up, tau0_up, tauS_up
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        # Parameter constraints (using sigmoid/tanh for bounded outputs)
        self.alpha_activation = nn.Sigmoid()  # alpha in [0, 1]
        self.tau_activation = nn.Softplus()   # tau > 0
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 6)
        Returns:
            parameters: Tensor of shape (batch_size, 8) containing correction parameters
        """
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        # For bidirectional LSTM, concatenate forward and backward hidden states
        if self.num_layers > 1:
            # Take the last layer's hidden state
            forward_hidden = hidden[-2, :, :]  # Last forward layer
            backward_hidden = hidden[-1, :, :] # Last backward layer
        else:
            forward_hidden = hidden[0, :, :]
            backward_hidden = hidden[1, :, :]
            
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Feature extraction
        features = self.feature_extractor(combined_hidden)
        
        # Output parameters
        raw_params = self.output_layer(features)
        
        # Apply constraints to parameters
        # alpha0_down, alphaS_down, tau0_down, tauS_down, alpha0_up, alphaS_up, tau0_up, tauS_up
        alpha0_down = self.alpha_activation(raw_params[:, 0])
        alphaS_down = self.alpha_activation(raw_params[:, 1])
        tau0_down = self.tau_activation(raw_params[:, 2])
        tauS_down = self.tau_activation(raw_params[:, 3])
        
        alpha0_up = self.alpha_activation(raw_params[:, 4])
        alphaS_up = self.alpha_activation(raw_params[:, 5])
        tau0_up = self.tau_activation(raw_params[:, 6])
        tauS_up = self.tau_activation(raw_params[:, 7])
        
        parameters = torch.stack([
            alpha0_down, alphaS_down, tau0_down, tauS_down,
            alpha0_up, alphaS_up, tau0_up, tauS_up
        ], dim=1)
        
        return parameters


def garau_correction(T, C, P, V, alpha0, alphaS, tau0, tauS, fs=20):
    """
    Apply Garau et al. 2011 thermal mass correction
    
    Args:
        T: Temperature time series (torch.Tensor)
        C: Conductivity time series (torch.Tensor)
        P: Pressure time series (torch.Tensor)
        V: flow speed
        alpha0: Alpha coefficient at surface pressure
        alphaS: Alpha salinity dependence
        tau0: Tau coefficient at surface pressure
        tauS: Tau salinity dependence
        fs: sampling frequency
        
    Returns:
        C_corrected: Corrected conductivity
    """
    batch_size, seq_len = T.shape
    
    # Initialize corrected temperature
    T_corrected = torch.clone(T)
        

    # For simplicity, using constant coefficients here
    alpha = alpha0.unsqueeze(1) + alphaS.unsqueeze(1) * V
    tau = tau0.unsqueeze(1) + tauS.unsqueeze(1) * V
    
    a = 4*fs*alpha*tau/(1+4*fs*tau)
    b = 1 - 2*a/alpha

    # Apply recursive correction
    for i in range(1, seq_len):
        T_corrected[:,i] = -b[:,i-1] * T_corrected[:,i-1] + a[:,i] * (T[:,i] - T[:,i-1])

    return T_corrected


def gsw_sp_from_c_torch(C, T, P):
    """
    Compute practical salinity from conductivity using GSW (approximate)
    This is a simplified version - in practice, you'd use the actual GSW formula
    """
    # Simplified formula for demonstration
    # In practice, implement the full GSW SP_from_C algorithm
    R = C / 42.914  # Approximate conductivity ratio
    rt = torch.sqrt(R)
    S = ((((2.7081e-1 * rt - 7.0261e-1) * rt + 1.4692e-1) * rt + 
          (2.5986e-2 - 6.0269e-3 * rt) * rt) * rt + 3.5000e-1) * rt
    return S * 35.0


class ThermalMassLoss(nn.Module):
    """
    Loss function comparing corrected salinity with CTD reference
    """
    
    def __init__(self):
        super(ThermalMassLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_params, T_down, C_down, P_down, V_down, T_up, C_up, P_up, V_up,
                S_ctd_down, S_ctd_up, valid_mask_down=None, valid_mask_up=None):
        """
        Compute loss based on salinity difference with CTD
        
        Args:
            predicted_params: (batch_size, 8) - predicted correction parameters
            T_down, C_down, P_down: Downcast data
            T_up, C_up, P_up: Upcast data  
            S_ctd_down, S_ctd_up: CTD reference salinity
            valid_mask_down, valid_mask_up: Masks for valid data points
            
        Returns:
            loss: Total loss value
        """
        batch_size = T_down.shape[0]
        
        # Extract parameters
        alpha0_down = predicted_params[:, 0]
        alphaS_down = predicted_params[:, 1]
        tau0_down = predicted_params[:, 2]
        tauS_down = predicted_params[:, 3]
        
        alpha0_up = predicted_params[:, 4]
        alphaS_up = predicted_params[:, 5]
        tau0_up = predicted_params[:, 6]
        tauS_up = predicted_params[:, 7]
        
        # Apply corrections
        T_down_corrected = garau_correction(T_down, C_down, P_down, V_down,
                                          alpha0_down, alphaS_down, tau0_down, tauS_down)
        T_up_corrected = garau_correction(T_up, C_up, P_up, V_up,
                                        alpha0_up, alphaS_up, tau0_up, tauS_up)
        
        # Compute corrected salinity
        S_down_corrected = gsw_sp_from_c_torch(C_down, T_down_corrected, P_down)
        S_up_corrected = gsw_sp_from_c_torch(C_up, T_up_corrected, P_up)
        
        # Compute losses
        if valid_mask_down is not None:
            loss_down = self.mse_loss(S_down_corrected[valid_mask_down], 
                                    S_ctd_down[valid_mask_down])
        else:
            loss_down = self.mse_loss(S_down_corrected, S_ctd_down)
            
        if valid_mask_up is not None:
            loss_up = self.mse_loss(S_up_corrected[valid_mask_up], 
                                  S_ctd_up[valid_mask_up])
        else:
            loss_up = self.mse_loss(S_up_corrected, S_ctd_up)
        
        # Total loss
        total_loss = loss_down + loss_up
        
        # Add regularization on parameters to keep them reasonable
        param_reg = 0.001 * torch.mean(predicted_params**2)
        
        return total_loss + param_reg


class ThermalMassDataset(torch.utils.data.Dataset):
    """
    Dataset class for thermal mass correction data
    """
    
    def __init__(self, mvp_data, ctd_data, sequence_length=1000):
        """
        Args:
            mvp_data: Dictionary with keys ['TEMP_down', 'COND_down', 'PRES_down', 
                                          'TEMP_up', 'COND_up', 'PRES_up']
            ctd_data: Dictionary with keys ['SALT_down', 'SALT_up']
            sequence_length: Length of input sequences
        """
        self.mvp_data = mvp_data
        self.ctd_data = ctd_data
        self.sequence_length = sequence_length
        
        # Number of profiles
        self.n_profiles = len(mvp_data['TEMP_down'])
        
    def __len__(self):
        return self.n_profiles
    
    def __getitem__(self, idx):
        # Get MVP data
        temp_down = torch.FloatTensor(self.mvp_data['TEMP_down'][idx])
        cond_down = torch.FloatTensor(self.mvp_data['COND_down'][idx])
        pres_down = torch.FloatTensor(self.mvp_data['PRES_down'][idx])
        speed_down = torch.FloatTensor(self.mvp_data['SPEED_down'][idx])
        
        temp_up = torch.FloatTensor(self.mvp_data['TEMP_up'][idx])
        cond_up = torch.FloatTensor(self.mvp_data['COND_up'][idx])
        pres_up = torch.FloatTensor(self.mvp_data['PRES_up'][idx])
        speed_up = torch.FloatTensor(self.mvp_data['SPEED_up'][idx])
        
        # Get CTD reference
        salt_ctd_down = torch.FloatTensor(self.ctd_data['SALT_down'][idx])
        salt_ctd_up = torch.FloatTensor(self.ctd_data['SALT_up'][idx])
        
        # Handle variable length sequences by padding/truncating
        seq_len = min(len(temp_down), self.sequence_length)
        
        # Pad or truncate to fixed length
        if len(temp_down) < self.sequence_length:
            pad_len = self.sequence_length - len(temp_down)
            temp_down = torch.cat([temp_down, torch.zeros(pad_len)])
            cond_down = torch.cat([cond_down, torch.zeros(pad_len)])
            pres_down = torch.cat([pres_down, torch.zeros(pad_len)])
            speed_down = torch.cat([speed_down, torch.zeros(pad_len)])
            
            temp_up = torch.cat([temp_up, torch.zeros(pad_len)])
            cond_up = torch.cat([cond_up, torch.zeros(pad_len)])
            pres_up = torch.cat([pres_up, torch.zeros(pad_len)])
            speed_up = torch.cat([speed_up, torch.zeros(pad_len)])
            
            salt_ctd_down = torch.cat([salt_ctd_down, torch.zeros(pad_len)])
            salt_ctd_up = torch.cat([salt_ctd_up, torch.zeros(pad_len)])
        else:
            temp_down = temp_down[:self.sequence_length]
            cond_down = cond_down[:self.sequence_length]
            pres_down = pres_down[:self.sequence_length]
            speed_down = speed_down[:self.sequence_length]
            
            temp_up = temp_up[:self.sequence_length]
            cond_up = cond_up[:self.sequence_length]
            pres_up = pres_up[:self.sequence_length]
            speed_up = speed_up[:self.sequence_length]
            
            salt_ctd_down = salt_ctd_down[:self.sequence_length]
            salt_ctd_up = salt_ctd_up[:self.sequence_length]
        
        # Stack input features
        input_features = torch.stack([
            temp_down, cond_down, pres_down,
            temp_up, cond_up, pres_up
        ], dim=1)
        
        return {
            'input_features': input_features,
            'temp_down': temp_down,
            'cond_down': cond_down,
            'pres_down': pres_down,
            'speed_down': speed_down,
            'temp_up': temp_up,
            'cond_up': cond_up,
            'pres_up': pres_up,
            'speed_up': speed_up,
            'salt_ctd_down': salt_ctd_down,
            'salt_ctd_up': salt_ctd_up,
            'valid_length': seq_len
        }

