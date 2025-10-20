"""
Neural Network for Thermal Mass Correction based on Garau et al. 2011
Input: 6 time series (T, C, P for downcast and upcast)
Output: 8 parameters (alpha0, alphaS, tau0, tauS for up and down)
"""

from operator import le
from turtle import speed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SP_from_C_torch import gsw_sp_from_c_torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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


def garau_correction(T, C, P, V, t, alpha0, alphaS, tau0, tauS, fs=20):
    """
    Apply Garau correction (COMPLETELY SAFE VERSION - no loops, no in-place)
    """
    batch_size, seq_len = T.shape
    device = T.device

    
    # Use mean flow speed for each profile
    V_mean = torch.mean(V, dim=1)  # [batch_size]
    
    # Compute coefficients with aggressive clamping
    alpha = torch.clamp(alpha0 + alphaS * V_mean, min=1e-3, max=0.5)  # Smaller range
    tau = torch.clamp(tau0 + tauS * V_mean, min=1.0, max=50.0)        # Smaller range
    
    # Compute correction coefficients with safety
    denominator = torch.clamp(1 + 4 * fs * tau, min=1e-3)
    a = torch.clamp(4 * fs * alpha * tau / denominator, min=1e-6, max=1.0)
    
    # Avoid division by alpha - use alternative formulation
    # Original: b = 1 - 2*a/alpha  ← PROBLÉMATIQUE
    # Alternative: b = (1 + 4*fs*tau - 2*4*fs*tau) / (1 + 4*fs*tau)
    b = torch.clamp((1 - 8 * fs * tau) / denominator, min=-5.0, max=5.0)
    
    # Expand for broadcasting
    a_exp = a.unsqueeze(1)  # [batch_size, 1]
    b_exp = b.unsqueeze(1)  # [batch_size, 1]
    
    # Compute temperature differences (vectorized)
    dT = torch.zeros_like(T)
    dT[:, 1:] = T[:, 1:] - T[:, :-1]
    
    # SIMPLIFIED NON-RECURSIVE CORRECTION (no loops!)
    # Instead of recursive formula, use first-order approximation
    correction_factor = a_exp * tau.unsqueeze(1)  # Combined correction
    T_corrected = T + correction_factor * dT
    
    # Ensure output is reasonable
    T_corrected = torch.clamp(T_corrected, min=T.min()-5.0, max=T.max()+5.0)
    
    return T_corrected




class ThermalMassLoss(nn.Module):
    """
    Loss function comparing corrected salinity with CTD reference
    """
    
    def __init__(self):
        super(ThermalMassLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_params, T_down, C_down, P_down, V_down, t_down, T_up, C_up, P_up, V_up, t_up,
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
        T_down_corrected = garau_correction(T_down, C_down, P_down, V_down, t_down,
                                          alpha0_down, alphaS_down, tau0_down, tauS_down)
        T_up_corrected = garau_correction(T_up, C_up, P_up, V_up, t_up,  
                                        alpha0_up, alphaS_up, tau0_up, tauS_up)
        

        
        try:
            # Calculer la salinité
            S_down_corrected = gsw_sp_from_c_torch(C_down, T_down_corrected, P_down)
            S_up_corrected = gsw_sp_from_c_torch(C_up, T_up_corrected, P_up)
            
            # MASQUES AMÉLIORÉS pour ignorer le padding NaN
            valid_down = (torch.isfinite(S_down_corrected) & 
                         torch.isfinite(S_ctd_down) &
                         torch.isfinite(T_down) &
                         torch.isfinite(C_down) &
                         torch.isfinite(P_down) &
                         (C_down > 0.1))  # Éviter conductivité nulle
            
            valid_up = (torch.isfinite(S_up_corrected) & 
                       torch.isfinite(S_ctd_up) &
                       torch.isfinite(T_up) &
                       torch.isfinite(C_up) &
                       torch.isfinite(P_up) &
                       (C_up > 0.1))
            
            # print(f"Valid points - Down: {valid_down.sum().item()}/{valid_down.numel()}, Up: {valid_up.sum().item()}/{valid_up.numel()}")
            
            # Calculer loss seulement sur points valides - CORRECTION ICI
            if valid_down.sum() > 50:  # Au moins 50 points valides
                # Extraire seulement les points valides et calculer MSE directement
                S_down_valid = S_down_corrected[valid_down]
                S_ctd_down_valid = S_ctd_down[valid_down]
                loss_down = torch.mean((S_down_valid - S_ctd_down_valid)**2)
            else:
                print(f"⚠️  Pas assez de points valides down: {valid_down.sum().item()}")
                # Utiliser une pénalité qui a des gradients
                loss_down = torch.mean(predicted_params[:, :4]**2) * 10.0
                
            if valid_up.sum() > 50:  # Au moins 50 points valides  
                # Extraire seulement les points valides et calculer MSE directement
                S_up_valid = S_up_corrected[valid_up]
                S_ctd_up_valid = S_ctd_up[valid_up]
                loss_up = torch.mean((S_up_valid - S_ctd_up_valid)**2)
            else:
                print(f"⚠️  Pas assez de points valides up: {valid_up.sum().item()}")
                # Utiliser une pénalité qui a des gradients
                loss_up = torch.mean(predicted_params[:, 4:]**2) * 10.0
            
            # Loss totale avec régularisation plus faible
            reg_loss = 0.0001 * torch.mean(predicted_params**2)  # Régularisation plus faible
            total_loss = loss_down + loss_up + reg_loss
            
            # print(f"Loss components: down={loss_down.item():.6f}, up={loss_up.item():.6f}, reg={reg_loss.item():.6f}, total={total_loss.item():.6f}")
            
            # Vérifier que la loss est finie, sinon utiliser une pénalité avec gradient
            if not torch.isfinite(total_loss):
                print("❌ Loss non-finie!")
                total_loss = torch.mean(predicted_params**2) * 100.0
            
            return total_loss
            
        except Exception as e:
            print(f"Erreur dans loss: {e}")
            import traceback
            traceback.print_exc()
            # Retourner une pénalité basée sur les paramètres prédits pour avoir des gradients
            return torch.mean(predicted_params**2) * 100.0



class ThermalMassLossDebug(nn.Module):
    def __init__(self):
        super(ThermalMassLossDebug, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, predicted_params, T_down, C_down, P_down, V_down, T_up, C_up, P_up, V_up,
                S_ctd_down, S_ctd_up, valid_mask_down=None, valid_mask_up=None):
        
        # DEBUG: Vérifier predicted_params
        if torch.any(torch.isnan(predicted_params)):
            print(f"❌ predicted_params contient des NaN: {torch.isnan(predicted_params).sum().item()}/{predicted_params.numel()}")
            print(f"   Range: {predicted_params.min().item():.6f} - {predicted_params.max().item():.6f}")
            # Nettoyer les NaN dans predicted_params
            predicted_params = torch.where(torch.isnan(predicted_params), 
                                         torch.zeros_like(predicted_params), 
                                         predicted_params)
        
        # Extraire les paramètres
        alpha0_down = predicted_params[:, 0]
        alphaS_down = predicted_params[:, 1]
        tau0_down = predicted_params[:, 2]
        tauS_down = predicted_params[:, 3]
        alpha0_up = predicted_params[:, 4]
        alphaS_up = predicted_params[:, 5]
        tau0_up = predicted_params[:, 6]
        tauS_up = predicted_params[:, 7]

        # Utiliser la correction Garau et al
        T_down_corrected = garau_correction(T_down, C_down, P_down, V_down,
                                          alpha0_down, alphaS_down, tau0_down, tauS_down)
        T_up_corrected = garau_correction(T_up, C_up, P_up, V_up,
                                        alpha0_up, alphaS_up, tau0_up, tauS_up)

        # T_down_corrected = T_down
        # T_up_corrected = T_up      
        
        try:

            # Nettoyer les salinités si nécessaire
            if torch.any(torch.isnan(S_down_corrected)):
                print("⚠️  Nettoyage S_down_corrected")
                S_down_corrected = torch.where(torch.isnan(S_down_corrected), 
                                             torch.full_like(S_down_corrected, 35.0), 
                                             S_down_corrected)
            
            if torch.any(torch.isnan(S_up_corrected)):
                print("⚠️  Nettoyage S_up_corrected")
                S_up_corrected = torch.where(torch.isnan(S_up_corrected), 
                                           torch.full_like(S_up_corrected, 35.0), 
                                           S_up_corrected)
            
            # MASQUES stricts
            valid_down = (torch.isfinite(S_down_corrected) & 
                         torch.isfinite(S_ctd_down) &
                         torch.isfinite(T_down) &
                         torch.isfinite(C_down) &
                         torch.isfinite(P_down) &
                         (C_down > 0.1) &
                         (S_down_corrected > 20.0) &
                         (S_down_corrected < 45.0))
            
            valid_up = (torch.isfinite(S_up_corrected) & 
                       torch.isfinite(S_ctd_up) &
                       torch.isfinite(T_up) &
                       torch.isfinite(C_up) &
                       torch.isfinite(P_up) &
                       (C_up > 0.1) &
                       (S_up_corrected > 20.0) &
                       (S_up_corrected < 45.0))
            
            # print(f"Valid points - Down: {valid_down.sum().item()}/{valid_down.numel()}, Up: {valid_up.sum().item()}/{valid_up.numel()}")
            
            # Calculer losses avec vérifications multiples
            loss_down = torch.tensor(0.0, device=T_down.device)
            loss_up = torch.tensor(0.0, device=T_down.device)
            
            if valid_down.sum() > 50:
                try:
                    # Extraire seulement les points valides
                    S_down_valid = S_down_corrected[valid_down]
                    S_ctd_down_valid = S_ctd_down[valid_down]
                    
                    # Calculer MSE sur points valides
                    loss_down = torch.mean((S_down_valid - S_ctd_down_valid)**2)
                    
                    if torch.isnan(loss_down):
                        print("❌ loss_down est NaN après calcul")
                        loss_down = torch.tensor(1.0, device=T_down.device)
                    
                except Exception as e:
                    print(f"❌ Erreur calcul loss_down: {e}")
                    loss_down = torch.tensor(1.0, device=T_down.device)
            else:
                loss_down = torch.tensor(1.0, device=T_down.device)
                
            if valid_up.sum() > 50:
                try:
                    # Extraire seulement les points valides
                    S_up_valid = S_up_corrected[valid_up]
                    S_ctd_up_valid = S_ctd_up[valid_up]
                    
                    # Calculer MSE sur points valides
                    loss_up = torch.mean((S_up_valid - S_ctd_up_valid)**2)
                    
                    if torch.isnan(loss_up):
                        print("❌ loss_up est NaN après calcul")
                        loss_up = torch.tensor(1.0, device=T_down.device)
                    
                except Exception as e:
                    print(f"❌ Erreur calcul loss_up: {e}")
                    loss_up = torch.tensor(1.0, device=T_down.device)
            else:
                loss_up = torch.tensor(1.0, device=T_down.device)
            
            # Régularisation avec nettoyage
            if torch.any(torch.isnan(predicted_params)):
                reg_loss = torch.tensor(0.0, device=T_down.device)
            else:
                reg_loss = 0.0001 * torch.mean(predicted_params**2)
                if torch.isnan(reg_loss):
                    reg_loss = torch.tensor(0.0, device=T_down.device)
            
            # Loss totale
            total_loss = loss_down + loss_up + reg_loss
            
            # print(f"Loss components: down={loss_down.item():.6f}, up={loss_up.item():.6f}, reg={reg_loss.item():.6f}, total={total_loss.item():.6f}")
            
            if torch.isnan(total_loss):
                print("❌ Total loss est NaN!")
                return torch.tensor(1.0, device=T_down.device, requires_grad=True)
            
            return total_loss
            
        except Exception as e:
            print(f"❌ Exception in loss: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(1.0, device=T_down.device, requires_grad=True)



class ThermalMassDataset(torch.utils.data.Dataset):
    """
    Dataset class for thermal mass correction data
    """
    def __init__(self, mvp_data, ctd_data):
        """
        Args:
            mvp_data: Dictionary with keys ['TEMP_down', 'COND_down', 'PRES_down', 
                                          'TEMP_up', 'COND_up', 'PRES_up']
            ctd_data: Dictionary with keys ['SALT_down', 'SALT_up']
            sequence_length: Length of input sequences
        """
        self.mvp_data = mvp_data
        self.ctd_data = ctd_data
     
        
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
        t_down = torch.FloatTensor(self.mvp_data['TIME_down'][idx])
        
        temp_up = torch.FloatTensor(self.mvp_data['TEMP_up'][idx])
        cond_up = torch.FloatTensor(self.mvp_data['COND_up'][idx])
        pres_up = torch.FloatTensor(self.mvp_data['PRES_up'][idx])
        speed_up = torch.FloatTensor(self.mvp_data['SPEED_up'][idx])
        t_up = torch.FloatTensor(self.mvp_data['TIME_up'][idx])
        
        salt_ctd_down = torch.FloatTensor(self.ctd_data['SALT_down'][idx])
        salt_ctd_up = torch.FloatTensor(self.ctd_data['SALT_up'][idx])
        
        # ✅ CALCULER LA VRAIE LONGUEUR DE DONNÉES VALIDES
        valid_mask = (~torch.isnan(temp_down)) & (~torch.isnan(cond_down)) & \
                     (~torch.isnan(pres_down)) & (~torch.isnan(temp_up)) & \
                     (~torch.isnan(cond_up)) & (~torch.isnan(pres_up)) & \
                     (~torch.isnan(speed_down)) & (~torch.isnan(speed_up)) & \
                     (~torch.isnan(t_down)) & (~torch.isnan(t_up))
        
        # Compter le NOMBRE TOTAL de points valides (pas le dernier index!)
        num_valid_points = valid_mask.sum().item()
        
        if num_valid_points > 0:
            # Trouver le premier et dernier index valide
            valid_indices = torch.where(valid_mask)[0]
            first_valid_idx = valid_indices[0].item()
            last_valid_idx = valid_indices[-1].item()
            
            # La longueur est la plage du premier au dernier point valide
            sequence_length = last_valid_idx - first_valid_idx + 1
            
            # ✅ OPTION 1: Extraire seulement la partie valide (RECOMMANDÉ)
            # Cela évite d'avoir des NaN au début
            temp_down = temp_down[first_valid_idx:last_valid_idx+1]
            cond_down = cond_down[first_valid_idx:last_valid_idx+1]
            pres_down = pres_down[first_valid_idx:last_valid_idx+1]
            speed_down = speed_down[first_valid_idx:last_valid_idx+1]
            t_down = t_down[first_valid_idx:last_valid_idx+1]
            temp_up = temp_up[first_valid_idx:last_valid_idx+1]
            cond_up = cond_up[first_valid_idx:last_valid_idx+1]
            pres_up = pres_up[first_valid_idx:last_valid_idx+1]
            speed_up = speed_up[first_valid_idx:last_valid_idx+1]
            t_up = t_up[first_valid_idx:last_valid_idx+1]
            salt_ctd_down = salt_ctd_down[first_valid_idx:last_valid_idx+1]
            salt_ctd_up = salt_ctd_up[first_valid_idx:last_valid_idx+1]
            
            # Maintenant sequence_length correspond vraiment à la taille des tensors
            # Et il peut encore y avoir des NaN INTERNES qu'il faut nettoyer
            
        else:
            # Pas de données valides - lever uene exception
            raise ValueError(f"Profile {idx} has no valid data points.")
        
        # ✅ NETTOYER les NaN internes par interpolation (si il en reste)
        def fill_internal_nans(tensor):
            """Remplace les NaN internes par interpolation linéaire"""
            if not torch.any(torch.isnan(tensor)):
                return tensor
            
            mask = torch.isnan(tensor)
            if mask.all():
                return torch.zeros_like(tensor)
            
            # Créer une copie
            filled = tensor.clone()
            indices = torch.arange(len(tensor))
            
            # Interpolation linéaire
            valid_mask = ~mask
            if valid_mask.sum() >= 2:
                # Utiliser interpolation PyTorch
                valid_indices = indices[valid_mask]
                valid_values = tensor[valid_mask]
                
                # Pour chaque NaN, trouver les voisins valides
                for i in range(len(tensor)):
                    if mask[i]:
                        # Trouver le voisin valide avant et après
                        before = valid_indices[valid_indices < i]
                        after = valid_indices[valid_indices > i]
                        
                        if len(before) > 0 and len(after) > 0:
                            # Interpolation linéaire
                            idx_before = before[-1]
                            idx_after = after[0]
                            val_before = tensor[idx_before]
                            val_after = tensor[idx_after]
                            
                            # Interpolation
                            alpha = (i - idx_before) / (idx_after - idx_before)
                            filled[i] = val_before * (1 - alpha) + val_after * alpha
                        elif len(before) > 0:
                            # Forward fill
                            filled[i] = tensor[before[-1]]
                        elif len(after) > 0:
                            # Backward fill
                            filled[i] = tensor[after[0]]
            elif valid_mask.sum() == 1:
                # Un seul point valide - propager
                filled[:] = tensor[valid_mask][0]
            
            return filled
        
        temp_down = fill_internal_nans(temp_down)
        cond_down = fill_internal_nans(cond_down)
        pres_down = fill_internal_nans(pres_down)
        speed_down = fill_internal_nans(speed_down)
        t_down
        temp_up = fill_internal_nans(temp_up)
        cond_up = fill_internal_nans(cond_up)
        pres_up = fill_internal_nans(pres_up)
        speed_up = fill_internal_nans(speed_up)
        t_up = fill_internal_nans(t_up)
        salt_ctd_down = fill_internal_nans(salt_ctd_down)
        salt_ctd_up = fill_internal_nans(salt_ctd_up)
        
        # Stack features
        input_features = torch.stack([
            temp_down, cond_down, pres_down,
            temp_up, cond_up, pres_up
        ], dim=0)  # [num_features=6, sequence_length]
        
        return {
            'input_features': input_features,
            'temp_down': temp_down,
            'cond_down': cond_down,
            'pres_down': pres_down,
            'speed_down': speed_down,
            'time_down': t_down,
            'temp_up': temp_up,
            'cond_up': cond_up,
            'pres_up': pres_up,
            'speed_up': speed_up,
            'time_up': t_up,
            'salt_ctd_down': salt_ctd_down,
            'salt_ctd_up': salt_ctd_up,
            'sequence_length': sequence_length  # ← Maintenant c'est la vraie longueur!
        }


class ThermalMassCorrectionNetFixed(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super(ThermalMassCorrectionNetFixed, self).__init__()
        
        self.hidden_size = hidden_size
        
        # LSTM simple (pas bidirectionnel pour éviter les NaN)
        self.lstm = nn.LSTM(
            input_size=6,  # T_down, C_down, P_down, T_up, C_up, P_up
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,  # ✅ CORRECTION: dropout=0 pour num_layers=1
            bidirectional=False
        )
        
        # ✅ CORRECTION: Feature extractor adapté pour LSTM non-bidirectionnel
        # Input: hidden_size (pas hidden_size * 2)
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 64 -> 32
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 32 -> 16
            nn.ReLU()
        )
        
        # Couche de sortie
        self.output_layer = nn.Linear(hidden_size // 4, 8)  # 16 -> 8 paramètres
        
        # INITIALISATION CRITIQUE
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialisation très conservative"""
        
        # LSTM: initialisation Xavier avec gain réduit
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data, gain=0.5)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data, gain=0.5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # Feature extractor
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0)
        
        # Couche de sortie: très conservative
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.001)
        
        # Biais de sortie: valeurs physiquement raisonnables
        with torch.no_grad():
            initial_bias = torch.tensor([0.02, 0.001, 5.0, 0.1,  # down
                                       0.02, 0.001, 5.0, 0.1], # up
                                      dtype=torch.float32)
            self.output_layer.bias.copy_(initial_bias)
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor [batch_size, num_features, sequence_length]
            lengths: Tensor of actual sequence lengths [batch_size]
        """
        # x shape: [batch_size, num_features, sequence_length]
        # LSTM expects: [batch_size, sequence_length, num_features]
        
        # ✅ CORRECTION: Transposer correctement
        # De [batch_size, num_features=6, sequence_length] 
        # À [batch_size, sequence_length, num_features=6]
        if x.dim() == 3:
            x = x.transpose(1, 2)  # [batch_size, sequence_length, num_features]
        
        batch_size = x.size(0)


        if torch.any(torch.isnan(x)):
            print("⚠️  NaN détecté, remplacement par zéros")
            x = torch.nan_to_num(x, nan=0.0)




        # Si lengths fourni, utiliser pack_padded_sequence
        if lengths is not None:
            # Trier par longueur décroissante (requis par pack_padded_sequence)
            lengths_clamped = torch.clamp(lengths, min=1, max=x.size(1))


            # Pack les séquences
            packed_input = pack_padded_sequence(
                x, 
                lengths_clamped.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            # LSTM sur séquences packées
            packed_output, (hidden, cell) = self.lstm(packed_input)
            
            # Unpack
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
            

            # Prendre le dernier output valide pour chaque séquence
            last_output = torch.zeros(batch_size, self.hidden_size, device=x.device)
            for i in range(batch_size):
                valid_len = min(max(0, lengths[i].item() - 1), lstm_out.size(1) - 1)
                last_output[i] = lstm_out[i, valid_len, :]
        else:
            # Mode sans pack (backward compatibility)
            lstm_out, (hidden, cell) = self.lstm(x)
            last_output = lstm_out[:, -1, :]
        
        # Vérification NaN
        if torch.any(torch.isnan(last_output)):
            print("⚠️  NaN détecté après LSTM, remplacement par zéros")
            print(torch.isnan(last_output).sum().item())
            last_output = torch.zeros_like(last_output)
        
        # Feature extraction
        features = self.feature_extractor(last_output)
        
        # Output
        output = self.output_layer(features)
        
        # Clamp pour éviter les valeurs extrêmes
        output = torch.clamp(output, min=-5.0, max=5.0)
        
        return output