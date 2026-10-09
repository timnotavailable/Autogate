import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import find_peaks,peak_widths,butter,sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
import torch.fft
from scipy.signal import periodogram

#-----------------------------------------------------------------------------------------------
## Autoencoder related code
#-----------------------------------------------------------------------------------------------
    
class WeightTiedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, sampling_rate, taper_slope=100.0, ranges=None):
        """
        Args:
            sampling_rate: Sampling frequency in Hz (1/TR).
            taper_slope: Controls edge sharpness of the frequency masks.
            ranges: A list of tuples defining the frequency bands for each latent vector.
                    E.g., [(0.0, 0.1), (0.1, 0.7), (0.7, 2.0), (2.0, 10.0)].
        """
        super(WeightTiedAutoencoder, self).__init__()
        
        # --- Default Ranges ---
        default_ranges = [
            (0.0, 0.1),  # Latent 0: Baseline / Drift
            (0.1, 0.7),  # Latent 1: Respiration
            (0.7, 2.0),  # Latent 2: Cardiac
            (2.0, 10.0)  # Latent 3: High Freq Noise / Harmonics
        ]
        
        self.sampling_rate = sampling_rate
        self.taper_slope = taper_slope
        self.ranges = ranges if ranges is not None else default_ranges
        
        # Dynamic Latent Dimension based on the number of ranges provided
        self.latent_dim = len(self.ranges)
        
        # --- Encoder ---
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim2, self.latent_dim) # Latent size is dynamic
        )
        
        # --- Decoder Biases ---
        # The sizes of these biases remain fixed based on the hidden layers
        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dim1))
        self.decoder_bias2 = nn.Parameter(torch.zeros(input_dim))
        self.decoder_bias3 = nn.Parameter(torch.zeros(hidden_dim2))
        
        # Access weights for tying
        self.enc_l1 = self.encoder_net[0]
        self.enc_l2 = self.encoder_net[2]
        self.enc_l3 = self.encoder_net[4]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.decoder_bias1)
        nn.init.zeros_(self.decoder_bias2)
        nn.init.zeros_(self.decoder_bias3)

    def get_mask(self, n_samples, device):
        """
        Generates a Windowed Frequency Mask based on self.ranges.
        """
        # --- Dynamic Mask Generation ---
        freqs = torch.fft.rfftfreq(n_samples, d=1/self.sampling_rate).to(device)
        mask = torch.zeros(len(freqs), self.latent_dim, device=device)
        
        for i, (low_cut, high_cut) in enumerate(self.ranges):
            # 1. High-Pass Edge (Left side)
            if low_cut == 0.0:
                hp_edge = torch.ones_like(freqs)
            else:
                hp_edge = torch.sigmoid(self.taper_slope * (freqs - low_cut))
            
            # 2. Low-Pass Edge (Right side)
            lp_edge = torch.sigmoid(self.taper_slope * (high_cut - freqs))
            
            # 3. Band-Pass Window
            mask[:, i] = hp_edge * lp_edge
            
        return mask

    def forward(self, x):
        n_samples = x.shape[0]
        
        # 1. Encode
        h1 = F.leaky_relu(self.enc_l1(x), 0.1)
        h2 = F.leaky_relu(self.enc_l2(h1), 0.1)
        encoded = self.enc_l3(h2) 
        
        # 2. --- WINDOWED FREQUENCY FILTERING ---
        encoded_fft = torch.fft.rfft(encoded, dim=0)
        windowed_mask = self.get_mask(n_samples, x.device)
        filtered_fft = encoded_fft * windowed_mask
        encoded_filtered = torch.fft.irfft(filtered_fft, n=n_samples, dim=0)
        
        # 3. Decode (Weight Tied)
        d_h2 = F.linear(encoded_filtered, self.enc_l3.weight.t(), self.decoder_bias3)
        d_h2 = F.leaky_relu(d_h2, 0.1)
        
        d_h1 = F.linear(d_h2, self.enc_l2.weight.t(), self.decoder_bias1)
        d_h1 = F.leaky_relu(d_h1, 0.1)

        decoded = F.linear(d_h1, self.enc_l1.weight.t(), self.decoder_bias2)
        
        return decoded, encoded_filtered

    def ae_loss(self, x):
        decoded, encoded = self.forward(x)
        loss = nn.MSELoss()(decoded, x)
        # Return total loss and L2 loss to match training loop expectations
        return loss
#-----------------------------------------------------------------------------------------------
# Variational Autoencoder related code
class WeightTiedVAE(nn.Module):
    """
    A Variational Autoencoder with Frequency Constraints and Tied Weights.
    The separation of frequency bands is enforced architecturally.
    """
    # --- MODIFIED __init__ ---
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, sampling_rate, 
                 taper_slope=100.0, ranges=None, beta=1.0, Sigma_noise=None):
        super(WeightTiedVAE, self).__init__()
        
        # Hyperparameters
        self.beta = beta
        self.sampling_rate = sampling_rate
        self.taper_slope = taper_slope
        
        # Define Ranges (Must be consistent with AE)
        default_ranges = [
            (0.0, 0.1), (0.1, 0.7), (0.7, 2.0), (2.0, 10.0) 
        ]
        self.ranges = ranges if ranges is not None else default_ranges
        self.latent_dim = len(self.ranges) # Dynamic latent dim
        
        # --- Encoder Layers (Using LeakyReLU from AE) ---
        self.enc_l1 = nn.Linear(input_dim, hidden_dim1)
        self.enc_l2 = nn.Linear(hidden_dim1, hidden_dim2)
        # VAE split layer (Outputs latent_dim for mean and log_var)
        self.fc_mean = nn.Linear(hidden_dim2, self.latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim2, self.latent_dim)

        # --- Decoder Biases (Matching AE) ---
        # The decoder now has 3 steps (Latent -> H2, H2 -> H1, H1 -> Input)
        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dim1)) # For H2 -> H1 step
        self.decoder_bias2 = nn.Parameter(torch.zeros(input_dim)) # For H1 -> Input step
        self.decoder_bias3 = nn.Parameter(torch.zeros(hidden_dim2)) # For Latent -> H2 step
        
        # --- Handle Noise Covariance (Unchanged) ---
        if Sigma_noise is not None:
             sigma_stable = Sigma_noise + torch.eye(Sigma_noise.shape[0], device=Sigma_noise.device) * 1e-6
             self.register_buffer("Sigma_cholesky", torch.linalg.cholesky(sigma_stable))
        else:
             self.register_buffer('Sigma_cholesky', torch.eye(input_dim))

        self._init_weights()

    def _init_weights(self):
        # Kaiming Init for everything (LeakyReLU)
        for m in [self.enc_l1, self.enc_l2, self.fc_mean, self.fc_log_var]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        
        nn.init.zeros_(self.decoder_bias1)
        nn.init.zeros_(self.decoder_bias2)
        nn.init.zeros_(self.decoder_bias3)
    
    # --- NEW: Mask Generation Function (From AE) ---
    def get_mask(self, n_samples, device):
        freqs = torch.fft.rfftfreq(n_samples, d=1/self.sampling_rate).to(device)
        mask = torch.zeros(len(freqs), self.latent_dim, device=device)
        
        for i, (low_cut, high_cut) in enumerate(self.ranges):
            if low_cut == 0.0:
                hp_edge = torch.ones_like(freqs)
            else:
                hp_edge = torch.sigmoid(self.taper_slope * (freqs - low_cut))
            
            lp_edge = torch.sigmoid(self.taper_slope * (high_cut - freqs))
            mask[:, i] = hp_edge * lp_edge
            
        return mask

    def encode(self, x):
        # Use LeakyReLU activations
        h1 = F.leaky_relu(self.enc_l1(x), 0.1)
        h2 = F.leaky_relu(self.enc_l2(h1), 0.1)
        mean = self.fc_mean(h2)
        log_var = self.fc_log_var(h2)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        # --- Decoder Steps (Weight Tied, Matching AE structure) ---
        
        # Step 1: Latent -> H2 (Tied to fc_mean/fc_log_var weights, but we need one set)
        # Note: We must pick one weight matrix for tying. We use fc_mean.
        d_h2 = F.linear(z, self.fc_mean.weight.t(), self.decoder_bias3) 
        d_h2 = F.leaky_relu(d_h2, 0.1)
        
        # Step 2: H2 -> H1 (Tied to enc_l2 weight)
        d_h1 = F.linear(d_h2, self.enc_l2.weight.t(), self.decoder_bias1)
        d_h1 = F.leaky_relu(d_h1, 0.1)
        
        # Step 3: H1 -> Input (Tied to enc_l1 weight)
        decoded = F.linear(d_h1, self.enc_l1.weight.t(), self.decoder_bias2)
        return decoded

    def forward(self, x):
        n_samples = x.shape[0]
        
        # 1. Encode and Reparameterize
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        
        # 2. --- FREQUENCY FILTERING (Hard Constraint) ---
        encoded_fft = torch.fft.rfft(z, dim=0) # FFT on sampled latent
        freq_mask = self.get_mask(n_samples, x.device)
        filtered_fft = encoded_fft * freq_mask
        z_filtered = torch.fft.irfft(filtered_fft, n=n_samples, dim=0) # Final constrained latent
        
        # 3. Decode
        reconstructed = self.decode(z_filtered)
        
        # Return constrained latent 'z_filtered' and parameters
        return reconstructed, mean, log_var, z_filtered 

    # --- MODIFIED LOSS FUNCTION ---
    def vae_loss_mahalanobis(self, x):
        # Use z_filtered for decoding/reconstruction loss
        reconstructed, mean, logvar, z_filtered = self.forward(x) 
        
        # 1. Reconstruction Loss (Mahalanobis)
        diff = x - reconstructed
        # Ensure diff has correct dimensions for Cholesky solve: (input_dim, Batch)
        y = torch.linalg.solve_triangular(self.Sigma_cholesky, diff.T, upper=False) 
        maha_squared = torch.sum(y**2, dim=0)
        recon_loss = 0.5 * torch.mean(maha_squared)
        
        # 2. KL Divergence Loss (Standard VAE regularization)
        kl_element = 1 + logvar - mean.pow(2) - logvar.exp()
        kl_loss = -0.5 * self.beta * torch.mean(torch.sum(kl_element, dim=1))
        
        total_loss = recon_loss + kl_loss
        
        return total_loss,kl_loss
#-----------------------------------------------------------------------------------------------
def train_model(model, data: torch.Tensor, model_type: str, max_epoch: int, learning_rate: float = 0.001, max_iter: int = 15):
    """
    Trains a PyTorch model (AE or VAE) using the L-BFGS optimizer.
    """
    if model_type not in ['ae', 'vae']:
        raise ValueError("model_type must be either 'ae' or 'vae'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    # L-BFGS optimizer configuration
    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter, history_size=100, line_search_fn='strong_wolfe')
    loss_history = []
    
    print(f"Starting {model_type.upper()} Training... 🚀")
    
    model.train()
    for epoch in range(max_epoch):
        # Dictionary to store loss components for printing
        epoch_losses = {}

        def closure():
            optimizer.zero_grad()
            
            if model_type == 'vae':
                # --- MODIFIED: VAE returns (total_loss, kl_loss) ---
                total_loss, kl_loss = model.vae_loss_mahalanobis(data)
                
                # Store for logging
                # Derived Recon Loss = Total - KL
                epoch_losses['kl'] = kl_loss.item()
                epoch_losses['recon'] = total_loss.item() - kl_loss.item() 
                
                loss = total_loss
            else: # 'ae'
                # --- MODIFIED: AE returns single loss ---
                loss = model.ae_loss(data)
                # No specific components to log for AE anymore
            
            loss.backward()
            return loss

        # Perform optimization step
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        
        # Print epoch results
        if model_type == 'vae':
            print(f'Epoch [{epoch+1:02}/{max_epoch}], Total: {loss.item():.6f}, '
                  f'Recon: {epoch_losses.get("recon", 0):.6f}, KL: {epoch_losses.get("kl", 0):.6f}')
        else: # 'ae'
            print(f'Epoch [{epoch+1}/{max_epoch}], Total Loss: {loss.item():.8f}')
    
    return model, loss_history


def plot_loss(loss_history: list, model_type: str):
    """
    Plots the training loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', label='Total Loss')
    
    title = f'{model_type.upper()} Training Loss Over Time'
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.show()


def run_training(nav_signal_tensor: torch.Tensor, model_type: str, 
                 sampling_rate: float,
                 noise_real: torch.Tensor = None, beta=1,
                 max_epoch=45, learning_rate=1.0, kick_out_num=0, 
                 hidden_dim1=None, hidden_dim2=None,
                 taper_slope=100.0,
                 ranges=None
                 ):
    
    # Preprocessing
    data = nav_signal_tensor.clone()
    data /= data.max() # Normalize
    data = data.T # (Nsample, Ncoil)
    data = data[kick_out_num:]
    print("Input data shape:", data.shape)
    
    if hidden_dim1 is None: hidden_dim1 = data.shape[1] // 2
    if hidden_dim2 is None: hidden_dim2 = data.shape[1] // 4

    model = None

    if model_type == 'vae':
        if noise_real is None: raise ValueError("noise needed for VAE")
        noise_tensor = noise_real / nav_signal_tensor.max()     
        noise_cov_tensor = torch.cov(noise_tensor)
        
        # Initialize VAE with frequency constraint parameters
        model = WeightTiedVAE(
            input_dim=data.shape[1],
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            beta=beta,
            Sigma_noise=noise_cov_tensor,
            sampling_rate=sampling_rate,  # Passed to VAE
            taper_slope=taper_slope,          # Default Harder mask
            ranges=ranges
        )
        
    elif model_type == 'ae':
        # Initialize AE with frequency constraint parameters
        model = WeightTiedAutoencoder(
            input_dim=data.shape[1],
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            sampling_rate=sampling_rate,  # Passed to AE
            taper_slope=taper_slope,          # Default Harder mask 
            ranges=ranges        
        )
    else:
        raise ValueError("Invalid model_type")

    # Training
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")
    
    trained_model, loss_history = train_model(
        model=model,
        data=data,
        model_type=model_type,
        max_epoch=max_epoch,
        learning_rate=learning_rate
    )
    print(f"\n✅ Training {model_type.upper()} finished.")
    
    plot_loss(loss_history, model_type)
    
    return trained_model

#-----------------------------------------------------------------------------------------------
# inferring
def find_clean_peaks(sig, distance=None, edge_margin=5):
    """
    Finds peaks and valleys but removes those too close to the start or end of the signal.
    This prevents edge artifacts (like a high starting value) from being counted as a peak.
    """
    # 1. Find raw peaks
    peaks, _ = find_peaks(sig, distance=distance)
    
    # 2. Find raw valleys (invert signal)
    valleys, _ = find_peaks(-sig, distance=distance)
    
    # 3. Edge Filtering
    if len(peaks) > 0:
        mask_p = (peaks > edge_margin) & (peaks < len(sig) - edge_margin)
        peaks = peaks[mask_p]
        
    if len(valleys) > 0:
        mask_v = (valleys > edge_margin) & (valleys < len(sig) - edge_margin)
        valleys = valleys[mask_v]
        
    return peaks, valleys

def find_peak_freq(latent, TR, freq_min=0.7, freq_max=2.0):
    """
    Finds the dominant frequency in a clean signal using the Periodogram method.

    This method calculates the Power Spectral Density (PSD) without averaging 
    windows, providing the highest possible frequency resolution for the given 
    signal length.

    Args:
        latent (np.ndarray): The input time-series signal.
        TR (float): The sampling interval (Time Repetition), in seconds.
        freq_min (float, optional): The minimum frequency of the search range (Hz). Defaults to 0.7.
        freq_max (float, optional): The maximum frequency of the search range (Hz). Defaults to 2.0.

    Raises:
        ValueError: If no frequencies in the signal are found within the specified range.

    Returns:
        float: The frequency (in Hz) with the maximum power density in the specified range.
    """
    # Convert Time Repetition (TR) to Sampling Frequency (fs)
    fs = 1.0 / TR
    # Calculate PSD using the Periodogram method
    # By default, this uses a 'boxcar' window (rectangular), which means
    # it uses the raw signal as is, maximizing resolution.
    freqs, psd = signal.periodogram(latent, fs=fs)

    # Create a mask to focus on the requested frequency range
    search_mask = (freqs >= freq_min) & (freqs <= freq_max)

    # Handle case where the mask is empty
    if not np.any(search_mask):
        raise ValueError(f"No frequencies found in the search range {freq_min} Hz to {freq_max} Hz.")

    # Extract the PSD values and Frequencies within the range
    psd_search_area = psd[search_mask]
    freq_search_area = freqs[search_mask]

    # Find the peak
    relative_peak_index = np.argmax(psd_search_area)
    peak_freq = freq_search_area[relative_peak_index]
    print(f"Peak frequency (Periodogram): {peak_freq:.2f} Hz (from range {freq_min}-{freq_max} Hz)")
    return peak_freq

def _preprocess_gating_signal(nav_signal_tensor, kick_out_num):
    """Prepares the navigator signal for processing."""
    nav_signal_tensor /= nav_signal_tensor.max() # Normalize
    
    signal = nav_signal_tensor.T  # (Nsample, Ncoil)
    signal = signal[kick_out_num:]
    print("Processed nav_signal_tensor shape: ", signal.shape)
    return signal

def _get_latent_representation(model, model_type, signal_tensor, device):
    """Extracts the latent representation from the model."""
    model = model.to(device)
    signal_tensor = signal_tensor.to(device)
    
    with torch.no_grad():
        if model_type == 'vae':
            _, mean, _, _ = model(signal_tensor)
            latent_z = mean.squeeze().cpu().numpy()
        elif model_type == 'ae':
            _, encoded = model(signal_tensor)
            latent_z = encoded.squeeze().cpu().numpy()
        else:
            raise ValueError("model_type must be 'ae' or 'vae'")     
    return latent_z 

def _calculate_hrv_metrics(extrema_indices, TR):
    """Calculates heart rate and variability metrics from peaks or valleys."""
    if len(extrema_indices) < 2:
        return {'rr_intervals': np.array([]), 'heart_rate_bpm': np.nan, 'std_rr_ms': np.nan}
    
    rr_intervals = np.diff(extrema_indices) * TR
    heart_rate_bpm = 60 / np.mean(rr_intervals)
    std_rr_ms = np.std(rr_intervals) * 1000
    
    return {'rr_intervals': rr_intervals, 'heart_rate_bpm': heart_rate_bpm, 'std_rr_ms': std_rr_ms}

def _analyze_and_plot_gating_signal(latent_z, peaks, valleys, metrics_peaks, metrics_valleys, TR,is_cardiac=True):
    """Prints analysis results and plots the latent signal with its PSD (Periodogram)."""
    peaks = peaks.astype(int)
    valleys = valleys.astype(int)
    if is_cardiac:        
        print(f"\n--- Cardiac Gating Signal Analysis ---")
        print(f"Estimated RR interval from peaks: {np.mean(metrics_peaks['rr_intervals'])*1000:.4f} ms")
        print(f"Estimated RR interval from valleys: {np.mean(metrics_valleys['rr_intervals'])*1000:.4f} ms")
        print(f"Estimated heart rate from peaks: {metrics_peaks['heart_rate_bpm']:.4f} BPM")
        print(f"Estimated heart rate from valleys: {metrics_valleys['heart_rate_bpm']:.4f} BPM")
        print("\nHeart Rate Variability (HRV)")
        print(f"RR interval std from peaks: {metrics_peaks['std_rr_ms']:.2f} ms")
        print(f"RR interval std from valleys: {metrics_valleys['std_rr_ms']:.2f} ms")
        
        fs = 1.0 / TR
        freqs, psd = signal.periodogram(latent_z, fs=fs)
        
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Create a time axis in milliseconds
        time_axis = np.arange(len(latent_z)) * TR * 1000
        
        # Plot 1: Time Domain
        axs[0].plot(time_axis, latent_z, label='Encoded Latent', color='black')
        axs[0].plot(peaks * TR * 1000, latent_z[peaks], "ro", label='Peaks')
        axs[0].plot(valleys * TR * 1000, latent_z[valleys], "bo", label='Valleys')
        axs[0].set_title(f"Peaks & Valleys\nHR from Valleys: {metrics_valleys['heart_rate_bpm']:.2f} BPM", fontsize=14)
        axs[0].set_xlabel("Time (ms)", fontsize=12)
        axs[0].set_ylabel("Value", fontsize=12)
        axs[0].tick_params(axis='both', which='major', labelsize=11)
        axs[0].legend(fontsize=12)

        # Plot 2: Frequency Domain (PSD)
        axs[1].plot(freqs, psd, label='PSD (Periodogram)', color='purple')
        axs[1].set_xlim([0, 5])  # Common display range (Hz)
        axs[1].set_title("Power Spectral Density", fontsize=14)
        axs[1].set_xlabel("Frequency [Hz]", fontsize=12)
        axs[1].set_ylabel("Power Density", fontsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=11)
        axs[1].legend(fontsize=12)

        plt.tight_layout()
        plt.show()
    if not is_cardiac:  ### respiratory
        print(f"\n--- Respiratory Gating Signal Analysis ---")
        print(f"Estimated Respiratory Rate from peaks: {metrics_peaks['heart_rate_bpm']:.4f} BrPM")
        print(f"Estimated Respiratory Rate from valleys: {metrics_valleys['heart_rate_bpm']:.4f} BrPM")
        
        fs = 1.0 / TR

        freqs, psd = signal.periodogram(latent_z, fs=fs)
        
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Create a time axis in milliseconds
        time_axis = np.arange(len(latent_z)) * TR * 1000
        
        # Plot 1: Time Domain
        axs[0].plot(time_axis, latent_z, label='Encoded Latent', color='black')
        axs[0].plot(peaks * TR * 1000, latent_z[peaks], "ro", label='Peaks')
        axs[0].plot(valleys * TR * 1000, latent_z[valleys], "bo", label='Valleys')
        axs[0].set_title(f"Peaks & Valleys\nHR from Valleys: {metrics_valleys['heart_rate_bpm']:.2f} BPM", fontsize=14)
        axs[0].set_xlabel("Time (ms)", fontsize=12)
        axs[0].set_ylabel("Value", fontsize=12)
        axs[0].tick_params(axis='both', which='major', labelsize=11)
        axs[0].legend(fontsize=12)

        # Plot 2: Frequency Domain (PSD)
        axs[1].plot(freqs, psd, label='PSD (Periodogram)', color='purple')
        axs[1].set_xlim([0, 5])  # Common display range (Hz)
        axs[1].set_title("Power Spectral Density", fontsize=14)
        axs[1].set_xlabel("Frequency [Hz]", fontsize=12)
        axs[1].set_ylabel("Power Density", fontsize=12)
        axs[1].tick_params(axis='both', which='major', labelsize=11)
        axs[1].legend(fontsize=12)
        plt.tight_layout()
        plt.show()

def _apply_filter(input_signal, fs, freq_peak, order, bandwidth=0.2):
    """
    Applies a zero-phase Butterworth bandpass filter (SOS format) centered around freq_peak.

    The passband is defined as [freq_peak - bandwidth, freq_peak + bandwidth].
    Uses Second-Order Sections (SOS) for better numerical stability.

    Args:
        input_signal (np.ndarray): The input time-series signal.
        fs (float): Sampling frequency (Hz).
        freq_peak (float): The center frequency of the bandpass filter (Hz).
        order (int): The order of the filter.
        bandwidth (float, optional): The half-width of the passband (Hz). Defaults to 0.2.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyquist = 0.5 * fs
    low_cut = freq_peak - bandwidth
    high_cut = freq_peak + bandwidth

    # --- Safety Checks ---
    if low_cut <= 0:
        low_cut = 0.05
        print(f"Warning: Filter low cutoff adjusted to {low_cut} Hz")
    
    if high_cut >= nyquist:
        high_cut = nyquist - 0.05
        print(f"Warning: Filter high cutoff adjusted to {high_cut} Hz")

    if low_cut >= high_cut:
        raise ValueError(f"Invalid filter range: [{low_cut:.2f}, {high_cut:.2f}] Hz")

    # --- Filter Design ---
    sos = signal.butter(order, [low_cut, high_cut], btype='bandpass', fs=fs, output='sos')

    # --- FIX: Calculate Custom Padding Length ---
    # 1. Estimate the "memory" duration of the filter (Impulse Response).
    #    T ≈ 1 / (Total Bandwidth)
    total_bandwidth = high_cut - low_cut
    impulse_duration = 1.0 / total_bandwidth  # e.g., 1 / 0.4 = 2.5 seconds
    
    # 2. Set padding to cover roughly 3x the impulse duration to be safe
    pad_seconds = 3 * impulse_duration
    pad_len = int(pad_seconds * fs)
    
    # 3. Scipy constraint: padlen must be < signal length
    #    (If signal is too short, we clamp it to len-1)
    if pad_len >= len(input_signal):
        pad_len = len(input_signal) - 1

    # --- Filter Application ---
    # We pass the custom padlen here. 
    # padtype='odd' is usually best for periodic/oscillating signals.
    filtered_signal = signal.sosfiltfilt(sos, input_signal, padlen=pad_len, padtype='odd')

    return filtered_signal



def find_gating_index(model,
                       model_type,
                       TR, 
                       nav_signal_tensor, 
                       kick_out_num=0, 
                       plot_all=False, 
                       order=6, 
                       bandwidth=0.2, 
                       include_breath=True,
                       plot_overlay_psd=True
                       ):
    """
    Finds gating indices from a navigator signal using a trained AE or VAE model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = 1 / TR
    
    # 1. Get Latent Representation
    processed_signal = _preprocess_gating_signal(nav_signal_tensor, kick_out_num)
    latent_z = _get_latent_representation(model, model_type, processed_signal, device)
    
    # Expect model.ranges to be a list of tuples: [(min, max), (min, max), ...]
    model_ranges = model.ranges 
    N, n_components = latent_z.shape

    if plot_all: 
        t = np.arange(N) * TR
        fig, axes = plt.subplots(n_components, 3, figsize=(20, 4 * n_components))
        fig.suptitle(f'Signal Analysis (TR={TR}s)', fontsize=18, y=1.02)

        for i in range(n_components):
            # Safe handling if model_ranges is smaller than n_components
            if i < len(model_ranges):
                low_cut, high_cut = model_ranges[i]
                title_suffix = f' (Target: {low_cut:.2f}–{high_cut:.2f} Hz)'
            else:
                title_suffix = " (No range defined)"

            sig_comp = latent_z[:, i] # Extract specific component
            
            # A. Time Domain
            axes[i, 0].plot(t, sig_comp, color='#1f77b4', linewidth=1)
            axes[i, 0].set_title(f'Vector {i}: Time Domain{title_suffix}', fontsize=14)
            axes[i, 0].set_xlabel('Time (s)', fontsize=12)
            axes[i, 0].set_ylabel('Amplitude', fontsize=12)
            axes[i, 0].tick_params(axis='both', which='major', labelsize=11)
            axes[i, 0].grid(True, alpha=0.3)

            # B. FFT Magnitude
            fft_vals = np.fft.fft(sig_comp)
            freqs = np.fft.fftfreq(N, d=TR)
            half_n = N // 2
            axes[i, 1].plot(freqs[:half_n], np.abs(fft_vals)[:half_n], color='#ff7f0e', linewidth=1)
            axes[i, 1].set_title(f'Vector {i}: FFT Magnitude', fontsize=14)
            axes[i, 1].set_xlabel('Frequency (Hz)', fontsize=12)
            axes[i, 1].set_ylabel('Magnitude', fontsize=12)
            axes[i, 1].tick_params(axis='both', which='major', labelsize=11)
            axes[i, 1].set_xlim(0, 4)
            axes[i, 1].grid(True, alpha=0.3)

            # C. PSD (Periodogram)
            f_psd, Pxx = periodogram(sig_comp, fs=fs)
            axes[i, 2].plot(f_psd, Pxx, color='#2ca02c', linewidth=1)
            axes[i, 2].set_title(f'Vector {i}: PSD', fontsize=14)
            axes[i, 2].set_xlabel('Frequency (Hz)', fontsize=12)
            axes[i, 2].set_ylabel('PSD', fontsize=12)
            axes[i, 2].tick_params(axis='both', which='major', labelsize=11)
            axes[i, 2].set_xlim(0, 4)
            axes[i, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    if plot_overlay_psd:
            plt.figure(figsize=(12, 6))
            
            # Get a color map with enough unique colors
            cmap = plt.get_cmap('tab10') 
            
            for i in range(n_components):
                sig_comp = latent_z[:, i]
                f_psd, Pxx = periodogram(sig_comp, fs=fs)
                
                # Determine label based on range
                if i < len(model_ranges):
                    low_cut, high_cut = model_ranges[i]
                    range_str = f"[{low_cut:.2f}-{high_cut:.2f} Hz]"
                else:
                    range_str = "[No Range]"
                
                # Plot with unique color and formatted label
                plt.plot(f_psd, Pxx, 
                        color=cmap(i % 10), 
                        label=f'Vector {i} {range_str}', 
                        alpha=0.8, linewidth=2)

            plt.title(f' Power Spectral Density of Latent Vectors', fontsize=16)
            plt.xlabel('Frequency (Hz)', fontsize=14)
            plt.ylabel('Power Spectral Density', fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.xlim(0, 4)  # Focus on physiological range
            
            # Move legend to upper left
            plt.legend(loc='upper left', frameon=True, fontsize=12)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    # ============================================================
    # CARDIAC PROCESSING
    # ============================================================
    print("\n--- Processing Cardiac Signal ---")
    # CORRECTION 1: Select the specific cardiac component 
    idx_cardiac = 2 
    raw_cardiac = latent_z[:, idx_cardiac]
    
    freq_min_c = model_ranges[idx_cardiac][0]
    freq_max_c = model_ranges[idx_cardiac][1]

    # Find Peak Freq
    cardiac_peak_freq = find_peak_freq(raw_cardiac, TR, freq_min=freq_min_c, freq_max=freq_max_c)
    # Apply Filter
    if bandwidth is not None:
        cardiac_signal = _apply_filter(raw_cardiac, fs, cardiac_peak_freq, order, bandwidth=bandwidth)
        print(f"Applied bandpass filter around {cardiac_peak_freq:.2f} Hz")
    else:
        cardiac_signal = raw_cardiac
    
    if bandwidth is not None:
        freq_max_c=cardiac_peak_freq+bandwidth
    else:
        freq_max_c=cardiac_peak_freq+0.2

    min_samples_c = int(fs / (freq_max_c )) 
    
    peaks_c, valleys_c = find_clean_peaks(cardiac_signal, distance=min_samples_c, edge_margin=5)

    metrics_peaks_c = _calculate_hrv_metrics(peaks_c, TR)
    metrics_valleys_c = _calculate_hrv_metrics(valleys_c, TR)

    _analyze_and_plot_gating_signal(cardiac_signal, peaks_c, valleys_c, metrics_peaks_c, metrics_valleys_c, TR, is_cardiac=True)

    # Select best extrema (lowest std dev)
    std_peaks = metrics_peaks_c['std_rr_ms']
    std_valleys = metrics_valleys_c['std_rr_ms']
    
    if std_peaks < std_valleys:
        cardiac_gating_array = peaks_c + kick_out_num
        cardiac_peaks = peaks_c + kick_out_num
        cardiac_valleys = valleys_c + kick_out_num
        print("Selected: Peaks (lower variability)")
    else:
        cardiac_gating_array = valleys_c + kick_out_num
        cardiac_peaks = peaks_c + kick_out_num
        cardiac_valleys = valleys_c + kick_out_num
        print("Selected: Valleys (lower variability)")


    # ============================================================
    # RESPIRATORY PROCESSING
    # ============================================================
    breath_gating_array = None
    breath_peaks = None
    breath_valleys = None

    if include_breath:
        print("\n--- Processing Respiratory Signal ---")

        idx_resp = 1
        raw_resp = latent_z[:, idx_resp]
        
        freq_min_r = model_ranges[idx_resp][0]
        freq_max_r = model_ranges[idx_resp][1]

        respiration_peak_freq = find_peak_freq(raw_resp, TR, freq_min=freq_min_r, freq_max=freq_max_r)
        
        if bandwidth is not None:
            resp_signal = _apply_filter(raw_resp, fs, respiration_peak_freq, order, bandwidth=bandwidth)
            print(f"Applied bandpass filter around {respiration_peak_freq:.2f} Hz")
        else:
            resp_signal = raw_resp

        ## deriving the min sample.
        if bandwidth is not None:
            freq_max_r=freq_max_r+bandwidth
        else:
            freq_max_r=freq_max_r+0.2
        min_samples_r = int(fs / (freq_max_r )) 
        

        peaks_r, valleys_r = find_clean_peaks(resp_signal, distance=min_samples_r, edge_margin=5)
        
        metrics_peaks_r = _calculate_hrv_metrics(peaks_r, TR)
        metrics_valleys_r = _calculate_hrv_metrics(valleys_r, TR)
        
        _analyze_and_plot_gating_signal(resp_signal, peaks_r, valleys_r, metrics_peaks_r, metrics_valleys_r, TR, is_cardiac=False)
        
        std_peaks_r = metrics_peaks_r['std_rr_ms']
        std_valleys_r = metrics_valleys_r['std_rr_ms']
        
        std_peaks_r = np.inf if np.isnan(std_peaks_r) else std_peaks_r
        std_valleys_r = np.inf if np.isnan(std_valleys_r) else std_valleys_r

        if std_peaks_r < std_valleys_r:
            breath_gating_array = peaks_r + kick_out_num
            breath_peaks = peaks_r + kick_out_num
            breath_valleys = valleys_r + kick_out_num
            print("Selected: Peaks")
        else:
            breath_gating_array = valleys_r + kick_out_num
            breath_peaks = peaks_r + kick_out_num
            breath_valleys = valleys_r + kick_out_num
            print("Selected: Valleys")

    # ============================================================
    # RETURN
    # ============================================================
    if not include_breath:
        return cardiac_gating_array, cardiac_peaks, cardiac_valleys
    else:
        return cardiac_gating_array, cardiac_peaks, cardiac_valleys, breath_gating_array, breath_peaks, breath_valleys
    
def get_original_indices(gating_array, discard_num, numtaps):
    return gating_array + discard_num + (numtaps - 1) // 2
##-------------------------------------------------------------------------------------------------------
## compare the metrics of the gating
def plot_gating(self_gating_times_ms=None, TTL_first_trigger=None, ECG_trigger_ms=None):
    """
    Plots vertical lines for different gating/trigger signals over time.

    Skips any signal that is None or empty.

    Args:
        self_gating_times_ms (list-like, optional): 
            Timestamps for self-gating (in ms). Defaults to None.
        TTL_first_trigger (list-like, optional): 
            Timestamps for TTL first trigger (in ms). Defaults to None.
        ECG_trigger_ms (list-like, optional): 
            Timestamps for ECG trigger (in ms). Defaults to None.
    """
    plt.figure(figsize=(20, 6))
    
    has_label = False # Track if any label was added

    # Plot vertical lines for TTL_first_trigger (green)
    if TTL_first_trigger is not None and len(TTL_first_trigger) > 0:
        has_label = True
        for t in TTL_first_trigger:
            plt.axvline(x=t, color='green', linestyle='--',
                        label='TTL first trigger' if t == TTL_first_trigger[0] else "")

    # Plot vertical lines for ECG_trigger_ms (blue)
    if ECG_trigger_ms is not None and len(ECG_trigger_ms) > 0:
        has_label = True
        for t in ECG_trigger_ms:
            plt.axvline(x=t, color='blue', linestyle='-.',
                        label='ECG trigger' if t == ECG_trigger_ms[0] else "")

    # Plot vertical lines for self-gating (red)
    if self_gating_times_ms is not None:
        try:
            # Find the first valid (non-zero) timestamp to attach the label to
            first_gating_time = next(t for t in self_gating_times_ms if t > 0)
        except StopIteration:
            first_gating_time = None  # Handle case with no positive gating times
        
        if first_gating_time is not None:
            has_label = True
            for g in self_gating_times_ms:
                if g > 0:
                    plt.axvline(x=g, color='red', linestyle=':',
                                label='Self-gating' if g == first_gating_time else "")
    
    # --- Styling ---
    plt.xlabel("Time (ms)", fontsize=14)
    plt.yticks([])  # Hide y-axis ticks
    plt.tick_params(axis='x', which='major', labelsize=12)
    plt.title("TTL vs ECG vs Self-gating", fontsize=16)
    
    if has_label: # Only show legend if a label was actually added
        plt.legend(fontsize=12)
        
    plt.tight_layout()
    plt.show()

def find_align_RR_intervals(self_gating_times_ms, ecg_trigger_times_ms, k=5):
    """
    Aligns self-gating and ECG timestamps, calculates RR intervals,
    filters out mistriggers based on ECG intervals, and returns
    the filtered, aligned RR intervals for both signals.

    Mistriggers are identified based on the ECG RR intervals (using a window 
    of 2*k+1, e.g., k=5 -> 11 neighbors):
    - Interval > 1.5 * local median
    - Interval < 0.5 * local median
    
    The filter mask derived from the ECG intervals is then applied to both
    the ECG and self-gating interval arrays.

    Args:
        self_gating_times_ms (np.array): The reference timestamp array (self-gating).
        ecg_trigger_times_ms (np.array): The ECG timestamp array to be aligned.
        k (int, optional): The number of neighbors on each side for the
                           median filter window (window_size = 2*k + 1).
                           Defaults to 5, (10 neighbors + self = window size 11).

    Returns:
        tuple (np.array, np.array):
            - filtered_sg_intervals: The filtered self-gating RR intervals (ms).
            - filtered_ecg_intervals: The filtered ECG RR intervals (ms).
    """
    
    # --- 1. Alignment (from original function) ---
    
    # Handle empty arrays
    if len(self_gating_times_ms) == 0 or len(ecg_trigger_times_ms) == 0:
        print("One or both input timestamp arrays are empty.")
        return np.array([]), np.array([])

    # Find the target time from the first element
    target_time_ms = self_gating_times_ms[0]

    # Find the index in ecg_trigger_times_ms closest to the target time
    closest_index = np.argmin(np.abs(ecg_trigger_times_ms - target_time_ms))
    
    print(f"Target Time (first SG trigger): {target_time_ms}")
    print(f"Closest ECG trigger found: {ecg_trigger_times_ms[closest_index]} (at index {closest_index})")

    # Slice the ECG array from that index
    aligned_ecg_times = ecg_trigger_times_ms[closest_index:]

    # --- Trimming Logic (from original function) ---
    trimmed_gating_times = self_gating_times_ms
    len_gating = len(trimmed_gating_times)
    len_aligned = len(aligned_ecg_times)

    if len_gating > len_aligned:
        print(f"Trimming self_gating_times_ms from {len_gating} to {len_aligned} elements.")
        trimmed_gating_times = trimmed_gating_times[:len_aligned]
    elif len_aligned > len_gating:
        print(f"Trimming aligned_ecg_times from {len_aligned} to {len_gating} elements.")
        aligned_ecg_times = aligned_ecg_times[:len_gating]
    else:
        print("Aligned timestamp arrays are already the same length.")
        
    print(f"Final aligned timestamp array length: {len(trimmed_gating_times)}")

    # Check if we have enough data to calculate intervals
    if len(trimmed_gating_times) < 2:
         print("Aligned arrays have less than 2 elements, cannot calculate intervals.")
         return np.array([]), np.array([])

    # --- 2. Calculate RR Intervals ---
    # np.diff() calculates the difference between adjacent elements
    sg_intervals = np.diff(trimmed_gating_times)
    ecg_intervals = np.diff(aligned_ecg_times)
    
    print(f"Calculated {len(sg_intervals)} intervals for both signals.")

    # --- 3. Identify Mistriggers using Median Filter (NEW) ---
    
    # Window size is 2*k + 1. For k=5, size=11 (the point + 5 neighbors on each side)
    window_size = 2 * k + 1 
    
    if len(ecg_intervals) < window_size:
        print(f"Warning: Interval array length ({len(ecg_intervals)}) is smaller than filter window ({window_size}).")
        # Fallback: use the median of the entire array if it's too small for the window
        median_local_intervals = np.median(ecg_intervals)
    else:
        # Calculate the local median for each interval using a sliding window
        # 'mode=reflect' handles the edges of the array gracefully
        median_local_intervals = median_filter(ecg_intervals, size=window_size, mode='reflect')

    # --- 4. Create Filter Mask (NEW) ---
    
    # Define upper and lower bounds based on the local median
    upper_threshold = 1.5 * median_local_intervals
    lower_threshold = 0.5 * median_local_intervals

    # Find intervals that are *outside* these bounds (i.e., "bad" intervals)
    too_long_mask = ecg_intervals > upper_threshold
    too_short_mask = ecg_intervals < lower_threshold
    
    bad_interval_mask = too_long_mask | too_short_mask # Combine masks
    
    # Invert the mask to get the "good" intervals
    good_interval_mask = ~bad_interval_mask
    
    num_rejected = np.sum(bad_interval_mask)
    num_kept = np.sum(good_interval_mask)
    print(f"Filtering complete. Rejected {num_rejected} mistriggers. Kept {num_kept} intervals.")

    # --- 5. Apply Filter and Return (NEW) ---
    
    # Apply the same 'good' mask (derived from ECG) to both interval arrays
    filtered_sg_intervals = sg_intervals[good_interval_mask]
    filtered_ecg_intervals = ecg_intervals[good_interval_mask]

    # Return the filtered SG intervals and filtered ECG RR intervals
    return filtered_sg_intervals, filtered_ecg_intervals,num_rejected


def calc_metrics(rr_ext,rr_sg):
    """calculate the accuracy and precision
    Args:
    ECG triggering time or ACT
    self gating triggering time  
    """
    mu=np.mean(np.abs(rr_ext - rr_sg))
    rms=np.std(rr_ext-rr_sg)
    return mu, rms

import os
import pickle

def save_dict(self_gating_times_ms, TTL_first_trigger, processed_trigger_ms, metadata: dict, save_path="processed_RR"):
    rr_sg1, rr_ecg1, miss_trigger1 = find_align_RR_intervals(self_gating_times_ms, processed_trigger_ms)
    rr_sg2, rr_act2, miss_trigger2 = find_align_RR_intervals(self_gating_times_ms, TTL_first_trigger)
    rr_act3, rr_ecg3, miss_trigger3 = find_align_RR_intervals(TTL_first_trigger, processed_trigger_ms)

    sg_vs_ecg = {
        "SG": rr_sg1[:-1],
        "ECG": rr_ecg1[:-1],
        "Miss Trigger": miss_trigger1
    }
    sg_vs_act = {
        "SG": rr_sg2[:-1],
        "ACT": rr_act2[:-1],
        "Miss Trigger": miss_trigger2
    }
    act_vs_ecg = {
        "ACT": rr_act3,
        "ECG": rr_ecg3,
        "Miss Trigger": miss_trigger3
    }

    data_dict = {
        "SG vs ECG": sg_vs_ecg,
        "SG vs ACT": sg_vs_act,
        "ACT vs ECG": act_vs_ecg,
        "meta_data": metadata
    }

    filename = f"{metadata['name']}_{metadata['position']}_{metadata['type']}.pkl"
    full_path = os.path.join(save_path, filename)
    os.makedirs(save_path, exist_ok=True)
    with open(full_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved to {full_path}")
