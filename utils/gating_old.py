import torch
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import find_peaks,peak_widths,butter,sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
import torch.fft
#-----------------------------------------------------------------------------------------------
## Autoencoder related code
class WeightTiedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2):
        super(WeightTiedAutoencoder, self).__init__()
        
        # For weight tying to work with a deeper network, we need to handle layers individually
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.activation1 = nn.Sigmoid()
        # self.encoder_layer2 = nn.Linear(hidden_dim, 1)  # bottleneck
        self.encoder_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.activation2 = nn.Sigmoid()
        self.encoder_layer3=nn.Linear(hidden_dim2,1)
        # Decoder bias for the output layer
        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dim1))
        self.decoder_bias2 = nn.Parameter(torch.zeros(input_dim))
        self.decoder_bias3 = nn.Parameter(torch.zeros(hidden_dim2))
        # Custom positive initialization
        self._init_weights()
## must be positive initiazation for numerical stability
    def _init_weights(self):
        # Positive initialization (e.g., uniform in (0.01, 1.0))
        nn.init.uniform_(self.encoder_layer1.weight, a=0.01, b=0.1)
        nn.init.uniform_(self.encoder_layer2.weight, a=0.01, b=0.1)
        nn.init.uniform_(self.encoder_layer3.weight, a=0.01, b=0.1)
        # Optionally: ensure biases are positive too
        nn.init.uniform_(self.encoder_layer1.bias, a=0.01, b=0.1)
        nn.init.uniform_(self.encoder_layer2.bias, a=0.01, b=0.1)
        nn.init.uniform_(self.encoder_layer3.bias, a=0.01, b=0.1)
    def forward(self, x):
        # Encoder forward pass
        h1 = self.activation1(self.encoder_layer1(x))
        h2= self.activation2(self.encoder_layer2(h1))
        encoded = self.encoder_layer3(h2)
        
        # Decoder using transposed weights (weight tying)
        # 1. From bottleneck (1) to hidden_dim2
        # Use the transpose of the third encoder layer's weights
        d_h2 = nn.functional.linear(encoded, self.encoder_layer3.weight.t(), self.decoder_bias3)
        d_h2 = self.activation2(d_h2) # Apply activation
        
        # 2. From hidden_dim2 to hidden_dim1
        # Use the transpose of the second encoder layer's weights
        d_h1 = nn.functional.linear(d_h2, self.encoder_layer2.weight.t(), self.decoder_bias1)
        d_h1 = self.activation1(d_h1) # Apply activation

        # 3. From hidden_dim1 to input_dim (reconstruction)
        # Use the transpose of the first encoder layer's weights
        decoded = nn.functional.linear(d_h1, self.encoder_layer1.weight.t(), self.decoder_bias2)
        # No activation is typically used on the final reconstructed output
        
        return decoded, encoded
#-----------------------------------------------------------------------------------------------
# Variational Autoencoder related code
class WeightTiedVAE(nn.Module):
    """
    A Variational Autoencoder with tied weights.
    The noise covariance matrix for the Mahalanobis distance can be passed at initialization.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, latent_dim=1, beta=1.0, Sigma_noise=None):
        super(WeightTiedVAE, self).__init__()
        self.beta = beta

        # --- Encoder ---
        self.encoder_layer1 = nn.Linear(input_dim, hidden_dim1)
        self.activation1 = nn.Sigmoid()
        self.encoder_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.activation2 = nn.Sigmoid()
        self.fc_mean = nn.Linear(hidden_dim2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim2, latent_dim)

        # --- Decoder (with weight tying) ---
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim2)
        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dim1))
        self.decoder_bias2 = nn.Parameter(torch.zeros(input_dim))
        
        # --- Handle Noise Covariance ---
        if Sigma_noise is not None:
            # Use the provided covariance matrix
            # Add a small value to the diagonal for numerical stability
            sigma_stable = Sigma_noise + torch.eye(Sigma_noise.shape[0], device=Sigma_noise.device) * 1e-6
            self.register_buffer("Sigma_cholesky", torch.linalg.cholesky(sigma_stable))
        else:
            # Initialize as identity; can be set later with set_covariance()
            self.register_buffer('Sigma_cholesky', torch.eye(input_dim))

        self._init_weights()

    def _init_weights(self):
        for layer in [self.encoder_layer1, self.encoder_layer2, self.fc_mean, self.fc_log_var, self.decoder_fc1]:
            nn.init.uniform_(layer.weight, a=0.01, b=0.1)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, a=0.01, b=0.1)

    def set_covariance(self, data):
        """
        Computes and sets the Cholesky decomposition of the data's covariance matrix.
        Useful if a covariance matrix was not provided at initialization.
        """
        print("Computing and setting data covariance matrix from data...")
        with torch.no_grad():
            cov_matrix = torch.cov(data.T)
            cov_matrix += torch.eye(data.shape[1], device=data.device) * 1e-6
            self.Sigma_cholesky = torch.linalg.cholesky(cov_matrix)
    
    def encode(self, x):
        h1 = self.activation1(self.encoder_layer1(x))
        h2 = self.activation2(self.encoder_layer2(h1))
        mean = self.fc_mean(h2)
        log_var = self.fc_log_var(h2)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        d_h2 = self.activation2(self.decoder_fc1(z))
        d_h1 = nn.functional.linear(d_h2, self.encoder_layer2.weight.t(), self.decoder_bias1)
        d_h1 = self.activation1(d_h1)
        decoded = nn.functional.linear(d_h1, self.encoder_layer1.weight.t(), self.decoder_bias2)
        return decoded

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mean, log_var, z

    def vae_loss_mahalanobis(self, x):
        reconstructed, mean, logvar, z = self.forward(x)
        diff = x - reconstructed
        y = torch.linalg.solve_triangular(self.Sigma_cholesky, diff.T, upper=False)
        maha_squared = torch.sum(y**2, dim=0)
        recon_loss = 0.5 * torch.mean(maha_squared)
        kl_element = 1 + logvar - mean.pow(2) - logvar.exp()
        kl_loss = -0.5 * self.beta * torch.mean(torch.sum(kl_element, dim=1))
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss
#-----------------------------------------------------------------------------------------------
def train_model(model, data:torch.Tensor, model_type: str, max_epoch: int, learning_rate: float=0.001, max_iter: int=15):
    """
    Trains a PyTorch model (AE or VAE) using the L-BFGS optimizer.

    Args:
        model (nn.Module): The model to train.
        data (torch.Tensor): The input data.
        model_type (str): The type of model, either 'ae' for Autoencoder or 
                          'vae' for Variational Autoencoder.
        max_epoch (int): The maximum number of training epochs.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        tuple: A tuple containing:
            - nn.Module: The trained model.
            - list: The history of loss values over epochs.
    """
    if model_type not in ['ae', 'vae']:
        raise ValueError("model_type must be either 'ae' or 'vae'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=max_iter)
    loss_history = []
    
    # Specific setup for 'ae' model type
    if model_type == 'ae':
        criterion = nn.MSELoss()

    print(f"Starting {model_type.upper()} Training... 🚀")
    for epoch in range(max_epoch):
        # Using a dictionary to store epoch-specific losses for logging
        epoch_losses = {}

        def closure():
            optimizer.zero_grad()
            
            if model_type == 'vae':
                total_loss, recon_loss, kl_loss = model.vae_loss_mahalanobis(data)
                # Store losses for detailed printing outside the closure
                epoch_losses['recon'] = recon_loss.item()
                epoch_losses['kl'] = kl_loss.item()
                loss = total_loss
            else: # 'ae'
                decoded, _ = model(data)
                loss = criterion(decoded, data)

            loss.backward()
            return loss.detach()
        
        # Perform optimization step
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        
        # Print epoch results
        if model_type == 'vae':
            print(f'Epoch [{epoch+1:02}/{max_epoch}], Total Loss: {loss.item():.4f}, '
                  f'Recon Loss: {epoch_losses.get("recon", 0):.4f}, KL Loss: {epoch_losses.get("kl", 0):.4f}')
        else: # 'ae'
            print(f'Epoch [{epoch+1}/{max_epoch}], Loss: {loss.item():.8f}')
    
    return model, loss_history


def plot_loss(loss_history: list, model_type: str):
    """
    Plots the training loss over epochs.

    Args:
        loss_history (list): A list of loss values.
        model_type (str): The type of model ('ae' or 'vae') to set the title.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', label='Total Loss')
    
    title = f'{model_type.upper()} Training Loss Over Time' if model_type in ['ae', 'vae'] else 'Training Loss Over Time'
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()

def run_training(nav_signal_tensor: torch.Tensor, model_type: str, noise_real: torch.Tensor = None, beta=1,
                         max_epoch=45, learning_rate=0.001, kick_out_num=0, 
                          hidden_dim1=None, hidden_dim2=None):
    """
    Initializes and trains either a VAE or an AE model.
    
    Args:
        nav_signal_tensor (torch.Tensor): The input signal tensor (Ncoil, Nsample).
        model_type (str): Type of model to train, either 'ae' or 'vae'.
        noise_real (torch.Tensor, optional): Real noise data, required for VAE. Defaults to None.
        beta (int, optional): Beta parameter for VAE. Defaults to 1.

        max_epoch (int, optional): Number of training epochs. Defaults to 45.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
        kick_out_num (int, optional): Number of initial samples to discard. Defaults to 0.
        hidden_dim1 (int, optional): Size of the first hidden layer. Defaults to None.
        hidden_dim2 (int, optional): Size of the second hidden layer. Defaults to None.
        
    Returns:
        nn.Module: The trained model.
    """
    # Common preprocessing
    nav_signal_tensor-= nav_signal_tensor.mean(axis=1,keepdim=True)  # Center
    nav_signal_tensor /= nav_signal_tensor.max() # Normalize

    nav_signal_tensor = nav_signal_tensor.T  # (Nsample, Ncoil)
    nav_signal_tensor = nav_signal_tensor[kick_out_num:]
    print("nav_signal_tensor shape: ", nav_signal_tensor.shape)
    


    # Set default hidden dimensions if not provided
    if hidden_dim1 is None:
        hidden_dim1 = nav_signal_tensor.shape[1] // 2
    if hidden_dim2 is None:
        hidden_dim2 = nav_signal_tensor.shape[1] // 4

    model = None
    # Model specific initialization
    if model_type == 'vae':
        if noise_real is None:
            raise ValueError("noise must be provided for VAE model.")
        noise_tensor = noise_real / nav_signal_tensor.max()  # (Ncoil, Nsample)       
        noise_cov_tensor = torch.cov(noise_tensor)
        print(noise_cov_tensor.shape)
        
        # NOTE: WeightTiedVAE class is assumed to be defined elsewhere and imported.
        model = WeightTiedVAE(
            input_dim=nav_signal_tensor.shape[1],
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            beta=beta,
            Sigma_noise=noise_cov_tensor
        )
    elif model_type == 'ae':
        # NOTE: WeightTiedAutoencoder class is assumed to be defined elsewhere and imported.
        model = WeightTiedAutoencoder(
            input_dim=nav_signal_tensor.shape[1],
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2
        )
    else:
        raise ValueError("model_type must be either 'ae' or 'vae'")

    # Common training and plotting logic
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} parameters")
    
    trained_model, loss_history = train_model(
        model=model,
        data=nav_signal_tensor,
        model_type=model_type,
        max_epoch=max_epoch,
        learning_rate=learning_rate
    )
    print(f"\n✅ Training {model_type.upper()} finished.")
    
    # Plot the captured loss history
    plot_loss(loss_history, model_type)
    
    return trained_model

#-----------------------------------------------------------------------------------------------
# inferring
def find_peak_freq(latent, TR, freq_min=0.7, freq_max=2.0):
    """
    Finds the dominant frequency in a signal within a specified range.

    This function calculates the Fast Fourier Transform (FFT) of the input
    signal and identifies the frequency with the highest magnitude strictly
    within the [freq_min, freq_max] range.

    Args:
        latent (np.ndarray): The input time-series signal.
        TR (float): The sampling interval (Time Repetition), in seconds.
        freq_min (float, optional): The minimum frequency of the search range (Hz). Defaults to 1.0.
        freq_max (float, optional): The maximum frequency of the search range (Hz). Defaults to 2.0.

    Raises:
        ValueError: If no frequencies in the signal are found within the
                    specified [freq_min, freq_max] range.

    Returns:
        float: The frequency (in Hz) with the maximum magnitude in the specified range.
    """
    # 1. Perform FFT
    fft_z = np.fft.fft(latent)
    fft_magnitude = np.abs(fft_z)
    fft_freq = np.fft.fftfreq(len(latent), d=TR)

    # 2. Create a boolean mask for the desired frequency range
    # We only care about positive frequencies, so we can also add fft_freq >= 0,
    # but the specified 1-2 Hz range already covers this.
    search_mask = (fft_freq >= freq_min) & (fft_freq <= freq_max)

    # 3. Check if any frequencies exist in this range
    if not np.any(search_mask):
        raise ValueError(f"No frequencies found in the search range {freq_min} Hz to {freq_max} Hz.")

    # 4. Apply the mask to the magnitudes and corresponding frequencies
    magnitude_search_area = fft_magnitude[search_mask]
    freq_search_area = fft_freq[search_mask]
    
    # 5. Find the index of the highest peak *within the search area*
    # np.argmax() guarantees finding the highest peak in the provided array
    relative_peak_index = np.argmax(magnitude_search_area)

    # 6. Get the corresponding frequency from the *masked frequency array*
    peak_freq = freq_search_area[relative_peak_index]
    
    print(f"Peak frequency: {peak_freq:.2f} Hz (from range {freq_min}-{freq_max} Hz)")
    return peak_freq


def dynamic_find_min_RR(latent,TR,freq_threshold = 0.7):
    """
    Given a 1D latent signal and the repetition time (TR), this function computes the
    minimum RR interval based on the dominant frequency component above a specified threshold."""
    ## 1. do fft
    fft_z = np.fft.fft(latent)
    fft_magnitude = np.abs(fft_z)
    fft_freq = np.fft.fftfreq(len(latent), d=TR)
    ## 2. Find Index in fft_freq > 0.8 Hz --which means that the heart rate is at least 48 BPM---
    try:
        freq_start_index = np.where(fft_freq > freq_threshold)[0][0]
        print(f"Algorithm starts searching from index: {freq_start_index} (Frequency: {fft_freq[freq_start_index]:.2f} Hz)")
    except IndexError:
        raise ValueError(f"No frequency found in the array greater than {freq_threshold} Hz.")
    # 3. Find the Highest Peak in the Search Area ---
    # The search area for the peak is from `freq_start_index` to the end of the array.
    # The positive frequencies are in the first half of the array
    n_samples = len(latent)
    positive_freq_end_index = n_samples // 2
    magnitude_search_area = fft_magnitude[freq_start_index:positive_freq_end_index]
    # Find the index of the max value *within the search area*
    relative_peak_index = np.argmax(magnitude_search_area)
    # Convert this to the absolute index in the original `fft_magnitude` array
    peak_index = freq_start_index + relative_peak_index

    # print(f"Highest peak after {freq_threshold} Hz found at index: {peak_index}")
    # print(f"Peak corresponds to frequency: {fft_freq[peak_index]:.2f} Hz with magnitude: {fft_magnitude[peak_index]:.2f}")
    # --- 4. Calculate Peak Width ---
    # Use peak_widths to find the width of our identified peak.
    # We are interested in the right-side intersection point (right_ips).
    widths, _, _, right_ips = peak_widths(fft_magnitude, [peak_index], rel_height=0.85)
    # right_ips is an array; we get the first element as we only analyzed one peak
    right_ip_index = int(right_ips[0])
    # --- 5. Index fft_freq with the Result ---
    final_frequency = fft_freq[right_ip_index]*1.1
    min_rr= 1/final_frequency
    return min_rr
def _filter_extrema(latent_z, peaks, valleys):
    """
    Filters peaks and valleys to ensure they alternate and keeps the most extreme ones.
    """
    combined_indices = np.sort(np.concatenate([peaks, valleys]))
    if len(combined_indices) == 0:
        return np.array([]), np.array([])

    # Initialize lists for the filtered indices and their types
    filtered_indices = []
    filtered_types = []  # 1 for peak, -1 for valley

    # Get the first point and its type
    first_index = combined_indices[0]
    if first_index in peaks:
        filtered_indices.append(first_index)
        filtered_types.append(1)
        last_type = 1
    else: # it's a valley
        filtered_indices.append(first_index)
        filtered_types.append(-1)
        last_type = -1

    # Iterate through the rest of the combined indices
    for i in range(1, len(combined_indices)):
        current_index = combined_indices[i]
        is_peak = current_index in peaks
        
        current_type = 1 if is_peak else -1

        # Check for consecutive points of the same type
        if current_type == last_type:
            # Get the index of the last point added
            last_index = filtered_indices[-1]
            
            # Compare and update if the current point is more extreme
            if (is_peak and latent_z[current_index] > latent_z[last_index]) or \
               (not is_peak and latent_z[current_index] < latent_z[last_index]):
                filtered_indices[-1] = current_index
        else:
            # Add the new point since it's an alternating type
            filtered_indices.append(current_index)
            filtered_types.append(current_type)
            last_type = current_type

    # Separate the filtered lists back into peaks and valleys
    filtered_peaks = [filtered_indices[i] for i, t in enumerate(filtered_types) if t == 1]
    filtered_valleys = [filtered_indices[i] for i, t in enumerate(filtered_types) if t == -1]

    return np.array(filtered_peaks), np.array(filtered_valleys)

def _preprocess_gating_signal(nav_signal_tensor, kick_out_num):
    """Prepares the navigator signal for processing."""
    nav_signal_tensor-= nav_signal_tensor.mean(axis=1,keepdim=True)  # Center
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
    return latent_z - np.mean(latent_z)

def _calculate_hrv_metrics(extrema_indices, tr_value):
    """Calculates heart rate and variability metrics from peaks or valleys."""
    if len(extrema_indices) < 2:
        return {'rr_intervals': np.array([]), 'heart_rate_bpm': np.nan, 'std_rr_ms': np.nan}
    
    rr_intervals = np.diff(extrema_indices) * tr_value
    heart_rate_bpm = 60 / np.mean(rr_intervals)
    std_rr_ms = np.std(rr_intervals) * 1000
    
    return {'rr_intervals': rr_intervals, 'heart_rate_bpm': heart_rate_bpm, 'std_rr_ms': std_rr_ms}

def _analyze_and_plot_gating_signal(latent_z, peaks, valleys, metrics_peaks, metrics_valleys, TR):
    """Prints analysis results and plots the latent signal with its FFT."""
    peaks = peaks.astype(int)
    valleys = valleys.astype(int)
    print(f"\n--- Gating Signal Analysis ---")
    print(f"Estimated RR interval from peaks: {np.mean(metrics_peaks['rr_intervals'])*1000:.4f} ms")
    print(f"Estimated RR interval from valleys: {np.mean(metrics_valleys['rr_intervals'])*1000:.4f} ms")
    print(f"Estimated heart rate from peaks: {metrics_peaks['heart_rate_bpm']:.4f} BPM")
    print(f"Estimated heart rate from valleys: {metrics_valleys['heart_rate_bpm']:.4f} BPM")
    print("\nHeart Rate Variability (HRV)")
    print(f"RR interval std from peaks: {metrics_peaks['std_rr_ms']:.2f} ms")
    print(f"RR interval std from valleys: {metrics_valleys['std_rr_ms']:.2f} ms")
    
    # FFT analysis
    fft_z = np.fft.fft(latent_z)
    fft_magnitude = np.abs(fft_z)
    fft_freq = np.fft.fftfreq(len(latent_z), d=TR)
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Create a time axis in miniseconds
    time_axis = np.arange(len(latent_z)) * TR*1000
    
    axs[0].plot(time_axis, latent_z, label='Encoded Latent', color='black')
    axs[0].plot(peaks * TR*1000, latent_z[peaks], "ro", label='Peaks')
    axs[0].plot(valleys * TR*1000, latent_z[valleys], "bo", label='Valleys')
    axs[0].set_title(f"Peaks & Valleys\nHR from Valleys: {metrics_valleys['heart_rate_bpm']:.2f} BPM")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Value")
    axs[0].legend()

    axs[1].plot(fft_freq, fft_magnitude, label='FFT Magnitude')
    axs[1].set_xlim([0, 5])  # Common display range
    axs[1].set_title("FFT of Latent Representation")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
def _apply_filter(signal, fs, filter_params):
    """
    Applies a filter to the signal based on the provided parameters.
    (Using numerically stable SOS implementation)
    """
    if filter_params is None:
        return signal # Return original signal if no params are given

    filter_type = filter_params.get('type')
    if not filter_type:
        raise ValueError("'type' must be specified in filter_params (e.g., 'bandpass', 'lowpass').")

    order = filter_params.get('order', 3)
    
    if filter_type == 'bandpass':
        lowcut = filter_params.get('lowcut')
        highcut = filter_params.get('highcut')
        if lowcut is None or highcut is None:
            raise ValueError("Bandpass filter requires 'lowcut' and 'highcut' in filter_params.")
        
        # --- FIX 1 ---
        sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
        print(f"Applied {order}th-order bandpass filter ({lowcut}-{highcut} Hz).")

    elif filter_type == 'lowpass':
        highcut = filter_params.get('highcut')
        if highcut is None:
            raise ValueError("Lowpass filter requires 'highcut' in filter_params.")
            
        # --- FIX 2 ---
        sos = butter(order, highcut, btype='low', fs=fs, output='sos')
        print(f"Applied {order}th-order lowpass filter (cutoff: {highcut} Hz).")

    elif filter_type == 'highpass':
        lowcut = filter_params.get('lowcut')
        if lowcut is None:
            raise ValueError("Highpass filter requires 'lowcut' in filter_params.")
            
        # --- FIX 3 ---
        sos = butter(order, lowcut, btype='high', fs=fs, output='sos')
        print(f"Applied {order}th-order highpass filter (cutoff: {lowcut} Hz).")

    else:
        raise NotImplementedError(f"Filter type '{filter_type}' is not supported. Use 'bandpass', 'lowpass', or 'highpass'.")
    
    # --- FIX 4: Apply the filter using sosfiltfilt ---
    filtered_signal = sosfiltfilt(sos, signal)
    return filtered_signal
# def apply_harmonic_filter(signal_to_filter, f0, TR_s, bandwidth_num_TRs, num_harmonics_to_remove, filter_order=8):
#     """
#     Applies a series of Butterworth band-stop filters to remove a base frequency and its harmonics.

#     An "aggressive" (higher-order, wider-bandwidth) filter is used for the base frequency (f0)
#     to ensure maximum attenuation. A "standard" filter is used for all other harmonics.

#     Args:
#         signal_to_filter (np.ndarray): The 1D input signal array.
#         f0 (float): The base frequency (in Hz) of the harmonic series to remove.
#         TR_s (float): The Repetition Time (in seconds), used for sampling rate and bandwidth.
#         bandwidth_num_TRs (float): The desired filter bandwidth, expressed as a multiple of TR_s.
#         num_harmonics_to_remove (int): The number of integer harmonics of f0 to remove.
#         filter_order (int, optional): The order for the standard Butterworth filter. Defaults to 8.

#     Returns:
#         np.ndarray: The filtered signal.
#     """
#     # --- 1. Calculate Filter Parameters ---
#     sampling_freq = 1 / TR_s
#     nyquist_freq = sampling_freq / 2.0
#     bandwidth_hz = bandwidth_num_TRs * TR_s * 2

#     # print("--- Butterworth Harmonic Filter Configuration ---")
#     # print(f"Base Frequency (f0): {f0:.4f} Hz")
#     # print(f"Sampling Frequency (fs): {sampling_freq:.2f} Hz")
#     # print(f"Base Bandwidth per Harmonic: {bandwidth_hz:.4f} Hz")
#     # print(f"Base Filter Order: {filter_order}")
#     # print("-----------------------------------------------\n")

#     filtered_signal = signal_to_filter.copy()
    
#     print("Applying comb filters...")
#     for n in range(1, num_harmonics_to_remove + 1):
#         # if n == 2:
#         #     # print(f"  - Skipped harmonic {n} as requested.")
#         #     continue       
#         harmonic_freq = n * f0
        
#         # Use a wider bandwidth and higher order for the first harmonic (aggressive)
#         # Use a standard configuration for others (easy)
#         if n == 1:
#             current_bandwidth = bandwidth_hz * 1.5
#             current_filter_order = filter_order * 2
#         else:
#             current_bandwidth = bandwidth_hz
#             current_filter_order = filter_order

#         low_cutoff = harmonic_freq - (current_bandwidth / 2.0)
#         high_cutoff = harmonic_freq + (current_bandwidth / 2.0)

#         # Safety checks
#         if low_cutoff <= 0:
#             print(f"  - Skipped harmonic {n} ({harmonic_freq:.2f} Hz): Lower cutoff is <= 0 Hz.")
#             continue
#         if high_cutoff >= nyquist_freq:
#             print(f"  - Stopping at harmonic {n}. Upper cutoff ({high_cutoff:.2f} Hz) exceeds Nyquist frequency.")
#             break
        
#         try:
#             sos = signal.butter(current_filter_order, [low_cutoff, high_cutoff], btype='bandstop', fs=sampling_freq, output='sos')
#             filtered_signal = signal.sosfiltfilt(sos, filtered_signal)
            
#             print_msg = f"  - Removed band for harmonic {n}: {low_cutoff:.2f}-{high_cutoff:.2f} Hz (BW: {current_bandwidth:.2f} Hz)"
#             if n == 1:
#                 print_msg += f" (Aggressive - Order {current_filter_order})"
#             else:
#                  print_msg += f" (Easy - Order {current_filter_order})"
#             # print(print_msg)

#         except ValueError as e:
#             print(f"  - Could not design filter for harmonic {n} ({harmonic_freq:.2f} Hz): {e}")

#     print("\nFiltering complete.")
#     return filtered_signal

def find_gating_index(model, model_type, TR, nav_signal_tensor,kick_out_num=0, filter_params=None):
    """
    Finds gating indices from a navigator signal using a trained AE or VAE model.
    
    Args:
        model (nn.Module): The trained AE or VAE model.
        model_type (str): The type of model, 'ae' or 'vae'.
        TR (float): The repetition time of the MRI sequence.
        nav_signal_tensor (torch.Tensor): The raw navigator signal tensor.
        kick_out_num (int, optional): Number of initial samples to discard. Defaults to 0.
        filter_params (dict, optional): Parameters for filtering the latent signal.
                                        If provided, dynamic_find_min_RR is skipped.
                                        If None, no filter is applied. Defaults to None.

        bandwidth_num_TRs (float): The desired filter bandwidth, expressed as a multiple of TR_s.
                            For example, a value of 1.0 means the bandwidth will be TR_s Hz.
        num_harmonics_to_remove (int): The number of integer harmonics of f0 to remove.

    Returns:
        tuple: A tuple containing:
            - np.array: The final array of gating indices.
            - np.array: The detected peak indices (adjusted for kick_out_num).
            - np.array: The detected valley indices (adjusted for kick_out_num).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = 1 / TR
    
    processed_signal = _preprocess_gating_signal(nav_signal_tensor, kick_out_num)
    latent_z = _get_latent_representation(model, model_type, processed_signal, device)
    # latent_z=apply_harmonic_filter(signal_to_filter=latent_z,f0=base_f0,TR_s=TR,bandwidth_num_TRs=bandwidth_num_TRs,num_harmonics_to_remove=num_harmonics_to_remove)
    latent_z = _apply_filter(latent_z, fs, filter_params)
    cardiac_peak_freq = find_peak_freq(latent_z, TR, freq_min=0.7, freq_max=2.0)
    respiration_peak_freq = find_peak_freq(latent_z, TR, freq_min=0.1, freq_max=0.5)
    # 4. Determine minimum peak distance (min_rr)
    min_rr = 0.5  # Corresponds to a max heart rate of 120 BPM
    # if filter_params:
    #     # If a filter is applied, skip dynamic calculation and use a fixed default.
    #     # This assumes the filter has removed noise, making the signal reliable for a fixed threshold.
    #     min_rr = 0.5  # Corresponds to a max heart rate of 120 BPM

    #     print(f"Filter applied.")
    # else:
    #     # If no filter, use the dynamic FFT-based method to find min_rr.
    #     print("No filter applied. Using dynamic FFT-based method to find min_rr.")
    #     try:
    #         min_rr = 1/(cardiac_peak_freq+0.2)
    #     except (NameError, ValueError) as e:
    #         print(f"Warning: `dynamic_find_min_RR` failed with error: {e}. Using fixed min RR of 0.5s.")
    #         min_rr = 0.5
    # =================================================================
    # 5. Detect peaks and valleys on the signal
    min_samples = int(min_rr * fs)
    peaks, _ = find_peaks(latent_z, distance=min_samples)
    valleys, _ = find_peaks(-latent_z, distance=min_samples)  
    peaks, valleys = _filter_extrema(latent_z, peaks, valleys)

    # 6. Calculate metrics for both
    metrics_peaks = _calculate_hrv_metrics(peaks, TR)
    metrics_valleys = _calculate_hrv_metrics(valleys, TR)
    
    # 7. Analyze and plot
    _analyze_and_plot_gating_signal(latent_z, peaks, valleys, metrics_peaks, metrics_valleys, TR)
    
    # 8. Choose best gating array and adjust indices
    std_peaks_ms = metrics_peaks['std_rr_ms']
    std_valleys_ms = metrics_valleys['std_rr_ms']
    
    if np.isnan(std_peaks_ms) and not np.isnan(std_valleys_ms):
        chosen_extrema = valleys
        choice = 'valleys'
    elif not np.isnan(std_peaks_ms) and np.isnan(std_valleys_ms):
        chosen_extrema = peaks
        choice = 'peaks'
    elif std_peaks_ms < std_valleys_ms:
        chosen_extrema = peaks
        choice = 'peaks'
    else:
        chosen_extrema = valleys
        choice = 'valleys'

    gating_array = chosen_extrema + kick_out_num
    peaks_adjusted = peaks + kick_out_num
    valleys_adjusted = valleys + kick_out_num
    print(f"Final gating array chosen based on {choice} (lower RR interval standard deviation).")
    return gating_array, peaks_adjusted, valleys_adjusted
# Example 1: No Filtering (Default)
# gating_indices, peaks, valleys = find_gating_index(
#     model=my_model,
#     model_type='ae',
#     TR=0.05,
#     nav_signal_tensor=my_signal,
#     filter_params=None # or simply omit this argument
# )
# Example 2: Bandpass Filter
# bandpass_settings = {
#     'type': 'bandpass',
#     'lowcut': 0.7,      # In Hz
#     'highcut': 3.0,     # In Hz
#     'order': 3
# }
# gating_indices, peaks, valleys = find_gating_index(
#     model=my_model,
#     model_type='ae',
#     TR=0.05,
#     nav_signal_tensor=my_signal,
#     filter_params=bandpass_settings
# )
# Example 3: Lowpass Filter

# lowpass_settings = {
#     'type': 'lowpass',
#     'highcut': 4.0,     # Cutoff frequency in Hz
#     'order': 4
# }
# gating_indices, peaks, valleys = find_gating_index(
#     model=my_model,
#     model_type='ae',
#     TR=0.05,
#     nav_signal_tensor=my_signal,
#     filter_params=lowpass_settings
# )

# Example 4: Highpass Filter
# highpass_settings = {
#     'type': 'highpass',
#     'lowcut': 0.5,      # Cutoff frequency in Hz
#     'order': 2
# }
# gating_indices, peaks, valleys = find_gating_index(
#     model=my_model,
#     model_type='ae',
#     TR=0.05,
#     nav_signal_tensor=my_signal,
#     filter_params=highpass_settings
# )
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
    plt.xlabel("Time (ms)")
    plt.yticks([])  # Hide y-axis ticks
    plt.title("TTL vs ECG vs Self-gating")
    
    if has_label: # Only show legend if a label was actually added
        plt.legend()
        
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
