import twixtools
import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path
from tqdm import tqdm
import torch
import torch.fft as fft
from scipy.signal import find_peaks
from scipy import signal
import bart
from bart import bart

from matplotlib.animation import FuncAnimation,FFMpegWriter
from IPython.display import HTML
from ipywidgets import interactive
from ipywidgets import widgets
import gc
import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # or "forkserver"
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from collections import Counter

## read twix,process data
def read_path(data_dir):
    if not data_dir.exists():
        raise FileNotFoundError(f"File not found: {data_dir}")
    twix=twixtools.read_twix(str(data_dir))
    return twix
### For cartesian one need stitch the nav signal
def extract_nav_tr_cv(twix):   ## Scan counter=2 is beginning timestamp
    "Extract navigation signal and TR from twix data, for CV (retrospective) cine"
    
    print('\nnumber of separate scans (multi-raid):', len(twix))
    mdb_list = twix[-1]['mdb']
    print("MDB total number is: ", len(mdb_list))
    
    nav_label = 'PHASCOR'
    image_label = 'ONLINE'
    beginning_label='FIRSTSCANINSLICE'
    seg_counts = Counter(i.cSeg for i in twix[-1]['mdb'] if image_label in i.get_active_flags())

    print("Segment counts:", seg_counts)
    nav_data_list = []
    
    # This list will store the segment ID for every single navigator point we collect
    assigned_segment_ids = []
    
    # State variable to remember the last seen image segment
    current_image_seg_id = 0 
    beginning_timestamp = None

    print("Scanning MDB headers sequentially...")
    
    # --- 1. Sequential Scan (Single Pass) ---
    for i in tqdm(range(len(mdb_list))):
        active_flags = mdb_list[i].get_active_flags()
        
        # A. If Image Packet -> Update current Segment ID
        if image_label in active_flags:
            current_image_seg_id = mdb_list[i].cSeg
            assigned_segment_ids.append(current_image_seg_id)
            
            # Capture timestamp from the very first Image packet
            if beginning_label in active_flags:
                beginning_timestamp = mdb_list[i].mdh.TimeStamp

        # B. If Nav Packet -> Collect Data AND Assign Current ID
        elif nav_label in active_flags:
            nav_data_list.append(mdb_list[i].data)
            
    print("Time Stamp starts at: ", beginning_timestamp)
    
    assigned_segment_ids=assigned_segment_ids[1:] # Drop the very first ONLINE image packet because it is noise scan
    assigned_segment_ids[-1]=assigned_segment_ids[-2]   ## last segment belongs to the previous one

    # --- 2. Convert to Numpy ---
    nav_signal = np.stack(nav_data_list, axis=1)
    nav_signal = np.abs(nav_signal)
    print("before mean shape ", nav_signal.shape  )
    nav_signal = np.mean(nav_signal[:,:, 3:7], axis=-1)
    print("after mean shape ", nav_signal.shape )

    hdr = twix[-1]['hdr']['Phoenix']
    acq_segments = hdr['sFastImaging']['lSegments']  ## num of PE lines per segment
    TR = (hdr['alTR'][0]/1e6) / acq_segments
    num_navs = nav_signal.shape[1]
    
    # --- 3. Detect Stitching Boundaries ---
    # Convert list to array for fast diff
    assigned_segment_ids = np.array(assigned_segment_ids)
    # Find where the assigned segment ID changes
    boundary_indices = np.where(np.diff(assigned_segment_ids) != 0)[0]+1  # -5 works!

    print(f"Detected {len(boundary_indices)} segment boundaries")
    print("Number of PE lines per segment:", acq_segments )
    print("Boundary indices:", boundary_indices)
    print(f"Segment Repetition Time (TR): {hdr['alTR'][0]/1e6} s")
    # --- 5. Apply Stitching Logic ---   Notice the segment is not homogeneous, 
    print("--- Applying Stitching (De-Stepping) ---")
    
    stitched_signal = nav_signal.copy()
    window_size = acq_segments
    # Add the end of the file as a "virtual boundary" to define the last segment's range
    all_boundaries = np.append(boundary_indices, num_navs)
    # Iterate through boundaries
    for i in range(len(boundary_indices)):
        idx = boundary_indices[i]      # Start of the current jump
        next_idx = all_boundaries[i+1] # Start of the NEXT jump (or end of file)
        # Define Calculation Windows
        start_prev = max(0, idx- window_size)
        end_next = min(num_navs, idx + window_size)

    
        prev_chunk = stitched_signal[:, start_prev:idx]
        next_chunk = stitched_signal[:, idx:end_next]
        

        if prev_chunk.shape[1] > 1 and next_chunk.shape[1] > 1:
            val_prev = np.mean(prev_chunk, axis=1)
            val_next = np.mean(next_chunk, axis=1)            
            jump_height = val_next - val_prev
            stitched_signal[:, idx:] -= jump_height[:, np.newaxis]
# ##------------------------------------Debug code
#         print("current loop ", i)
#         print("boundary indice :", idx)
#         print("previous segment border start ",start_prev)
#         print("end border of the next segment ",end_next)

#         print("previous segment chunk ", prev_chunk[0])
#         print("next segment chunk ", next_chunk[0])
#         print(f"--- Debug Loop {i} ---")
#         print(f"Boundary Index: {idx}")
#         print(f"Window: {start_prev} -> {idx} -> {end_next}")
        
#         # --- DEBUG PLOTTING ---
#         plt.figure(figsize=(10, 5))
        
#         # 1. Create X-axis arrays so points appear in sequential time order
#         x_prev = np.arange(start_prev, idx)
#         x_next = np.arange(idx, end_next)
        
#         # 2. Plot the Signal Data (Dots)
#         # Prev Chunk in Blue, Next Chunk in Red
#         plt.plot(x_prev, prev_chunk[0], 'bo', markersize=3, alpha=0.6, label='Previous Chunk')
#         plt.plot(x_next, next_chunk[0], 'ro', markersize=3, alpha=0.6, label='Next Chunk')
        
#         # 3. Plot the Calculated Means (Horizontal Lines)
#         # hlines args: (y, xmin, xmax)
#         # We assume val_prev/val_next are arrays (channels), so we take [0]
#         v_p = val_prev[0] if isinstance(val_prev, (np.ndarray, list)) else val_prev
#         v_n = val_next[0] if isinstance(val_next, (np.ndarray, list)) else val_next
        
#         plt.hlines(v_p, xmin=start_prev, xmax=idx, colors='blue', linewidth=2, linestyle='-', label=f'Prev Mean ({v_p:.2e})')
#         plt.hlines(v_n, xmin=idx, xmax=end_next, colors='red', linewidth=2, linestyle='-', label=f'Next Mean ({v_n:.2e})')
        
#         # 4. Add Stitch Boundary Line
#         plt.axvline(x=idx, color='black', linestyle='--', alpha=0.5, label='Boundary')
        
#         plt.title(f"Stitching Boundary {i} at Index {idx}")
#         plt.xlabel("Sample Index")
#         plt.ylabel("Amplitude")
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.show() 
#         # Note: In a loop, you might want to break after the first few to avoid 16 popups
#         if i >= 4: break
##------------------------------------End of debug code
    nav_signal = stitched_signal

    return nav_signal, TR, beginning_timestamp

def extract_nav_tr(twix, if_stitch=False):   ### currently for gre sequence
    "Extract navigation signal and TR from twix data, for gre and trufi sequence, Siemens prospective k trajectory, average to control cPhy"
    nav_signal=[]
    nav_label=  'PHASCOR'
    print('\nnumber of separate scans (multi-raid):', len(twix))
    print("MDB total number is: ",len(twix[0]['mdb']))
    ## for cartesian data, you can use the following code to extract the navigation signal
    beginning_timestamp  = 0
    first_nav_found = False

    for i in tqdm(range(len(twix[-1]['mdb']))):
        if nav_label in twix[-1]['mdb'][i].get_active_flags():
            nav_signal.append(twix[-1]['mdb'][i].data)
            if not first_nav_found:
                beginning_timestamp= twix[-1]['mdb'][i].mdh.TimeStamp
                first_nav_found = True
    print("Time Stamp starts at: ",beginning_timestamp )
    nav_signal=np.stack(nav_signal,axis=1)
    nav_signal=np.abs(nav_signal)
    nav_signal=np.mean(nav_signal[:,:,3:7],axis=-1)
    acq_segments = twix[-1]['hdr']['Phoenix']['sFastImaging']['lSegments']
    print(f"Acquisition Segments: {acq_segments}")   
    partitions = twix[-1]['hdr']['Phoenix']['sKSpace']['lPartitions']
    print(f"Partitions: {partitions}")
    average= twix[-1]['hdr']['Phoenix']['lAverages']
    print(f"Average: {average}")
    TR= (twix[-1]['hdr']['Phoenix']['alTR'][0]/1e6)/acq_segments
    print("Segment Repetition Time (TR): ",twix[-1]['hdr']['Phoenix']['alTR'][0]/1e6," s")
    num_navs=nav_signal.shape[1]
    print(f"Total number of navigators: {num_navs}")
    ### get each segments PE lines number:  note ! For multiple slice and multi contrast please insert properly
    try:
        lRepetitions= twix[-1]['hdr']['Phoenix']['lRepetitions']
    except:
        lRepetitions= 0

    lContrasts= twix[-1]['hdr']['Phoenix']['lContrasts']
    print(f"Total number of repetitions: {lRepetitions+1}, contrasts: {lContrasts}")
# ---------------------------------------------------------
    # STITCHING LOGIC (With Repetition/Contrast Handling)
    # ---------------------------------------------------------   
    if if_stitch:
        print("--- Applying Stitching (De-Stepping) ---")
        pe_num_per_seg = acq_segments * average
        boundary_indices = np.arange(0, num_navs, pe_num_per_seg, dtype=int)[1:]
        print("boundary_indices is", boundary_indices)
        boundary_indices = boundary_indices[boundary_indices < num_navs]
        
        stitched_signal = nav_signal.copy()
        window_size = int(max(1, acq_segments))
        all_boundaries = np.append(boundary_indices, num_navs)
        
        for i in range(len(boundary_indices)):
            idx = boundary_indices[i]
            next_idx = all_boundaries[i+1]
            start_prev = max(0, idx - window_size)
            end_next = min(num_navs, idx + window_size)   
            
            prev_chunk = stitched_signal[:, start_prev:idx]
            next_chunk = stitched_signal[:, idx:end_next]     
            
            if prev_chunk.shape[1] > 1 and next_chunk.shape[1] > 1:
                val_prev = np.mean(prev_chunk, axis=1)
                val_next = np.mean(next_chunk, axis=1)    
                jump_height = val_next - val_prev     
                # Subtract jump from EVERYTHING after this point
                stitched_signal[:, idx:] -= jump_height[:, np.newaxis]      
        nav_signal = stitched_signal

    return nav_signal, TR, beginning_timestamp


def get_noise(twix):
    Noise_label='NOISEADJSCAN'
    for i in twix[0]['mdb']:
        if Noise_label in i.get_active_flags():
            noise=i.data

    print(noise.shape)
    return noise

def noise_to_bart_shape(noise):
    D0, D1 = noise.shape

    noise_bart = noise.transpose(1, 0).reshape(D1, 1, 1, D0, 1, 1, 1, 1, 1, 1, 1, 1)
    return noise_bart
def process_nav_signal(nav_signal, fs, free_breath=False, plot=False):
    """
    Applies IIR Zero-Phase filtering with Symmetric Reflective Padding.
    
    Guarantees:
    1. Output shape matches Input shape exactly.
    2. Zero Phase Distortion.
    3. Baseline (DC) preservation.
    4. Strong 0.7Hz aliasing rejection in strict mode.
    """
    # 1. Convert Input to Numpy
    if torch.is_tensor(nav_signal):
        x = nav_signal.detach().cpu().numpy()
    else:
        x = nav_signal.copy()

    # Ensure 2D shape (Channels, N)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    n_channels, n_samples = x.shape
    nyquist = 0.5 * fs
    
    # --- STEP A: Separate DC Offset ---
    # Calculate median to preserve baseline
    dc_offset = np.median(x, axis=1, keepdims=True)
    x_ac = x - dc_offset
    
    filtered_ac = np.zeros_like(x_ac)
    label_str = ""

    if free_breath:
        # --- Condition 1: Free Breath (Simple LPF) ---
        cut_off = 5.0 
        sos_lp = signal.butter(4, cut_off / nyquist, btype='low', output='sos')
        
        # sosfiltfilt maintains length automatically
        filtered_ac = signal.sosfiltfilt(sos_lp, x_ac, axis=-1)
        label_str = "Free Breath (LPF 5Hz)"

    else:
        # --- Condition 2: Strict Mode (Band Pass) ---
        
        # 1. Heavy Symmetric Reflective Padding (Half Length)
        # This extends the trend linearly to let the filter settle.
        pad_samples = n_samples // 2 

        # Apply Padding
        # mode='reflect' handles the mono-increasing/decreasing trend perfectly
        if pad_samples > 0:
            x_padded = np.pad(x_ac, ((0, 0), (pad_samples, pad_samples)), mode='reflect')
        else:
            x_padded = x_ac

        # 2. Design Filters
        # Strong HP at 0.7Hz (Order 2 -> Effective Order 4) to crush aliasing <0.1Hz
        sos_hp = signal.butter(2, 0.65 / nyquist, btype='high', output='sos')
        # Standard LP at 5Hz (Order 2 -> Effective Order 4)
        sos_lp = signal.butter(2, 5.0 / nyquist, btype='low', output='sos')

        # 3. Cascade Filtering (Zero-Phase)
        temp = signal.sosfiltfilt(sos_hp, x_padded, axis=-1)
        temp = signal.sosfiltfilt(sos_lp, temp, axis=-1)

        # 4. Remove Padding (Restore original length)
        # We slice off exactly what we added.
        if pad_samples > 0:
            filtered_ac = temp[:, pad_samples:-pad_samples]
        else:
            filtered_ac = temp
            
        label_str = " (BP 0.7-5Hz)"

    # --- STEP B: Restore DC Offset ---
    filtered_x = filtered_ac + dc_offset

    # --- STEP C: Validation (Length Check) ---
    if filtered_x.shape != x.shape:
        raise ValueError(f"Shape Mismatch! Input: {x.shape}, Output: {filtered_x.shape}")

    # --- Plotting ---
    if plot:
        t = np.arange(n_samples) / fs
        fig, axes = plt.subplots(n_channels, 1, figsize=(10, 3 * n_channels), sharex=True)
        if n_channels == 1: axes = [axes]
        
        for i in range(n_channels):
            ax = axes[i]
            # Plot Original
            ax.plot(t, x[i], label=f'Ch {i} Original', alpha=0.5, color='gray')
            # Plot Filtered
            ax.plot(t, filtered_x[i], label=f'Ch {i} Filtered', color='red', linewidth=1.5)
            
            ax.set_ylabel("Amplitude", fontsize=14)
            ax.set_xlabel("Time (s)", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12, labelbottom=True)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_title(f"IIR Zero-Phase Result: {label_str}", fontsize=16)

        plt.tight_layout()
        plt.show()
    
    return torch.tensor(filtered_x, dtype=torch.float32)
##--------------------------------------------------------------------------------------



def visualize_k_space_cardiac_phase(k_space):
    """
    Visualizes the k-space sampling pattern for each cardiac phase in a subplot grid.

    This function plots the sampling pattern of the last coil and calculates
    the number and percentage of k-space lines filled along the Phase Encode Y
    direction for each cardiac phase, displaying this info in the title.

    Args:
        k_space (np.ndarray): The k-space data, expected to have a shape like
                              (coils, readout, phase_y, phase_z, cardiac_phase, echo).
    """
    # --- 1. Get dimensions from the k-space data ---
    # The shape order is assumed to be: (coils, readout, phase_y, phase_z, cardiac_phase, echo)
    if k_space.ndim != 6:
        raise ValueError(f"Expected 6 dimensions for k_space, but got {k_space.ndim}")
        
    num_coils = k_space.shape[0]
    total_phase_encode_lines = k_space.shape[2]
    num_cardiac_phases = k_space.shape[4]

    # --- 2. Set up the subplot grid ---
    # We want a grid with 5 columns. The number of rows is calculated based on
    # how many cardiac phases there are.
    cols = 5
    rows = math.ceil(num_cardiac_phases / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.2))
    
    # Flatten the axes array for easier iteration, especially if there's only one row.
    if num_cardiac_phases > 1:
        axes = axes.flatten()
    else:
        axes = [axes] # Make it iterable for a single plot

    fig.suptitle(f'K-Space Filling Pattern (Last Coil - Coil {num_coils})', fontsize=18, y=1.0)

    # --- 3. Loop through each cardiac phase to process and plot ---
    for i in range(num_cardiac_phases):
        ax = axes[i]
        
        # Isolate the k-space data for the last coil and current cardiac phase.
        # The '!= 0' creates a boolean array (True/False) representing the sampling pattern.
        sampling_pattern = k_space[-1, :, :, 0, i, 0] != 0

        # --- 4. Calculate how many lines are filled ---
        # To count lines along 'Phase Encode Y', we check each column of our 2D pattern.
        # np.any(sampling_pattern, axis=0) creates a 1D array; True for columns with any data.
        filled_lines_mask = np.any(sampling_pattern, axis=0)
        num_filled_lines = np.sum(filled_lines_mask)
        
        # Calculate the fill percentage
        fill_percentage = (num_filled_lines / total_phase_encode_lines) * 100 if total_phase_encode_lines > 0 else 0

        # --- 5. Plot the sampling pattern ---
        ax.imshow(sampling_pattern, cmap='gray', aspect='auto', interpolation='none')
        
        # --- 6. Set a descriptive title for the subplot ---
        title_text = (
            f'Cardiac Phase {i+1}\n'
            f'{num_filled_lines}/{total_phase_encode_lines} lines filled ({fill_percentage:.1f}%)'
        )
        ax.set_title(title_text, fontsize=14)
        ax.set_xlabel('Phase Encode Y', fontsize=12)
        ax.set_ylabel('Readout', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=11)

    # --- 7. Clean up the figure ---
    # Hide any subplot axes that are not being used.
    for j in range(num_cardiac_phases, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make room for suptitle
    plt.show()

##--------------------------------------------------------------------------------------
# --- K-Space Filling ---   binning logic suitable for non-cartesian 
def extrapolate_map(gating_array, num_total_scans, num_phases=1, return_continuous=False):
    """
    Calculates phase indices (int) or normalized cardiac time (float).
    Based on the logic that data is strictly bounded between triggers [0, 1).
    """
    # Initialize with safety values
    if return_continuous:
        phase_map = np.full(num_total_scans, -100.0, dtype=np.float64)
    else:
        phase_map = np.full(num_total_scans, -1, dtype=np.int16)

    if gating_array is None or len(gating_array) < 2:
        return phase_map

    rr_intervals = np.diff(gating_array)
    valid_intervals = rr_intervals[rr_intervals > 0]
    
    # 1. Fill Known Intervals (Strictly 0.0 to 1.0)
    for i in range(len(rr_intervals)):
        start, end = gating_array[i], gating_array[i+1]
        duration = rr_intervals[i]
        if duration == 0: continue
        
        # Indices strictly between triggers
        acq_indices = np.arange(start, end)
        valid_mask = acq_indices < num_total_scans
        valid_indices = acq_indices[valid_mask]
        
        # Relative Time: Always [0.0, 1.0)
        rel_pos = (valid_indices - start) / duration
        
        if return_continuous:
            phase_map[valid_indices] = rel_pos
        else:
            # Binning: Simple multiplication
            phases = (rel_pos * num_phases).astype(np.int16)
            phase_map[valid_indices] = np.clip(phases, 0, num_phases - 1)

    # 2. Extrapolate (For data before first trigger or after last)
    # This handles the 'tails' of the acquisition.
    if len(valid_intervals) == 0: return phase_map
    avg_interval = np.mean(valid_intervals)
    
    # Find unassigned indices
    if return_continuous:
        unassigned = np.where(phase_map == -100.0)[0]
    else:
        unassigned = np.where(phase_map == -1)[0]
        
    if len(unassigned) == 0: return phase_map

    first_trig, last_trig = gating_array[0], gating_array[-1]

    for idx in unassigned:
        cycle_pos = -100.0
        
        # Backwards from first trigger (Phase 0.0)
        if idx < first_trig: 
            dist = first_trig - idx
            # e.g. -0.1
            cycle_pos = 1.0 - ((dist % avg_interval) / avg_interval)
            
        # Forwards from last trigger
        elif idx > last_trig:
            dist = idx - last_trig
            cycle_pos = (dist % avg_interval) / avg_interval
            
        # Gaps inside
        else:
            prev = np.searchsorted(gating_array, idx) - 1
            if prev >= 0:
                dist = idx - gating_array[prev]
                cycle_pos = (dist % avg_interval) / avg_interval
        
        # Assign
        if cycle_pos != -100.0:
            if return_continuous:
                phase_map[idx] = cycle_pos
            else:
                # Wrap cyclicly for binning safe-guard
                pos_wrapped = cycle_pos % 1.0
                p = int(pos_wrapped * num_phases)
                phase_map[idx] = np.clip(p, 0, num_phases - 1)

    return phase_map
## 3D function not tested yet!!!!
##k_space (coil,readout x,phase encode y, phase encode z, cardiac phase,respiration phase)
def fill_kspace_bin(twix, gating_array, num_cardiac_phases, breath_gating_array=None, num_breath_phases=1, discard_num=0, dimensions=2, stop_dimension=None):
    """
    Standard Binning Version (Discrete phases 0..N-1).
    Used for Compressed Sensing (BART).
    Now includes Asymmetric Echo & Oversampling support.
    """
    print(f"\n--- Starting K-Space Filling (Binning) [Discard: {discard_num}] ---")
    
    # --- 1. Validation ---
    img_label='ONLINE'
    image_scans = [mdb for mdb in twix[0]['mdb'] if img_label in mdb.get_active_flags()]
    image_scans=image_scans[1:]  #(the first ONLINE is noise scan)
    if stop_dimension is not None:
        image_scans = image_scans[:stop_dimension]
    if not image_scans: raise ValueError("No image scans found.")
    num_image_scans = len(image_scans)
    
    if gating_array is None or len(gating_array) < 2:
        raise ValueError("`gating_array` required.")

    # --- 2. Maps (Integer) ---
    cardiac_phase_map = extrapolate_map(
        gating_array, num_image_scans, num_phases=num_cardiac_phases, return_continuous=False
    )
    
    breath_phase_map = np.zeros(num_image_scans, dtype=np.int16)
    if num_breath_phases > 1:
        if breath_gating_array is None: raise ValueError("`breath_gating_array` required.")
        breath_phase_map = extrapolate_map(
            breath_gating_array, num_image_scans, num_phases=num_breath_phases, return_continuous=False
        )

    # --- 3. Setup (Oversampling & Asymmetric Echo) ---
    first_mdh = image_scans[0].mdh
    header = twix[-1]['hdr']
    n_c, n_ro = first_mdh.UsedChannels, first_mdh.SamplesInScan
    n_pe_y = header['Phoenix']['sKSpace']['lPhaseEncodingLines']

    n_base_ro = header['Phoenix']['sKSpace']['lBaseResolution']
    
    # Detect if oversampling is used (Siemens is almost always 2x)
    oversample_factor = 2 if n_ro > n_base_ro else 1
    full_ro = n_base_ro * oversample_factor
    
    # Calculate the correct offset based on the oversampled grid
    ro_offset = full_ro - n_ro 

    if dimensions == 2:
        n_pe_z = 1
    else:
        n_pe_z = header['Phoenix']['sKSpace'].get('lPartitions', 1)

    # Shape uses full_ro (576)
    k_space_shape = (n_c, full_ro, n_pe_y, n_pe_z, num_cardiac_phases, num_breath_phases)
    
    print(f"Initializing k-space: {k_space_shape}")
    print(f"Params: Acquired RO={n_ro}, Base RO={n_base_ro}, Full Oversampled RO={full_ro}, Offset={ro_offset}")

    # ADDED BACK: Initialization arrays were missing in your snippet!
    k_space_sum = np.zeros(k_space_shape, dtype=np.complex128)
    counter_shape = (n_pe_y, n_pe_z, num_cardiac_phases, num_breath_phases)
    k_space_counter = np.zeros(counter_shape, dtype=np.uint32)
    asymmetric_ratio= n_ro / full_ro
    # --- 4. Fill Loop ---
    for idx, mdb in enumerate(image_scans):
        if idx < discard_num: continue

        c = cardiac_phase_map[idx]
        b = breath_phase_map[idx]

        if c == -1 or b == -1: continue

        lin_y = mdb.cLin
        lin_z = mdb.cPar if hasattr(mdb, 'cPar') else 0
        if lin_y >= n_pe_y or lin_z >= n_pe_z: continue

        # ASYMMETRIC ECHO FIX: Insert data into [ro_offset : full_ro] (using full_ro, not n_base_ro)
        k_space_sum[:, ro_offset:full_ro, lin_y, lin_z, c, b] += mdb.data
        k_space_counter[lin_y, lin_z, c, b] += 1

    # --- 5. Average ---
    counter_broad = k_space_counter[np.newaxis, np.newaxis, :, :, :, :]
    k_space_avg = np.zeros_like(k_space_sum)
    np.divide(k_space_sum, counter_broad, out=k_space_avg, where=counter_broad!=0)

    print(f"Final k-space shape: {k_space_avg.shape}")
    return k_space_avg.astype(np.complex64)
### k space interpolation : used for cartesian grappa/sense style,along breath dimension is binning, along cardiac dimension is interpolation
##k_space (coil,readout x,phase encode y, phase encode z, cardiac phase,respiration phase)
##---------------------------------------------------------------------------------------

def fill_kspace_interpolate(twix, gating_array, num_cardiac_phases, breath_gating_array=None, num_breath_phases=1, discard_num=0, dimensions=2, stop_dimension=None):
    """
    Final Hybrid Logic (with Asymmetric Echo & Oversampling Support):
    - Cardiac: Cyclic Linear Interpolation (Smooth loop 0->1).
    - Respiratory: Binning with Extrapolation.
    - Groups data by dictionary for O(N) efficiency.
    - Correctly maps reduced readout data into the full oversampled Base Resolution.
    """
    print(f"\n--- Final Hybrid K-Space Interpolation [Discard: {discard_num}] ---")
    
    # --- 1. Validation ---
    if gating_array is None or len(gating_array) < 2:
        raise ValueError("`gating_array` required.")
        
    img_label = 'ONLINE'
    image_scans = [mdb for mdb in twix[0]['mdb'] if img_label in mdb.get_active_flags()]
    image_scans = image_scans[1:] #(the first ONLINE is noise scan)
    if stop_dimension is not None:
        image_scans = image_scans[:stop_dimension]
    if not image_scans: raise ValueError("No image scans found.")
    num_image_scans = len(image_scans)

    # --- 2. Generate Maps ---
    
    # A. Breath (Integer Bins)
    breath_phase_map = np.zeros(num_image_scans, dtype=np.int16)
    if num_breath_phases > 1:
        if breath_gating_array is None: raise ValueError("`breath_gating_array` required.")
        breath_phase_map = extrapolate_map(
            breath_gating_array, num_image_scans, num_phases=num_breath_phases, return_continuous=False
        )

    # B. Cardiac (Continuous Float [0.0, 1.0))
    print("Generating Cardiac Map...")
    scan_cardiac_times = extrapolate_map(
        gating_array, num_image_scans, num_phases=1, return_continuous=True
    )

    # --- 3. Setup (Oversampling & Asymmetric Echo) ---
    first_mdh = image_scans[0].mdh
    header = twix[-1]['hdr']
    n_c, n_ro = first_mdh.UsedChannels, first_mdh.SamplesInScan
    n_pe_y = header['Phoenix']['sKSpace']['lPhaseEncodingLines']
    
    # Calculate correct full readout size including oversampling
    n_base_ro = header['Phoenix']['sKSpace']['lBaseResolution']
    oversample_factor = 2 if n_ro > n_base_ro else 1
    full_ro = n_base_ro * oversample_factor
    
    # Offset handles the missing early echo lines
    ro_offset = full_ro - n_ro 
    
    if dimensions == 2:
        n_pe_z = 1
    else:
        n_pe_z = header['Phoenix']['sKSpace'].get('lPartitions', 1)

    # Initialize with full oversampled shape
    k_space_shape = (n_c, full_ro, n_pe_y, n_pe_z, num_cardiac_phases, num_breath_phases)
    print(f"Initializing k-space: {k_space_shape}")
    print(f"Params: Acquired RO={n_ro}, Base RO={n_base_ro}, Full Oversampled RO={full_ro}, Offset={ro_offset}")

    k_space_sum = np.zeros(k_space_shape, dtype=np.complex128)
    k_space_weights = np.zeros((n_pe_y, n_pe_z, num_cardiac_phases, num_breath_phases), dtype=np.float32)

    asymmetric_ratio= n_ro / full_ro

    # --- 4. Grouping Data ---
    print("Grouping data for cyclic interpolation...")
    data_groups = {} 
    
    for idx, mdb in enumerate(image_scans):
        if idx < discard_num: continue

        c_time = scan_cardiac_times[idx]
        b_phase = breath_phase_map[idx]
        
        if c_time == -100.0 or b_phase == -1: continue
            
        lin_y = mdb.cLin
        lin_z = mdb.cPar if hasattr(mdb, 'cPar') else 0
        
        if lin_y >= n_pe_y or lin_z >= n_pe_z: continue
            
        key = (b_phase, lin_y, lin_z)
        if key not in data_groups: data_groups[key] = []
        data_groups[key].append(idx)

    # --- 5. Cyclic Interpolation Loop ---
    target_times = np.linspace(0, 1, num_cardiac_phases, endpoint=False)

    for key, raw_indices in data_groups.items():
        b_phase, lin_y, lin_z = key
        
        raw_indices = np.array(raw_indices)
        raw_times = scan_cardiac_times[raw_indices]
        
        # Virtual Prev (-1.0) and Virtual Next (+1.0) for cyclic mapping
        t_prev = raw_times - 1.0
        t_next = raw_times + 1.0
        
        all_times = np.concatenate([t_prev, raw_times, t_next])
        all_indices = np.concatenate([raw_indices, raw_indices, raw_indices])
        
        sort_order = np.argsort(all_times)
        sorted_times = all_times[sort_order]
        sorted_indices = all_indices[sort_order]
        
        for c_phase in range(num_cardiac_phases):
            target_t = target_times[c_phase]
            idx_right = np.searchsorted(sorted_times, target_t)
            
            if idx_right == 0 or idx_right == len(sorted_times):
                continue

            idx_A = idx_right - 1
            idx_B = idx_right
            
            t_A = sorted_times[idx_A]
            t_B = sorted_times[idx_B]
            
            dist = t_B - t_A
            if dist > 1e-6:
                w_B = (target_t - t_A) / dist
                w_A = 1.0 - w_B
            else:
                w_A, w_B = 1.0, 0.0
                
            # Insert data into [ro_offset : full_ro]
            scan_A = sorted_indices[idx_A]
            data_A = image_scans[scan_A].data
            k_space_sum[:, ro_offset:full_ro, lin_y, lin_z, c_phase, b_phase] += data_A * w_A
            
            if w_B > 0:
                scan_B = sorted_indices[idx_B]
                data_B = image_scans[scan_B].data
                k_space_sum[:, ro_offset:full_ro, lin_y, lin_z, c_phase, b_phase] += data_B * w_B
                
            k_space_weights[lin_y, lin_z, c_phase, b_phase] += 1.0

    # --- 6. Final Averaging ---
    weights_broad = k_space_weights[np.newaxis, np.newaxis, :, :, :, :]
    k_space_final = np.zeros_like(k_space_sum)
    np.divide(k_space_sum, weights_broad, out=k_space_final, where=weights_broad > 0)
    
    print(f"Final K-Space Shape: {k_space_final.shape}")
    return k_space_final.astype(np.complex64)
## soft gating: along cardiac dimension is binning, along respiration dimension uses a weighted function for the reconstruction
## here I use exponential weight function 
## k_space (coil,readout x,phase encode y, phase encode z, cardiac phase,rep number)  , weight (1,PEy,PEz,cardiac phase,rep number)
def weight_func(current_pos, target_pos=0.0, alpha=-1.0, threshold=0.1):
    """
    Exponential weight function from Equation [4]:
    w[n] = exp(alpha * (dist - threshold)) if dist > threshold, else 1
    """
    dist = np.abs(current_pos - target_pos)
    dist = np.minimum(dist, 1.0 - dist) # Cyclic distance
    
    if dist > threshold:
        return np.exp(alpha * (dist - threshold))
    return 1.0
def fill_kspace_softbin(twix, gating_array, num_cardiac_phases, 
                        breath_gating_array=None, discard_num=0, 
                        dimensions=2, stop_dimension=None, alpha=-1.0, threshold=0.1):
    """
    Corrected Logic:
    - Ignores header-based repetition counts.
    - Dynamically collects all lines matching (lin_y, lin_z, c_bin).
    - Stacks them into the last dimension (Time/Repetition).
    - Now includes Asymmetric Echo & Oversampling support.
    """
    print(f"\n--- Starting Dynamic K-Space Collection ---")
    
    # --- 1. Setup & Filtering ---
    header = twix[0]['hdr']
    img_label = 'ONLINE'
    
    # BUG FIX: Define image_scans first, then slice it
    image_scans = [mdb for mdb in twix[0]['mdb'] if img_label in mdb.get_active_flags()]
    image_scans = image_scans[1:] #(the first ONLINE is noise scan)
    
    if stop_dimension is not None:
        image_scans = image_scans[:stop_dimension]
    
    num_image_scans = len(image_scans)
    if num_image_scans == 0:
        raise ValueError("No image scans found.")
    
    first_mdh = image_scans[0].mdh
    n_c, n_ro = first_mdh.UsedChannels, first_mdh.SamplesInScan
    n_pe_y = header['Phoenix']['sKSpace']['lPhaseEncodingLines']
    n_pe_z = 1 if dimensions == 2 else header['Phoenix']['sKSpace'].get('lPartitions', 1)

    # ASYMMETRIC ECHO + OVERSAMPLING FIX
    n_base_ro = header['Phoenix']['sKSpace']['lBaseResolution']
    oversample_factor = 2 if n_ro > n_base_ro else 1
    full_ro = n_base_ro * oversample_factor
    ro_offset = full_ro - n_ro 

    print(f"Params: Acquired RO={n_ro}, Base RO={n_base_ro}, Full Oversampled RO={full_ro}, Offset={ro_offset}")

    # --- 2. Generate Phase Maps ---
    cardiac_phase_map = extrapolate_map(
        gating_array, num_image_scans, num_phases=num_cardiac_phases, return_continuous=False
    )
    breath_continuous_map = extrapolate_map(
        breath_gating_array, num_image_scans, num_phases=1, return_continuous=True
    )

    # --- 3. First Pass: Count repetitions ---
    repetition_counts = {}
    valid_indices = []

    print("Pass 1: Analyzing data structure...")
    for idx, mdb in enumerate(image_scans):
        if idx < discard_num: continue

        c_bin = cardiac_phase_map[idx]
        r_val = breath_continuous_map[idx]
        
        if c_bin == -1 or r_val < 0: continue

        lin_y = mdb.cLin
        lin_z = mdb.cPar if hasattr(mdb, 'cPar') else 0
        
        if lin_y >= n_pe_y or lin_z >= n_pe_z: continue

        key = (lin_y, lin_z, c_bin)
        repetition_counts[key] = repetition_counts.get(key, 0) + 1
        
        valid_indices.append((idx, lin_y, lin_z, c_bin, r_val))

    if not repetition_counts:
        raise ValueError("No valid lines found after gating logic.")
        
    max_reps = max(repetition_counts.values())
    print(f"Max repetitions found for a single k-space location: {max_reps}")

    # --- 4. Allocation ---
    # ASYMMETRIC ECHO FIX: Use full_ro instead of n_ro
    k_space = np.zeros((n_c, full_ro, n_pe_y, n_pe_z, num_cardiac_phases, max_reps), dtype=np.complex64)
    weight_matrix = np.zeros((1, 1, n_pe_y, n_pe_z, num_cardiac_phases, max_reps), dtype=np.float32)
    asymmetric_ratio= n_ro / full_ro
    # --- 5. Second Pass: Fill arrays ---
    current_fill_counts = {}

    print("Pass 2: Filling arrays...")
    for (idx, lin_y, lin_z, c_bin, r_val) in valid_indices:
        mdb = image_scans[idx]
        key = (lin_y, lin_z, c_bin)
        
        slot_idx = current_fill_counts.get(key, 0)
        
        w = weight_func(r_val, alpha=alpha, threshold=threshold)

        # ASYMMETRIC ECHO FIX: Insert data into [ro_offset : full_ro]
        k_space[:, ro_offset:full_ro, lin_y, lin_z, c_bin, slot_idx] = mdb.data
        weight_matrix[0, 0, lin_y, lin_z, c_bin, slot_idx] = w
        
        current_fill_counts[key] = slot_idx + 1

    print(f"Final Output Shapes:\nK-Space: {k_space.shape}\nWeights: {weight_matrix.shape}")
    return k_space, weight_matrix


##--------------------------------------------------------------------------------------
# Reconstruction 
def _to_bart_shape(k_space):
    original_shape= k_space.shape
    expanded_shape = original_shape + (1,) * 6
    k_space_expanded = k_space.reshape(expanded_shape)
    target_order = (1, 2, 3, 0, 6, 7, 8, 9, 10, 11, 4, 5)
    k_space_bart = np.transpose(k_space_expanded, target_order)
    print(f"Original shape:    {k_space.shape}")
    print(f"Expanded shape:    {k_space_expanded.shape}")
    print(f"Final shape:       {k_space_bart.shape}")
    return k_space_bart
##------------------------------------------------------------------
## homodyne then eliminate oversampling
# img_homodyne=bart(1,"homodyne -C 0 {}".format(assymmetric_ratio), k_space_bart)
## one can also perform coil compression here for coil compression to k_space_cropped
# 1) Homodyne on raw inserted k-space
# cimg_hd = bart(
#     1,
#     f"homodyne -C 0 {asymmetric_ratio}",
#     k_space_bart
# )

# # 2) Crop oversampling + return to k-space
# k_space_cropped = homodyne_then_crop_to_kspace(
#     jnp.array(cimg_hd),
#     oversampling_factor=2,
#     axis=0
# )

@partial(jit, static_argnames=['oversampling_factor', 'axis'])
def crop_readout_image_domain(cimg, oversampling_factor=2, axis=0):
    """
    Crop readout oversampling in image domain.
    Input must be image-domain coil images (e.g. output of BART homodyne).
    """
    n_orig = cimg.shape[axis]
    n_new  = n_orig // oversampling_factor
    start  = (n_orig - n_new) // 2
    return jnp.take(cimg, jnp.arange(start, start + n_new), axis=axis)

@partial(jit, static_argnames=['axes'])
def fft_centered_image_to_kspace(cimg, axes=(0, 1)):
    """
    Centered n-dimensional FFT: image → k-space (BART-compatible).
    """
    # ifftshift correctly moves the center of the image to the origin (0,0) expected by FFT
    x = jnp.fft.ifftshift(cimg, axes=axes)
    # Use fftn to transform over both axis 0 and axis 1
    x = jnp.fft.fftn(x, axes=axes)
    # fftshift then correctly places the DC component back at the center of K-space
    x = jnp.fft.fftshift(x, axes=axes)
    return x

@partial(jit, static_argnames=['oversampling_factor', 'axis', 'fft_axes'])
def homodyne_then_crop_to_kspace(cimg_hd, oversampling_factor=2, axis=0, fft_axes=(0, 1)):
    """
    Input: image-domain coil images from BART homodyne
    Output: cropped k-space (readout oversampling removed)
    """
    cimg_crop = crop_readout_image_domain(
        cimg_hd,
        oversampling_factor=oversampling_factor,
        axis=axis
    )
    ksp_crop = fft_centered_image_to_kspace(cimg_crop, axes=fft_axes)
    return ksp_crop
##-------------------------------------------------------------------------------------
## no partial echo, eliminate oversampling with bart
def eliminate_oversampling(k_space_bart,oversampling_ratio=2):
    temp = bart(1,"fft -i 1 ", k_space_bart)
    temp=bart(1,"resize -c 0 {} ".format(k_space_bart.shape[0]//oversampling_ratio), temp)
    k_space_cropped=bart(1,"fft -u 1 ",temp)
    return k_space_cropped
# ## no partial echo, eliminate oversampling with jax
# @partial(jit, static_argnames=['oversampling_factor'])
# def process_single_coil(k_space_coil, oversampling_factor=2):
#     axis = 0
#     n_orig = k_space_coil.shape[axis]
#     n_new = n_orig // oversampling_factor
#     start = (n_orig - n_new) // 2

#     temp = jnp.fft.ifftshift(k_space_coil, axes=axis)
#     temp = jnp.fft.ifft(temp, axis=axis, norm="ortho")
#     temp = jnp.fft.fftshift(temp, axes=axis)

#     temp_cropped = jnp.take(temp, jnp.arange(start, start + n_new), axis=axis)

#     out = jnp.fft.ifftshift(temp_cropped, axes=axis)
#     out = jnp.fft.fft(out, axis=axis, norm="ortho")
#     out = jnp.fft.fftshift(out, axes=axis)
#     return out

# def eliminate_oversampling_gpu(k_space_bart, oversampling_factor=2):
#     # k_space_bart is BART order, coil on axis=3.
#     n_coils = k_space_bart.shape[3]
#     results = []

#     # Warm-up compile
#     _ = process_single_coil(k_space_bart[:, :, :, 0, ...], oversampling_factor=oversampling_factor).block_until_ready()

#     for c in range(n_coils):
#         coil = k_space_bart[:, :, :, c, ...]   # keep all other dims
#         processed = process_single_coil(coil, oversampling_factor=oversampling_factor)
#         results.append(np.array(processed.block_until_ready()))

#     # stack back coil
#     return np.stack(results, axis=3)
### ------------------- pytorch version of eliminate oversampling -------------------

def process_single_coil_pt(k_space_coil, oversampling_factor=2):
    axis = 0
    n_orig = k_space_coil.shape[axis]
    n_new = n_orig // oversampling_factor
    start = (n_orig - n_new) // 2

    # 1D Inverse FFT (PyTorch uses 'dim' instead of 'axes')
    temp = fft.ifftshift(k_space_coil, dim=axis)
    temp = fft.ifft(temp, dim=axis, norm="ortho")
    temp = fft.fftshift(temp, dim=axis)

    # PyTorch handles slicing much more elegantly than jnp.take.
    # This grabs the cropped region along axis=0 and keeps all other dims.
    temp_cropped = temp[start:start + n_new, ...]

    # 1D Forward FFT
    out = fft.ifftshift(temp_cropped, dim=axis)
    out = fft.fft(out, dim=axis, norm="ortho")
    out = fft.fftshift(out, dim=axis)
    
    return out

def eliminate_oversampling_gpu(k_space_bart, oversampling_factor=2, device="cuda"):
    """
    
    """
    n_coils = k_space_bart.shape[3]
    results = []

    # No "warm-up" compile needed in PyTorch! Eager execution runs instantly.

    for c in range(n_coils):
        # 1. Slice the NumPy array and push ONLY one coil to the GPU at a time.
        # Ensure it's complex to match k-space data types.
        coil_np = k_space_bart[:, :, :, c, ...]
        coil_pt = torch.tensor(coil_np, dtype=torch.complex64, device=device)

        # 2. Process on the GPU
        processed_pt = process_single_coil_pt(coil_pt, oversampling_factor)

        # 3. Pull the result immediately back to CPU System RAM
        results.append(processed_pt.cpu().numpy())

        # 4. Explicitly wipe the PyTorch tensors and clear the GPU cache 
        # so VRAM never exceeds the size of a single coil.
        del coil_pt
        del processed_pt
        torch.cuda.empty_cache()

    # Stack back into a standard CPU NumPy array for BART
    final_cpu_array = np.stack(results, axis=3)
    
    # One final garbage sweep to ensure the OS recognizes the freed memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return final_cpu_array

## this is for single breathing phase
def get_sensitivity(k_space_cropped,cal_size=28,kernel_size=6,threshold=5e-4, crop_value=0.2,num_maps=2,use_gpu=True):
    """
    Estimate coil sensitivity maps using ESPIRiT algorithm from BART.

    Args:
        k_space_bart (np.array): The k-space data in BART format.
        calibration_size (int, optional): Size of the calibration region. Defaults to 24.

    """
    ksp_t_avg=bart(1,"avg -w {}".format(1<<10), k_space_cropped)
    if use_gpu:
        sens_maps=bart(1,"ecalib -r {} -k {} -t {} -c {} -m {} -g ".format(cal_size,kernel_size,threshold,crop_value, num_maps), ksp_t_avg )
    else:
        sens_maps=bart(1,"ecalib -r {} -k {} -t {} -c {} -m {} ".format(cal_size,kernel_size,threshold,crop_value, num_maps), ksp_t_avg )
    return sens_maps

def visualize_sens_maps(sens_maps,ecalib_ind=0):
    for coil_i in range(8):
        sens_map_coil = sens_maps[:,:,0,coil_i,ecalib_ind].squeeze()
        plt.subplot(2,4,coil_i+1)
        plt.imshow(np.abs(sens_map_coil),cmap="gray")
        plt.axis('off')
        plt.title('coil {}'.format(coil_i+1))
    plt.show()

def espirit_recon(k_space_cropped,sens_maps,l2_reg=1e-3,l_TVtime=5e-4,l1_wav=0,use_gpu=True):
    if use_gpu:
        recon=bart(1,'pics -R Q:{} -R T:1024:0:{} -R W:7:0:{} -S -g '.format(l2_reg,l_TVtime,l1_wav), k_space_cropped, sens_maps)
    else:
        recon=bart(1,'pics -R Q:{} -R T:1024:0:{} -R W:7:0:{} -S '.format(l2_reg,l_TVtime,l1_wav), k_space_cropped, sens_maps)
    recon_rss = bart(1,'rss 16', recon)
    return recon_rss    
#--------------------------------------------------------------------------------------
# Visualization

def show_cine(cine, colors=False):
    """
    Displays a cine (movie) of image slices using an interactive slider.

    Args:
        cine (np.ndarray): A 3D numpy array (height, width, phases).
        colors (bool): If False, uses grayscale colormap. Otherwise, uses viridis.
    """
    # Define the function that will be called by the interactive widget
    def f(cardiac_phases_index):
        fig, ax = plt.subplots(figsize=(12, 12))
        
        
        # 1. Use the correct array 'cine'
        # 2. Use the correct argument name 'cardiac_phases_index'
        ax.imshow(cine[:, :, cardiac_phases_index], cmap="gray" if not colors else "viridis")
        
        ax.set_title(f"Cardiac Phase: {cardiac_phases_index}")
        ax.axis('off') # Hide axes for a cleaner look
        plt.show()

    # The keyword argument here 'cardiac_phases_index' MUST match the argument in f()
    # The slider will go from 0 to the last index of the 3rd dimension.
    interactive_plot = interactive(f, cardiac_phases_index=(0, cine.shape[-1] - 1))
    
    # Display the interactive widget
    display(interactive_plot)


def animate_3d_array(data, interval=100, cmap='gray', save_path=None, fps=10):
    """
    Animate a 3D NumPy array along the last axis (frames) inline in Jupyter.
    """
    if data.ndim != 3:
        raise ValueError("Input array must be 3D (H, W, N)")

    # Normalize data to [0, 1] for consistent display
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    fig, ax = plt.subplots(figsize=(10,10))
    img = ax.imshow(data[:, :, 0], cmap=cmap, animated=True)
    ax.axis('off')

    def update(frame):
        img.set_array(data[:, :, frame])
        return [img]

    ani = FuncAnimation(fig, update, frames=data.shape[2], interval=interval, blit=True)
       # Save if requested (accept both str and Path)
    if save_path:
        # Convert to string for writer APIs
        save_path_str = str(save_path)
        print(f"Saving animation to {save_path_str}...")

        # Determine extension robustly using pathlib
        from pathlib import Path as _Path
        ext = _Path(save_path_str).suffix.lower()

        if ext == '.avi':
            writer = FFMpegWriter(fps=fps, codec=None, bitrate=5000)
            ani.save(save_path_str, writer=writer)
        elif ext == '.mkv':
            writer = FFMpegWriter(fps=fps, codec='ffv1')
            ani.save(save_path_str, writer=writer)
        elif ext == '.mp4':
            writer = FFMpegWriter(
                fps=fps,
                codec='libx264',
                extra_args=['-pix_fmt', 'yuv420p']
            )
            ani.save(save_path_str, writer=writer)
        elif ext == '.gif':
            # Pillow handles GIF output
            ani.save(save_path_str, writer='pillow', fps=fps)
        else:
            raise ValueError("Unsupported file format. Use .avi, .mkv, .mp4, or .gif")
        print("Save complete.")

    plt.close(fig)
    return HTML(ani.to_jshtml())


class PMU_Processor:
    """
    A class to process and visualize physiological monitoring unit (PMU) data
    from a twixtools pmu object.
    """
    def __init__(self, pmu,beginning_timestamp):
        """
        Initializes the processor with a PMU object.

        Args:
            pmu: A pmu object from twixtools, e.g., tt.pmu.PMU(twix[-1]['mdb']).
        """
        # The channels we are interested in plotting
        self.available_channels = ['ECG1', 'ECG2', 'ECG3', 'ECG4', 'EXT1', 'EXT2', 'LEARN_PULS', 'LEARN_RESP', 'LEARN_EXT1']

        # A complete list of all possible channels for reference
        self.all_channels = ['LEARN_ECG1', 'LEARN_ECG2', 'LEARN_ECG3', 'LEARN_ECG4',
                             'LEARN_PULS', 'LEARN_RESP', 'LEARN_EXT1', 'LEARN_EXT2',
                             'ECG1', 'ECG2', 'ECG3', 'ECG4', 'PULS', 'RESP', 'EXT1', 'EXT2']

        # Store the data dictionaries from the pmu object
        self.signal = pmu.signal      # dict of pmu waveforms
        self.trigger = pmu.trigger    # dict of pmu triggers (boolean arrays)
        self.timestamp = pmu.timestamp  # dict of pmu timestamps
        self.beginning_timestamp=beginning_timestamp
        self.raster_time=2.5          # 2.5 ms, PMU label for Siemens is 2.5 ms

    def get_signal(self, channel_name):
        """Retrieves the signal waveform for a given channel."""
        if channel_name in self.signal:
            return self.signal[channel_name]
        else:
            raise ValueError(f"Channel {channel_name} not found in PMU signals.")

    def get_trigger(self, channel_name):
        """Retrieves the trigger data for a given channel."""
        if channel_name in self.trigger:
            return self.trigger[channel_name]
        else:
            raise ValueError(f"Channel {channel_name} not found in PMU triggers.")

    def get_timestamp(self, channel_name):
        """Retrieves the timestamps for a given channel."""
        if channel_name in self.timestamp:
            return self.timestamp[channel_name]
        else:
            raise ValueError(f"Channel {channel_name} not found in PMU timestamps.")

    def get_TTL_first_trigger(self):
        """
        Calculates the start time of each TTL pulse from the 'EXT1' signal.

        This method identifies each consecutive group of non-zero values in the
        'EXT1' signal, which corresponds to a TTL pulse. It then finds the
        timestamp of the *first* non-zero value in each group, converts it to
        milliseconds relative to the recording start, and returns these times.

        Returns:
            np.array: A NumPy array containing the start times of each TTL
                      pulse in milliseconds. Returns an empty array if the 'EXT1'
                      channel is not found or has no signal.
        """
        channel_name = 'EXT1'
        try:
            # Step 1: Get the signal data for the EXT1 channel
            signal_data = self.get_signal(channel_name)

            # Step 2: Find the indices of all non-zero elements
            non_zero_indices = np.where(signal_data != 0)[0]

            # If there are no non-zero elements, there are no triggers to process
            if non_zero_indices.size == 0:
                print(f"No non-zero signal found in '{channel_name}'.")
                return np.array([])

            # Step 3: Identify the first index of each consecutive group
            # A new group starts where the difference between consecutive indices is > 1.
            # The very first non-zero index is always the start of the first group.
            diffs = np.diff(non_zero_indices)
            
            # Create a boolean mask: True for the first element and for any element that follows a gap
            group_starts_mask = np.concatenate(([True], diffs > 1))
            
            # Apply the mask to get the index of the first trigger in each pulse group
            first_trigger_indices = non_zero_indices[group_starts_mask]

            # Step 4: Convert these indices to relative timestamps in milliseconds
            timestamp_data = self.get_timestamp(channel_name)
            start_time = self.beginning_timestamp
            
            # Select the timestamps corresponding to the identified trigger indices
            trigger_timestamps = timestamp_data[first_trigger_indices]
            
            # Calculate the time in ms relative to the beginning of the recording
            TTL_trigger_ms = (trigger_timestamps - start_time) * self.raster_time
            
            return TTL_trigger_ms

        except (ValueError, IndexError) as e:
            # Handle cases where the 'EXT1' channel or its data is not found
            print(f"Could not process TTL triggers for '{channel_name}': {e}")
            return np.array([])


    def _find_first_trigger_in_group(self, trigger_booleans):
            """
            Finds the *first* index of each consecutive group of True values
            in a boolean array.

            Args:
                trigger_booleans (np.array): A boolean array of triggers.

            Returns:
                np.array: An array of indices corresponding to the *first* trigger
                        of each consecutive group.
            """
            if not np.any(trigger_booleans):
                return np.array([], dtype=int)

            # Find all indices where the trigger is True
            trigger_indices_all = np.where(trigger_booleans)[0]

            # If there's only one trigger (or none), it's the first
            if len(trigger_indices_all) <= 1:
                return trigger_indices_all

            # Find breaks between consecutive sequences. A break is any gap where
            # the difference between indices is greater than 1.
            diffs = np.diff(trigger_indices_all)
            
            # A new group starts at the beginning (True)
            # and wherever the difference to the previous index was > 1
            group_starts_mask = np.concatenate(([True], diffs > 1))
            
            # Apply the mask to get the index of the first trigger in each group
            first_trigger_indices = trigger_indices_all[group_starts_mask]

            return first_trigger_indices

    def _filter_ecg_triggers(self, trigger_indices):
            """
            Filters ECG triggers to ensure a minimum separation of 60 indices.
            If multiple triggers are within this window, only the first one is kept.
            """
            if len(trigger_indices) <= 1:
                return trigger_indices

            filtered_indices = []
            
            # Start from the beginning of the list
            last_accepted_index = trigger_indices[0]
            filtered_indices.append(last_accepted_index)

            # Iterate forwards starting from the second index
            for i in range(1, len(trigger_indices)):
                current_index = trigger_indices[i]
                
                # Check the distance to the last trigger that was accepted
                if (current_index - last_accepted_index) >= 60:
                    filtered_indices.append(current_index)
                    last_accepted_index = current_index

            return np.array(filtered_indices)

    def get_origin_trigger_ms(self, channel_name):
        """
        Calculates the timestamps of trigger events in milliseconds, relative to the session's start time.
        """
        trigger_booleans = self.get_trigger(channel_name)
        trigger_indices_array = np.where(trigger_booleans)[0]
        if trigger_indices_array.size == 0:
            return np.array([]) # Return empty array if no triggers are found
        timestamp_data = self.get_timestamp(channel_name)
        start_time = self.beginning_timestamp
        trigger_timestamps = timestamp_data[trigger_indices_array]  
        relative_trigger_ms = (trigger_timestamps - start_time) * self.raster_time
        return relative_trigger_ms

    def get_processed_trigger_ms(self, channel_name):
            """
            Processes triggers for a given channel and returns their timestamps in milliseconds.

            This method replicates the full trigger processing pipeline from the 
            plot_all_channels method:
            1. Finds the *first* trigger in each consecutive group of active triggers.
            2. If the channel is an ECG channel, it filters triggers to ensure a minimum separation.
            3. Converts the final trigger indices to timestamps in milliseconds relative to the
            beginning of the recording.

            Args:
                channel_name (str): The name of the channel to process (e.g., 'ECG1', 'EXT1').

            Returns:
                np.array: A NumPy array containing the final, processed trigger times in milliseconds.
                        Returns an empty array if no triggers are found or the channel is invalid.
            """
            try:
                # Step 1: Get the raw boolean trigger data
                trigger_booleans = self.get_trigger(channel_name)
                if not np.any(trigger_booleans):
                    return np.array([])

                # Step 2: Find the *first* index of each consecutive group of 'True' values
                # This cleans up triggers that might span multiple time points
                trigger_indices = self._find_first_trigger_in_group(trigger_booleans)

                # Step 3: For ECG channels, apply the additional time-based filter
                if 'ECG' in channel_name:
                    trigger_indices = self._filter_ecg_triggers(trigger_indices)
                
                # If no triggers remain after processing, return an empty array
                if len(trigger_indices) == 0:
                    return np.array([])

                # Step 4: Convert the final indices to relative timestamps in milliseconds
                timestamp_data = self.get_timestamp(channel_name)
                start_time = self.beginning_timestamp
                
                # Create the full time axis in ms
                relative_timestamp_ms = (np.array(timestamp_data) - start_time) * self.raster_time
                
                # Select the timestamps corresponding to the processed trigger indices
                processed_trigger_times_ms = relative_timestamp_ms[trigger_indices]
                
                return processed_trigger_times_ms

            except (ValueError, IndexError) as e:
                # Handle cases where the channel name is not found
                print(f"Could not process triggers for '{channel_name}': {e}")
                return np.array([])

    def plot_all_channels(self):
        """
        Plots signals and their corresponding triggers for all available channels,
        each on a separate subplot. The time axis for each plot starts at 0 ms.
        Triggers are identified from a boolean list and plotted as vertical lines.
        """
        plt.style.use('seaborn-v0_8-whitegrid')

        # Filter for channels that actually have signal data to plot
        channels_to_plot = [ch for ch in self.available_channels if ch in self.signal and len(self.signal.get(ch, [])) > 0]
        if not channels_to_plot:
            print("No data found for any of the available channels to plot.")
            return

        fig, axes = plt.subplots(nrows=len(channels_to_plot), ncols=1,
                                 figsize=(15, 6 * len(channels_to_plot))) # Increased vertical space

        if len(channels_to_plot) == 1:
            axes = [axes]

        print("Plotting signals and triggers for channels:", ", ".join(channels_to_plot))

        # Use a color cycle for the signal plots
        color_cycle = plt.cm.get_cmap('viridis', len(channels_to_plot))

        for i, (ax, channel_name) in enumerate(zip(axes, channels_to_plot)):
            try:
                # Get all necessary data for the channel
                signal_data = self.get_signal(channel_name)
                timestamp_data = self.get_timestamp(channel_name)
                trigger_booleans = self.get_trigger(channel_name)

                # Normalize timestamps to start from 0 and convert to milliseconds
                start_time = self.beginning_timestamp
                relative_timestamp_ms = (np.array(timestamp_data) - start_time) * self.raster_time

                # Plot the signal with a unique color
                ax.plot(relative_timestamp_ms, signal_data, label=f'{channel_name} Signal', color=color_cycle(i))

                # Find the timestamps where the trigger is True
                if len(trigger_booleans) > 0:
                    # Use the helper function to get the first trigger in a consecutive group
                    trigger_indices = self._find_first_trigger_in_group(trigger_booleans)

                    # For ECG channels, apply the additional time-based filter
                    if 'ECG' in channel_name:
                        trigger_indices = self._filter_ecg_triggers(trigger_indices)

                    if len(trigger_indices) > 0:
                        trigger_times_ms = relative_timestamp_ms[trigger_indices]

                        # Get the y-limits from the signal plot for the vlines
                        ymin, ymax = ax.get_ylim()
                        # Plot dashed red vertical lines for triggers
                        ax.vlines(trigger_times_ms, ymin, ymax, colors='red', linestyles='--',
                                  label=f'{channel_name} Triggers')
                        # Plot red markers at the bottom of the plot to mark the x-axis
                        ax.plot(trigger_times_ms, np.full_like(trigger_times_ms, ymin), 'r|',
                                markersize=10, markeredgewidth=1.5, label='_nolegend_')

                        # Set the x-axis ticks to only be the trigger times
                        ax.set_xticks(trigger_times_ms)
                        ax.tick_params(axis='x', rotation=45) # Rotate labels for readability

                ax.set_title(f'Signal and Triggers for {channel_name}', fontsize=14)
                ax.set_ylabel('Amplitude (a.u.)', fontsize=12)
                # Set the x-axis label for each individual subplot
                ax.set_xlabel('time/ms', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.legend(fontsize=12)
                ax.grid(True)

            except (ValueError, IndexError) as e:
                print(f"  - Could not plot {channel_name}: {e}")

        fig.suptitle('PMU Signal and Trigger Events', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
    def get_ECG_extremes(self, channel_name, extreme='valleys',min_rrms=500):
            """
            Detects peaks or valleys in an ECG signal with filtering and returns
            their timestamps in milliseconds.

            This method finds all local extrema (peaks or valleys) and then
            filters them to ensure a minimum separation, based on a typical
            maximum heart rate (120 bpm or 2 Hz), to avoid noise.

            Args:
                channel_name (str): The ECG channel to process (e.g., 'ECG1').
                extreme (str): 'valleys' to find R-wave valleys (local minima)
                            or 'peaks' to find R-wave peaks (local maxima).

            Returns:
                np.array: A NumPy array of the timestamps (in ms) for the
                        detected and filtered extrema.
            """
            try:
                # Step 1: Get the signal data
                signal_data = self.get_signal(channel_name)
                if signal_data.size == 0:
                    print(f"No signal data found for '{channel_name}'.")
                    return np.array([])

                # Step 2: Prepare signal for peak finding
                if extreme == 'valleys':
                    # Invert the signal to find valleys (local minima)
                    processed_signal = -signal_data
                elif extreme == 'peaks':
                    processed_signal = signal_data
                else:
                    raise ValueError("Parameter 'extreme' must be 'peaks' or 'valleys'.")

                # Step 3: Calculate filtering parameters
                
                # "minimum RR interval is around 1-2Hz"
                # A 2 Hz frequency means a 500 ms period (1 / 2 Hz = 0.5s).
                # This is our minimum separation time.
                min_separation_ms = min_rrms
                
                # "each entry represents 2.5ms"
                # Convert the time separation (ms) into an index distance
                min_distance_indices = int(min_separation_ms / self.raster_time)
                
                # Step 4: Find the peaks (or valleys) using Scipy
                # We removed the 'height' parameter to be less restrictive.
                indices, _ = find_peaks(processed_signal, 
                                    distance=min_distance_indices)

                if indices.size == 0:
                    print(f"No {extreme} found for '{channel_name}' with current distance filter.")
                    return np.array([])

                # Step 5: Convert the final indices to relative timestamps in milliseconds
                timestamp_data = self.get_timestamp(channel_name)
                start_time = self.beginning_timestamp
                
                # Select the timestamps corresponding to the identified extrema indices
                extreme_timestamps = timestamp_data[indices]
                
                # Calculate the time in ms relative to the beginning of the recording
                extreme_times_ms = (extreme_timestamps - start_time) * self.raster_time
                
                return extreme_times_ms

            except (ValueError, IndexError) as e:
                print(f"Could not process ECG extrema for '{channel_name}': {e}")
                return np.array([])
            except ImportError:
                # Handle case where scipy is not installed
                print("Error: 'scipy' library not found.")
                print("Please install it to use get_ECG_extremes: pip install scipy")
                return np.array([])


