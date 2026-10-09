#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
breath_hold_cine.py
-------------------

Utility module that reproduces the workflow of the original notebook as a
single callable function.

The function :func:`run_slice` loads a Siemens raw data file, extracts the
navigation signal, converts it to a PyTorch tensor, runs the auto‑encoder
training routine, performs cardiac gating, reconstructs the image series,
and saves a video and a NumPy array of the reconstructed magnitude images.

Typical usage
~~~~~~~~~~~~~
>>> from utils.breath_hold_cine import run_slice
>>> run_slice("../CMRdata/.../meas_..._BreathHold.dat", save_dir="saved_files")
"""

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import bart
# ----------------------------------------------------------------------
# Import utilities from the repository (relative imports within the utils package)
# ----------------------------------------------------------------------
from .bin import (
    read_path,
    extract_nav_tr,
    process_nav_signal,
    visualize_k_space_cardiac_phase,
    fill_kspace_bin,
    fill_kspace_interpolate,
    _to_bart_shape,
    get_sensitivity,
    visualize_sens_maps,
    espirit_recon,
    show_cine,
    animate_3d_array,
    get_noise,
    noise_to_bart_shape,
    homodyne_then_crop_to_kspace,
    eliminate_oversampling_gpu,
)
from .gating import (
    run_training,
    find_gating_index,
    get_original_indices,
)
from bart import bart

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def run_slice(data_dir: str, save_dir: str = "../saved_files",ranges=None) -> None:
    """Execute the full reconstruction pipeline.

    Parameters
    ----------
    data_dir : str
        Path to the Siemens raw ``.dat`` file. The function wraps it with
        :class:`pathlib.Path` internally.
    save_dir : str, optional
        Directory where results will be saved. Defaults to "saved_files".
        Subdirectories "video" and "array" will be created within this directory.

    Returns
    -------
    None
        The function saves a video (AVI) and a NumPy ``.npy`` file
        containing the magnitude of the reconstructed cine series.
    """
    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    data_path = Path(data_dir)
    twix = read_path(data_path)

    # ------------------------------------------------------------------
    # 2. Extract navigation signal and repetition time (TR)
    # ------------------------------------------------------------------
    nav_signal, TR, beginning_timestamp = extract_nav_tr(twix)

    # ------------------------------------------------------------------
    # 3. Process navigation signal into a torch tensor
    # ------------------------------------------------------------------
    discard_num = 0  # keep all acquisitions (original notebook used 0)
    nav_signal_tensor = process_nav_signal(
        nav_signal[:, discard_num:],
        fs=1 / TR,
        free_breath=False,
        plot=False,
    )

    # ------------------------------------------------------------------
    # 4. Run the auto‑encoder training routine
    # ------------------------------------------------------------------
    ae_model = run_training(
        nav_signal_tensor=nav_signal_tensor,
        model_type="ae",
        learning_rate=0.002,
        max_epoch=30,
        hidden_dim1=12,
        hidden_dim2=6,
        sampling_rate=(1 / TR),
        ranges=ranges
    )

    # ------------------------------------------------------------------
    # 5. Cardiac gating using the trained model
    # ------------------------------------------------------------------
    cardiac_gating_array, cardiac_peaks, cardiac_valleys = find_gating_index(
        model=ae_model,
        model_type="ae",
        TR=TR,
        nav_signal_tensor=nav_signal_tensor,
        kick_out_num=0,
        plot_all=True,
        order=5,
        bandwidth=0.15,
        include_breath=False,
    )
    original_cardiac_gate = get_original_indices(
        cardiac_gating_array, discard_num, 0
    )

    # ------------------------------------------------------------------
    # 6. Reconstruct k‑space for each cardiac phase
    # ------------------------------------------------------------------
    k_space = fill_kspace_interpolate(
        twix=twix,
        gating_array=original_cardiac_gate,
        num_cardiac_phases=30,
        discard_num=discard_num,
    )

    # ------------------------------------------------------------------
    # 7. Pre‑process k‑space (oversampling removal, whitening, sensitivity)
    # ------------------------------------------------------------------
    k_space_bart=_to_bart_shape(k_space)
    k_space_cropped=eliminate_oversampling_gpu(k_space_bart)
    noise = get_noise(twix)
    noise_bart = noise_to_bart_shape(noise)
    k_space_whiten = bart(1, "whiten", k_space_cropped, noise_bart)
    sens_maps = get_sensitivity(
        k_space_whiten,
        cal_size=32,    ### ACS lines
        kernel_size=6,
        threshold=1e-4,
        crop_value=0.2,
        num_maps=2,
        use_gpu=True,
    )
    torch.cuda.empty_cache()
    # ------------------------------------------------------------------
    # 8. Image reconstruction (ESPIRiT)
    # ------------------------------------------------------------------
    l2_reg = 1e-3
    recon=bart(1,'pics -R Q:{} -S -g '.format(l2_reg), k_space_cropped, sens_maps)
    recon_rss=bart(1,'rss 16', recon)
    
    # l_TVtime = 0
    # recon_rss = espirit_recon(
    #     k_space_whiten,
    #     sens_maps,
    #     l2_reg=l2_reg,
    #     l_TVtime=l_TVtime,
    #     use_gpu=False,
    # )

    # ------------------------------------------------------------------
    # 9. Save results (video and NumPy array)
    # ------------------------------------------------------------------
    # The new structure set up the same whole subfolder path after CMRdata under saved_files
    parts = data_path.parent.parts
    idx = -1
    for i, part in enumerate(parts):
        if part.lower() == "cmrdata":
            idx = i
            break
    if idx != -1:
        sub_path = Path(*parts[idx+1:])
    else:
        sub_path = Path(data_path.parent.name if data_path.parent.name else "unknown")
    base_save_path = Path(save_dir) / sub_path
    
    video_dir = base_save_path / "video"
    array_dir = base_save_path / "array"
    
    video_dir.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)

    animate_3d_array(
        data=np.abs(recon_rss.squeeze()),
        interval=100,
        cmap="gray",
        save_path=str(video_dir / f"{data_path.stem}.avi"),
        fps=10,
    )
    np.save(str(array_dir / f"{data_path.stem}.npy"), np.abs(recon_rss.squeeze()))

    # No explicit return – results are saved to disk.

def process_BH_folder(data_dir: str, save_dir: str = "../saved_files",ranges=None) -> None:
    """Process all Siemens raw data files in a directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing Siemens raw ``.dat`` files.
    save_dir : str, optional
        Directory where results will be saved. Defaults to "../saved_files".
    """
    data_path = Path(data_dir)
    # Find all .dat files in the folder (Siemens raw data)
    dat_files = sorted(list(data_path.glob("*.dat")))

    if not dat_files:
        print(f"No .dat files found in {data_dir}")
        return

    print(f"Found {len(dat_files)} files in {data_dir}. Starting processing...")

    for i, file_path in enumerate(dat_files):
        print(f"\n--- [{i+1}/{len(dat_files)}] Processing {file_path.name} ---")
        try:
            run_slice(str(file_path), save_dir=save_dir,ranges=ranges)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

# ----------------------------------------------------------------------
# Optional: quick sanity‑check when executed as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python breath_hold_cine.py <path_to_dat_file_or_folder> [save_directory]")
        sys.exit(1)

    input_path = sys.argv[1]
    save_folder = sys.argv[2] if len(sys.argv) == 3 else "../saved_files"

    path_obj = Path(input_path)
    if path_obj.is_dir():
        process_BH_folder(input_path, save_dir=save_folder)
    else:
        run_slice(input_path, save_dir=save_folder)
    print("\nProcessing completed.")