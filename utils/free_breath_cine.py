#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
free_breath_cine.py
-------------------

Utility module that reproduces the workflow of the free breath cine notebook as a
single callable function.

The function `run_slice_FB` executes the full cardiac MRI reconstruction pipeline
for free-breathing data. It loads a Siemens raw data file, processes the navigation
signal, determines both cardiac and respiratory gating indices, and reconstructs the
images using BART. It saves a video and a NumPy array of the reconstructed series.
"""

from pathlib import Path
import os
import numpy as np
import cfl
import torch

from .bin import (
    read_path,
    extract_nav_tr,
    process_nav_signal,
    fill_kspace_softbin,
    _to_bart_shape,
    eliminate_oversampling_gpu,
    visualize_sens_maps,
    animate_3d_array,
)

# Attempt to load the CPU fallback if it exists
try:
    from .bin import eliminate_oversampling
except ImportError:
    eliminate_oversampling = None

from .gating import (
    run_training,
    find_gating_index,
    get_original_indices,
)

from bart import bart
os.environ['BART_TOOLBOX_PATH'] = '/opt/bart'
os.environ['BART_GPU_GLOBAL_MEMORY'] = '1'
def run_slice_FB(data_dir: str, save_dir: str = "../saved_files") -> None:
    """Execute the full reconstruction pipeline for free-breathing cine MRI.

    Parameters
    ----------
    data_dir : str
        Path to the Siemens raw `.dat` file.
    save_dir : str, optional
        Directory where results will be saved. Defaults to "../saved_files".
        Subdirectories "video" and "array" will be created.

    Returns
    -------
    None
        The function saves a video (AVI) and a NumPy `.npy` file
        containing the magnitude of the reconstructed cine series.
    """


    data_path = Path(data_dir)
    twix = read_path(data_path)

    nav_signal, TR, beginning_timestamp = extract_nav_tr(twix)
    total_num_acqs = nav_signal.shape[1]

    # Process navigation signal
    discard_num = int(40 * 5)
    nav_signal_tensor = process_nav_signal(
        nav_signal[:, discard_num:],
        fs=1 / TR,
        free_breath=True,
        plot=False
    )

    # Train AE model
    ae_model = run_training(
        nav_signal_tensor=nav_signal_tensor,
        model_type='ae',
        learning_rate=0.001,
        max_epoch=50,
        hidden_dim1=14,
        hidden_dim2=6,
        sampling_rate=(1 / TR)
    )

    # Gating indices array
    (cardiac_gating_array, cardiac_peaks, cardiac_valleys, 
     breath_gating_array, breath_peaks, breath_valleys) = find_gating_index(
        model=ae_model,
        model_type='ae',
        TR=TR,
        nav_signal_tensor=nav_signal_tensor,
        kick_out_num=0,
        plot_all=False,
        order=5,
        bandwidth=0.2,
        include_breath=True
    )

    original_cardiac_gate = get_original_indices(cardiac_peaks, discard_num, 0)

    # Soft weight K-space
    num_cardiac_phases = 30
    k_space, weight_matrix = fill_kspace_softbin(
        twix=twix,
        gating_array=original_cardiac_gate,
        breath_gating_array=breath_gating_array,
        num_cardiac_phases=num_cardiac_phases,
        discard_num=discard_num,
        alpha=-1.6,
        threshold=0.10
    )

    # Convert to BART shape
    k_space_bart = _to_bart_shape(k_space)
    
    # GPU / CPU fallback for oversampling
    k_space_bart_all = eliminate_oversampling_gpu(k_space_bart)
    if k_space_bart_all is None:
        if eliminate_oversampling is not None:
            print("GPU oversampling failed or returned None. Falling back to CPU...")
            k_space_bart_all = eliminate_oversampling(k_space_bart)
        else:
            raise RuntimeError("GPU oversampling failed and CPU fallback eliminate_oversampling is not available.")

    torch.cuda.empty_cache()

    weights_bart_all = _to_bart_shape(weight_matrix)

    # Square weights
    weights_sq = weights_bart_all ** 2

    # Collapse Repetition Dimension
    sum_weighted_data = np.sum(k_space_bart_all * weights_sq, axis=11)
    sum_weights_sq = np.sum(weights_sq, axis=11)

    epsilon = 1e-10
    k_space_bart_collapsed = sum_weighted_data / (sum_weights_sq + epsilon)
    weights_bart_collapsed = np.sqrt(sum_weights_sq)

    # BART Calibration
    ksp_t_avg = bart(1, "avg -w {}".format((1<<10)+(1<<11)), k_space_bart_all * (weights_bart_all >= 0.95))
    cal_size = 32
    kernel_size = 6
    threshold = 5e-5
    crop_value = 0.2
    num_maps = 2
    sens_maps = bart(1, "ecalib -r {} -k {} -t {} -c {} -m {} -g ".format(cal_size, kernel_size, threshold, crop_value, num_maps), ksp_t_avg)

    # BART PICs reconstruction with GPU, fallback to CPU if it crashes
    cfl.writecfl('weights', weights_bart_collapsed)
    recon = None
    try:
        recon = bart(1, 'pics -S -g -R Q:0.001 -R T:1024:0:0.04 -p weights', 
                     k_space_bart_collapsed, 
                     sens_maps)
    except Exception as e:
        print(f"GPU pics failed: {e}")
        
    if recon is None:
        print("GPU pics failed or returned None. Falling back to CPU...")
        recon = bart(1, 'pics -S -R Q:0.001 -R T:1024:0:0.04 -p weights', 
                     k_space_bart_collapsed, 
                     sens_maps)

    recon_rss = bart(1, 'rss 16', recon)
    
    if os.path.exists("weights.cfl"):
        os.remove("weights.cfl")
    if os.path.exists("weights.hdr"):
        os.remove("weights.hdr")

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

    # Save animation and .npy
    out_video_path = str(video_dir / f"{data_path.stem}.avi")
    out_array_path = str(array_dir / f"{data_path.stem}.npy")
    
    animate_3d_array(
        data=np.abs(recon_rss.squeeze()),
        interval=100,
        cmap="gray",
        save_path=out_video_path,
        fps=10,
    )
    np.save(out_array_path, np.abs(recon_rss.squeeze()))
    print(f"Results saved to:\n  Video: {out_video_path}\n  Array: {out_array_path}")

def process_FB_folder(data_dir: str, save_dir: str = "../saved_files") -> None:
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
            run_slice_FB(str(file_path), save_dir=save_dir)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python free_breath_cine.py <path_to_dat_file_or_folder> [save_directory]")
        sys.exit(1)

    input_path = sys.argv[1]
    save_folder = sys.argv[2] if len(sys.argv) == 3 else "../saved_files"

    path_obj = Path(input_path)
    if path_obj.is_dir():
        process_FB_folder(input_path, save_dir=save_folder)
    else:
        run_slice_FB(input_path, save_dir=save_folder)
    print("\nProcessing completed.")

