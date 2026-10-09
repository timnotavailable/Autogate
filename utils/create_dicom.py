#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_dicom.py
---------------

Utility module to create DICOM files from NumPy arrays using existing DICOM 
files as metadata templates.

3D array 
Implementation Details:
Normalization: The BART magnitudes are mapped to a stored range of 0 → 32767.
Physical Units: I added the RescaleSlope and RescaleIntercept tags. This means any DICOM viewer will automatically perform the math $I_{real} = I_{stored} \\cdot slope + intercept$ to show you the exact original BART intensities.
Metadata Updates:
Data Type: Set to int16 signed integer (PixelRepresentation = 1).
Bit Depth: set to 16-bit (BitsAllocated = 16, BitsStored = 16, HighBit = 15).
Viewing: WindowCenter and WindowWidth are now updated to match the real BART units $(I_{min}, I_{max})$, ensuring the image looks perfect instantly.


"""

import pydicom
import numpy as np
from pathlib import Path
import os
from pydicom.uid import generate_uid

def numpy_to_dicom(npy_path: str, dicom_dir: str, save_dir: str = "../saved_files") -> None:
    """
    Converts a 3D numpy array to DICOM files by using metadata from a template DICOM folder.
    
    The function reads all DICOM files in the template folder to determine the global
    intensity range (min/max), normalizes the input 3D array to this range, and saves
    each cardiac phase into a new DICOM file with original metadata.

    Parameters
    ----------
    npy_path : str
        Path to the .npy file containing the 3D array (H, W, Phase).
    dicom_dir : str
        Path to the folder containing template DICOM files (.dcm).
    save_dir : str
        Root folder where the results will be saved.
    """
    # 1. Load the input numpy array
    print(f"Loading numpy array from {npy_path}...")
    try:
        volume = np.load(npy_path)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    if volume.ndim != 3:
        print(f"Error: Expected a 3D array, but got shape {volume.shape}")
        return

    # 2. Get the template DICOM files
    dicom_path = Path(dicom_dir)
    print(f"Reading template DICOMs from {dicom_dir}...")
    
    # Use sorted glob to match the zero-padded filenames (e.g., 0001, 0002...)
    dcm_files = sorted(
        list(dicom_path.glob("*.dcm")) + 
        list(dicom_path.glob("*.DCM")) + 
        list(dicom_path.glob("*.IMA")) + 
        list(dicom_path.glob("*.ima"))
    )
    
    if not dcm_files:
        print(f"Error: No DICOM files (*.dcm or *.IMA) found in {dicom_dir}")
        return

    # Read DICOM data into a list (each file is a cardiac phase)
    dicom_data = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            dicom_data.append({
                'path': f,
                'ds': ds
            })
        except Exception as e:
            print(f"Warning: Skipping {f.name}: {e}")
            continue

    if not dicom_data:
        print("Error: No valid DICOM data could be read.")
        return
    
    # Detect target DICOM array dimension and rescale volume if needed
    try:
        import scipy.ndimage
        first_ds = dicom_data[0]['ds']
        target_shape = first_ds.pixel_array.shape
        current_shape = volume.shape[:2]
        
        # Auto-align orientation by checking if transposing increases matching dimensions
        match_before = (current_shape[0] == target_shape[0]) + (current_shape[1] == target_shape[1])
        match_after = (current_shape[1] == target_shape[0]) + (current_shape[0] == target_shape[1])
        
        if match_after > match_before:
            print(f"Auto-transposing volume from {current_shape} to {(current_shape[1], current_shape[0])} to align shared dimension...")
            volume = np.transpose(volume, (1, 0, 2))
            current_shape = volume.shape[:2]
        
        if current_shape != target_shape:
            print(f"Rescaling volume from {current_shape} to {target_shape} to match DICOM template...")
            # We only rescale the dimension that is not aligned; the aligned dimension gets a factor of 1.0
            zoom_y = target_shape[0] / current_shape[0] if current_shape[0] != target_shape[0] else 1.0
            zoom_x = target_shape[1] / current_shape[1] if current_shape[1] != target_shape[1] else 1.0
            zoom_factors = (zoom_y, zoom_x, 1.0)
            
            volume = scipy.ndimage.zoom(volume, zoom_factors, order=1)
    except Exception as e:
        print(f"Warning: Could not rescale/align volume: {e}")

    # Rotate volume 180 degrees
    print("Rotating volume by 180 degrees...")
    volume = np.rot90(volume, k=2, axes=(0, 1))

    num_dicoms = len(dicom_data)
    # The user specifies that the array is 3D and saved along the last dimension (Phases)
    num_phases = volume.shape[-1]
    
    if num_dicoms != num_phases:
        print(f"Warning: Number of DICOM templates ({num_dicoms}) does not match volume phases ({num_phases}).")
        limit = min(num_dicoms, num_phases)
        dicom_data = dicom_data[:limit]
        volume = volume[..., :limit]
        print(f"Processing only the first {limit} phases.")

    # 3. Use metadata from the first template to define output format
    print("Detecting template metadata format...")
    first_ds = dicom_data[0]['ds']
    target_bits = getattr(first_ds, 'BitsStored', 16)
    target_rep = getattr(first_ds, 'PixelRepresentation', 0)
    
    # Target range for stored intensity
    stored_max = float((1 << (target_bits - (1 if target_rep == 1 else 0))) - 1)
    
    if target_rep == 1:
        target_dtype = np.int16
    else:
        target_dtype = np.uint16
    
    # 4. Normalize and Scale
    v_min = float(volume.min())
    v_max = float(np.percentile(volume, 99.9))
    print(f"BART volume range (for scaling): [{v_min}, {v_max}]")
    print(f"Target format: {target_bits}-bit {'signed' if target_rep == 1 else 'unsigned'} (max: {stored_max})")
    
    if v_max - v_min > 1e-10:
        normalized_volume = (volume - v_min) / (v_max - v_min) * stored_max
    else:
        normalized_volume = np.zeros_like(volume)
        
    stored_volume = np.clip(np.round(normalized_volume), 0, stored_max).astype(target_dtype)
    
    # Generate new Series UID to prevent viewer caching issues
    new_series_uid = generate_uid()
    
    # 5. Create output directory
    input_folder_name = dicom_path.name
    output_folder_name = "autogate_dicom_" + input_folder_name
    output_path = Path(save_dir) / output_folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_path}")

    # 6. Save results with Rescale tags
    for idx, item in enumerate(dicom_data):
        ds = item['ds']
        orig_file = item['path']
        
        # Extract new frame [H, W, Phase]
        new_frame = stored_volume[..., idx]
        
        # Ensure array is C-contiguous
        new_frame_contiguous = np.ascontiguousarray(new_frame)
        
        # Inject updated pixel data
        ds.PixelData = new_frame_contiguous.tobytes()
        
        # Update metadata to match template
        ds.BitsAllocated = 16
        ds.BitsStored = target_bits
        ds.HighBit = target_bits - 1
        ds.PixelRepresentation = target_rep
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # UIDs to prevent caching
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        
        # Windowing
        if 'RescaleSlope' in ds:
            del ds.RescaleSlope
        if 'RescaleIntercept' in ds:
            del ds.RescaleIntercept
            
        frame_min_val = float(new_frame_contiguous.min())
        frame_max_val = float(new_frame_contiguous.max())
        
        if frame_max_val - frame_min_val > 0:
            ds.WindowCenter = (frame_max_val + frame_min_val) / 2.0
            ds.WindowWidth = frame_max_val - frame_min_val
        else:
            ds.WindowCenter = stored_max / 2.0
            ds.WindowWidth = stored_max
            
        # Update min/max pixel tags
        if 'SmallestImagePixelValue' in ds:
            ds.SmallestImagePixelValue = int(frame_min_val)
        if 'LargestImagePixelValue' in ds:
            ds.LargestImagePixelValue = int(frame_max_val)
            
        # Optional: update Rows/Columns
        ds.Rows, ds.Columns = new_frame_contiguous.shape

        # Save
        new_filename = f"Autogate_{orig_file.name}"
        if not new_filename.lower().endswith(".dcm"):
            new_filename += ".dcm"
            
        ds.save_as(str(output_path / new_filename))

    print(f"Successfully saved {len(dicom_data)} DICOM files to {output_path}")

def process_dicom_batch(recon_results_dir: str, template_dicom_root: str) -> None:
    """
    Batch processes multiple reconstructed slices.
    
    Parameters
    ----------
    recon_results_dir : str
        Path to the folder containing 'video' and 'array' subfolders (e.g., A02_A049_0_SelfGated_raw).
    template_dicom_root : str
        Root folder containing Siemens template DICOM slice subfolders (e.g., A02_A049_CV_SG_DICOM).
    """
    import re
    
    recon_path = Path(recon_results_dir)
    array_dir = recon_path / "array"
    template_root = Path(template_dicom_root)
    
    if not array_dir.exists():
        print(f"Error: {array_dir} does not exist.")
        return
    if not template_root.exists():
        print(f"Error: {template_root} does not exist.")
        return
        
    # 1. Collect all .npy files and extract slice numbers
    npy_files = list(array_dir.glob("*.npy"))
    npy_map = {}
    for f in npy_files:
        # Extract number after 'slc_'
        match = re.search(r'slc_(\d+)', f.name)
        if match:
            key = int(match.group(1))
            npy_map[key] = f
        else:
            # Fallback to matching by the part after the last '_'
            key = f.stem.split('_')[-1]
            npy_map[key] = f
            
    # 2. Collect all template subfolders and extract slice numbers
    template_folders = [d for d in template_root.iterdir() if d.is_dir()]
    template_map = {}
    for d in template_folders:
        match = re.search(r'slc_(\d+)', d.name)
        if match:
            key = int(match.group(1))
            template_map[key] = d
        else:
            # Fallback to matching by the part after the last '_'
            key = d.name.split('_')[-1]
            template_map[key] = d
            
    # 3. Match and process
    print(f"Found {len(npy_map)} arrays and {len(template_map)} template folders.")
    
    for key, npy_path in npy_map.items():
        if key in template_map:
            template_dir = template_map[key]
            print(f"\n--- Processing Match: {key} ---")
            print(f"Array: {npy_path.name}")
            print(f"Template: {template_dir.name}")
            
            # Invoke numpy_to_dicom
            # Results will be saved inside recon_results_dir (e.g., autogate_dicom_slc_4)
            numpy_to_dicom(
                npy_path=str(npy_path),
                dicom_dir=str(template_dir),
                save_dir=str(recon_path)
            )
        else:
            print(f"Warning: No template folder found for match {key} (Array: {npy_path.name})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage (Single): python create_dicom.py <npy_path> <dicom_dir> [save_dir]")
        print("Usage (Batch):  python create_dicom.py --batch <recon_results_dir> <template_dicom_root>")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        recon_results_dir = sys.argv[2]
        template_dicom_root = sys.argv[3]
        process_dicom_batch(recon_results_dir, template_dicom_root)
    else:
        npy_path = sys.argv[1]
        dicom_dir = sys.argv[2]
        save_dir = sys.argv[3] if len(sys.argv) == 4 else "../saved_files"
        numpy_to_dicom(npy_path, dicom_dir, save_dir)
