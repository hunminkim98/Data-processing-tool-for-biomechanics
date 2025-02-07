# SPM.py
"""
This script performs Statistical Parametric Mapping (SPM) analysis on raw joint angle waveforms
extracted from Excel files—the same raw data used for CMC calculation in stat_main.py and CMC2.py.
The get_waveforms and extract_subject_motion functions (imported from stat_main.py) are used to extract
waveforms and group files by subject and motion. The data are assumed to be normalized and filtered.
An independent samples t-test is performed using SPM1D following Pataky's methodology:
"One-dimensional statistical parametric mapping in Python".
Official documentation: https://spm1d.org/
"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from spm1d import stats
from collections import defaultdict

# Import functions from stat_main.py
from stat_main import get_waveforms, extract_subject_motion

def plot_spm_results(mean_markerless, mean_markerbased, std_markerless, std_markerbased, 
                    SPMti, title, output_path, show_std=True):
    """Helper function to create and save combined waveform and SPM plots"""
    plt.figure(figsize=(12, 8))
    
    # Upper subplot: Mean waveforms with std bands
    plt.subplot(2, 1, 1)
    x = np.linspace(0, 100, mean_markerless.shape[0])
    plt.plot(x, mean_markerless, label='Markerless', linewidth=2)
    if show_std:
        plt.fill_between(x, mean_markerless - std_markerless, mean_markerless + std_markerless, 
                        alpha=0.2, label='Markerless SD')
    plt.plot(x, mean_markerbased, label='Marker-based', color='r', linewidth=2)
    if show_std:
        plt.fill_between(x, mean_markerbased - std_markerbased, mean_markerbased + std_markerbased,
                        color='r', alpha=0.2, label='Marker-based SD')
    plt.xlabel('Normalized Time (%)', fontsize=12)
    plt.ylabel('Joint Angle (deg)', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Lower subplot: SPM results
    plt.subplot(2, 1, 2)
    SPMti.plot()
    plt.xlabel('Normalized Time (%)', fontsize=12)
    plt.ylabel('SPM{t}', fontsize=12)
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create directories for saving plots
    output_folder = "SPM"
    # Removed individual_folder related lines
    os.makedirs(output_folder, exist_ok=True)

    
    # Specify the parent folder containing raw Excel files (same as used for CMC calculation)
    parent_folder = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check'
    
    # Get all Excel trial files
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    trial_files.sort()
    
    if not trial_files:
        print('No trial files found in the specified folder.')
        return
    
    # Create nested defaultdict to store waveforms by motion, joint, and axis
    # Structure: motion -> joint -> axis -> [markerbased_list, markerless_list, file_paths]
    waveform_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[], [], []])))
    
    # Define joints and axes to process
    joints = ["Left_Ankle", "Left_Hip", "Left_Knee", "Right_Ankle", "Right_Hip", "Right_Knee", "Trunk"]
    axes = ["X", "Y", "Z"]
    
    # First pass: collect all waveforms grouped by motion, joint, and axis
    print("Collecting waveforms from all trials...")
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        print(f"Processing file from Subject: {subject}, Motion: {motion}")
        
        for joint in joints:
            for axis in axes:
                try:
                    mb_wave, ml_wave = get_waveforms(file, joint, axis)
                    waveform_data[motion][joint][axis][0].append(mb_wave)  # markerbased
                    waveform_data[motion][joint][axis][1].append(ml_wave)  # markerless
                    waveform_data[motion][joint][axis][2].append(file)     # store file path for subject info
                except Exception as e:
                    print(f"Error processing file {file} for Joint: {joint}, Axis: {axis}: {e}")
    
    # Second pass: perform SPM analysis for each motion-joint-axis combination
    print("\nPerforming SPM analysis for each motion-joint-axis combination...")
    for motion in waveform_data:
        print(f"\nProcessing Motion: {motion}")
        # Removed motion_folder related lines
        
        for joint in waveform_data[motion]:
            print(f"  Joint: {joint}")
            # Removed joint_folder related lines
            
            for axis in waveform_data[motion][joint]:
                markerbased_list, markerless_list, file_paths = waveform_data[motion][joint][axis]
                
                if not markerbased_list or not markerless_list:
                    print(f"    No valid data for Axis: {axis}")
                    continue
                
                # Removed individual trials processing section
                
                # Process combined data
                print(f"    Processing combined data for Axis: {axis}")
                
                markerbased_array = np.array(markerbased_list)
                markerless_array = np.array(markerless_list)

                # Combined SPM analysis (paired test)
                SPMt = stats.ttest_paired(markerless_array, markerbased_array)
                SPMti = SPMt.inference(alpha=0.05, two_tailed=True, interp=True)
                
                # Calculate mean waveforms and std
                mean_markerless = np.mean(markerless_array, axis=0)
                mean_markerbased = np.mean(markerbased_array, axis=0)
                std_markerless = np.std(markerless_array, axis=0)
                std_markerbased = np.std(markerbased_array, axis=0)
                
                # Plot combined results
                title = f'Combined Analysis\nMotion: {motion}, Joint: {joint}, Axis: {axis}\n(All Subjects Combined)'
                output_path = os.path.join(output_folder, f"combined_{motion}_{joint}_{axis}.png")
                plot_spm_results(mean_markerless, mean_markerbased, 
                               std_markerless, std_markerbased,
                               SPMti, title, output_path, show_std=True)
                
                print(f"    Saved combined plot for Motion: {motion}, Joint: {joint}, Axis: {axis}")
                print(f"    Critical threshold: {SPMti.zstar}")
                print(f"    P-values for clusters: {[f'{p:.4f}' for p in SPMti.p]}")
    
    print("\nAll SPM analyses completed. Results saved in 'SPM' folder.") # Modified the message


if __name__ == '__main__':
    main()
