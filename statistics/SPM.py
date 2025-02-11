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
import matplotlib.pyplot as plt
from collections import defaultdict
import spm1d
from spm1d import stats
from scipy import stats as scipy_stats
from stat_main import get_waveforms, extract_subject_motion

def resample_to_standard_rate(data, original_rate=1.0, target_rate=120.0):
    """Resample data to standard rate using interpolation"""
    if original_rate == target_rate:
        return data
    
    # Calculate number of points after resampling
    original_points = data.shape[1]
    target_points = int(original_points * (target_rate / original_rate))
    
    # Create time vectors
    original_time = np.linspace(0, 1, original_points)
    target_time = np.linspace(0, 1, target_points)
    
    # Resample each row (subject)
    resampled_data = np.zeros((data.shape[0], target_points))
    for i in range(data.shape[0]):
        resampled_data[i] = np.interp(target_time, original_time, data[i])
    
    return resampled_data

def get_effect_size_interpretation(effect_sizes):
    """Get the most common effect size interpretation using pandas"""
    import pandas as pd
    
    # Convert effect sizes to interpretations
    interpretations = np.array(['negligible' if abs(x) < 0.2 else
                              'small' if abs(x) < 0.5 else
                              'medium' if abs(x) < 0.8 else
                              'large' for x in effect_sizes])
    
    # Use pandas mode
    mode_result = pd.Series(interpretations).mode()
    return mode_result.iloc[0] if not mode_result.empty else 'negligible'

def plot_spm_results(mean_markerless, mean_markerbased, std_markerless, std_markerbased, SPMti, title, output_path, show_std=True):
    """Helper function to create and save combined waveform and SPM plots"""
    plt.figure(figsize=(12,6))  # Increased height to accommodate decorations
    
    # Plot waveforms
    ax0 = plt.subplot(121)
    if show_std:
        ax0.fill_between(range(len(mean_markerless)), 
                        mean_markerless - std_markerless,
                        mean_markerless + std_markerless,
                        color='blue', alpha=0.2)
        ax0.fill_between(range(len(mean_markerbased)), 
                        mean_markerbased - std_markerbased,
                        mean_markerbased + std_markerbased,
                        color='red', alpha=0.2)
    ax0.plot(mean_markerless, 'b-', label='Markerless')
    ax0.plot(mean_markerbased, 'r-', label='Marker-based')
    ax0.legend()
    ax0.set_title('Mean Trajectories')
    
    # Plot SPM results
    ax1 = plt.subplot(122)
    SPMti.plot(ax=ax1)
    SPMti.plot_threshold_label(ax=ax1, fontsize=8)
    SPMti.plot_p_values(ax=ax1, size=10)
    ax1.set_title('SPM Analysis')
    
    plt.suptitle(title, y=1.05)  # Moved title up
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def check_normality(data1, data2=None):
    """Check normality of data using SPM1D's normality test for paired data if two datasets are provided,
    otherwise use Shapiro-Wilk test for single dataset"""
    try:
        if data2 is not None:
            # For paired data, use SPM1D's normality test
            spmi = spm1d.stats.normality.ttest_paired(data1, data2).inference(0.05)
            # The test statistic exceeding the critical value indicates non-normality
            is_normal = spmi.z < spmi.zstar
            return is_normal, spmi
        else:
            # For single dataset, use Shapiro-Wilk test
            if len(data1.shape) > 1:
                p_values = np.array([scipy_stats.shapiro(data1[:, i])[1] for i in range(data1.shape[1])])
            else:
                p_values = np.array([scipy_stats.shapiro(data1)[1]])
            is_normal = p_values > 0.05
            return is_normal, p_values
    except Exception as e:
        print(f"Warning: Error in normality check - {str(e)}")
        return np.array([False]), None

def plot_normality_test(data1, data2, title, output_path):
    """Plot normality test results including data, residuals, and test statistics"""
    spmi = spm1d.stats.normality.ttest_paired(data1, data2).inference(0.05)
    
    plt.figure(figsize=(14,4))
    # Plot original data
    ax = plt.subplot(131)
    ax.plot(data1.T, 'k', lw=0.5, label='Data 1')
    ax.plot(data2.T, 'r', lw=0.5, label='Data 2')
    ax.set_title('Data')
    ax.legend()
    
    # Plot residuals
    ax = plt.subplot(132)
    ax.plot(spmi.residuals.T, 'k', lw=0.5)
    ax.set_title('Residuals')
    
    # Plot test statistics
    ax = plt.subplot(133)
    spmi.plot(ax=ax)
    spmi.plot_threshold_label(ax=ax)
    ax.set_title('Normality Test')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_effect_size(markerless_array, markerbased_array):
    """Calculate Cohen's d effect size for paired data"""
    d = (markerless_array - markerbased_array).mean(axis=0) / np.std(markerless_array - markerbased_array, axis=0)
    return d

def main():
    # Create directories for saving plots
    output_folder = "SPM"
    os.makedirs(output_folder, exist_ok=True)
    
    # Specify the parent folder containing raw Excel files
    parent_folder = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check'
    
    # Get all Excel trial files
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    trial_files.sort()
    
    if not trial_files:
        print('No trial files found in the specified folder.')
        return
    
    # Create nested defaultdict to store waveforms
    waveform_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [[], [], []])))
    
    # Define joints and axes to process
    joints = ["Left_Ankle", "Left_Hip", "Left_Knee", "Right_Ankle", "Right_Hip", "Right_Knee", "Trunk"]
    axes = ["X", "Y", "Z"]
    
    # First pass: collect all waveforms
    print("Collecting waveforms from all trials...")
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        print(f"Processing file from Subject: {subject}, Motion: {motion}")
        
        for joint in joints:
            for axis in axes:
                try:
                    mb_wave, ml_wave = get_waveforms(file, joint, axis)
                    waveform_data[motion][joint][axis][0].append(mb_wave)
                    waveform_data[motion][joint][axis][1].append(ml_wave)
                    waveform_data[motion][joint][axis][2].append(file)
                except Exception as e:
                    print(f"Error processing file {file} for Joint: {joint}, Axis: {axis}: {e}")
    
    # Second pass: perform SPM analysis
    print("\nPerforming SPM analysis for each motion-joint-axis combination...")
    for motion in waveform_data:
        print(f"\nProcessing Motion: {motion}")
        
        for joint in waveform_data[motion]:
            print(f"  Joint: {joint}")
            
            for axis in waveform_data[motion][joint]:
                markerbased_list, markerless_list, file_paths = waveform_data[motion][joint][axis]
                
                if not markerbased_list or not markerless_list:
                    print(f"    No valid data for Axis: {axis}")
                    continue
                
                print(f"    Processing combined data for Axis: {axis}")
                
                # Convert lists to numpy arrays for analysis
                mb_array = np.array(markerbased_list)
                ml_array = np.array(markerless_list)
                
                # Resample data to standard rate if needed
                mb_array = resample_to_standard_rate(mb_array)
                ml_array = resample_to_standard_rate(ml_array)
                
                # Check normality of the paired difference
                is_normal, spmi = check_normality(ml_array, mb_array)
                
                # Plot normality test results
                normality_plot_path = os.path.join(output_folder, f'normality_{motion}_{joint}_{axis}.png')
                plot_normality_test(ml_array, mb_array, f"{motion} - {joint} {axis} Normality Test", normality_plot_path)
                
                # Calculate effect size
                effect_size = calculate_effect_size(ml_array, mb_array)
                
                # Get effect size interpretation
                effect_interpretation = get_effect_size_interpretation(effect_size)
                
                # Perform SPM analysis
                if np.all(is_normal):
                    # Use parametric test (SPM t-test)
                    SPMt = stats.ttest_paired(ml_array, mb_array)
                    SPMti = SPMt.inference(alpha=0.05, two_tailed=False)  # Use one-tailed test
                else:
                    # Use non-parametric test (SPM Wilcoxon)
                    print(f"Using non-parametric test for {motion} {joint} {axis} due to non-normal distribution")
                    print(f"P-values: min={np.min(spmi.p):.4f}")
                    
                    # Use all possible permutations for small datasets, or a large number for big datasets
                    n_subjects = ml_array.shape[0]
                    if n_subjects <= 12:  # For small samples, use all permutations
                        iterations = -1
                    else:  # For larger samples, use 10000 permutations
                        iterations = 10000
                    
                    SPMt = stats.nonparam.ttest_paired(ml_array, mb_array)
                    SPMti = SPMt.inference(alpha=0.05, two_tailed=False, iterations=iterations)
                
                # Save results
                title = f"{motion} - {joint} {axis}\nMean Effect Size: {np.mean(effect_size):.2f} ({effect_interpretation})"
                output_path = os.path.join(output_folder, f"{motion}_{joint}_{axis}_spm.png")
                
                # Calculate means and standard deviations for plotting
                mean_ml = ml_array.mean(axis=0)
                mean_mb = mb_array.mean(axis=0)
                std_ml = ml_array.std(axis=0)
                std_mb = mb_array.std(axis=0)
                
                plot_spm_results(mean_ml, mean_mb, std_ml, std_mb, SPMti, title, output_path)
                
                print(f"Completed analysis for {motion} {joint} {axis}")
    
    print("\nAll SPM analyses completed. Results saved in 'SPM' folder.")

if __name__ == '__main__':
    main()
