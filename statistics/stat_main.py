import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from CMC2 import calculate_cmc  # Import the calculate_cmc function

# Set Korean font and configure minus sign display
plt.rcParams['font.family'] = 'Malgun Gothic'  # For Windows
# For Mac: plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# Sheet index mapping for each joint (global constant)
JOINT_SHEET_MAPPING = {
    "Left_Ankle": 0,
    "Left_Hip": 1,
    "Left_Knee": 2,
    "Right_Ankle": 4,
    "Right_Hip": 5,
    "Right_Knee": 6,
    "Trunk": 8
}

def extract_subject_motion(file_path):
    """
    Extract subject and motion information from the file path.
    Example: parent_folder/subject/motion/trial.xlsx
    """
    parts = os.path.normpath(file_path).split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else 'unknown'
    motion = parts[-2] if len(parts) >= 3 else 'unknown'
    return subject, motion

def get_waveforms(file_path, joint='Ankle', coordinate='X'):
    """
    Read joint and coordinate data from the specified file to extract marker-based and markerless waveform data.
    If the data lengths differ, they are trimmed to the shorter length, and NaN values are removed.
    The data is then filtered with a 6Hz lowpass filter and normalized to 101 points using linear interpolation.
    """
    sheet_idx = JOINT_SHEET_MAPPING.get(joint)
    if sheet_idx is None:
        raise ValueError(f"Unsupported joint: {joint}. Available options: {list(JOINT_SHEET_MAPPING.keys())}.")
    
    df = pd.read_excel(file_path, header=[0, 1, 2], sheet_name=sheet_idx)
    
    # marker-based: columns 1 to 3, markerless: columns 5 to 7
    marker_based = df.iloc[:, 1:4]
    markerless = df.iloc[:, 5:8]
    
    coord_idx = {"X": 0, "Y": 1, "Z": 2}.get(coordinate)
    if coord_idx is None:
        raise ValueError(f"Unsupported coordinate: {coordinate}. Available options: 'X', 'Y', 'Z'.")
    
    marker_based_wave = marker_based.iloc[:, coord_idx].values
    markerless_wave = markerless.iloc[:, coord_idx].values

    # Trim to the shorter length if lengths differ
    len_mb, len_ml = len(marker_based_wave), len(markerless_wave)
    if len_mb != len_ml:
        print(f"Warning: In file '{file_path}', joint {joint} and coordinate {coordinate} have mismatched lengths (Marker-based: {len_mb}, Markerless: {len_ml}). Using the minimum length.")
    common_length = min(len_mb, len_ml)
    marker_based_wave = marker_based_wave[:common_length]
    markerless_wave = markerless_wave[:common_length]
    
    # Remove NaN values
    mask = ~np.isnan(marker_based_wave) & ~np.isnan(markerless_wave)
    if not np.all(mask):
        print(f"Warning: In file '{file_path}', joint {joint} and coordinate {coordinate} had {np.count_nonzero(~mask)} NaN values removed.")
    marker_based_wave = marker_based_wave[mask]
    markerless_wave = markerless_wave[mask]
    
    if marker_based_wave.size == 0:
        raise ValueError(f"All data for joint {joint} in file {file_path} has been removed due to NaN values.")
    
    # Apply 6Hz lowpass filter
    # Assuming 100Hz sampling rate (can be adjusted if needed)
    fs = 100  # sampling frequency in Hz
    fc = 6    # cutoff frequency in Hz
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(4, w, 'low')
    
    marker_based_filtered = signal.filtfilt(b, a, marker_based_wave)
    markerless_filtered = signal.filtfilt(b, a, markerless_wave)
    
    # Normalize to 101 points using linear interpolation
    old_indices = np.linspace(0, 100, len(marker_based_filtered))
    new_indices = np.linspace(0, 100, 101)
    
    marker_based_normalized = np.interp(new_indices, old_indices, marker_based_filtered)
    markerless_normalized = np.interp(new_indices, old_indices, markerless_filtered)
    
    return marker_based_normalized, markerless_normalized

def process_trial(file_path, joint='Ankle', coordinate='X'):
    """
    Extract waveform data from the specified file and calculate the CMC value.
    """
    mb_wave, ml_wave = get_waveforms(file_path, joint, coordinate)
    waveform = np.vstack([mb_wave, ml_wave])
    cmc_value = calculate_cmc([waveform])
    return cmc_value

def visualize_trial(file_path, joint='Ankle', coordinate='X'):
    """
    Visualize the waveform data for the specified joint and coordinate from the file,
    displaying the CMC value in the title.
    """
    mb_wave, ml_wave = get_waveforms(file_path, joint, coordinate)
    waveform = np.vstack([mb_wave, ml_wave])
    cmc_value = calculate_cmc([waveform])
    subject, motion = extract_subject_motion(file_path)
    
    frames = np.arange(len(mb_wave))
    plt.figure(figsize=(10, 6))
    plt.plot(frames, mb_wave, label='Marker-Based', marker='o', linestyle='-')
    plt.plot(frames, ml_wave, label='Markerless', marker='x', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel(f'{coordinate} Value')
    plt.title(f'Subject: {subject}, Motion: {motion}, Joint: {joint} | CMC: {cmc_value:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()

def aggregate_CMC(parent_folder, 
                  joints=['Left_Ankle', 'Left_Hip', 'Left_Knee', 'Right_Ankle', 'Right_Hip', 'Right_Knee', 'Trunk'], 
                  axes=['X', 'Y', 'Z']):
    """
    Calculate the CMC value for each joint and coordinate for all Excel files within parent_folder
    and return a DataFrame.
    Folder structure example: parent_folder/subject/motion/*.xlsx
    """
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    print(f"Found {len(trial_files)} trial files in '{parent_folder}'.")
    trial_files.sort()
    records = []
    
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        for joint in joints:
            for axis in axes:
                try:
                    cmc_val = process_trial(file, joint, axis)
                    records.append({
                        'subject': subject, 
                        'motion': motion, 
                        'joint': joint, 
                        'axis': axis, 
                        'trial_file': file, 
                        'cmc': cmc_val
                    })
                except Exception as e:
                    print(f"Error processing {file} (joint: {joint}, axis: {axis}): {e}")
    return pd.DataFrame(records)

def plot_aggregate_CMC(df):
    """
    Visualize the aggregated CMC values using boxplots grouped by subject, motion, joint, and axis.
    """
    for col, title in zip(['subject', 'motion', 'joint', 'axis'], 
                            ["CMC Distribution by Subject", "CMC Distribution by Motion", "CMC Distribution by Joint", "CMC Distribution by Axis"]):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=col, y='cmc', data=df)
        plt.title(title)
        plt.ylabel("CMC Value")
        plt.show()

def main():
    # Specify the parent folder path (multiple subject folders exist under this path)
    parent_folder = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check'
    
    do_visualization = True
    do_aggregate = True

    if do_aggregate:
        df_cmc = aggregate_CMC(parent_folder)
        print("Aggregated results for each trial:")
        print(df_cmc)
        
        if not df_cmc.empty:
            subject_summary = df_cmc.groupby('subject').agg(mean=('cmc', 'mean'), std=('cmc', 'std'), count=('trial_file', lambda x: x.nunique()))
            motion_summary = df_cmc.groupby('motion').agg(mean=('cmc', 'mean'), std=('cmc', 'std'), count=('trial_file', lambda x: x.nunique()))
            joint_summary = df_cmc.groupby('joint').agg(mean=('cmc', 'mean'), std=('cmc', 'std'), count=('trial_file', lambda x: x.nunique()))
            joint_axis_pivot = df_cmc.pivot_table(index='joint', columns='axis', values='cmc', aggfunc=[np.mean, np.std])
            
            print("\nAverage CMC by Subject:")
            print(subject_summary)
            print("\nAverage CMC by Motion:")
            print(motion_summary)
            print("\nAverage CMC by Joint:")
            print(joint_summary)
            print("\nAverage CMC by Joint and Axis:")
            print(joint_axis_pivot)
            
            # Save aggregated results to an Excel file (with formatting adjustments)
            output_excel_path = os.path.join(os.path.dirname(parent_folder), "cmc_aggregated_results.xlsx")
            with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
                df_cmc.to_excel(writer, sheet_name='Trial Data', index=False)
                subject_summary.to_excel(writer, sheet_name='Subject Summary')
                motion_summary.to_excel(writer, sheet_name='Motion Summary')
                joint_summary.to_excel(writer, sheet_name='Joint Summary')
                joint_axis_pivot.to_excel(writer, sheet_name='Joint Axis Summary')
                
                # Adjust column widths for the Trial Data sheet
                worksheet = writer.sheets['Trial Data']
                worksheet.set_column('A:A', 15)   # subject
                worksheet.set_column('B:B', 15)   # motion
                worksheet.set_column('C:C', 15)   # joint
                worksheet.set_column('D:D', 10)   # axis
                worksheet.set_column('E:E', 50)   # trial_file
                worksheet.set_column('F:F', 10)   # cmc

                # Adjust column widths for summary sheets
                for sheet in ['Subject Summary', 'Motion Summary', 'Joint Summary']:
                    ws = writer.sheets[sheet]
                    ws.set_column('A:A', 15)
                    ws.set_column('B:D', 12)
                writer.sheets['Joint Axis Summary'].set_column('A:A', 15)
                
            print(f"\nAggregated results have been saved to '{output_excel_path}'.")
            plot_aggregate_CMC(df_cmc)
        else:
            print("No trial files available for aggregation.")
    
    if do_visualization:
        trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
        trial_files.sort()
        if trial_files:
            # Example: Visualize the 'X' coordinate for all joints in the first trial file
            for joint in ['Left_Ankle', 'Left_Hip', 'Left_Knee', 'Right_Ankle', 'Right_Hip', 'Right_Knee', 'Trunk']:
                visualize_trial(trial_files[0], joint, coordinate='X')

if __name__ == '__main__':
    main()
