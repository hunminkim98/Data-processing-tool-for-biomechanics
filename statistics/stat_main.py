import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from CMC2 import calculate_cmc  # Import the calculate_cmc function from CMC2.py

def get_waveforms(file_path, coordinate='X'):
    """
    Reads the Excel file at the specified file path and extracts the marker-based and markerless waveform data.
    It then trims both waveforms to the length of the shorter one and returns them.
    
    Parameters:
      file_path: The file path to the trial .xlsx file.
      coordinate: The coordinate to compare (e.g., 'X', 'Y', 'Z').
    
    Returns:
      marker_based_wave, markerless_wave (numpy arrays)
    """
    df = pd.read_excel(file_path, header=[0, 1, 2])
    
    # Marker-based data: columns 1 to 3
    marker_based = df.iloc[:, 1:4]
    # Markerless data: columns 5 to 7
    markerless = df.iloc[:, 5:8]
    
    # Fixed index mapping: both marker-based and markerless are assumed to be in the order X, Y, Z
    coord_index = {"X": 0, "Y": 1, "Z": 2}.get(coordinate)
    if coord_index is None:
        raise ValueError(f"Unsupported coordinate: {coordinate}. Must be one of 'X', 'Y', or 'Z'.")
    
    marker_based_wave = marker_based.iloc[:, coord_index].values
    markerless_wave = markerless.iloc[:, coord_index].values
    
    # Trim the waveforms to the length of the shorter one
    common_length = min(len(marker_based_wave), len(markerless_wave))
    marker_based_wave = marker_based_wave[:common_length]
    markerless_wave = markerless_wave[:common_length]
    return marker_based_wave, markerless_wave

def process_trial(file_path, coordinate='X'):
    """
    Extracts marker-based and markerless waveforms from the given trial file and computes the CMC value.
    
    Parameters:
      file_path: The file path to the trial .xlsx file.
      coordinate: The coordinate to compare (e.g., 'X', 'Y', 'Z').
      
    Returns:
      The CMC value comparing the marker-based and markerless waveforms.
    """
    marker_based_wave, markerless_wave = get_waveforms(file_path, coordinate)
    waveform = np.vstack([marker_based_wave, markerless_wave])
    cmc_value = calculate_cmc([waveform])
    return cmc_value

def visualize_trial(file_path, coordinate='X'):
    """
    Reads the data from the given trial file, visualizes the marker-based and markerless waveforms,
    and displays the computed CMC value in the graph title.
    
    Parameters:
      file_path: The file path to the trial .xlsx file.
      coordinate: The coordinate to compare (e.g., 'X', 'Y', 'Z').
    """
    marker_based_wave, markerless_wave = get_waveforms(file_path, coordinate)
    waveform = np.vstack([marker_based_wave, markerless_wave])
    cmc_value = calculate_cmc([waveform])
    
    # Extract subject and motion information from the file path (folder structure: parent_folder/subject/motion/trial.xlsx)
    parts = os.path.normpath(file_path).split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else 'unknown'
    motion = parts[-2] if len(parts) >= 3 else 'unknown'
    
    frames = np.arange(len(marker_based_wave))
    plt.figure(figsize=(10, 6))
    plt.plot(frames, marker_based_wave, label='Marker-Based', marker='o', linestyle='-')
    plt.plot(frames, markerless_wave, label='Markerless', marker='x', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel(f'{coordinate} Value')
    plt.title(f'Subject: {subject}, Motion: {motion} | CMC: {cmc_value:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()

def aggregate_CMC(parent_folder, coordinate='X'):
    """
    Calculates the CMC values for all trial files under the specified parent folder and aggregates the results into a DataFrame.
    Folder structure example: parent_folder/subject/motion/*.xlsx
    
    Returns:
      pandas DataFrame: Each row corresponds to a trial and includes subject, motion, trial_file, and the CMC value.
    """
    # Find all .xlsx files located in the motion folders under each subject folder in the parent folder
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    trial_files.sort()
    records = []
    for file in trial_files:
        parts = os.path.normpath(file).split(os.sep)
        subject = parts[-3] if len(parts) >= 3 else 'unknown'
        motion = parts[-2] if len(parts) >= 3 else 'unknown'
        try:
            cmc_val = process_trial(file, coordinate)
            records.append({'subject': subject, 'motion': motion, 'trial_file': file, 'cmc': cmc_val})
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return pd.DataFrame(records)

def plot_aggregate_CMC(df):
    """
    Generates boxplots for the aggregated CMC results by subject and motion.
    
    Parameters:
      df: The pandas DataFrame returned by aggregate_CMC().
    """
    # CMC distribution by subject
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='subject', y='cmc', data=df)
    plt.title("CMC Distribution by Subject")
    plt.ylabel("CMC Value")
    plt.show()
    
    # CMC distribution by motion
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='motion', y='cmc', data=df)
    plt.title("CMC Distribution by Motion")
    plt.ylabel("CMC Value")
    plt.show()

if __name__ == '__main__':
    # Specify the parent folder path (multiple subject folders exist under this path)
    parent_folder = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\test'
    
    # The following block uses a correct glob pattern to locate xlsx trial files for visualization
    do_visualization = False
    do_aggregate = True

    if do_aggregate:
        df_cmc = aggregate_CMC(parent_folder, coordinate='X')
        print("Aggregated results for each trial:")
        print(df_cmc)
        if not df_cmc.empty:
            print("\nAverage CMC per subject:")
            print(df_cmc.groupby('subject')['cmc'].agg(['mean', 'std', 'count']))
            print("\nAverage CMC per motion:")
            print(df_cmc.groupby('motion')['cmc'].agg(['mean', 'std', 'count']))
            # Visualize aggregated results (boxplots)
            plot_aggregate_CMC(df_cmc)
        else:
            print("No trial files found for aggregation.")

    if do_visualization:
        # Visualize individual trial (set to True if needed)
        trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
        trial_files.sort()
        for file in trial_files:
            visualize_trial(file, coordinate='X')