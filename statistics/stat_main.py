import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from CMC import calculate_cmc  # Import the calculate_cmc function
from collections import defaultdict
from scipy.interpolate import CubicSpline

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
    Uses standard motion capture frame rate of 120Hz for filtering.
    """
    sheet_idx = JOINT_SHEET_MAPPING.get(joint)
    if sheet_idx is None:
        raise ValueError(f"Unsupported joint: {joint}. Available options: {list(JOINT_SHEET_MAPPING.keys())}.")
    
    # Read the Excel file
    df = pd.read_excel(file_path, header=[0, 1, 2], sheet_name=sheet_idx)
    
    # Use standard motion capture frame rate
    fs = 120  # Hz (standard motion capture frame rate)
    
    # Extract marker-based and markerless data
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
    
    # Remove NaN values while preserving temporal order
    mask = ~np.isnan(marker_based_wave) & ~np.isnan(markerless_wave)
    if not np.all(mask):
        print(f"Warning: In file '{file_path}', joint {joint} and coordinate {coordinate} had {np.count_nonzero(~mask)} NaN values removed.")
        # Instead of removing NaN values, interpolate them
        marker_based_wave = pd.Series(marker_based_wave).interpolate().values
        markerless_wave = pd.Series(markerless_wave).interpolate().values
    
    if marker_based_wave.size == 0:
        raise ValueError(f"All data for joint {joint} in file {file_path} has been removed due to NaN values.")
    
    # Check for normality before filtering
    from scipy import stats
    _, p_value_mb = stats.shapiro(marker_based_wave)
    _, p_value_ml = stats.shapiro(markerless_wave)
    
    normality_threshold = 0.05
    is_normal = p_value_mb >= normality_threshold and p_value_ml >= normality_threshold
    
    if not is_normal:
        print(f"Warning: Data not normally distributed (p-values: marker-based={p_value_mb:.4f}, markerless={p_value_ml:.4f})")
        print("Applying robust preprocessing steps...")
        
        # Use median filtering for non-normal data
        window_size = 5  # Adjust as needed
        marker_based_wave = signal.medfilt(marker_based_wave, window_size)
        markerless_wave = signal.medfilt(markerless_wave, window_size)
    
    # Apply 6Hz lowpass filter with standard frame rate
    fc = 6    # cutoff frequency in Hz
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(4, w, 'low')
    
    marker_based_filtered = signal.filtfilt(b, a, marker_based_wave)
    markerless_filtered = signal.filtfilt(b, a, markerless_wave)
    
    # Calculate actual sampling rate based on time stamps if available
    try:
        time_stamps = df.index.values
        actual_fs = 1 / np.mean(np.diff(time_stamps))
        if abs(actual_fs - fs) > 5:  # If difference is more than 5 Hz
            print(f"Warning: Actual sampling rate ({actual_fs:.1f} Hz) differs from standard rate ({fs} Hz)")
    except:
        print("Warning: Could not calculate actual sampling rate from time stamps")
    
    # Normalize to 101 points using cubic spline interpolation
    # First, create time vector based on actual sampling rate
    original_time = np.linspace(0, 100, len(marker_based_filtered))
    cs_mb = CubicSpline(original_time, marker_based_filtered)
    cs_ml = CubicSpline(original_time, markerless_filtered)
    
    new_indices = np.linspace(0, 100, 101)
    marker_based_normalized = cs_mb(new_indices)
    markerless_normalized = cs_ml(new_indices)
    
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
    모든 trials의 웨이브폰을 통합하여 CMC 계산
    """
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    print(f"Found {len(trial_files)} trial files in '{parent_folder}'.")
    
    records = []
    
    # 관절별/축별로 모든 trials의 웨이브폰 수집
    waveform_dict = defaultdict(list)
    
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        for joint in joints:
            for axis in axes:
                try:
                    mb_wave, ml_wave = get_waveforms(file, joint, axis)
                    combined_wave = np.vstack([mb_wave, ml_wave])
                    waveform_dict[(joint, axis)].append(combined_wave)
                except Exception as e:
                    print(f"Error processing {file} (joint: {joint}, axis: {axis}): {e}")
    
    # 통합 웨이브폰으로 CMC 계산
    for (joint, axis), waveforms in waveform_dict.items():
        if len(waveforms) < 2:
            print(f"Warning: Insufficient waveforms for {joint}-{axis} (n={len(waveforms)})")
            continue
            
        try:
            # 모든 trials의 웨이브폰을 하나의 gait_cycles 리스트로 결합
            cmc_val = calculate_cmc(waveforms)
            
            # 실수부만 추출 (복소수 결과 방지)
            cmc_val = cmc_val.real if isinstance(cmc_val, complex) else cmc_val
            
            records.append({
                'joint': joint,
                'axis': axis,
                'num_trials': len(waveforms),
                'cmc': cmc_val
            })
        except Exception as e:
            print(f"Error calculating CMC for {joint}-{axis}: {e}")
    
    return pd.DataFrame(records)

def plot_aggregate_CMC(df):
    """
    시각화 함수 수정
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='joint', y='cmc', hue='axis', data=df)
    plt.title("CMC Distribution by Joint and Axis")
    plt.ylabel("CMC Value")
    plt.legend(title='Axis')
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
            # 새로운 요약 방식으로 변경
            joint_axis_summary = df_cmc.groupby(['joint', 'axis']).agg(
                mean_cmc=('cmc', 'mean'),
                std_cmc=('cmc', 'std'),
                trial_count=('num_trials', 'sum')
            ).reset_index()
            
            print("\nAverage CMC by Joint and Axis:")
            print(joint_axis_summary)
            
            # Excel 저장 부분도 수정
            output_excel_path = os.path.join(os.path.dirname(parent_folder), "cmc_aggregated_results.xlsx")
            with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
                df_cmc.to_excel(writer, sheet_name='Combined Analysis', index=False)
                joint_axis_summary.to_excel(writer, sheet_name='Summary', index=False)
                
                # 컬럼 너비 조정
                worksheet = writer.sheets['Combined Analysis']
                worksheet.set_column('A:A', 15)  # joint
                worksheet.set_column('B:B', 10)  # axis
                worksheet.set_column('C:C', 12)  # num_trials
                worksheet.set_column('D:D', 10)  # cmc
                
                writer.sheets['Summary'].set_column('A:B', 15)
                writer.sheets['Summary'].set_column('C:E', 12)
            
            plot_aggregate_CMC(df_cmc)  # 이 함수도 수정 필요
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
