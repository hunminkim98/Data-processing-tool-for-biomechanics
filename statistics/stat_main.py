import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from CMC import calculate_cmc  # CMC.py에서 calculate_cmc 함수 가져오기
from collections import defaultdict
from scipy.interpolate import CubicSpline

# 한글 폰트 설정 및 음수 기호 표시 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
# Mac용: plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 각 관절에 대한 시트 인덱스 매핑 (전역 상수)
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
    파일 경로에서 피험자와 동작 정보를 추출합니다.
    예: parent_folder/subject/motion/trial.xlsx
    """
    parts = os.path.normpath(file_path).split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else 'unknown'
    motion = parts[-2] if len(parts) >= 3 else 'unknown'
    return subject, motion

def get_waveforms(file_path, joint='Ankle', coordinate='X', SPM=False, filtering=False):
    """
    지정된 파일에서 관절 및 좌표 데이터를 읽어 마커 기반 및 마커리스 파형 데이터를 추출합니다.
    필터링에는 표준 모션 캡처 프레임 속도인 120Hz를 사용합니다.
    """
    subject, motion = extract_subject_motion(file_path)
    
    # 제외 기준 적용
    if subject == "성기훈" and motion == "swing" and joint == "Left_Knee":
        raise ValueError(f"Excluding {joint} {coordinate}-axis data for subject 성기훈 in swing motion")
    if subject == "김리언" and motion == "swing" and "Ankle" in joint:
        raise ValueError(f"Excluding {joint} {coordinate}-axis data for subject 김리언 in swing motion")
    if subject == "김태형" and motion == "swing" and "Trunk" in joint:
        raise ValueError(f"Excluding {joint} {coordinate}-axis data for subject 김태형 in swing motion")
    
    sheet_idx = JOINT_SHEET_MAPPING.get(joint)
    if sheet_idx is None:
        raise ValueError(f"Unsupported joint: {joint}. Available options: {list(JOINT_SHEET_MAPPING.keys())}.")
    
    df = pd.read_excel(file_path, header=[0, 1, 2], sheet_name=sheet_idx)
    
    fs = 120  # Hz (표준 모션 캡처 프레임 속도)
    
    marker_based = df.iloc[:, 1:4]
    markerless = df.iloc[:, 5:8]
    
    coord_idx = {"X": 0, "Y": 1, "Z": 2}.get(coordinate)
    if coord_idx is None:
        raise ValueError(f"Unsupported coordinate: {coordinate}. Available options: 'X', 'Y', 'Z'.")
    
    marker_based_wave = marker_based.iloc[:, coord_idx].values
    markerless_wave = markerless.iloc[:, coord_idx].values

    len_mb, len_ml = len(marker_based_wave), len(markerless_wave)
    common_length = min(len_mb, len_ml)
    marker_based_wave = marker_based_wave[:common_length]
    markerless_wave = markerless_wave[:common_length]
    
    mask = ~np.isnan(marker_based_wave) & ~np.isnan(markerless_wave)
    if not np.all(mask):
        marker_based_wave = pd.Series(marker_based_wave).interpolate().values
        markerless_wave = pd.Series(markerless_wave).interpolate().values
    
    if marker_based_wave.size == 0:
        raise ValueError(f"All data for joint {joint} in file {file_path} has been removed due to NaN values.")
    
    if SPM:
        from scipy import stats
        _, p_value_mb = stats.shapiro(marker_based_wave)
        _, p_value_ml = stats.shapiro(markerless_wave)
        
        normality_threshold = 0.05
        is_normal = p_value_mb >= normality_threshold and p_value_ml >= normality_threshold
        
        if not is_normal:
            print(f"Warning: Data not normally distributed (p-values: marker-based={p_value_mb:.4f}, markerless={p_value_ml:.4f})")
            print("Applying robust preprocessing steps...")
            
            window_size = 5
            marker_based_wave = signal.medfilt(marker_based_wave, window_size)
            markerless_wave = signal.medfilt(markerless_wave, window_size)
    
    if filtering:
        fc = 6
        w = fc / (fs / 2)
        b, a = signal.butter(4, w, 'low')
        
        marker_based_filtered = signal.filtfilt(b, a, marker_based_wave)
        markerless_filtered = signal.filtfilt(b, a, markerless_wave)
    else:
        marker_based_filtered = marker_based_wave
        markerless_filtered = markerless_wave
    
    original_time = np.linspace(0, 100, len(marker_based_filtered))
    cs_mb = CubicSpline(original_time, marker_based_filtered)
    cs_ml = CubicSpline(original_time, markerless_filtered)
    
    new_indices = np.linspace(0, 100, 101)
    marker_based_normalized = cs_mb(new_indices)
    markerless_normalized = cs_ml(new_indices)
    
    return marker_based_normalized, markerless_normalized

def process_trial(file_path, joint='Ankle', coordinate='X'):
    """
    지정된 파일에서 파형 데이터를 추출하고 CMC 값을 계산합니다.
    """
    mb_wave, ml_wave = get_waveforms(file_path, joint, coordinate)
    waveform = np.vstack([mb_wave, ml_wave])
    cmc_value = calculate_cmc([waveform])  # calculate_cmc는 파형 리스트를 기대합니다
    return cmc_value

def visualize_trial(file_path, joint='Ankle', coordinate='X'):
    """
    지정된 관절 및 좌표에 대한 파형 데이터를 시각화하며, 제목에 CMC 값을 표시합니다.
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

def remove_outliers(df, group_cols, value_col='cmc', multiplier=3):
    def outlier_mask(group):
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (group[value_col] >= lower_bound) & (group[value_col] <= upper_bound)
    
    # Add include_groups=False to suppress the DeprecationWarning
    mask = df.groupby(group_cols).apply(outlier_mask, include_groups=False).reset_index(level=group_cols, drop=True)
    return df[mask]


def aggregate_CMC(parent_folder, 
                  joints=['Left_Ankle', 'Left_Hip', 'Left_Knee', 'Right_Ankle', 'Right_Hip', 'Right_Knee', 'Trunk'], 
                  axes=['X', 'Y', 'Z']):
    """
    각 트라이얼에 대해 CMC를 계산하고, IQR 3rule로 이상치를 제거한 후 결과를 집계합니다.
    
    Parameters:
    - parent_folder: 트라이얼 파일이 있는 상위 폴더 경로
    - joints: 분석할 관절 리스트
    - axes: 분석할 축 리스트
    
    Returns:
    - df_before: 이상치 제거 전 데이터프레임
    - df_cmc: 이상치 제거 후 데이터프레임
    - df_outliers: 제거된 이상치 데이터프레임
    """
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    print(f"Found {len(trial_files)} trial files in '{parent_folder}'.")
    
    records = []
    
    # CMC 값 추출
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        for joint in joints:
            for axis in axes:
                try:
                    cmc_val = process_trial(file, joint, axis)
                    cmc_val = cmc_val.real if isinstance(cmc_val, complex) else cmc_val
                    records.append({
                        'motion': motion,
                        'joint': joint,
                        'axis': axis,
                        'trial_file': file,
                        'cmc': cmc_val
                    })
                except Exception as e:
                    print(f"Error processing {file} (joint: {joint}, axis: {axis}): {e}")
    
    df_before = pd.DataFrame(records)
    
    if not df_before.empty:
        # 이상치 제거
        df_cmc = remove_outliers(df_before, group_cols=['motion', 'joint', 'axis'], multiplier=3)
        # 제거된 이상치 추출
        df_outliers = df_before[~df_before.index.isin(df_cmc.index)]
        
        # 왼쪽/오른쪽 관절 평균 계산 (이상치 제거 후 데이터 사용)
        joint_pairs = {}
        for joint in df_cmc['joint'].unique():
            if joint.startswith('Left_'):
                right_joint = 'Right_' + joint[5:]
                if right_joint in df_cmc['joint'].unique():
                    joint_pairs[joint] = right_joint
        
        avg_records = []
        for trial_file in df_cmc['trial_file'].unique():
            trial_df = df_cmc[df_cmc['trial_file'] == trial_file]
            for axis in axes:
                for left_joint, right_joint in joint_pairs.items():
                    left_cmc = trial_df[(trial_df['joint'] == left_joint) & (trial_df['axis'] == axis)]['cmc']
                    right_cmc = trial_df[(trial_df['joint'] == right_joint) & (trial_df['axis'] == axis)]['cmc']
                    if not left_cmc.empty and not right_cmc.empty:
                        avg_cmc = (left_cmc.values[0] + right_cmc.values[0]) / 2
                        base_joint = left_joint[5:]
                        avg_records.append({
                            'motion': trial_df['motion'].iloc[0],
                            'joint': f'Avg_{base_joint}',
                            'axis': axis,
                            'trial_file': trial_file,
                            'cmc': avg_cmc
                        })
        
        if avg_records:
            df_avg = pd.DataFrame(avg_records)
            # 평균값에 대해서도 이상치 제거
            df_avg_clean = remove_outliers(df_avg, group_cols=['motion', 'joint', 'axis'], multiplier=3)
            df_cmc = pd.concat([df_cmc, df_avg_clean], ignore_index=True)
    
    else:
        df_outliers = pd.DataFrame(columns=df_before.columns)
        df_cmc = pd.DataFrame(columns=df_before.columns)
    
    return df_before, df_cmc, df_outliers


def visualize_outliers(df_before, df_after, df_outliers):
    """
    이상치 제거 전후의 CMC 분포를 시각화합니다.
    
    Parameters:
    - df_before: 이상치 제거 전 데이터프레임
    - df_after: 이상치 제거 후 데이터프레임
    - df_outliers: 제거된 이상치 데이터프레임
    """
    for motion in df_before['motion'].unique():
        for joint in df_before['joint'].unique():
            for axis in df_before['axis'].unique():
                # 특정 motion, joint, axis에 대한 데이터 필터링
                before = df_before[(df_before['motion'] == motion) & 
                                 (df_before['joint'] == joint) & 
                                 (df_before['axis'] == axis)]
                after = df_after[(df_after['motion'] == motion) & 
                               (df_after['joint'] == joint) & 
                               (df_after['axis'] == axis)]
                outliers = df_outliers[(df_outliers['motion'] == motion) & 
                                     (df_outliers['joint'] == joint) & 
                                     (df_outliers['axis'] == axis)]
                
                if not before.empty:
                    plt.figure(figsize=(10, 6))
                    # 제거 전 박스플롯 (파란색)
                    sns.boxplot(data=before, x='joint', y='cmc', color='lightblue', showfliers=False)
                    # 제거 후 박스플롯 (초록색)
                    sns.boxplot(data=after, x='joint', y='cmc', color='lightgreen', showfliers=False)
                    # 제거된 이상치 (빨간색 x)
                    if not outliers.empty:
                        sns.scatterplot(data=outliers, x='joint', y='cmc', color='red', 
                                      marker='x', s=100, label='Outliers')
                    plt.title(f'CMC Distribution for {motion} - {joint} - {axis}')
                    plt.ylabel('CMC Value')
                    plt.xlabel('Joint')
                    plt.legend()
                    plt.show()


def plot_aggregate_CMC(df):
    """
    모션별로 관절과 축에 따른 CMC 분포를 시각화합니다.
    """
    for motion in df['motion'].unique():
        motion_df = df[df['motion'] == motion]
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='joint', y='cmc', hue='axis', data=motion_df)
        plt.title(f"CMC Distribution by Joint and Axis for Motion: {motion}")
        plt.ylabel("CMC Value")
        plt.xlabel("Joint")
        plt.legend(title='Axis')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    parent_folder = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check_interpolated'
    
    do_visualization = True
    do_aggregate = True

    if do_aggregate:
        df_before, df_cmc, df_outliers = aggregate_CMC(parent_folder)
        visualize_outliers(df_before, df_cmc, df_outliers)
        print("CMC results for each trial (after outlier removal):")
        print(df_cmc)
        
        if not df_cmc.empty:
            joint_axis_summary = df_cmc.groupby(['motion', 'joint', 'axis']).agg(
                mean_cmc=('cmc', 'mean'),
                std_cmc=('cmc', 'std'),
                trial_count=('trial_file', 'nunique')
            ).reset_index()
            
            print("\nAverage CMC and SD by Motion, Joint and Axis:")
            print(joint_axis_summary)
            
            output_excel_path = os.path.join(os.path.dirname(parent_folder), "cmc_aggregated_results.xlsx")
            with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
                df_cmc.to_excel(writer, sheet_name='Combined Analysis', index=False)
                joint_axis_summary.to_excel(writer, sheet_name='Summary', index=False)
                # 열 너비 조정
                worksheet = writer.sheets['Combined Analysis']
                worksheet.set_column('A:A', 15)  # motion
                worksheet.set_column('B:B', 15)  # joint
                worksheet.set_column('C:C', 10)  # axis
                worksheet.set_column('D:D', 50)  # trial_file
                worksheet.set_column('E:E', 10)  # cmc
                writer.sheets['Summary'].set_column('A:A', 15)  # motion
                writer.sheets['Summary'].set_column('B:B', 15)  # joint
                writer.sheets['Summary'].set_column('C:C', 10)  # axis
                writer.sheets['Summary'].set_column('D:F', 12)  # 통계값
            
            plot_aggregate_CMC(df_cmc)
        else:
            print("No trial files available for aggregation.")
    
    if do_visualization:
        trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
        trial_files.sort()
        if trial_files:
            for joint in ['Left_Ankle', 'Left_Hip', 'Left_Knee', 'Right_Ankle', 'Right_Hip', 'Right_Knee', 'Trunk']:
                visualize_trial(trial_files[0], joint, coordinate='X')

if __name__ == '__main__':
    main()