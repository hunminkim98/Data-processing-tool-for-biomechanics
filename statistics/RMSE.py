import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from stat_main import get_waveforms, extract_subject_motion
from scipy import stats

# 좌우 관절 매핑 정의
JOINT_PAIRS = {
    'Ankle': ['Left_Ankle', 'Right_Ankle'],
    'Knee': ['Left_Knee', 'Right_Knee'],
    'Hip': ['Left_Hip', 'Right_Hip'],
    'Trunk': ['Trunk']
}

def calculate_rmse(true_values, predicted_values):
    """Calculate Root Mean Square Error between two arrays"""
    return np.sqrt(np.mean((true_values - predicted_values)**2))

def process_trial_waveforms(file_path, joint='Ankle', coordinate='X'):
    """Process trial file to get averaged marker-based and markerless waveforms for paired joints"""
    paired_joints = JOINT_PAIRS.get(joint, [])
    if not paired_joints:
        raise ValueError(f"Unsupported joint: {joint}")
    
    mb_waves = []
    ml_waves = []
    
    for j in paired_joints:
        try:
            mb_wave, ml_wave = get_waveforms(file_path, j, coordinate)
            mb_waves.append(mb_wave)
            ml_waves.append(ml_wave)
        except Exception as e:
            print(f"Error loading {j} from {file_path}: {e}")
            return None, None
    
    min_length = min(len(wave) for wave in mb_waves + ml_waves)
    mb_waves = [wave[:min_length] for wave in mb_waves]
    ml_waves = [wave[:min_length] for wave in ml_waves]
    
    mb_avg = np.mean(mb_waves, axis=0)
    ml_avg = np.mean(ml_waves, axis=0)
    
    return mb_avg, ml_avg

def apply_iqr_rule(data):
    """
    IQR 2.0 rule을 적용하여 이상치를 식별합니다.
    data: 2D numpy array (trials x segments)
    return: 정상 데이터, 이상치 마스크
    """
    if len(data) == 0:
        return data, np.array([])
    
    # 각 trial의 모든 segment RMSE를 일렬로 나열
    flattened_data = data.flatten()
    
    # IQR 계산
    q1 = np.percentile(flattened_data, 25)
    q3 = np.percentile(flattened_data, 75)
    iqr = q3 - q1
    
    # 경계값 계산
    lower_bound = q1 - 2.0 * iqr
    upper_bound = q3 + 2.0 * iqr
    
    # 이상치 마스크 생성
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    inlier_data = np.where(outlier_mask, np.nan, data)
    
    return inlier_data, outlier_mask

def calculate_segment_rmse(mb_wave, ml_wave, segment_size=10):
    """Calculate RMSE for each segment (10 frames)"""
    if len(mb_wave) != 101 or len(ml_wave) != 101:
        raise ValueError("입력 데이터는 반드시 101개 프레임이어야 합니다.")
    rmse_segments = []
    for i in range(0, 100, segment_size):
        start = i + 1
        end = i + segment_size + 1
        segment_rmse = calculate_rmse(mb_wave[start:end], ml_wave[start:end])
        rmse_segments.append(segment_rmse)
    return rmse_segments

def perform_statistical_test(rmse_data, alpha=0.05):
    """Perform t-test for each segment to identify significant differences"""
    significant_segments = []
    for segment_idx in range(rmse_data.shape[1]):
        segment_rmse = rmse_data[:, segment_idx]
        segment_rmse = segment_rmse[~np.isnan(segment_rmse)]  # NaN 제거 (이상치 제외)
        if len(segment_rmse) > 1:  # 통계 검정을 위해 데이터가 충분해야 함
            t_stat, p_val = stats.ttest_1samp(segment_rmse, popmean=0, nan_policy='omit')
            if p_val < alpha:
                significant_segments.append(segment_idx)
    return significant_segments

def aggregate_segment_rmse(parent_folder, joints=['Ankle', 'Knee', 'Hip', 'Trunk'], axes=['X', 'Y', 'Z']):
    """Aggregate segment-wise RMSE values across all trials, grouped by motion"""
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    
    all_rmse = {}
    raw_rmse = {}  # 이상치 포함 원본 데이터 저장
    
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        if motion not in all_rmse:
            all_rmse[motion] = {joint: {axis: [] for axis in axes} for joint in joints}
            raw_rmse[motion] = {joint: {axis: [] for axis in axes} for joint in joints}
        
        for joint in joints:
            for axis in axes:
                try:
                    mb_wave, ml_wave = process_trial_waveforms(file, joint, axis)
                    if mb_wave is not None and ml_wave is not None:
                        rmse_segments = calculate_segment_rmse(mb_wave, ml_wave)
                        all_rmse[motion][joint][axis].append(rmse_segments)
                        raw_rmse[motion][joint][axis].append(rmse_segments)
                except Exception as e:
                    print(f"Error processing {file} (joint: {joint}, axis: {axis}): {e}")
    
    # IQR 적용 및 이상치 제거
    for motion in all_rmse:
        for joint in joints:
            for axis in axes:
                if all_rmse[motion][joint][axis]:
                    data = np.array(all_rmse[motion][joint][axis])
                    inlier_data, outlier_mask = apply_iqr_rule(data)
                    all_rmse[motion][joint][axis] = inlier_data
                    raw_rmse[motion][joint][axis] = np.array(raw_rmse[motion][joint][axis])
    
    return all_rmse, raw_rmse

def plot_rmse(all_rmse, output_dir, joints=['Ankle', 'Knee', 'Hip', 'Trunk'], axes=['X', 'Y', 'Z'], alpha=0.05):
    """Plot average RMSE with error bars for each motion, and save all graphs"""
    rmse_data_list = []
    
    for motion, motion_data in all_rmse.items():
        for joint in joints:
            for axis in axes:
                rmse_data = motion_data[joint][axis]
                if rmse_data.size == 0:
                    continue
                mean_rmse = np.nanmean(rmse_data, axis=0)
                std_rmse = np.nanstd(rmse_data, axis=0)
                segments = np.arange(0, 100, 10)
                
                # 통계적 유의미성 검정
                significant_segments = perform_statistical_test(rmse_data, alpha=alpha)
                
                # 엑셀 데이터에 Motion 열 추가
                for seg_idx, (seg, mean, std) in enumerate(zip(segments, mean_rmse, std_rmse)):
                    rmse_data_list.append({
                        'Motion': motion,
                        'Joint': joint,
                        'Axis': axis,
                        'Segment': f'{seg}-{seg+10}%',
                        'Mean_RMSE': mean,
                        'Std_RMSE': std
                    })
                
                plt.figure(figsize=(10, 6))
                plt.errorbar(segments, mean_rmse, yerr=std_rmse, fmt='-o', capsize=5, label='Average RMSE ± Std')
                
                # 유의미한 구간 표시
                for seg_idx in significant_segments:
                    plt.scatter(segments[seg_idx], mean_rmse[seg_idx], color='red', marker='*', s=200, 
                               label='Significant (p<0.05)' if seg_idx == significant_segments[0] else "")
                
                plt.xlabel('Phase (%)')
                plt.ylabel('Average RMSE')
                plt.title(f'Average RMSE for {motion} - {joint} - {axis} axis')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                
                plot_filename = f'RMSE_{motion}_{joint}_{axis}.png'
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()
    
    df = pd.DataFrame(rmse_data_list)
    excel_path = os.path.join(output_dir, 'segment_wise_RMSE_by_motion.xlsx')
    df.to_excel(excel_path, index=False)
    return df

def plot_rmse_outliers(raw_rmse, output_dir):
    """모든 RMSE를 scatter plot으로 그리고 IQR에 의해 제거될 이상치는 빨간색으로 표시"""
    plt.figure(figsize=(15, 8))
    
    for motion in raw_rmse:
        for joint in raw_rmse[motion]:
            for axis in raw_rmse[motion][joint]:
                if raw_rmse[motion][joint][axis].size > 0:
                    data = raw_rmse[motion][joint][axis]
                    original_data, outlier_mask = apply_iqr_rule(data)
                    
                    segments = np.arange(0, 100, 10)
                    for seg_idx in range(len(segments)):
                        seg_data = original_data[:, seg_idx]
                        seg_mask = outlier_mask[:, seg_idx]
                        
                        # 정상 데이터 플롯 (파란색)
                        normal_data = seg_data[~seg_mask]
                        plt.scatter([segments[seg_idx]] * len(normal_data), normal_data, 
                                  c='blue', alpha=0.5, s=30)
                        
                        # 이상치 플롯 (빨간색)
                        outlier_data = seg_data[seg_mask]
                        plt.scatter([segments[seg_idx]] * len(outlier_data), outlier_data, 
                                  c='red', alpha=0.5, s=30)
    
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('RMSE')
    plt.title('RMSE Distribution with Outliers (Red)')
    plt.grid(True, alpha=0.3)
    
    plt.scatter([], [], c='blue', alpha=0.5, label='Normal Data')
    plt.scatter([], [], c='red', alpha=0.5, label='Outliers')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'rmse_outliers.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        parent_folder = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\_normalized\merged_check_interpolated'
        output_dir = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\results'
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_rmse, raw_rmse = aggregate_segment_rmse(parent_folder)
        
        df = plot_rmse(all_rmse, output_dir)
        plot_rmse_outliers(raw_rmse, output_dir)
        
        print(f"Results have been saved to {output_dir}")
        print(f"Excel file has been saved as: {os.path.join(output_dir, 'segment_wise_RMSE_by_motion.xlsx')}")
    
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == '__main__':
    main()