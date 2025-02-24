import numpy as np
from stat_main import get_waveforms, extract_subject_motion
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 좌우 관절 매핑 정의
JOINT_PAIRS = {
    'Ankle': ['Left_Ankle', 'Right_Ankle'],
    'Knee': ['Left_Knee', 'Right_Knee'],
    'Hip': ['Left_Hip', 'Right_Hip'],
    'Trunk': ['Trunk']  # 단일 관절은 리스트에 하나만 포함
}

def calculate_rmse(true_values, predicted_values):
    """Calculate Root Mean Square Error between two arrays"""
    return np.sqrt(np.mean((true_values - predicted_values)**2))

def process_trial_rmse(file_path, joint='Ankle', coordinate='X'):
    """Process trial file to calculate RMSE between averaged marker-based and markerless data for paired joints"""
    paired_joints = JOINT_PAIRS.get(joint, [])
    if not paired_joints:
        raise ValueError(f"Unsupported joint: {joint}")
    
    mb_waves = []
    ml_waves = []
    
    # 좌우 관절 데이터 로드
    for j in paired_joints:
        try:
            mb_wave, ml_wave = get_waveforms(file_path, j, coordinate)
            mb_waves.append(mb_wave)
            ml_waves.append(ml_wave)
        except Exception as e:
            print(f"Error loading {j} from {file_path}: {e}")
            return None
    
    # 데이터 길이 맞추기 (최소 길이 기준)
    min_length = min(len(wave) for wave in mb_waves + ml_waves)
    mb_waves = [wave[:min_length] for wave in mb_waves]
    ml_waves = [wave[:min_length] for wave in ml_waves]
    
    # 좌우 데이터 평균화
    mb_avg = np.mean(mb_waves, axis=0)
    ml_avg = np.mean(ml_waves, axis=0)
    
    # 평균화된 데이터로 RMSE 계산
    return calculate_rmse(mb_avg, ml_avg)

def aggregate_rmse(parent_folder, 
                  joints=['Ankle', 'Knee', 'Hip', 'Trunk'], 
                  axes=['X', 'Y', 'Z']):
    """Aggregate RMSE values across all trials and joints"""
    trial_files = glob.glob(os.path.join(parent_folder, '*', '*', '*.xlsx'))
    
    records = []
    
    for file in trial_files:
        subject, motion = extract_subject_motion(file)
        for joint in joints:
            for axis in axes:
                try:
                    rmse_value = process_trial_rmse(file, joint, axis)
                    if rmse_value is not None:
                        records.append({
                            'subject': subject,
                            'motion': motion,
                            'joint': joint,  # 평균화된 관절 이름 사용
                            'axis': axis,
                            'rmse': rmse_value
                        })
                except Exception as e:
                    print(f"Error processing {file} (joint: {joint}, axis: {axis}): {e}")
    
    return pd.DataFrame(records)

def identify_outliers_iqr(df, motion, threshold=2.0):
    """Non-parametric outlier detection using interquartile range
    
    Args:
        df: DataFrame containing RMSE data
        motion: Specific motion category to analyze
        threshold: IQR multiplier (default=2)
    
    Returns:
        Boolean Series indicating non-outlier values
    """
    motion_group = df[df['motion'] == motion]
    
    if len(motion_group) < 4:  # Require minimum 4 data points
        return pd.Series([True]*len(df), index=df.index)
    
    Q1 = motion_group['rmse'].quantile(0.25)
    Q3 = motion_group['rmse'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return df['rmse'].between(lower_bound, upper_bound, inclusive='both')

def visualize_outliers(df, output_dir):
    """Visualize RMSE distribution with outliers highlighted"""
    plt.figure(figsize=(14, 8))
    
    # Use strip plot with optimized parameters
    ax = sns.stripplot(x='motion', y='rmse', hue='outlier', data=df,
                      palette={True: 'red', False: 'blue'},
                      size=3, jitter=0.3, alpha=0.7, 
                      dodge=False, linewidth=0.3)
    
    # Add IQR box visualization
    sns.boxplot(x='motion', y='rmse', data=df, showfliers=False,
               boxprops={'facecolor':'None', 'linewidth': 1.2}, 
               whiskerprops={'linewidth': 1.2},
               ax=ax)
    
    plt.title('RMSE Distribution (IQR 3σ Rule)')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE Value')
    plt.xlabel('Motion Type')
    plt.tight_layout()
    
    # Save and show
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(output_dir, f'outlier_visualization_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    return plot_path

def calculate_average_rmse(df):
    """Calculate average RMSE values for each motion, joint, and axis combination after outlier removal"""
    # Group by motion, joint, and axis to calculate mean and std
    summary = df.groupby(['motion', 'joint', 'axis'])['rmse'].agg(['mean', 'std']).reset_index()
    
    # Format the results as "mean ± std" with 2 decimal places
    summary['result'] = summary['mean'].round(2).astype(str) + ' ± ' + summary['std'].round(2).astype(str)
    
    # Create pivot tables for each motion
    motion_tables = {}
    for motion in summary['motion'].unique():
        motion_data = summary[summary['motion'] == motion]
        pivot = motion_data.pivot(index='joint', columns='axis', values='result')
        motion_tables[motion] = pivot
    
    return motion_tables

def main():
    try:
        # Configure output paths
        results_dir = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\results'
        data_source = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\_normalized\merged_check_interpolated'
        
        # Create results directory if needed
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(results_dir, f'RMSE_Results_{timestamp}.xlsx')
        
        # Aggregate RMSE values
        df = aggregate_rmse(data_source)
        
        # Add outlier detection
        df['outlier'] = False
        for motion in df['motion'].unique():
            motion_mask = df['motion'] == motion
            is_outlier = ~identify_outliers_iqr(df, motion)
            df.loc[motion_mask & is_outlier, 'outlier'] = True
        
        # Create cleaned dataset (non-outliers only)
        clean_df = df[~df['outlier']].copy()
        
        # Calculate average RMSE values for each motion
        motion_tables = calculate_average_rmse(clean_df)
        
        # Save to Excel with formatting
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Save full data with outliers
            df.to_excel(writer, index=False, sheet_name='All Data')
            
            # Save cleaned data
            clean_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            # Save average RMSE values for each motion
            for motion, pivot_table in motion_tables.items():
                sheet_name = f'{motion} Summary'
                pivot_table.to_excel(writer, sheet_name=sheet_name)
            
            workbook = writer.book
            
            # Format trial data sheets
            for sheet_name in ['All Data', 'Cleaned Data']:
                worksheet = writer.sheets[sheet_name]
                worksheet.autofilter(0, 0, len(df), len(df.columns)-1)
                num_format = workbook.add_format({'num_format': '0.000'})
                worksheet.set_column('E:E', 12, num_format)  # RMSE column
        
        print(f"Successfully saved results to:\n{output_file}")
        
        # Generate outlier visualization
        plot_path = visualize_outliers(df, results_dir)
        print(f"Generated visualization:\n{plot_path}")
    
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == '__main__':
    main()