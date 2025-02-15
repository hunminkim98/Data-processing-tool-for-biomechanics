import numpy as np
from stat_main import get_waveforms, extract_subject_motion
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rmse(true_values, predicted_values):
    """Calculate Root Mean Square Error between two arrays"""
    return np.sqrt(np.mean((true_values - predicted_values)**2))

def process_trial_rmse(file_path, joint='Ankle', coordinate='X'):
    """Process trial file to calculate RMSE between marker-based and markerless data"""
    mb_wave, ml_wave = get_waveforms(file_path, joint, coordinate)
    return calculate_rmse(mb_wave, ml_wave)

def aggregate_rmse(parent_folder, 
                  joints=['Left_Ankle', 'Left_Hip', 'Left_Knee', 'Right_Ankle', 'Right_Hip', 'Right_Knee', 'Trunk'], 
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
                    records.append({
                        'subject': subject,
                        'motion': motion,
                        'joint': joint,
                        'axis': axis,
                        'rmse': rmse_value
                    })
                except Exception as e:
                    print(f"Error processing {file} (joint: {joint}, axis: {axis}): {e}")
    
    return pd.DataFrame(records)

def identify_outliers_iqr(df, motion, threshold=3.0):
    """Non-parametric outlier detection using interquartile range
    
    Args:
        df: DataFrame containing RMSE data
        motion: Specific motion category to analyze
        threshold: IQR multiplier (default=3)
    
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

if __name__ == '__main__':
    # Configure output paths
    results_dir = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\results'
    data_source = r'C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check'
    
    # Create results directory if needed
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'RMSE_Results_{timestamp}.xlsx')
    
    # Process data and save results
    try:
        print(f"Processing data from: {data_source}")
        df = aggregate_rmse(data_source)
        
        # Add outlier detection
        df['outlier'] = False
        for motion in df['motion'].unique():
            motion_mask = df['motion'] == motion
            is_outlier = ~identify_outliers_iqr(df, motion)
            df.loc[motion_mask & is_outlier, 'outlier'] = True
        
        # Create cleaned dataset
        clean_df = df[~df['outlier']].copy()
        
        # Save to Excel with formatting
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Save full data with outliers
            df.to_excel(writer, index=False, sheet_name='All Data')
            
            # Save cleaned data
            clean_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            workbook = writer.book
            
            # Format both sheets
            for sheet_name in ['All Data', 'Cleaned Data']:
                worksheet = writer.sheets[sheet_name]
                worksheet.autofilter(0, 0, len(df), len(df.columns)-1)
                num_format = workbook.add_format({'num_format': '0.000'})
                worksheet.set_column('E:E', 12, num_format)  # RMSE column
        
        print(f"Successfully saved results to:\n{output_file}")
        
        # Generate visualization
        plot_path = visualize_outliers(df, results_dir)
        print(f"Generated visualization:\n{plot_path}")
    
    except Exception as e:
        print(f"Error processing data: {e}")