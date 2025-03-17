import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CI import calculate_confidence_interval, plot_confidence_interval
from scipy import stats

# Parse the data
data_str = '''motion	joint	axis	mean_cmc	std_cmc	trial_count
kicking	Avg_Ankle	X	0.919431711	0.070166995	92
kicking	Avg_Ankle	Y	0.78289507	0.145405871	97
kicking	Avg_Ankle	Z	0.977812785	0.018854821	73
kicking	Avg_Hip	X	0.986020147	0.009105742	89
kicking	Avg_Hip	Y	0.952476307	0.027176663	96
kicking	Avg_Hip	Z	0.946293914	0.048318235	96
kicking	Avg_Knee	X	0.988563092	0.005594517	94
kicking	Avg_Knee	Y	0.987248216	0.00489977	96
kicking	Avg_Knee	Z	0.943565369	0.044129817	89
kicking	Trunk	X	0.89082644	0.067208251	97
kicking	Trunk	Y	0.865523186	0.09773638	96
kicking	Trunk	Z	0.96587529	0.035688141	97


pitching	Avg_Ankle	X	0.923215244	0.057191799	97
pitching	Avg_Ankle	Y	0.937805073	0.029197447	99
pitching	Avg_Ankle	Z	0.995776593	0.00198286	88
pitching	Avg_Hip	X	0.982743234	0.011733887	94
pitching	Avg_Hip	Y	0.97314507	0.023364904	83
pitching	Avg_Hip	Z	0.990849261	0.00390905	98
pitching	Avg_Knee	X	0.991367335	0.004769975	99
pitching	Avg_Knee	Y	0.987207136	0.004863074	93
pitching	Avg_Knee	Z	0.991296446	0.003407602	88
pitching	Trunk	X	0.960353884	0.031103027	99
pitching	Trunk	Y	0.950168509	0.036755845	98
pitching	Trunk	Z	0.998021643	0.001056487	98


swing	Avg_Ankle	X	0.739634979	0.160395829	84
swing	Avg_Ankle	Y	0.900736036	0.050309214	83
swing	Avg_Ankle	Z	0.994625686	0.003203134	68
swing	Avg_Hip	X	0.931234278	0.039632033	94
swing	Avg_Hip	Y	0.986634675	0.006835563	89
swing	Avg_Hip	Z	0.992756222	0.002982017	89
swing	Avg_Knee	X	0.975379175	0.012491392	88
swing	Avg_Knee	Y	0.993810479	0.002458203	81
swing	Avg_Knee	Z	0.986160399	0.005790256	88
swing	Trunk	X	0.90384743	0.080556519	87
swing	Trunk	Y	0.970585542	0.02195982	86
swing	Trunk	Z	0.996655588	0.001801557	88'''

# Load data into a pandas DataFrame
df = pd.DataFrame([line.split('\t') for line in data_str.strip().split('\n') if line.strip()])
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# Convert numeric columns to float and int
df['mean_cmc'] = df['mean_cmc'].astype(float)
df['std_cmc'] = df['std_cmc'].astype(float)
df['trial_count'] = df['trial_count'].astype(int)

# Function to calculate confidence intervals for each row
def calculate_ci_for_row(row, confidence_level=0.95):
    """
    Calculate confidence interval for a row in the dataframe using Fisher's z-transformation.
    This is more appropriate for CMC values which behave like correlation coefficients.
    
    Parameters:
    -----------
    row : pandas.Series
        Row from the dataframe containing mean_cmc, std_cmc, and trial_count
    confidence_level : float
        Confidence level (between 0 and 1)
    
    Returns:
    --------
    pandas.Series
        Series containing lower_bound and upper_bound of the confidence interval
    """
    r = row['mean_cmc']  # Correlation coefficient (CMC)
    n = row['trial_count']  # Sample size
    
    # Apply Fisher's z-transformation
    # This transforms correlation coefficient to z which is approximately normally distributed
    z = 0.5 * np.log((1 + r) / (1 - r))  # Fisher's z-transformation
    
    # Standard error for z
    se_z = 1 / np.sqrt(n - 3)
    
    # Z-value for the given confidence level
    z_alpha = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Calculate confidence interval in z-space
    z_lower = z - z_alpha * se_z
    z_upper = z + z_alpha * se_z
    
    # Transform back to correlation coefficient space
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    # Ensure bounds are within [0, 1]
    r_lower = max(0, r_lower)
    r_upper = min(1, r_upper)
    
    return pd.Series({
        'lower_bound': r_lower,
        'upper_bound': r_upper
    })

# Calculate 95% confidence intervals for all rows
ci_df = df.apply(lambda row: calculate_ci_for_row(row, confidence_level=0.95), axis=1)
df = pd.concat([df, ci_df], axis=1)

# Group by motion and joint to prepare for plotting
def create_motion_joint_plots():
    plt.figure(figsize=(15, 10))
    
    # Get unique motions and joints
    motions = df['motion'].unique()
    joints = df['joint'].unique()
    
    # Create a figure with subplots for each motion
    fig, axes = plt.subplots(len(motions), 1, figsize=(15, 5 * len(motions)))
    
    # For each motion, plot all joints with their X, Y, Z axes
    for i, motion in enumerate(motions):
        ax = axes[i] if len(motions) > 1 else axes
        
        motion_data = df[df['motion'] == motion]
        
        # Set up positions for grouped bars
        n_joints = len(joints)
        n_axes = 3  # X, Y, Z
        width = 0.25  # width of the bars
        
        # Create positions for bars
        positions = []
        for j in range(n_joints):
            for k in range(n_axes):
                positions.append(j * (n_axes + 1) * width + k * width)
        
        # Prepare data for plotting
        means = []
        errors = []
        labels = []
        colors = []
        
        color_map = {'X': 'red', 'Y': 'green', 'Z': 'blue'}
        
        # For each joint and axis, get the data
        for joint in joints:
            for axis in ['X', 'Y', 'Z']:
                row = motion_data[(motion_data['joint'] == joint) & (motion_data['axis'] == axis)]
                if not row.empty:
                    means.append(row['mean_cmc'].values[0])
                    errors.append([
                        row['mean_cmc'].values[0] - row['lower_bound'].values[0],
                        row['upper_bound'].values[0] - row['mean_cmc'].values[0]
                    ])
                    labels.append(f"{joint}-{axis}")
                    colors.append(color_map[axis])
        
        # Plot bars with error bars
        for j, (pos, mean, error, color) in enumerate(zip(positions, means, errors, colors)):
            ax.bar(pos, mean, width, color=color, alpha=0.7)
            ax.errorbar(pos, mean, yerr=[[error[0]], [error[1]]], fmt='none', color='black', capsize=5)
        
        # Set x-ticks and labels
        x_positions = []
        x_labels = []
        for j in range(n_joints):
            x_positions.append(j * (n_axes + 1) * width + width)
            x_labels.append(joints[j])
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
        
        # Set title and labels
        ax.set_title(f"{motion.capitalize()} Motion - 95% Confidence Intervals for CMC Values")
        ax.set_xlabel("Joint")
        ax.set_ylabel("CMC Value")
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=color_map[axis]) for axis in ['X', 'Y', 'Z']]
        ax.legend(handles, ['X-axis', 'Y-axis', 'Z-axis'])
        
        # Set y-limits
        ax.set_ylim(0, 1.05)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('motion_joint_cmc_confidence_intervals.png', dpi=300)
    plt.close()

def create_joint_comparison_plots():
    # For each joint, compare the CMC values across motions
    joints = df['joint'].unique()
    axes = ['X', 'Y', 'Z']
    
    for joint in joints:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, axis in enumerate(axes):
            # Get data for this joint and axis across all motions
            joint_axis_data = df[(df['joint'] == joint) & (df['axis'] == axis)]
            
            if not joint_axis_data.empty:
                motions = joint_axis_data['motion'].tolist()
                means = joint_axis_data['mean_cmc'].tolist()
                lower_bounds = joint_axis_data['lower_bound'].tolist()
                upper_bounds = joint_axis_data['upper_bound'].tolist()
                
                # Calculate error bars
                yerr = np.array([
                    [m - l for m, l in zip(means, lower_bounds)],
                    [u - m for m, u in zip(means, upper_bounds)]
                ])
                
                # Plot bars with error bars
                x_pos = np.arange(len(motions))
                axs[i].bar(x_pos, means, width=0.5, alpha=0.7, 
                           color=['#ff9999', '#66b3ff', '#99ff99'])
                axs[i].errorbar(x_pos, means, yerr=yerr, fmt='none', color='black', capsize=5)
                
                # Add labels and title
                axs[i].set_title(f"{joint} - {axis} axis")
                axs[i].set_ylabel("CMC Value")
                axs[i].set_xticks(x_pos)
                axs[i].set_xticklabels([m.capitalize() for m in motions])
                
                # Set y-limits
                axs[i].set_ylim(0, 1.05)
                
                # Add grid
                axs[i].grid(True, linestyle='--', alpha=0.7)
                
                # Add text annotations with exact values
                for j, (x, y, l, u) in enumerate(zip(x_pos, means, lower_bounds, upper_bounds)):
                    axs[i].text(x, y - 0.15, f"{y:.3f}\n[{l:.3f}, {u:.3f}]", 
                               ha='center', va='center', fontsize=9,
                               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.suptitle(f"Comparison of {joint} CMC Values Across Motions (95% CI)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{joint}_comparison_across_motions.png', dpi=300)
        plt.close()

def create_axis_comparison_plots():
    # For each axis, compare the CMC values across joints and motions
    axes = ['X', 'Y', 'Z']
    motions = df['motion'].unique()
    
    for axis in axes:
        fig, axs = plt.subplots(1, len(motions), figsize=(18, 6))
        
        for i, motion in enumerate(motions):
            # Get data for this axis and motion across all joints
            axis_motion_data = df[(df['axis'] == axis) & (df['motion'] == motion)]
            
            if not axis_motion_data.empty:
                joints = axis_motion_data['joint'].tolist()
                means = axis_motion_data['mean_cmc'].tolist()
                lower_bounds = axis_motion_data['lower_bound'].tolist()
                upper_bounds = axis_motion_data['upper_bound'].tolist()
                
                # Calculate error bars
                yerr = np.array([
                    [m - l for m, l in zip(means, lower_bounds)],
                    [u - m for m, u in zip(means, upper_bounds)]
                ])
                
                # Plot bars with error bars
                x_pos = np.arange(len(joints))
                axs[i].bar(x_pos, means, width=0.5, alpha=0.7,
                           color=['#ffcc99', '#cc99ff', '#99ffcc', '#ffff99'])
                axs[i].errorbar(x_pos, means, yerr=yerr, fmt='none', color='black', capsize=5)
                
                # Add labels and title
                axs[i].set_title(f"{motion.capitalize()} - {axis} axis")
                axs[i].set_ylabel("CMC Value")
                axs[i].set_xticks(x_pos)
                axs[i].set_xticklabels(joints)
                
                # Set y-limits
                axs[i].set_ylim(0, 1.05)
                
                # Add grid
                axs[i].grid(True, linestyle='--', alpha=0.7)
                
                # Add text annotations with exact values
                for j, (x, y, l, u) in enumerate(zip(x_pos, means, lower_bounds, upper_bounds)):
                    axs[i].text(x, y - 0.15, f"{y:.3f}\n[{l:.3f}, {u:.3f}]", 
                               ha='center', va='center', fontsize=9,
                               bbox=dict(facecolor='white', alpha=0.7))
        
        plt.suptitle(f"Comparison of {axis}-axis CMC Values Across Joints and Motions (95% CI)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{axis}_axis_comparison.png', dpi=300)
        plt.close()

# Create all plots
create_motion_joint_plots()
create_joint_comparison_plots()
create_axis_comparison_plots()

# Save processed data to CSV
df.to_csv('processed_cmc_data_with_fisher_ci.csv', index=False)

print("Analysis complete! The following files have been created:")
print("1. processed_cmc_data_with_fisher_ci.csv - processed data with confidence intervals using Fisher's z-transformation")
print("2. motion_joint_cmc_confidence_intervals.png - comparison of joints and axes within each motion")
print("3. Joint-specific comparison plots (e.g., Avg_Ankle_comparison_across_motions.png)")
print("4. Axis-specific comparison plots (e.g., X_axis_comparison.png)")
