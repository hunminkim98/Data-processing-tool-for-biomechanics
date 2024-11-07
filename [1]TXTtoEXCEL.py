import os
import pandas as pd

def read_excel_file(file_path):
    """
    Read Excel file and separate header and data sections
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        tuple: (header_data, frame_info, main_data)
            - header_data: First 5 rows of the Excel file
            - frame_info: First column data (frame information)
            - main_data: Data starting from row 6, column 2
    """

    
    # Read the entire Excel file using openpyxl engine for xlsx files
    df = pd.read_excel(file_path, engine='openpyxl', header=None)
    
    # Extract header (first 5 rows)
    header_data = df.iloc[:4]
    
    # Extract frame information (first column)
    frame_info = df.iloc[4:, 0]
    
    # Extract main data (from row 6, column 2)
    main_data = df.iloc[4:, 1:]
    
    return header_data, frame_info, main_data

def classify_data_by_trial(header_data):
    """
    Classify data based on trial information from header
    
    Args:
        header_data: First rows of Excel file containing file path information
        
    Returns:
        dict: Dictionary with trial names as keys and lists of corresponding indices as values
    """
    # Get the first row excluding first column
    file_paths = header_data.iloc[0, 1:].tolist()
    print(file_paths)
    
    # Dictionary to store trial classifications
    trial_groups = {}
    
    # Process each file path
    for idx, path in enumerate(file_paths):
        if isinstance(path, str) and '\\' in path:
            # Extract full trial name including the sequence number (e.g. 'gait_001_001' from 'gait_001_001.c3d')
            trial_name = path.split('\\')[-1].split('.')[0]
            
            # Add index to corresponding trial group
            if trial_name not in trial_groups:
                trial_groups[trial_name] = []
            trial_groups[trial_name].append(idx)
    
    return trial_groups

def save_trial_data_to_excel(header_data, frame_info, main_data, trial_groups, output_dir):
    """
    Save data for each trial group to separate Excel files
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each trial group
    for trial_name, indices in trial_groups.items():
        # Create new dataframe for this trial
        trial_df = pd.DataFrame()
        
        # Add frame info as first column
        trial_df['Frame'] = frame_info.reset_index(drop=True)
        
        # Add header rows (only first 4 rows, excluding the bold header row)
        header_rows = []
        for i in range(4):  # Changed from header_data.shape[0] to 4
            row_data = ['Frame']  # First column header
            row_data.extend([header_data.iloc[i, idx+1] for idx in indices])
            header_rows.append(row_data)
        
        # Add data columns for this trial
        for idx in indices:
            trial_df[f'Column_{idx+1}'] = main_data.iloc[:, idx].reset_index(drop=True)
        
        # Save to Excel
        output_file = os.path.join(output_dir, f'{trial_name}.xlsx')
        
        # Create Excel writer object
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write the header rows first (only first 4 rows)
            header_df = pd.DataFrame(header_rows)
            header_df.to_excel(writer, index=False, header=False)
            
            # Write the main data, starting after the header
            trial_df.to_excel(writer, startrow=4, index=False)

            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            # Delete the 4th row (index 3)
            worksheet.delete_rows(5)
            
        print(f'Saved {trial_name} to {output_file}')

# Example usage:
# Get directory path from file_path
directory = r"D:\석사\석사3차\Markerless validation\Results\Final2\Final_marker2\고승준"

# Get all xlsx files in the directory
xlsx_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]


# Process each xlsx file
for xlsx_file in xlsx_files:
    file_path = os.path.join(directory, xlsx_file)
    
    # output directory: remove file extension from file_path
    output_dir = file_path.rsplit('.', 1)[0]  # Split at last '.' and take first part
    
    # Read and process the data
    header_data, frame_info, main_data = read_excel_file(file_path)
    trial_groups = classify_data_by_trial(header_data)
    
    # Save trial data to separate Excel files
    save_trial_data_to_excel(header_data, frame_info, main_data, trial_groups, output_dir)
    
    print(f'Processed {xlsx_file}')

