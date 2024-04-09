import os
import numpy as np
import pandas as pd
import c3d
from c3d import DataTypes
import glob
import toml
import struct

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = toml.load(f)
    return config

def trc_to_c3d(config_dict):
    # Read config
    project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
    print(f"Project directory: {project_dir}")
    frame_rate = config_dict.get('project').get('frame_rate')
    print(f"Frame rate: {frame_rate}")

    # Get the last folder name of project_dir
    last_folder_name = os.path.basename(project_dir)
    print(f"Last folder name: {last_folder_name}")

    pose3d_dir = os.path.join(project_dir, 'pose-3d')
    print(f"pose3d_dir: {pose3d_dir}")

    # Determine the .trc file name to read
    trc_files = []
    if config_dict.get('triangulation').get('save_to_c3d'):
        trc_pattern = f"{last_folder_name}_*.trc"
        trc_files = glob.glob(os.path.join(pose3d_dir, trc_pattern))
        print(f"Successfully read {trc_files}, caused by save_to_c3d = True in triangulation.")
    elif config_dict.get('filtering').get('save_to_c3d'):
        trc_pattern = f"{last_folder_name}*_filt_*.trc"
        trc_files = glob.glob(os.path.join(pose3d_dir, trc_pattern))
        print(f"Successfully read {trc_files}, caused by save_to_c3d = True in filtering.")
    else:
        print("No valid save_to_c3d configuration found.")
        return

    print(f"Read trc files: {[os.path.basename(file) for file in trc_files]}")

    for trc_file in trc_files:
        # Extract marker names from the 4th row of the TRC file
        with open(os.path.join(pose3d_dir, trc_file), 'r') as file:
            lines = file.readlines()
            marker_names_line = lines[3]
            marker_names = marker_names_line.strip().split('\t')[2::3]
            print(f"Marker names: {marker_names}")

        # Read the data frame (skiprows=5)
        trc_data = pd.read_csv(os.path.join(pose3d_dir, trc_file), sep='\t', skiprows=5)

        # Extract marker coordinates
        marker_coords = trc_data.iloc[:, 2:].to_numpy().reshape(-1, len(marker_names), 3)
        marker_coords = np.nan_to_num(marker_coords, nan=0.0)

        # scale_factor = 100
        # marker_coords = marker_coords * scale_factor

        # Create a C3D writer
        writer = c3d.Writer(point_rate=frame_rate, analog_rate=0, point_scale=1.0, point_units='m', gen_scale=1.0)

        # Add marker parameters
        writer.set_point_labels(marker_names)

        # Add marker descriptions (optional)
        marker_descriptions = [''] * len(marker_names)
        writer.point_group.add_param('DESCRIPTIONS', desc='Marker descriptions', 
                                    bytes_per_element=-1, dimensions=[len(marker_names)], 
                                    bytes=np.array(marker_descriptions, dtype=object))
        
        # Set the data start parameter
        data_start = writer.header.data_block
        writer.point_group.add_param('DATA_START', desc='Data start parameter',
                                    bytes_per_element=2, dimensions=[], bytes=struct.pack('<H', data_start))
        
        # Create a C3D group for markers
        markers_group = writer.point_group

        # Add frame data
        for frame in marker_coords:
            # Add residual and camera columns
            residuals = np.full((frame.shape[0], 1), 0.0)  # Set residuals to 0.0
            cameras = np.zeros((frame.shape[0], 1))  # Set cameras to 0
            points = np.hstack((frame, residuals, cameras))
            writer.add_frames([(points, np.array([]))])

        # Set the trial start and end frames
        writer.set_start_frame(1)
        writer._set_last_frame(len(marker_coords))

        # Write the C3D file
        c3d_file_path = trc_file.replace('.trc', '.c3d')
        with open(os.path.join(pose3d_dir, c3d_file_path), 'wb') as handle:
            writer.write(handle)
        print(f"C3D file saved to: {c3d_file_path}")


        # # Reshape the marker coordinates and add empty analog data
        # num_frames = len(marker_coords)
        # num_markers = marker_coords.shape[1]
        # frames_data = marker_coords.reshape(num_frames, num_markers, 3)
        # frames_data = np.concatenate((frames_data, np.zeros((num_frames, num_markers, 2))), axis=2)
        # print(f"frames_data: {frames_data}")

        # # Add frame data
        # for frame in frames_data:
        #     writer.add_frames([(frame, np.array([]))])

        # # # Add frame data

        # # Write the C3D file
        # c3d_file_path = trc_file.replace('.trc', '.c3d')
        # with open(os.path.join(pose3d_dir, c3d_file_path), 'wb') as handle:
        #     writer.write(handle)
        # print(f"C3D file saved to: {c3d_file_path}")

if __name__ == '__main__':
    config_file = 'config.toml'
    config = load_config(config_file)
    trc_to_c3d(config)


        # markers_group = c3d.Group(name='MARKERS', dtypes=-1)
        # writer.add_group(group_id=0, name='MARKERS', desc='Markers data')
            # Create a C3D parameter for marker names
        # max_length = max([len(name) for name in marker_names])
        # labels_param = c3d.Param('LABELS', '', -1, [max_length, len(marker_names)])
        # labels_param.set_string_array(marker_names)
        # markers_group.add_param(labels_param)

        # max_length = max([len(name) for name in marker_names])
        # labels_param = c3d.Param(name='LABELS', dtype=DataTypes.CHAR, dimensions=[max_length, len(marker_names)])
        # labels_param.string_array[:] = marker_names
        # markers_group.add_parameter(labels_param)
        # # Reshape the marker coordinates and add empty analog data
        # num_frames = len(marker_coords)
        # num_markers = marker_coords.shape[1]
        # frames_data = marker_coords.reshape(num_frames, num_markers * 3)
        # frames_data = np.hstack((frames_data, np.zeros((frames_data.shape[0], 1))))