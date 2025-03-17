import numpy as np
import pandas as pd

# Euclidean distance function
def euclidean_distance(q1, q2):
    '''
    Euclidean distance between 2 points (N-dim).
    
    INPUTS:
    - q1: list of N_dimensional coordinates of point
         or list of N points of N_dimensional coordinates
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    '''
    
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    if np.isnan(dist).all():
        dist =  np.empty_like(dist)
        dist[...] = np.inf
    
    if len(dist.shape)==1:
        euc_dist = np.sqrt(np.nansum( [d**2 for d in dist]))
    else:
        euc_dist = np.sqrt(np.nansum( [d**2 for d in dist], axis=1))
    
    return euc_dist

# Read TRC file
def read_trc(file_path):
    # read the TRC file(header is 4 lines)
    df = pd.read_csv(file_path, sep='\s+', skiprows=4, on_bad_lines='skip')

    return df

# Calculat mean height
def calculate_mean_height(trc_file):
    '''
    Calculate mean Euclidean distance for height estimation using specific keypoints.
    '''
    # Read TRC file
    data = read_trc(trc_file)
    
    # Select keypoints (e.g., RAnkle, RHeel, LAnkle, LHeel, RShoulder, LShoulder, Nose etc.)
    RAnkle = data[['X3', 'Y3', 'Z3']].values
    RHeel = data[['X6', 'Y6', 'Z6']].values
    RKnee = data[['X2', 'Y2', 'Z2']].values
    RHip = data[['X1', 'Y1', 'Z1']].values
    RShoulder = data[['X16', 'Y16', 'Z16']].values

    LAnkle = data[['X9', 'Y9', 'Z9']].values
    LHeel = data[['X12', 'Y12', 'Z12']].values
    LKnee = data[['X8', 'Y8', 'Z8']].values
    LHip = data[['X7', 'Y7', 'Z7']].values
    LShoulder = data[['X22', 'Y22', 'Z22']].values
    
    Nose = data[['X13', 'Y13', 'Z13']].values

    # Calculate Euclidean distances for each body segment
    right_leg_distances = []
    right_body_distances = []
    left_leg_distances = []
    left_body_distances = []
    head_distances = []
    
    for i in range(len(RAnkle)):
        # Right leg distance calculation (ankle-heel, knee-ankle, hip-knee)
        right_leg = (
            euclidean_distance(RAnkle[i], RHeel[i]) +
            euclidean_distance(RKnee[i], RAnkle[i]) +
            euclidean_distance(RHip[i], RKnee[i])
        )
        right_leg_distances.append(right_leg)

        # Right body distance calculation (shoulder-hip)
        right_body = euclidean_distance(RShoulder[i], RHip[i])
        right_body_distances.append(right_body)

        # Left leg distance calculation (ankle-heel, knee-ankle, hip-knee)
        left_leg = (
            euclidean_distance(LAnkle[i], LHeel[i]) +
            euclidean_distance(LKnee[i], LAnkle[i]) +
            euclidean_distance(LHip[i], LKnee[i])
        )
        left_leg_distances.append(left_leg)

        # Left body distance calculation (shoulder-hip)
        left_body = euclidean_distance(LShoulder[i], LHip[i])
        left_body_distances.append(left_body)

        # Neck position calculation (estimated as the average of both shoulders)
        neck = (RShoulder[i] + LShoulder[i]) / 2

        # Head length calculation (1.5 times)
        head_length = 1.5 * euclidean_distance(neck, Nose[i])
        head_distances.append(head_length)

    # Calculate the average of each body segment
    average_right_leg = np.mean(right_leg_distances)
    average_right_body = np.mean(right_body_distances)
    average_left_leg = np.mean(left_leg_distances)
    average_left_body = np.mean(left_body_distances)
    average_head = np.mean(head_distances)

    # Whole height calculation (sum of the averages of each body segment)
    mean_height = (average_right_leg + average_left_leg + average_right_body + average_left_body) / 2 + average_head

    body_segment_averages = {
        'Average Right Leg': average_right_leg,
        'Average Right Body': average_right_body,
        'Average Left Leg': average_left_leg,
        'Average Left Body': average_left_body,
        'Average Head Length': average_head
    }

    return mean_height, body_segment_averages

# Example usage
trc_file_path = r'C:\Users\5W555A\Desktop\Challenge_Article\For_Sending\2-Cameras\2_Cameras_without_LSTM\Subject1(woman)\Trc(unfiltered+filtered)\marche\sub1_marche_0-1200_filt_butterworth.trc'
mean_height, body_segment_averages = calculate_mean_height(trc_file_path)

# Print the results
print(f"Estimated mean height: {mean_height}")
print("Average heights for each body segment:")
for segment, avg in body_segment_averages.items():
    print(f"{segment}: {avg}")
