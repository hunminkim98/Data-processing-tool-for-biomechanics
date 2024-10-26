import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import os

##############################
## Step 1. data preparation ##
##############################

def read_data_from_csv(csv_file_path):
    """
    CSV 파일에서 데이터와 헤더를 함께 읽어오는 함수

    Args:
        csv_file_path (str): CSV 파일 경로

    Returns:
        tuple: (헤더, 데이터)
    """
    # 헤더 읽기 (첫 7줄)
    with open(csv_file_path, 'r') as f:
        header_lines = ''.join([f.readline() for _ in range(7)])

    # 데이터 읽기
    data = pd.read_csv(csv_file_path, skiprows=7)

    return header_lines, data

def read_data_from_trc(trc_file_path):
    with open(trc_file_path, 'r') as f:
        lines = f.readlines()

    # 헤더 라인 추출
    header_lines = ''.join(lines[:5])
    
    # 마커 이름 추출
    marker_names_line = lines[3]
    marker_names = marker_names_line.strip().split()

    # 데이터 레이블 추출
    data_labels_line = lines[4]
    data_labels = data_labels_line.strip().split()

    # 컬럼 이름 생성
    column_names = []
    for i in range(len(marker_names)):
        if i < 2:
            # 'Frame#', 'Time'
            column_names.append(marker_names[i])
        else:
            # 각 마커에 대해 X, Y, Z 컬럼 생성
            marker = marker_names[i]
            column_names.append(f'{marker}_X')
            column_names.append(f'{marker}_Y')
            column_names.append(f'{marker}_Z')

    # 데이터 읽기
    data = pd.read_csv(trc_file_path, sep='\t', header=None, skiprows=6, names=column_names)

    return header_lines, data

def find_csv_marker_columns(header_lines, data, marker_names):
    """
    CSV 데이터에서 특정 마커의 X,Y,Z 컬럼 위치와 이름을 찾는 함수

    Args:
        header_lines (str): CSV 파일의 헤더 정보
        data (pd.DataFrame): CSV 데이터
        marker_names (list): 찾고자 하는 마커 이름들의 리스트

    Returns:
        dict: 마커 이름을 키로 하고 (컬럼 인덱스 리스트, 컬럼 이름 리스트)를 값으로 가지는 딕셔너리
    """
    # 헤더 라인을 줄 단위로 분리
    header_list = header_lines.strip().split('\n')
    
    # 마커 정보가 있는 줄 찾기 (4번째 줄, 0-based index로는 3)
    marker_line = header_list[3]
    marker_positions = marker_line.split(',')
    
    marker_dict = {}
    for marker_name in marker_names:
        # 마커 컬럼 찾기 (발견 순서대로 X,Y,Z)
        marker_indices = []
        marker_data = []
        
        for idx, marker in enumerate(marker_positions):
            if marker_name in marker:
                marker_indices.append(idx)
                marker_data.append(data.columns[idx])
                if len(marker_indices) == 3:  # X,Y,Z 3개를 찾으면 중단
                    break
        
        marker_dict[marker_name] = (marker_indices, marker_data)
    
    return marker_dict

def find_trc_marker_columns(header_lines, data, marker_names):
    # 헤더 라인을 분리
    header_list = header_lines.strip().split('\n')
    
    # 마커 이름이 있는 네 번째 헤더 라인 가져오기
    if len(header_list) < 5:
        return {}
    
    marker_names_line = header_list[3]
    marker_names_in_file = marker_names_line.strip().split()
    
    marker_dict = {}
    for marker_name in marker_names:
        try:
            # 대상 마커의 인덱스 찾기 (0-based)
            marker_idx = marker_names_in_file.index(marker_name)

            if marker_idx < 2:
                # 'Frame#' 또는 'Time'인 경우
                col_idx = marker_idx
                marker_indices = [col_idx]
                marker_data = [data.columns[col_idx]]
            else:
                # 'Frame#', 'Time' 이후 각 마커는 X, Y, Z로 구성
                col_x = 2 + (marker_idx - 2) * 3
                col_y = col_x + 1
                col_z = col_x + 2

                marker_indices = [col_x, col_y, col_z]
                marker_data = [data.columns[col_x], data.columns[col_y], data.columns[col_z]]

            marker_dict[marker_name] = (marker_indices, marker_data)
        except ValueError:
            print(f"[ERROR] 마커 '{marker_name}'을(를) 헤더에서 찾을 수 없습니다.")
            marker_dict[marker_name] = ([], [])
    
    return marker_dict

########################################################
## Step 2. compute euclidean distance and correlation ##
########################################################    

# Euclidean distance for compute marker speed made by David Pagnon
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

# Visualize euclidean distance
def visualize_euclidean_distance(csv_distances, trc_distances, optimal_offset, max_correlation):
    """
    Visualize Euclidean distances between CSV and TRC data before/after synchronization

    Args:
        csv_distances (list): List of Euclidean distances from CSV data
        trc_distances (list): List of Euclidean distances from TRC data
        optimal_offset (int): Optimal synchronization offset value
        max_correlation (float): Maximum correlation coefficient
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Visualize data before synchronization
    ax1.plot(csv_distances, label='Marker-based', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(trc_distances, label='Markerless', color='orange', linewidth=2, alpha=0.7)

    # Mark offset points
    if optimal_offset < 0:
        ax1.axvline(x=abs(optimal_offset), color='blue', linestyle='--', alpha=0.5, 
                   label=f'Marker-based Start (offset={optimal_offset})')
        ax1.axvline(x=0, color='orange', linestyle='--', alpha=0.5,
                   label='Markerless Start')
    else:
        ax1.axvline(x=0, color='blue', linestyle='--', alpha=0.5,
                   label='Marker-based Start')
        ax1.axvline(x=optimal_offset, color='orange', linestyle='--', alpha=0.5,
                   label=f'Markerless Start (offset={optimal_offset})')

    ax1.set_title('Euclidean Distance Comparison Before Sync', fontsize=14)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Distance (m)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # Visualize data after synchronization
    if optimal_offset < 0:
        synced_csv = csv_distances[abs(optimal_offset):]
        synced_trc = trc_distances[:len(synced_csv)]
    else:
        synced_csv = csv_distances[:len(csv_distances)-optimal_offset]
        synced_trc = trc_distances[optimal_offset:optimal_offset+len(synced_csv)]

    ax2.plot(synced_csv, label='Marker-based(synced)', color='blue', linewidth=2, alpha=0.7)
    ax2.plot(synced_trc, label='Markerless(synced)', color='orange', linewidth=2, alpha=0.7)
    ax2.set_title(f'Euclidean Distance Comparison After Sync (Offset: {optimal_offset} frames)', fontsize=14)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Distance (m)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)

    # display correlation coefficient
    correlation = max_correlation
    ax2.text(0.02, 0.98, f'Correlation: {correlation:.4f}', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top')

    plt.tight_layout()
    plt.show()

# Compute correlation coefficient and find optimal offset
def find_optimal_offset(csv_distances, trc_distances):
    """
    CSV와 TRC 데이터 간의 최적 오프셋을 찾는 함수

    Args:
        csv_distances (list): CSV 데이터의 유클리디안 거리 리스트
        trc_distances (list): TRC 데이터의 유클리디안 거리 리스트

    Returns:
        tuple: (최적 오프셋, 최대 상관계수)
    """
    max_corr = -1
    optimal_offset = 0

    # 더 짧은 데이터의 길이를 기준으로 검사 범위 설정
    min_length = min(len(csv_distances), len(trc_distances))
    max_offset = min_length

    # 양수, 음수 오프셋에 대해 상관계수 계산
    for offset in range(-max_offset, max_offset+1):
        if offset < 0:
            # CSV 데이터를 뒤로 이동
            csv_slice = csv_distances[abs(offset):min_length]
            trc_slice = trc_distances[:len(csv_slice)]
        else:
            # CSV 데이터를 앞으로 이동
            csv_slice = csv_distances[:min_length-offset]
            trc_slice = trc_distances[offset:offset+len(csv_slice)]

        if len(csv_slice) > min_length * 0.5:  # 최소 50% 이상의 데이터가 있을 때만 계산
            # 피어슨 상관계수 계산
            corr = np.corrcoef(csv_slice, trc_slice)[0,1]
            if not np.isnan(corr) and corr > max_corr:
                max_corr = corr
                optimal_offset = offset

    print(f"CSV 데이터 길이: {len(csv_distances)} 프레임")
    print(f"TRC 데이터 길이: {len(trc_distances)} 프레임") 
    print(f"최적 오프셋: {optimal_offset} 프레임")
    print(f"최대 상관계수: {max_corr:.4f}")

    return optimal_offset, max_corr

#########################
## Batch Processing ##
#########################

def batch_process(csv_files, trc_files, csv_marker_names, trc_marker_names, csv_frame_range=None, trc_frame_range=None):
    """
    매칭된 CSV와 TRC 파일 목록을 받아서 동기화를 수행하는 함수

    Args:
        csv_files (list): CSV 파일 경로들의 리스트
        trc_files (list): TRC 파일 경로들의 리스트
        csv_marker_names (list): CSV 파일에서 사용할 마커 이름들의 리스트
        trc_marker_names (list): TRC 파일에서 사용할 마커 이름들의 리스트
        csv_frame_range (tuple): CSV 데이터의 (시작 프레임, 종료 프레임)
        trc_frame_range (tuple): TRC 데이터의 (시작 프레임, 종료 프레임)
    """
    for csv_file_path, trc_file_path in zip(csv_files, trc_files):
        print(f"\nProcessing:\nCSV: {csv_file_path}\nTRC: {trc_file_path}")

        csv_header, csv_data = read_data_from_csv(csv_file_path)
        trc_header, trc_data = read_data_from_trc(trc_file_path)

        # 프레임 범위 적용
        if csv_frame_range is not None:
            csv_start_frame, csv_end_frame = csv_frame_range
            csv_data = csv_data.iloc[csv_start_frame:csv_end_frame].reset_index(drop=True)

        if trc_frame_range is not None:
            trc_start_frame, trc_end_frame = trc_frame_range
            trc_data = trc_data.iloc[trc_start_frame:trc_end_frame].reset_index(drop=True)

        # find marker columns for csv
        csv_marker_dict = find_csv_marker_columns(csv_header, csv_data, csv_marker_names)

        # find marker columns for trc
        trc_marker_dict = find_trc_marker_columns(trc_header, trc_data, trc_marker_names)

        # 각 마커별로 유클리디안 거리 계산 후 합산
        csv_total_distances = None
        trc_total_distances = None

        for marker_name in csv_marker_names:
            # CSV 마커 데이터 추출
            marker_indices, marker_data = csv_marker_dict[marker_name]
            if len(marker_indices) < 3:
                print(f"[ERROR] CSV 데이터에서 마커 '{marker_name}'의 좌표를 찾을 수 없습니다.")
                continue

            x_data = csv_data.iloc[:, marker_indices[0]]  # X 좌표 데이터
            y_data = csv_data.iloc[:, marker_indices[1]]  # Y 좌표 데이터
            z_data = csv_data.iloc[:, marker_indices[2]]  # Z 좌표 데이터

            # 현재 프레임과 다음 프레임 간의 유클리디안 거리 계산 (CSV 데이터)
            csv_distances = []
            for i in range(len(x_data)-1):
                point1 = [x_data[i], y_data[i], z_data[i]]
                point2 = [x_data[i+1], y_data[i+1], z_data[i+1]]
                distance = euclidean_distance(point1, point2)
                csv_distances.append(distance)
            
            csv_distances = np.array(csv_distances)
            
            if csv_total_distances is None:
                csv_total_distances = csv_distances
            else:
                csv_total_distances += csv_distances

        for marker_name in trc_marker_names:
            # TRC 마커 데이터 추출
            marker_indices, marker_data = trc_marker_dict[marker_name]
            if len(marker_indices) < 3:
                print(f"[ERROR] TRC 데이터에서 마커 '{marker_name}'의 좌표를 찾을 수 없습니다.")
                continue

            x_data_trc = trc_data.iloc[:, marker_indices[0]]  # X 좌표 데이터
            y_data_trc = trc_data.iloc[:, marker_indices[1]]  # Y 좌표 데이터
            z_data_trc = trc_data.iloc[:, marker_indices[2]]  # Z 좌표 데이터

            # 현재 프레임과 다음 프레임 간의 유클리디안 거리 계산 (TRC 데이터)
            trc_distances = []
            for i in range(len(x_data_trc)-1):
                point1 = [x_data_trc[i], y_data_trc[i], z_data_trc[i]]
                point2 = [x_data_trc[i+1], y_data_trc[i+1], z_data_trc[i+1]]
                distance = euclidean_distance(point1, point2)
                trc_distances.append(distance)
            
            trc_distances = np.array(trc_distances)
            
            if trc_total_distances is None:
                trc_total_distances = trc_distances
            else:
                trc_total_distances += trc_distances

        # 저주파 통과 필터 적용을 위한 파라미터 설정
        sampling_rate = 120  # 샘플링 레이트 (Hz)
        cutoff_freq = 6  # 차단 주파수 (Hz)
        nyquist_freq = sampling_rate / 2
        normalized_cutoff_freq = cutoff_freq / nyquist_freq

        # Butterworth 저주파 통과 필터 설계
        b, a = signal.butter(4, normalized_cutoff_freq, btype='low')

        # CSV 데이터에 필터 적용
        filtered_csv_distances = signal.filtfilt(b, a, csv_total_distances)

        # TRC 데이터에 필터 적용  
        filtered_trc_distances = signal.filtfilt(b, a, trc_total_distances)

        # 필터링된 데이터로 원본 데이터 업데이트
        csv_distances = filtered_csv_distances
        trc_distances = filtered_trc_distances

        # find optimal offset
        optimal_offset, max_correlation = find_optimal_offset(csv_distances, trc_distances)

        # visualize synchronization
        visualize_euclidean_distance(csv_distances, trc_distances, optimal_offset, max_correlation)

        # 동기화된 CSV 데이터 저장
        synced_csv_data = synchronize_csv_data(csv_data, optimal_offset)

        # 동기화된 CSV 파일 저장
        save_synchronized_csv(csv_file_path, csv_header, synced_csv_data)
        print(f"Synchronized CSV saved as: {csv_file_path[:-4]}_sync.csv")

def synchronize_csv_data(csv_data, optimal_offset):
    """
    CSV 데이터를 최적의 오프셋에 따라 동기화하는 함수

    Args:
        csv_data (pd.DataFrame): 원본 CSV 데이터
        optimal_offset (int): 최적의 오프셋 값

    Returns:
        pd.DataFrame: 동기화된 CSV 데이터
    """
    if optimal_offset < 0:
        print(f"CSV 데이터를 뒤로 {abs(optimal_offset)} 프레임 이동")
        synced_csv_data = csv_data.iloc[abs(optimal_offset):].reset_index(drop=True)
    else:
        # 앞에 빈 행 추가
        empty_rows = pd.DataFrame(np.nan, index=range(optimal_offset), columns=csv_data.columns)
        synced_csv_data = pd.concat([empty_rows, csv_data], ignore_index=True)
        # 데이터 길이를 원본과 맞추기 위해 잘라냄
        synced_csv_data = synced_csv_data.iloc[:len(csv_data)].reset_index(drop=True)
    return synced_csv_data

def save_synchronized_csv(csv_file_path, header_lines, synced_csv_data):
    """
    동기화된 CSV 데이터를 파일로 저장하는 함수

    Args:
        csv_file_path (str): 원본 CSV 파일 경로
        header_lines (str): CSV 헤더 정보
        synced_csv_data (pd.DataFrame): 동기화된 CSV 데이터
    """
    output_file_path = f"{csv_file_path[:-4]}_sync.csv"

    # newline='' 파라미터 추가
    with open(output_file_path, 'w', newline='') as f:
        f.write(header_lines)
        synced_csv_data.to_csv(f, index=False)


########################
##  Start functions   ##
########################

if __name__ == "__main__":
    # 사용자에게 CSV 파일들과 TRC 파일들을 선택하도록 요청
    root = tk.Tk()
    root.withdraw()

    print("CSV 파일들을 선택하세요:")
    csv_files = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    csv_files = list(csv_files)
    csv_files.sort()  # 오름차순 정렬

    print("TRC 파일들을 선택하세요:")
    trc_files = filedialog.askopenfilenames(title="Select TRC files", filetypes=[("TRC files", "*.trc")])
    trc_files = list(trc_files)
    trc_files.sort()  # 오름차순 정렬

    # 파일 수 확인
    if len(csv_files) != len(trc_files):
        print("[ERROR] 선택한 CSV와 TRC 파일의 수가 일치하지 않습니다.")
        exit()

    # 여러 개의 마커 이름을 리스트로 지정
    csv_marker_names = ['INHA_FULL2:RELB', 'INHA_FULL2:LELB']
    trc_marker_names = ['RElbow', 'LElbow']

    # 프레임 범위 입력 여부 확인
    use_frame_range = 'y'

    if use_frame_range == 'y':
        csv_start_frame = 0
        csv_end_frame = 720
        trc_start_frame = 0
        trc_end_frame = 360

        csv_frame_range = (csv_start_frame, csv_end_frame)
        trc_frame_range = (trc_start_frame, trc_end_frame)
    else:
        csv_frame_range = None
        trc_frame_range = None

    # 배치 프로세스 실행
    batch_process(csv_files, trc_files, csv_marker_names, trc_marker_names, csv_frame_range, trc_frame_range)
