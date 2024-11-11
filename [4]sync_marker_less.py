import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tkinter as tk
from tkinter import filedialog              
import os
from natsort import natsorted

#############################
## Step 1. data preparation ##
#############################

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
        header_lines = ''.join([f.readline() for _ in range(6)])

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

def read_and_apply_offset_target_from_xlsx(xlsx_file_path, offset):
    """
    xlsx 파일에서 데이터를 읽어오는 함수
    첫 2행은 헤더, 첫 2열은 filename과 frame 정보
    실제 데이터는 3행 3열부터 시작
    총 8개의 Sheet가 있으며, 각 Sheet는 하나의 joint angle 정보를 담고 있음
    각 시트의 데이터에 오프셋만큼 데이터를 자른 후 다시 저장.
    Args:
        xlsx_file_path (str): xlsx 파일 경로
        offset (int): 오프셋 값
    Returns:
        pd.DataFrame: xlsx 데이터
    """
    # 엑셀 파일 읽기
    excel_file = pd.ExcelFile(xlsx_file_path)
    
    # 모든 시트에 대해 처리
    for sheet_name in excel_file.sheet_names:
        # 시트 데이터 읽기 
        df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
        
        # 헤더와 데이터 분리
        headers = df.iloc[:2]  # 첫 2행은 헤더
        data = df.iloc[2:]     # 3행부터 데이터
        
        # 오프셋 적용
        if offset < 0:
            # 음수 오프셋: 데이터를 뒤에서 자름
            data = data.iloc[abs(offset)-1:]
        else:
            # 양수 오프셋: 데이터 앞에 빈 행 추가
            empty_rows = pd.DataFrame(np.nan, index=range(offset), columns=data.columns)
            data = pd.concat([empty_rows, data], ignore_index=True)
        
        # 헤더와 데이터 다시 합치기
        final_df = pd.concat([headers, data], ignore_index=True)
        
        # 새로운 폴더에 결과 저장
        output_path = os.path.join(os.path.dirname(xlsx_file_path), 'synced_data')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, f'{os.path.basename(xlsx_file_path)[:-5]}_sync.xlsx')
        
        # 첫 시트면 새로 만들고, 아니면 기존 파일에 추가
        if sheet_name == excel_file.sheet_names[0]:
            final_df.to_excel(output_path, sheet_name=sheet_name, index=False)
        else:
            with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
    return pd.read_excel(output_path)  # 동기화된 데이터 반환

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
    # plt.show(block=False)  # block=False로 설정하여 비동기적으로 표시
    # plt.pause(2)  # 2초 동안 대기
    # plt.close()  # 2초 후 자동으로 닫기


def find_peak_position(distances):
    threshold_factor = 1
    # 임계값 설정: 평균의 threshold_factor 배수
    threshold = np.mean(distances) * threshold_factor*1.5

    # 임계값을 초과하는 첫 번째 위치 찾기
    for i, value in enumerate(distances):
        if i >= 60 and value > threshold:
            return i

def find_optimal_offset(csv_distances, trc_distances, window_size=240):
    max_corr = -1
    optimal_offset = 0

    # 각 데이터에서 피크 위치 찾기
    csv_peak = find_peak_position(csv_distances)
    print(f"CSV 피크 위치: {csv_peak}")
    trc_peak = find_peak_position(trc_distances)
    print(f"TRC 피크 위치: {trc_peak}")

    # 초기 오프셋 설정 (두 피크를 일치시키기 위해)
    initial_offset = trc_peak - csv_peak

    # lag 범위 설정: 초기 오프셋 기준으로 좌우 window_size만큼 설정
    lag_range = [initial_offset - window_size, initial_offset + window_size]
    print(f"초기 오프셋: {initial_offset}, lag_range: {lag_range}")

    # 시리즈 생성
    csv_series = pd.Series(csv_distances)
    trc_series = pd.Series(trc_distances)
    
    for lag in range(lag_range[0], lag_range[1] + 1):
        # csv 데이터를 양수 방향으로 이동하여 lag 값 조정
        shifted_csv = csv_series.shift(lag)
        
        # NaN 값 제거 후 공통 인덱스만 사용
        valid_indices = ~(shifted_csv.isna() | trc_series.isna())
        csv_temp = shifted_csv[valid_indices]
        trc_temp = trc_series[valid_indices]
        
        # 데이터가 충분하지 않으면 건너뛰기
        if len(csv_temp) < 10:  # 최소 10개 이상의 데이터 포인트 필요
            continue

        # 상관계수의 절댓값을 비교하여 최대값 갱신
        corr = csv_temp.corr(trc_temp)
        
        # NaN 값이면 건너뛰기
        if np.isnan(corr):
            continue

        # 최대 상관계수 업데이트 (절댓값을 기준으로)
        if abs(corr) > max_corr:
            max_corr = abs(corr)
            optimal_offset = lag

    print(f"CSV 데이터 길이: {len(csv_distances)} 프레임")
    print(f"TRC 데이터 길이: {len(trc_distances)} 프레임") 
    print(f"최적 오프셋: {optimal_offset} 프레임")
    print(f"최대 상관계수: {max_corr:.4f}")

    return optimal_offset, max_corr


#########################
## Batch Processing ##
#########################

def batch_process(csv_files, trc_files, target_files, csv_marker_names, trc_marker_names, csv_frame_range=None, trc_frame_range=None):
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
    for csv_file_path, trc_file_path, target_file_path in zip(csv_files, trc_files, target_files):
        print(f"\nProcessing:\nCSV: {csv_file_path}\nTRC: {trc_file_path}\nTarget: {target_file_path}")

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

        # warning
        if max_correlation < 0.85:
            print(f"[WARNING] Correlation coefficient is less than 0.85. Please check the data.")

        # visualize synchronization
        visualize_euclidean_distance(csv_distances, trc_distances, optimal_offset, max_correlation)

        # 동기화된 CSV 데이터 저장
        synced_csv_data = synchronize_csv_data(csv_data, optimal_offset)

        # 동기화된 CSV 파일 저장
        save_synchronized_csv(csv_file_path, csv_header, synced_csv_data)
        print(f"Synchronized CSV saved as: {csv_file_path[:-4]}_sync.csv")

        # 동기화된 xlsx 파일 저장
        read_and_apply_offset_target_from_xlsx(target_file_path, optimal_offset)

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
        synced_csv_data = csv_data.iloc[abs(optimal_offset)-1:].reset_index(drop=True)
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
    # 동기화된 CSV 파일을 저장할 폴더 생성
    sync_folder = os.path.join(os.path.dirname(csv_file_path), "sync_results")
    if not os.path.exists(sync_folder):
        os.makedirs(sync_folder)

    # 동기화된 CSV 파일 경로 설정
    filename = os.path.basename(csv_file_path)
    output_file_path = os.path.join(sync_folder, f"{os.path.splitext(filename)[0]}_sync.csv")

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
    csv_files = natsorted(csv_files)  # 자연 정렬 적용

    print("TRC 파일들을 선택하세요:")
    trc_files = filedialog.askopenfilenames(title="Select TRC files", filetypes=[("TRC files", "*.trc")])
    trc_files = list(trc_files)
    trc_files = natsorted(trc_files)  # 자연 정렬 적용

    print(f"offset을 적용할 angle을 포함한 xlsx 파일을 선택하세요.")
    target_files = filedialog.askopenfilenames(title="Select xlsx files", filetypes=[("xlsx files", "*.xlsx")])
    target_files = list(target_files)
    target_files = natsorted(target_files)  # 자연 정렬 적용

    # 파일 수 확인
    if len(csv_files) != len(trc_files) != len(target_files):
        print("[ERROR] 선택한 CSV, TRC, 그리고 타겟 파일의 수가 일치하지 않습니다.")
        exit()

    # 여러 개의 마커 이름을 리스트로 지정
    csv_marker_names = ['INHA_FULL2:RELB', 'INHA_FULL2:LELB']
    trc_marker_names = ['RElbow', 'LElbow']

    # 프레임 범위 입력 여부 확인
    use_frame_range = 'n'

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
    batch_process(csv_files, trc_files, target_files, csv_marker_names, trc_marker_names, csv_frame_range, trc_frame_range)
