import pandas as pd
import numpy as np
import os
from pathlib import Path

def read_data_from_csv(file_path):
    """
    CSV 파일에서 데이터와 헤더를 함께 읽어오는 함수

    Args:
        file_path (str): CSV 파일 경로

    Returns:
        tuple: (헤더, 데이터)
    """
    # 헤더 읽기 (첫 7줄)
    with open(file_path, 'r') as f:
        header_lines = ''.join([f.readline() for _ in range(7)])

    # 데이터 읽기
    data = pd.read_csv(file_path, skiprows=7)

    return header_lines, data


def resample_data(data, original_hz, target_hz):
    """
    데이터를 목표 샘플링 레이트로 리샘플링하는 함수
    
    Args:
        data (pd.DataFrame): 리샘플링할 데이터
        original_hz (int): 원본 데이터의 샘플링 레이트 (Hz)
        target_hz (int): 목표 샘플링 레이트 (Hz)
        
    Returns:
        pd.DataFrame: 리샘플링된 데이터
    """
    if original_hz == target_hz:
        return data
        
    if target_hz > original_hz:
        new_length = int(len(data) * target_hz / original_hz)
        new_index = pd.Index(np.linspace(0, len(data)-1, new_length))
        
        resampled_data = pd.DataFrame()
        for column in data.columns:
            resampled_data[column] = data[column].reindex(new_index, method='linear')
    else:
        downsample_rate = original_hz // target_hz
        resampled_data = data.iloc[::downsample_rate, :]
    
    return resampled_data

def save_resampled_data(header_lines, resampled_data, save_path, target_hz):
    """
    리샘플링된 데이터를 CSV 파일로 저장하는 함수

    Args:
        header_lines (str): 원본 파일의 헤더
        resampled_data (pd.DataFrame): 리샘플링된 데이터
        save_path (str): 저장할 파일 경로
        target_hz (int): 목표 샘플링 레이트 (Hz)
    """
    # 프레임 번호 생성 (1부터 시작)
    frame = np.arange(1, len(resampled_data) + 1)
    # 시간 계산 (초 단위)
    time = (frame - 1) / target_hz
    
    # Frame과 Time (Seconds) 열의 값을 새로 계산된 값으로 교체
    resampled_data['Frame'] = frame
    resampled_data['Time (Seconds)'] = time

    with open(save_path, 'w', newline='') as f:
        f.write(header_lines)
        resampled_data.to_csv(f, index=False, lineterminator='\n')


def process_folder(parent_folder_path, original_hz, target_hz):
    """
    폴더 내의 모든 CSV 파일을 처리하는 함수
    
    Args:
        parent_folder_path (str): 처리할 폴더 경로
        original_hz (int): 원본 샘플링 레이트
        target_hz (int): 목표 샘플링 레이트
    """
    parent_path = Path(parent_folder_path)
    parent_name = parent_path.name
    
    # 저장 폴더 생성
    suffix = '_upsampling' if target_hz > original_hz else '_downsampling'
    save_folder = parent_path.parent / f"{parent_name}{suffix}"
    save_folder.mkdir(exist_ok=True)
    
    # 모든 CSV 파일 처리
    for csv_file in parent_path.rglob('*.csv'):
        # 상대 경로 유지를 위한 처리
        rel_path = csv_file.relative_to(parent_path)
        save_path = save_folder / rel_path
        
        # 저장 디렉토리 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터 처리
        header_lines, data = read_data_from_csv(str(csv_file))
        resampled_data = resample_data(data, original_hz, target_hz)
        save_resampled_data(header_lines, resampled_data, str(save_path), target_hz)
        
        print(f"Processed: {csv_file.name}")
        print(f"Original shape: {data.shape}, Resampled shape: {resampled_data.shape}")

def main():
    # 사용자 입력
    parent_folder = r'C:\Users\gns15\OneDrive\Desktop\Exported_csv (1)\test'
    original_hz = 240
    target_hz = 120
    
    # 처리 시작
    print("\n처리를 시작합니다...")
    process_folder(parent_folder, original_hz, target_hz)
    print("\n모든 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()
