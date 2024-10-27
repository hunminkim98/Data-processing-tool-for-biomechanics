import pandas as pd
import numpy as np
import os
from pathlib import Path

def read_data_from_excel(file_path):
    """
    엑셀 파일의 모든 시트를 읽어 딕셔너리로 반환하는 함수
    """
    # 헤더 없이 모든 시트를 읽음
    df_dict = pd.read_excel(file_path, sheet_name=None, engine='openpyxl', header=None)
    return df_dict

def resample_data(data, original_hz, target_hz):
    """
    데이터를 목표 샘플링 레이트로 리샘플링하는 함수
    """
    if original_hz == target_hz:
        return data.copy()

    if target_hz > original_hz:
        # 업샘플링
        new_length = int(len(data) * target_hz / original_hz)
        new_index = np.linspace(0, len(data)-1, new_length)
        resampled_data = pd.DataFrame()
        for column in data.columns:
            resampled_data[column] = np.interp(new_index, np.arange(len(data)), data[column])
    else:
        # 다운샘플링
        downsample_rate = original_hz // target_hz
        resampled_data = data.iloc[::downsample_rate, :].reset_index(drop=True)

    return resampled_data

def process_and_save_excel(file_path, save_path, original_hz, target_hz):
    """
    엑셀 파일의 모든 시트를 처리하고 저장하는 함수
    """
    df_dict = read_data_from_excel(file_path)

    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            # 헤더와 데이터 분리
            headers = df.iloc[:2]  # 첫 2행은 헤더
            data = df.iloc[2:].reset_index(drop=True)  # 3행부터 데이터
            print(f"기존 데이터 행의 수: {len(data)}")

            # 데이터 리샘플링
            resampled_data = resample_data(data, original_hz, target_hz)
            print(f"리샘플링 후 데이터 행의 수: {len(resampled_data)}")

            # 헤더와 데이터 결합
            headers_df = headers.copy()
            final_df = pd.concat([headers_df, resampled_data], ignore_index=True)

            # 엑셀 파일에 시트로 저장
            final_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    print(f"Processed and saved: {os.path.basename(save_path)}")

def process_folder(parent_folder_path, original_hz, target_hz):
    """
    폴더 내의 모든 엑셀 파일을 처리하는 함수 (여러 시트 포함)
    """
    parent_path = Path(parent_folder_path)
    parent_name = parent_path.name

    # 저장 폴더 생성
    suffix = '_upsampling' if target_hz > original_hz else '_downsampling'
    save_folder = parent_path.parent / f"{parent_name}{suffix}"
    save_folder.mkdir(exist_ok=True)

    # 모든 엑셀 파일 처리
    for excel_file in parent_path.rglob('*.xlsx'):
        # 상대 경로 유지를 위한 처리
        rel_path = excel_file.relative_to(parent_path)
        save_path = save_folder / rel_path
        # 저장 디렉토리 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 엑셀 파일 처리 및 저장
        process_and_save_excel(str(excel_file), str(save_path), original_hz, target_hz)

def main():
    # 사용자 입력
    parent_folder = r'D:\Validation\marker-based'
    original_hz = 240
    target_hz = 120

    # 처리 시작
    print("\n처리를 시작합니다...")
    process_folder(parent_folder, original_hz, target_hz)
    print("\n모든 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()
