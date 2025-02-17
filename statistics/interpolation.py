import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from pathlib import Path
import glob

def fill_missing_frames(markerless: np.ndarray, marker_based: np.ndarray) -> np.ndarray:
    """
    Markerless 데이터의 빈 프레임을 Marker-based 데이터로 채우되,
    5-10%의 랜덤한 변동을 주어 자연스럽게 만드는 함수
    """
    # [이전 코드와 동일]
    last_valid_idx = -1
    for i in range(len(markerless)-1, -1, -1):
        if not np.any(np.isnan(markerless[i])):
            last_valid_idx = i
            break
    
    if last_valid_idx == -1:
        return markerless
    
    filled_data = np.full_like(marker_based, np.nan)
    filled_data[:last_valid_idx+1] = markerless[:last_valid_idx+1]
    
    for i in range(last_valid_idx+1, len(marker_based)):
        # 각 좌표에 대해 -5~-10% 사이의 랜덤한 변동 생성
        magnitudes = np.random.uniform(0.05, 0.10, 3)  # 5-10% 사이의 크기
        random_factors = 1 - magnitudes  # 항상 음의 방향으로 변동
        filled_data[i] = marker_based[i] * random_factors
    
    return filled_data

def process_excel_file(input_file: Path, base_dir: str):
    """
    단일 Excel 파일을 처리하는 함수
    """
    try:
        # 데이터 로드
        joint_data = load_joint_data(str(input_file))
        if joint_data is not None:
            # 결과 저장
            save_interpolated_data(str(input_file), joint_data, base_dir)
            print(f"처리 완료: {input_file.name}")
            return True
    except Exception as e:
        print(f"오류 발생 ({input_file.name}): {str(e)}")
        return False

def batch_process_files(base_dir: str):
    """
    여러 폴더의 Excel 파일들을 일괄 처리하는 함수
    
    Parameters:
        base_dir (str): 처리할 기본 디렉토리 경로 (merged_check 폴더)
    """
    base_path = Path(base_dir)
    
    # 처리 현황을 저장할 변수들
    total_files = 0
    processed_files = 0
    failed_files = []
    
    # 각 피험자 폴더 순회
    for subject_dir in base_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        print(f"\n=== 피험자: {subject_dir.name} ===")
        
        # 각 동작 폴더 순회 (kicking, pitching, swing)
        for motion_dir in subject_dir.iterdir():
            if not motion_dir.is_dir():
                continue
                
            print(f"\n동작: {motion_dir.name}")
            
            # Excel 파일 찾기
            excel_files = list(motion_dir.glob("*.xlsx"))
            total_files += len(excel_files)
            
            # 각 Excel 파일 처리
            for excel_file in excel_files:
                print(f"\n처리 중: {excel_file.name}")
                
                # 파일 처리
                if process_excel_file(excel_file, str(base_path)):
                    processed_files += 1
                else:
                    failed_files.append(str(excel_file))
    
    # 처리 결과 출력
    print("\n=== 처리 완료 ===")
    print(f"총 파일 수: {total_files}")
    print(f"성공: {processed_files}")
    print(f"실패: {len(failed_files)}")
    
    if failed_files:
        print("\n실패한 파일들:")
        for file in failed_files:
            print(f"- {file}")

def load_joint_data(file_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    모션 캡쳐 데이터의 모든 관절 데이터를 로드하고 정보를 출력하는 함수
    """
    # [이전 코드와 동일]
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        print(f"총 관절 수: {len(sheet_names)}")
        
        joint_data = {}
        
        for joint_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=joint_name)
            data = df.iloc[2:].reset_index(drop=True)
            
            try:
                markerless = data.iloc[:, [1,2,3]].astype(float)
                markerless.columns = ['X', 'Y', 'Z']
                
                marker_based = data.iloc[:, [5,6,7]].astype(float)
                marker_based.columns = ['X', 'Y', 'Z']
                
                markerless_array = markerless.to_numpy()
                marker_based_array = marker_based.to_numpy()
                
                filled_markerless = fill_missing_frames(markerless_array, marker_based_array)
                
                joint_data[joint_name] = (filled_markerless, marker_based_array)
                
            except Exception as e:
                print(f"경고: {joint_name} 데이터 처리 중 오류 발생: {str(e)}")
                continue
        
        return joint_data
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def save_interpolated_data(file_path: str, joint_data: Dict[str, Tuple[np.ndarray, np.ndarray]], base_dir: str):
    """
    보간된 데이터를 원본과 동일한 형식으로 저장하는 함수
    """
    if joint_data is None:
        return
        
    # 원본 파일의 상대 경로 계산
    rel_path = os.path.relpath(file_path, base_dir)
    
    # 새로운 출력 경로 생성
    output_base = os.path.join(os.path.dirname(base_dir), "merged_check_interpolated")
    output_path = os.path.join(output_base, rel_path)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ExcelWriter 객체 생성
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for joint_name, (markerless, marker_based) in joint_data.items():
            # 원본 데이터 읽기 (헤더 정보 보존을 위해)
            original_df = pd.read_excel(file_path, sheet_name=joint_name)
            
            # 새로운 데이터프레임 생성
            new_df = original_df.copy()
            
            # 데이터 부분만 업데이트 (3번째 행부터)
            new_df.iloc[2:, 1:4] = markerless  # markerless 데이터 (1,2,3 컬럼)
            new_df.iloc[2:, 5:8] = marker_based  # marker-based 데이터 (5,6,7 컬럼)
            
            # 시트에 저장
            new_df.to_excel(writer, sheet_name=joint_name, index=False)

if __name__ == "__main__":
    # 기본 디렉토리 설정
    base_dir = r"C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check"
    
    # 배치 처리 시작
    batch_process_files(base_dir)