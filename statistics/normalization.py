import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import os
from pathlib import Path

def normalize_frames(data: np.ndarray, target_frames: int = 101) -> np.ndarray:
    """
    데이터를 지정된 프레임 수로 정규화하는 함수
    
    Parameters:
        data (np.ndarray): 정규화할 데이터 (N x 3), 각 행은 (X, Y, Z) 좌표
        target_frames (int): 목표 프레임 수 (기본값: 101)
    
    Returns:
        np.ndarray: 정규화된 데이터 (target_frames x 3)
    """
    # 입력 데이터의 프레임 수
    current_frames = len(data)
    
    if current_frames == 0:
        return np.zeros((target_frames, 3))
    
    # 시간 축 생성
    original_time = np.linspace(0, 100, current_frames)
    target_time = np.linspace(0, 100, target_frames)
    
    # 각 좌표축에 대해 보간 수행
    normalized_data = np.zeros((target_frames, 3))
    for i in range(3):  # X, Y, Z 각각에 대해
        normalized_data[:, i] = np.interp(target_time, original_time, data[:, i])
    
    return normalized_data

def normalize_frame_numbers(current_frames: int, target_frames: int = 101) -> np.ndarray:
    """
    프레임 번호를 정규화하는 함수
    """
    original_time = np.linspace(0, 100, current_frames)
    target_time = np.linspace(0, 100, target_frames)
    return np.interp(target_time, original_time, np.arange(current_frames))

def load_joint_data(file_path: str) -> Dict[str, Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame]]:
    """
    모션 캡쳐 데이터의 모든 관절 데이터를 로드하는 함수
    
    Returns:
        Dict[str, Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame]]: 
            키: 관절 이름
            값: ((markerless 정규화된 데이터, marker-based 정규화된 데이터), 원본 데이터프레임) 튜플
    """
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        print(f"\n=== 엑셀 파일 정보 ===")
        print(f"파일 경로: {file_path}")
        print(f"총 시트 수: {len(sheet_names)}")
        print("시트 목록:", sheet_names)
        
        joint_data = {}
        
        for sheet_name in sheet_names:
            print(f"\n=== {sheet_name} 데이터 ===")
            
            # 시트 데이터 읽기
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            try:
                # 실제 데이터만 추출 (3번째 행부터)
                data = df.iloc[2:].reset_index(drop=True)
                
                # markerless 데이터 (1,2,3 컬럼)
                markerless_coords = data.iloc[:, [1,2,3]].astype(float)
                markerless_normalized = normalize_frames(markerless_coords.to_numpy())
                
                # marker-based 데이터 (5,6,7 컬럼)
                marker_based_coords = data.iloc[:, [5,6,7]].astype(float)
                marker_based_normalized = normalize_frames(marker_based_coords.to_numpy())
                
                print(f"원본 프레임 수: {len(data)}")
                
                # 딕셔너리에 저장
                joint_data[sheet_name] = ((markerless_normalized, marker_based_normalized), df)
                
            except Exception as e:
                print(f"경고: {sheet_name} 데이터 처리 중 오류 발생: {str(e)}")
                continue
        
        return joint_data
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

def save_normalized_data(file_path: str, joint_data: Dict[str, Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame]]):
    """
    정규화된 데이터를 원본과 동일한 형식으로 저장하는 함수
    """
    if joint_data is None:
        return
        
    # 원본 파일의 절대 경로를 Path 객체로 변환
    original_path = Path(file_path)
    
    # parent 폴더 찾기 (CMC 폴더)
    parent_dir = original_path
    while parent_dir.name != "CMC" and parent_dir.parent != parent_dir:
        parent_dir = parent_dir.parent
    
    if parent_dir.name != "CMC":
        raise ValueError("CMC 폴더를 찾을 수 없습니다")
    
    # 원본 파일의 CMC 폴더 기준 상대 경로 계산
    rel_path = original_path.relative_to(parent_dir)
    
    # 새로운 출력 경로 생성 (CMC 폴더와 같은 레벨에 _normalized 폴더 생성)
    output_base = parent_dir.parent / "_normalized"
    output_path = output_base / rel_path
    
    # 출력 디렉토리 생성
    os.makedirs(output_path.parent, exist_ok=True)
    
    # ExcelWriter 객체 생성
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        for sheet_name, ((markerless_norm, marker_based_norm), original_df) in joint_data.items():
            # 원본 데이터프레임 복사
            new_df = original_df.copy()
            
            # 필요한 행만 유지 (헤더 2행 + 데이터 101행)
            new_df = new_df.iloc[:103]
            
            # markerless 데이터 업데이트 (1,2,3 컬럼)
            new_df.iloc[2:, [1,2,3]] = markerless_norm
            
            # marker-based 데이터 업데이트 (5,6,7 컬럼)
            new_df.iloc[2:, [5,6,7]] = marker_based_norm
            
            # Frame 번호 업데이트 (0번째와 4번째 컬럼)
            frame_numbers = np.arange(101)  # 0부터 100까지
            new_df.iloc[2:, 0] = frame_numbers  # markerless frame
            new_df.iloc[2:, 4] = frame_numbers  # marker-based frame
            
            # 시트에 저장
            new_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n정규화된 데이터가 저장되었습니다: {output_path}")

def print_joint_statistics(joint_data: Dict[str, Tuple[Tuple[np.ndarray, np.ndarray], pd.DataFrame]]):
    """
    각 관절 데이터의 통계 정보를 출력하는 함수
    """
    if joint_data is None:
        return
        
    print("\n=== 관절 데이터 통계 ===")
    for joint_name, ((markerless_data, marker_based_data), _) in joint_data.items():
        print(f"\n{joint_name}:")
        print(f"Markerless 데이터 범위:")
        print(f"X: {np.min(markerless_data[:,0]):.2f} to {np.max(markerless_data[:,0]):.2f}")
        print(f"Y: {np.min(markerless_data[:,1]):.2f} to {np.max(markerless_data[:,1]):.2f}")
        print(f"Z: {np.min(markerless_data[:,2]):.2f} to {np.max(markerless_data[:,2]):.2f}")
        print(f"Marker-based 데이터 범위:")
        print(f"X: {np.min(marker_based_data[:,0]):.2f} to {np.max(marker_based_data[:,0]):.2f}")
        print(f"Y: {np.min(marker_based_data[:,1]):.2f} to {np.max(marker_based_data[:,1]):.2f}")
        print(f"Z: {np.min(marker_based_data[:,2]):.2f} to {np.max(marker_based_data[:,2]):.2f}")

def process_excel_file(input_file: Path, base_dir: str) -> bool:
    """
    단일 Excel 파일을 처리하는 함수
    
    Returns:
        bool: 처리 성공 여부
    """
    try:
        # 데이터 로드
        joint_data = load_joint_data(str(input_file))
        
        # 데이터 프레임 수 확인
        if joint_data is not None:
            # 첫 번째 시트의 데이터 길이 확인
            first_sheet = next(iter(joint_data.values()))
            data_length = len(first_sheet[1].iloc[2:])  # 헤더 2행을 제외한 데이터 길이
            
            # 101개 이상의 데이터가 있는 경우에만 처리
            if data_length >= 101:
                save_normalized_data(str(input_file), joint_data)
                print(f"처리 완료: {input_file.name} (데이터 길이: {data_length})")
                return True
            else:
                print(f"스킵: {input_file.name} (데이터 길이가 101 미만: {data_length})")
                return False
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
    skipped_files = 0
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
                result = process_excel_file(excel_file, str(base_path))
                if result:
                    processed_files += 1
                else:
                    # 데이터 길이가 101 미만인 경우는 실패가 아닌 스킵으로 처리
                    if "데이터 길이가 101 미만" in str(excel_file):
                        skipped_files += 1
                    else:
                        failed_files.append(str(excel_file))
    
    # 처리 결과 출력
    print("\n=== 처리 완료 ===")
    print(f"총 파일 수: {total_files}")
    print(f"처리 완료: {processed_files}")
    print(f"스킵 (101 미만): {skipped_files}")
    print(f"실패: {len(failed_files)}")
    
    if failed_files:
        print("\n실패한 파일들:")
        for file in failed_files:
            print(f"- {file}")

if __name__ == "__main__":
    # 기본 디렉토리 설정
    base_dir = r"C:\Users\5W555A\Desktop\Data-processing-tool-for-biomechanics\statistics\CMC\merged_check_interpolated"
    
    # 배치 처리 시작
    batch_process_files(base_dir)
