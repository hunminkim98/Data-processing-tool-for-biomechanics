import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

def validate_filename(filename):
    """파일명 패턴 검증 (예: gait_001_001.xlsx, jogging_001_001.xlsx)"""
    pattern = r'^[a-zA-Z]+_\d{3}_\d{3}\.xlsx$'
    return bool(re.match(pattern, filename))

def get_matching_files(base_path, joint_folders):
    """각 동작별로 매칭되는 파일들을 찾아서 반환"""
    file_matches = defaultdict(dict)
    missing_files = []

    # 모든 폴더의 파일 검사
    for joint in joint_folders:
        folder_path = Path(base_path) / joint
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {joint}")

        for file in folder_path.glob('*.xlsx'):
            if validate_filename(file.name):
                file_matches[file.name][joint] = file

    # 완전한 매칭 세트만 유지
    complete_matches = {}
    for filename, joints in file_matches.items():
        if len(joints) == len(joint_folders):
            complete_matches[filename] = joints
        else:
            missing = set(joint_folders) - set(joints.keys())
            missing_files.append((filename, missing))

    return complete_matches, missing_files

def merge_joint_angles(base_path):
    # 관절 폴더 목록
    joint_folders = [
        'Left_Ankle', 'Left_Hip', 'Left_Knee', 'Left_Shoulder',
        'Right_Ankle', 'Right_Hip', 'Right_Knee', 'Right_Shoulder',
        'Trunk'
    ]
    
    try:
        # 매칭되는 파일들 찾기
        matching_files, missing_files = get_matching_files(base_path, joint_folders)
        
        if not matching_files:
            raise ValueError("No complete matching sets found!")

        # 매칭 결과 로깅
        log_path = Path(base_path) / 'merge_log.txt'
        with open(log_path, 'w', encoding='utf-8') as log:
            log.write("=== Matching Results ===\n")
            log.write(f"Complete matches found: {len(matching_files)}\n\n")
            
            if missing_files:
                log.write("=== Incomplete Matches ===\n")
                for filename, missing in missing_files:
                    log.write(f"File: {filename}\n")
                    log.write(f"Missing joints: {', '.join(missing)}\n\n")

        # 완전한 매칭 세트에 대해서만 병합 수행
        output_folder = Path(base_path) / 'merged_results'
        output_folder.mkdir(exist_ok=True)

        for filename, joint_files in matching_files.items():
            output_path = output_folder / f'{filename}'
            
            # 데이터 검증을 위한 행 수 확인
            row_counts = {}
            dfs = {}  # 데이터프레임을 저장할 딕셔너리
            
            # 첫 번째 패스: 데이터 읽기 및 검증
            for joint, file_path in joint_files.items():
                try:
                    # engine='openpyxl' 명시적 지정
                    df = pd.read_excel(file_path, engine='openpyxl', header=None)
                    row_counts[joint] = len(df)
                    dfs[joint] = df
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
            
            # 모든 데이터프레임의 행 수가 동일한지 확인
            if len(set(row_counts.values())) != 1:
                with open(log_path, 'a', encoding='utf-8') as log:
                    log.write(f"\nWarning: Row count mismatch in {filename}\n")
                    for joint, count in row_counts.items():
                        log.write(f"{joint}: {count} rows\n")
                continue
            
            try:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # 저장된 데이터프레임을 시트에 쓰기 (index=False로 설정)
                    for joint, df in dfs.items():
                        df.to_excel(writer, sheet_name=joint, index=False, header=False)
                        writer.sheets[joint].sheet_state = 'visible'
                    
                    # 첫 번째 시트를 활성 시트로 설정
                    writer.sheets[list(dfs.keys())[0]].active = True
                
                print(f'Successfully merged: {filename}')
            
            except Exception as e:
                print(f"Error writing {filename}: {str(e)}")
                continue

        print(f"\nMerge completed. Please check the log file at {log_path}")
        print(f"Merged files are saved in: {output_folder}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    base_path = r'D:\석사\석사3차\Markerless validation\Results\Final2\Final_marker2\고승준\Joints'
    merge_joint_angles(base_path)
