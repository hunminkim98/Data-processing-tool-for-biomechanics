import os

# 파일이 있는 디렉토리 경로를 설정하세요
folder_path = r'C:\Users\5W555A\Desktop\pose2sim\Pose2Sim\Demo_Batch\s4\pose\swing4_1_tracked'

# 디렉토리 안의 모든 json 파일을 가져옵니다
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# 정렬하여 순서대로 이름 변경
for i, filename in enumerate(sorted(json_files), start=1):
    new_filename = f"frame_{i:04d}.json"
    old_file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, new_filename)
    
    # 파일 이름 변경
    os.rename(old_file_path, new_file_path)

print("파일 이름 변경 완료!")
