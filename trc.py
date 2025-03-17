def inspect_trc_header(trc_path):
    with open(trc_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:10]):
            print(f"Line {i+1}: {line.strip()}")

# TRC 파일 경로를 제공하고 이 함수를 호출하여 첫 10줄을 출력해보세요.
inspect_trc_header(r'C:\Users\5W555A\Desktop\Bundle\pose2sim-w-Marker-Augmenter-Sync\Pose2Sim\S00_Demo_BatchSession\S00_P00_SingleParticipant\S00_P00_T01_BalancingTrial\pose-3d\S00_P00_T01_BalancingTrial_0-100.trc')
