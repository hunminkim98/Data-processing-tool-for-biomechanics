import pandas as pd
import os
import numpy as np
from scipy import signal
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted

def select_files(file_type):
    # 기존 코드 유지
    title = f"{file_type} 데이터 엑셀 파일들을 선택하세요:"
    print(title)
    files = filedialog.askopenfilenames(
        title=f"{file_type} 데이터 파일 선택", 
        filetypes=[("Excel files", "*.xlsx")]
    )
    return natsorted(list(files))

def apply_lowpass_filter(data, cutoff=6, fs=120, order=4):
    # 기존 코드 유지
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def filter_markerless_data(df, columns_to_plot=None, plot_first_n=0):
    # 수정된 함수
    import matplotlib.pyplot as plt

    filtered_df = df.copy()
    
    # 필터를 적용할 열 선택
    numeric_columns = [col for col in filtered_df.columns if col not in ['File', 'Frame'] and filtered_df[col].dtype in [np.float64, np.int64]]
    
    # 플롯할 열 설정
    if columns_to_plot is not None:
        plot_columns = [col for col in numeric_columns if col in columns_to_plot]
    else:
        plot_columns = numeric_columns[:plot_first_n]
    
    for col in numeric_columns:
        original_data = filtered_df[col].values
        filtered_data = apply_lowpass_filter(original_data)
        filtered_df[col] = filtered_data

        if col in plot_columns:
            plt.figure()
            plt.plot(original_data, label='Original')
            plt.plot(filtered_data, label='Filtered')
            plt.title(f'Filtering Result for {col}')
            plt.xlabel('Sample Index')
            plt.ylabel(col)
            plt.legend()
            plt.show()

    return filtered_df

def merge_excel_pair(file1, file2):
    # 수정된 함수
    excel1 = pd.ExcelFile(file1)
    excel2 = pd.ExcelFile(file2)

    base_name = os.path.basename(file1)
    output_dir = os.path.join(os.path.dirname(file1), 'merged')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"merged_{base_name}")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        sheet_order = excel1.sheet_names
        for sheet in excel2.sheet_names:
            if sheet not in sheet_order:
                sheet_order.append(sheet)
        
        for sheet_name in sheet_order:
            # 데이터 읽기 수정: header=1로 설정
            df1 = pd.read_excel(file1, sheet_name=sheet_name, header=1) if sheet_name in excel1.sheet_names else pd.DataFrame()
            df2 = pd.read_excel(file2, sheet_name=sheet_name, header=1) if sheet_name in excel2.sheet_names else pd.DataFrame()
            
            # markerless 데이터(df2)에 필터 적용 및 플롯
            filtered_df2 = filter_markerless_data(df2, plot_first_n=0)  # 처음 세 개의 숫자형 열을 플롯
            
            # 데이터 병합 및 저장
            merged_df = pd.concat([df1, filtered_df2], axis=1)
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"병합 완료: {output_path}")

def merge_excel_files():
    # 기존 코드 유지
    root = tk.Tk()
    root.withdraw()

    files1 = select_files("marker-based")
    files2 = select_files("markerless")

    if len(files1) != len(files2):
        print("[오류] 선택한 파일의 개수가 일치하지 않습니다.")
        return

    for file1, file2 in zip(files1, files2):
        merge_excel_pair(file1, file2)

if __name__ == "__main__":
    merge_excel_files()
    print("모든 작업이 완료되었습니다!")
