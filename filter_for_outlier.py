import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import re
from tkinter import filedialog
from tkinter import *
from tkinter import messagebox

## AUTHORSHIP INFORMATION
__author__ = "Hun Min Kim"
__version__ = "0.1"
__email__ = "gns5758@inha.edu"
__status__ = "Development"

## Functions
def browse_folder():
    folder_path = filedialog.askdirectory(title="데이터 폴더 경로를 선택하세요")
    folder_path_entry.delete(0, END)
    folder_path_entry.insert(0, folder_path)

def process_files():
    folder_path = folder_path_entry.get()
    file_extension = file_extension_var.get()
    data_start_col = int(data_start_col_entry.get())
    header_end_row = int(header_end_row_entry.get())
    outlier_handling = outlier_handling_var.get()
    interpolation_method = interpolation_method_var.get()
    outlier_detection_method = outlier_detection_method_var.get()
    save_image = save_image_var.get()

    if outlier_detection_method == 1:
        threshold = float(threshold_entry.get())
    else:
        z_score_threshold = float(z_score_threshold_entry.get())

    # 파일 처리 함수 (기존 코드와 동일)
    def find_files(directory, extension):
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files

    def plot_quartiles(data, title):
        fig, ax = plt.subplots()
        ax.boxplot(data.T)
        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Value')
        plt.show()

    files = find_files(folder_path, file_extension)
    print(f"Found {len(files)} files to process")

    outlier_folder = os.path.join(folder_path, 'outlier')
    os.makedirs(outlier_folder, exist_ok=True)

    total_outliers = 0

    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_outlier_folder = os.path.join(outlier_folder, file_name)
        os.makedirs(file_outlier_folder, exist_ok=True)

        if file_extension == '.csv':
            df = pd.read_csv(file, header=list(range(header_end_row+1)), index_col=0)
            
            data_columns = df.columns[data_start_col:]
            
            for column in data_columns:
                column_name = re.sub(r'[^a-zA-Z0-9]', '_', str(column))
                if outlier_detection_method == 1:
                    Q1 = df[column].quantile(0.25)
                    Q3 = df[column].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                else:
                    mean = df[column].mean()
                    std = df[column].std()
                    
                    z_scores = (df[column] - mean) / std
                    outliers = abs(z_scores) > z_score_threshold
                
                total_outliers += outliers.sum()
                
                print(f"Found {outliers.sum()} outliers in column {column}")
                print(f"발견된 아웃라이어 값: {df.loc[outliers, column].values}")
                
                # 아웃라이어 처리 전 데이터 시각화
                if save_image == 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df.index, df[column], label='Original Data')
                    plt.plot(df.index[outliers], df.loc[outliers, column], 'ro', label='Outliers')
                    plt.xlabel('Index')
                    plt.ylabel(column)
                    plt.title(f'Outliers in {column} (Before Treatment)')
                    plt.legend()
                    plt.tight_layout()
                    
                    output_filename = f"{file_name}_{column_name}_before_treatment.png"
                    output_path = os.path.join(file_outlier_folder, output_filename)
                    plt.savefig(output_path)
                    plt.close()
                
                if outlier_handling == 1:
                    df.loc[outliers, column] = np.nan
                elif outlier_handling == 2:
                    df = df[~outliers]
                elif outlier_handling == 3:
                    df.loc[outliers, column] = df[column].median()
                
                # 아웃라이어 처리 후 데이터 시각화
                if save_image == 1:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df.index, df[column], label='Treated Data')
                    plt.xlabel('Index')
                    plt.ylabel(column)
                    plt.title(f'Outliers in {column} (After Treatment)')
                    plt.legend()
                    plt.tight_layout()
                    
                    output_filename = f"{file_name}_{column_name}_after_treatment.png"
                    output_path = os.path.join(file_outlier_folder, output_filename)
                    plt.savefig(output_path)
                    plt.close()
            
            df.reset_index(drop=True, inplace=True)  # 인덱스 재설정
            
            if interpolation_method != 0:
                if interpolation_method == 1:
                    df.loc[:, data_columns] = df.loc[:, data_columns].interpolate(method='linear')
                elif interpolation_method == 2:
                    df.loc[:, data_columns] = df.loc[:, data_columns].interpolate(method='cubic')
            
            output_filename = file_name + '_outlier' + file_extension
            output_path = os.path.join(file_outlier_folder, output_filename)
            df.to_csv(output_path, index=True)
            
        else:
            wb = openpyxl.load_workbook(file)
            
            for sheet_name in wb.sheetnames:
                df = pd.read_excel(file, sheet_name, header=list(range(header_end_row+1)), index_col=0, engine='openpyxl')
                
                data_columns = df.columns[data_start_col:]
                
                for column in data_columns:
                    column_name = re.sub(r'[^a-zA-Z0-9]', '_', str(column))
                    if outlier_detection_method == 1:
                        Q1 = df[column].quantile(0.25)
                        Q3 = df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - threshold * IQR
                        print(f"Lower bound: {lower_bound}")
                        upper_bound = Q3 + threshold * IQR
                        print(f"Upper bound: {upper_bound}")
                        
                        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
                    else:
                        mean = df[column].mean()
                        std = df[column].std()
                        
                        z_scores = (df[column] - mean) / std
                        outliers = abs(z_scores) > z_score_threshold
                    
                    total_outliers += outliers.sum()
                    
                    print(f"Found {outliers.sum()} outliers in column {column}")
                    # print(f"발견된 아웃라이어 값: {df.loc[outliers, column].values}")
                    
                    # 아웃라이어 처리 전 데이터 시각화
                    if save_image == 1:
                        plt.figure(figsize=(10, 6))
                        plt.plot(df.index, df[column], label='Original Data')
                        plt.plot(df.index[outliers], df.loc[outliers, column], 'ro', label='Outliers')
                        plt.xlabel('Index')
                        plt.ylabel(column)
                        plt.title(f'Outliers in {column} (Before Treatment)')
                        plt.legend()
                        plt.tight_layout()
                    
                        output_filename = f"{file_name}_{column_name}_before_treatment.png"
                        output_path = os.path.join(file_outlier_folder, output_filename)
                        plt.savefig(output_path)
                        plt.close()
                    
                    if outlier_handling == 1:
                        df.loc[outliers, column] = np.nan
                    elif outlier_handling == 2:
                        df = df[~outliers]
                    elif outlier_handling == 3:
                        df.loc[outliers, column] = df[column].median()
                    
                    # 아웃라이어 처리 후 데이터 시각화
                    if save_image == 1:
                        plt.figure(figsize=(10, 6))
                        plt.plot(df.index, df[column], label='Treated Data')
                        plt.xlabel('Index')
                        plt.ylabel(column)
                        plt.title(f'Outliers in {column} (After Treatment)')
                        plt.legend()
                        plt.tight_layout()
                        
                        output_filename = f"{file_name}_{column_name}_after_treatment.png"
                        output_path = os.path.join(file_outlier_folder, output_filename)
                        plt.savefig(output_path)
                        plt.close()
                
                df.reset_index(drop=True, inplace=True)  # 인덱스 재설정
                
                if interpolation_method != 0:
                    if interpolation_method == 1:
                        df.loc[:, data_columns] = df.loc[:, data_columns].interpolate(method='linear')
                    elif interpolation_method == 2:
                        df.loc[:, data_columns] = df.loc[:, data_columns].interpolate(method='cubic')
                
                output_filename = file_name + '_' + sheet_name + '_outlier' + file_extension
                output_path = os.path.join(file_outlier_folder, output_filename)
                df.to_excel(output_path, sheet_name=sheet_name, index=True, engine='openpyxl')

    messagebox.showinfo("처리 완료", f"총 {total_outliers}개의 아웃라이어가 처리되었습니다.")

# GUI 설정
root = Tk()
root.title("아웃라이어 탐지 및 처리")
root.geometry("400x700")

# 폴더 경로 입력
folder_path_label = Label(root, text="데이터 폴더 경로:")
folder_path_label.pack()
folder_path_entry = Entry(root, width=40)
folder_path_entry.pack()
browse_button = Button(root, text="찾아보기", command=browse_folder)
browse_button.pack()

# 파일 확장자 선택
file_extension_label = Label(root, text="파일 확장자:")
file_extension_label.pack()
file_extension_var = StringVar(value=".csv")
csv_radio = Radiobutton(root, text=".csv", variable=file_extension_var, value=".csv")
csv_radio.pack()
xlsx_radio = Radiobutton(root, text=".xlsx", variable=file_extension_var, value=".xlsx")
xlsx_radio.pack()

# 데이터 시작 열 입력
data_start_col_label = Label(root, text="데이터 시작 열 번호 (어느 열부터 데이터가 시작 되나요?):")
data_start_col_label.pack()
data_start_col_entry = Entry(root)
data_start_col_entry.pack()

# 헤더 끝 행 입력
header_end_row_label = Label(root, text="헤더 끝 행 번호 (어느 행까지 헤더인가요?):")
header_end_row_label.pack()
header_end_row_entry = Entry(root)
header_end_row_entry.pack()

# 아웃라이어 처리 방법 선택
outlier_handling_label = Label(root, text="아웃라이어 처리 방법:")
outlier_handling_label.pack()
outlier_handling_var = IntVar(value=1)
marking_radio = Radiobutton(root, text="마킹", variable=outlier_handling_var, value=1)
marking_radio.pack()
removal_radio = Radiobutton(root, text="제거", variable=outlier_handling_var, value=2)
removal_radio.pack()
replacement_radio = Radiobutton(root, text="대체", variable=outlier_handling_var, value=3)
replacement_radio.pack()

# 보간 방법 선택
interpolation_method_label = Label(root, text="보간 방법:")
interpolation_method_label.pack()
interpolation_method_var = IntVar(value=0)
no_interpolation_radio = Radiobutton(root, text="안함", variable=interpolation_method_var, value=0)
no_interpolation_radio.pack()
linear_interpolation_radio = Radiobutton(root, text="Linear", variable=interpolation_method_var, value=1)
linear_interpolation_radio.pack()
cubic_interpolation_radio = Radiobutton(root, text="Cubic", variable=interpolation_method_var, value=2)
cubic_interpolation_radio.pack()

# 아웃라이어 탐지 방법 선택
outlier_detection_method_label = Label(root, text="아웃라이어 탐지 방법:")
outlier_detection_method_label.pack()
outlier_detection_method_var = IntVar(value=1)
iqr_radio = Radiobutton(root, text="IQR", variable=outlier_detection_method_var, value=1)
iqr_radio.pack()
z_score_radio = Radiobutton(root, text="Z-score", variable=outlier_detection_method_var, value=2)
z_score_radio.pack()

# 아웃라이어 임계치 입력 (IQR)
threshold_label = Label(root, text="아웃라이어 임계치 (IQR):")
threshold_label.pack()
threshold_entry = Entry(root)
threshold_entry.pack()

# Z-score 임계치 입력
z_score_threshold_label = Label(root, text="Z-score 임계치:")
z_score_threshold_label.pack()
z_score_threshold_entry = Entry(root)
z_score_threshold_entry.pack()

# 그래프 저장 여부 선택
save_image_label = Label(root, text="그래프 저장 여부:")
save_image_label.pack()
save_image_var = IntVar(value=1)
save_image_yes_radio = Radiobutton(root, text="예", variable=save_image_var, value=1)
save_image_yes_radio.pack()
save_image_no_radio = Radiobutton(root, text="아니요", variable=save_image_var, value=2)
save_image_no_radio.pack()

# 처리 시작 버튼
process_button = Button(root, text="처리 시작", command=process_files)
process_button.pack()

# 작성자 표시
author_label = Label(root, text="Author: hunminkim98", anchor="se")
author_label.pack(side=BOTTOM, fill=X)

root.mainloop()