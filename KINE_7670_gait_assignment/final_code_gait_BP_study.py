import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
from scipy.interpolate import interp1d

# read txt file
def process_data(file_path, cuttime1, cuttime2):
    # read txt file
    # def process_data(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    start_row = next(i for i, line in enumerate(lines) if "Sample #" in line)
    df = pd.read_csv(file_path, delimiter="\t", skiprows=start_row)
    
    # trasnfer colnums to variables 
    variables = {col: df[col].values[cuttime1:cuttime2] for col in df.columns}
    
    # interpolation -> gap filling method
    def interpolate_with_fallback(data):
        data = pd.DataFrame(data)
        data.replace(0, np.nan, inplace=True)
        data = data.interpolate(method='linear', axis=0)
        data.fillna(method='bfill', inplace=True)  
        data.fillna(method='ffill', inplace=True)  
        if data.isnull().values.any() or (data == 0).any().any():
            data = data.interpolate(method='polynomial', order=2, axis=0).fillna(method='bfill').fillna(method='ffill')
        return data.values
    
    # gap filling method -> apply to each variable 
    for key, value in variables.items():
        variables[key] = interpolate_with_fallback(value)
    
    
    # butterworth low pass filter method
    def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
        return filtfilt(b, a, data)
    
    fs = 100  # frequency of data
    fc = 6    # cutoff frequency
    
    # filter each variable 
    Rft_com_X = butterworth_filter((variables["Right_Foot_COM_X"].flatten())*100, fc, fs, order=4, filter_type='low')
    Rft_com_Y = butterworth_filter((variables["Right_Foot_COM_Y"].flatten())*100, fc, fs, order=4, filter_type='low')
    Rank_X = butterworth_filter((variables["RightAnklePosX"].flatten())*100, fc, fs, order=4, filter_type='low')
    Rank_Y = butterworth_filter((variables["RightAnklePosY"].flatten())*100, fc, fs, order=4, filter_type='low')
    
    Rknee_X = butterworth_filter((variables["RightKnee.Pos.X"].flatten())*100, fc, fs, order=4, filter_type='low')
    Rknee_Y = butterworth_filter((variables["RightKneePosY"].flatten())*100, fc, fs, order=4, filter_type='low')
    
    Rhip_X = butterworth_filter((variables["RightHip_PosX"].flatten())*100, fc, fs, order=4, filter_type='low')
    Rhip_Y = butterworth_filter((variables["RightHipPosY"].flatten())*100, fc, fs, order=4, filter_type='low')
    
    thorax_X = butterworth_filter((variables["ThoraxCOMPosX"].flatten())*100, fc, fs, order=4, filter_type='low')
    thorax_Y = butterworth_filter((variables["ThoraxCOMPosY"].flatten())*100, fc, fs, order=4, filter_type='low')
    
    FP_vertical = butterworth_filter(variables["GRF_MoundY"].flatten(), fc, fs, order=4, filter_type='low')
    FP_vertical = FP_vertical 
    sample = variables["Sample #"].flatten()
    
    
    # combine x and y point into a metrix for each point
    Rft = np.vstack((Rft_com_X, Rft_com_Y)).T
    Rank = np.vstack((Rank_X, Rank_Y)).T
    Rknee = np.vstack((Rknee_X, Rknee_Y)).T
    Rhip = np.vstack((Rhip_X, Rhip_Y)).T
    thorax = np.vstack((thorax_X, thorax_Y)).T
    
    
    # calcualate related angle
    n = np.shape(sample)[0]  # 取得 sample 的行數
    global_y = np.tile([0, 1], (n, 1))  # 創建 n x 2 的矩陣
    
    # create a line between two points
    Rft_ankle_line = Rft - Rank
    Rknee_ankle_line = Rknee - Rank
    Rankle_knee_line = Rank - Rknee
    Rhip_knee_line = Rhip - Rknee
    Rknee_hip_line = Rknee - Rhip
    Trunk_Rhip_line = thorax - Rhip
    
    # the related angle from line 1 to line 2 
    # you can watch: https://en.neurochispas.com/physics/direction-of-a-2d-vector-formulas-and-examples/
    def calculate_theta(data1, data2):
        m = np.shape(data1)[0]
        theta = np.zeros(m)
        R = np.zeros((m, 2, 2))
        c = np.zeros((m, 2))
        for i in range(m):
            A = data1[i, :]
            B = data2[i, :]
            nor_A = A / np.linalg.norm(A)
            nor_B = B / np.linalg.norm(B)
            R[i] = np.array([[nor_A[0], nor_A[1]], [-nor_A[1], nor_A[0]]])
            c[i] = np.dot(R[i], nor_B)
            theta[i] = np.arctan2(c[i, 1], c[i, 0])
        return theta
    
    Rankle_sagital_angle = calculate_theta(Rft_ankle_line, Rknee_ankle_line)
    Rknee_sagital_angle = calculate_theta(Rhip_knee_line, Rankle_knee_line)
    Rhip_sagital_angle = calculate_theta(Rknee_hip_line, Trunk_Rhip_line)
    trunk_sagital_angle = calculate_theta(Trunk_Rhip_line, global_y)
    
    
    def unwrap_deg(data):
        """
        input = data
        outcome = data without gimbal lock problem
        """
        # Calculate the difference between consecutive data points (angle changes)
        dp = np.diff(data)
        # Adjust the differences to be within the range of -π to π
        # First, add π to dp, then take the modulus with 2π, and subtract π to bring the angle change within the range of -π to π
        dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
        # Handle special case: when the difference is -π, and the original change was positive, fix it to π
        dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
        # Calculate the correction needed (difference between the adjusted angle change and the original angle change)
        dp_corr = dps - dp
        # For angle changes that are smaller than π, we set the correction to 0 (no need to fix)
        dp_corr[np.abs(dp) < np.pi] = 0
        # Accumulate the corrections into the original data starting from the second data point
        data[1:] += np.cumsum(dp_corr)
        # Return the corrected data
        return data
    
    
    Rhip_sagital_angle = unwrap_deg(Rhip_sagital_angle)
    Rknee_sagital_angle = unwrap_deg(Rknee_sagital_angle)
    
    
    Rankle_sagital_angle = Rankle_sagital_angle*(180/np.pi)
    Rknee_sagital_angle = Rknee_sagital_angle*(180/np.pi)
    Rhip_sagital_angle = Rhip_sagital_angle*(180/np.pi)
    trunk_sagital_angle = trunk_sagital_angle*(180/np.pi)
    
    
    
    def time_d(data, sampling_interval):
        length = len(data)
        velocity = np.zeros(length)
        for i in range(length):
            if i == 0 or i == length - 1:
                velocity[i] = 0
            else:
                velocity[i] = (data[i + 1] - data[i - 1]) / (2 * sampling_interval)
        return velocity
    
    sampling_interval = 1 / fs
    
    Rankle_sagital_AV = time_d(Rankle_sagital_angle, sampling_interval)
    Rknee_sagital_AV = time_d(Rknee_sagital_angle, sampling_interval)
    Rhip_sagital_AV = time_d(Rhip_sagital_angle, sampling_interval)
    trunk_sagital_AV = time_d(trunk_sagital_angle, sampling_interval)
    Rft_LV = time_d(Rft_com_X, sampling_interval)
        
    event1 = np.where(FP_vertical[1:] > 20)[0]
    if event1.size > 0:
        event1 = event1[0]  
        print(f"Touch down (event1): {event1}")
    
    
    event2 = np.where(FP_vertical[event1+1:] < 0)[0]
    if event2.size > 0:
        event2 = event2[0] + event1
        print(f" second toe off(event2): {event2}")
    
    event3 = np.where(Rft_LV[:event1] > 20)[0]
    if event3.size > 0:
        event3 = event3[0] 
        print(f" first toe off(event2)): {event3}")
        
    plt.figure(figsize=(10, 6))
    time = sample
    plt.plot(time, Rft_com_X, label="Right Foot_X", linewidth=2)
    plt.plot(time, Rank_X, label="Right Ankle_X", linewidth=2)
    plt.plot(time, Rknee_X, label="Right Knee_X", linewidth=2)
    plt.plot(time, Rhip_X, label="Right Hip_X", linewidth=2)
    plt.plot(time, thorax_X, label="Thorax_X", linewidth=2)
    plt.plot(time, Rft_com_Y, label="Right Foot_X", linewidth=2)
    plt.plot(time, Rank_Y, label="Right Ankle_X", linewidth=2)
    plt.plot(time, Rknee_Y, label="Right Knee_X", linewidth=2)
    plt.plot(time, Rhip_Y, label="Right Hip_X", linewidth=2)
    plt.plot(time, thorax_Y, label="Thorax_X", linewidth=2)
    plt.plot(time, FP_vertical/ 1000, label="FP_vertical", linewidth=2)
    plt.plot(time, Rft_LV, label="Rft_LV", linewidth=2)
    plt.axvline(x=sample[event3], color='red', linestyle='--', label='first toe off') # add event3
    plt.axvline(x=sample[event1], color='green', linestyle='--', label='Touch Down') # add event1
    plt.axvline(x=sample[event2], color='green', linestyle='--', label='second toe off') # add event2
    
    
    plt.xlabel("Time (frames)")
    plt.ylabel("Position (Y-axis)")
    plt.title("Joint Motion Over Time")
    plt.legend()
    plt.grid(True)
    
    # 顯示圖表
    plt.show()
    
    # the number we need 
    # AV -> positive = flexion or dorsiflexion / negative = extension or plantar flexion 
    # AV -> 
    

    max_extension_ankle_AV = np.min(Rankle_sagital_AV[event3+1:event2-1])
    max_extension_knee_AV = np.min(Rknee_sagital_AV[event3+1:event2-1])
    max_extension_hip_AV = np.min(Rhip_sagital_AV[event3+1:event2-1])
    max_extension_trunk_AV = np.min(trunk_sagital_AV[event3+1:event2-1])
    
    max_flexion_ankle_AV = np.max(Rankle_sagital_AV[event3+1:event2-1])
    max_flexion_knee_AV = np.max(Rknee_sagital_AV[event3+1:event2-1])
    max_flexion_hip_AV = np.max(Rhip_sagital_AV[event3+1:event2-1])
    max_flexion_turnk_AV = np.max(trunk_sagital_AV[event3+1:event2-1])
    
    ankle_AV_2TouchDown = Rankle_sagital_AV[event1]
    knee_AV_2TouchDown = Rknee_sagital_AV[event1]
    hip_AV_2TouchDown = Rhip_sagital_AV[event1]
    trunk_AV_2TouchDown = trunk_sagital_AV[event1]
    
    ankle_AV_ToeOff = Rankle_sagital_AV[event2]
    knee_AV_ToeOff = Rknee_sagital_AV[event2]
    hip_AV_ToeOff = Rhip_sagital_AV[event2]
    trunk_AV_ToeOff = trunk_sagital_AV[event2]
    
    ankle_AV_1TouchDown = Rankle_sagital_AV[event3]
    knee_AV_1TouchDown = Rknee_sagital_AV[event3]
    hip_AV_1TouchDown = Rhip_sagital_AV[event3]
    trunk_AV_1TouchDown = trunk_sagital_AV[event3]
    
    stride_length = Rank_X[event3] - Rank_X[event2]
    
    event = {
        
        "first toe off ": event3,
        "touch down ": event1,
        "second toe off ": event2
        
        
        }
    
    av_data = {
        "ankle_AV": Rankle_sagital_AV[event3+1:event2-1],
        "knee_AV":Rknee_sagital_AV[event3+1:event2-1],
        "hip_AV":Rhip_sagital_AV[event3+1:event2-1],
        "trunk_AV":trunk_sagital_AV[event3+1:event2-1]
            
            }
    
    data_for_event_setting = {
        "FP_V":FP_vertical[event3+1:event2-1],
        "Rft_LV":Rft_LV[event3+1:event2-1]
        
        }
    
    export_data = {
        "max_extension_ankle_AV": max_extension_ankle_AV,
        "max_extension_knee_AV": max_extension_knee_AV,
        "max_extension_hip_AV": max_extension_hip_AV,
        "max_extension_trunk_AV": max_extension_trunk_AV,
        
        "max_flexion_ankle_AV": max_flexion_ankle_AV,
        "max_flexion_knee_AV": max_flexion_knee_AV,
        "max_flexion_hip_AV": max_flexion_hip_AV,
        "max_flexion_turnk_AV": max_flexion_turnk_AV,
        
        "1TouchDown_ankle_AV": ankle_AV_1TouchDown,
        "1TouchDown_knee_AV": knee_AV_1TouchDown,
        "1TouchDown_hip_AV": hip_AV_1TouchDown,
        "1TouchDown_trunk_AV": trunk_AV_1TouchDown,
        
        "2TouchDown_ankle_AV": ankle_AV_2TouchDown,
        "2TouchDown_knee_AV": knee_AV_2TouchDown,
        "2TouchDown_hip_AV": hip_AV_2TouchDown,
        "2TouchDown_trunk_AV": trunk_AV_2TouchDown,
        
        "ToeOff_ankle_AV": ankle_AV_ToeOff,
        "ToeOff_knee_AV": knee_AV_ToeOff,
        "ToeOff_hip_AV": hip_AV_ToeOff,
        "ToeOfftrunk_AV": trunk_AV_ToeOff,
        "stride_length": stride_length,
        "stride_time": (event2 - event3) / 100    
        }

    return export_data, av_data, event, data_for_event_setting
def process_all_files(folder_path, cuttime1, cuttime2):
    results = {}
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path): 
            try:
                export_data, av_data, event, data_for_event_setting = process_data(file_path, cuttime1, cuttime2)
                results[file_name] = {'export_data': export_data, 'av_data': av_data, 'event': event, "data_eventsetting": data_for_event_setting}

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    return results



# the path of the fold
folder_path_IJ = '/Users/kairenzheng/Desktop/gait_study_assgiment/report_data/IJ_walk'
IJ_walk_results = process_all_files(folder_path_IJ , 140, 350)

folder_path_KJ = '/Users/kairenzheng/Desktop/gait_study_assgiment/report_data/KJ_walk'
KJ_walk_results = process_all_files(folder_path_KJ, 140, 350)

folder_path_KR = '/Users/kairenzheng/Desktop/gait_study_assgiment/report_data/KR_walk'
KR_walk_results = process_all_files(folder_path_KR, 140, 320)

#%%

# Data names and keys
data_names = [
    "BW1", "BW2", "BW3", "BP1", "BP2", "BP3"
]

# Initialize an empty list to store the DataFrames
all_data = []

# Process "IJ", "KJ", and "KR" sets
for group in ["IJ", "KJ", "KR"]:
    for data_name in data_names:
        # Build the key for the dictionary lookup
        key = f"{group}_Walk_{data_name} - Gait_Analysis.txt"
        
        # Initialize a flag to check if the key is found
        key_found = False

        # Check if the key exists in the corresponding dictionary
        if group == "IJ" and key in IJ_walk_results:
            data = IJ_walk_results[key]["export_data"]
            data_df = pd.DataFrame(list(data.items()), columns=["Parameter", "Value"])
            all_data.append({"name": f"{group}_walk_results_{data_name}", "data": data_df})
            key_found = True
        elif group == "KJ" and key in KJ_walk_results:
            data = KJ_walk_results[key]["export_data"]
            data_df = pd.DataFrame(list(data.items()), columns=["Parameter", "Value"])
            all_data.append({"name": f"{group}_walk_results_{data_name}", "data": data_df})
            key_found = True
        elif group == "KR" and key in KR_walk_results:
            data = KR_walk_results[key]["export_data"]
            data_df = pd.DataFrame(list(data.items()), columns=["Parameter", "Value"])
            all_data.append({"name": f"{group}_walk_results_{data_name}", "data": data_df})
            key_found = True
        
        # If the key was not found in any group, print a message
        if not key_found:
            print(f"Key {key} not found in {group}_walk_results. Skipping...")

# After running this code, `all_data` will contain dictionaries with "name" and "data" for each DataFrame.

#%%

import pandas as pd

# 初始化一個 DataFrame 用來存儲 BW 和 BP 資料
combined_df = pd.DataFrame()

# 處理每個資料集
for idx, data_info in enumerate(all_data):
    data = data_info["data"]
    trial_name = data_info["name"]  # 使用資料集的名稱作為 trial 名稱
    
    df = pd.DataFrame(data)
    
    if "BW" in trial_name:
        if idx == 0:
            # 第一個 BW 資料集保留 'Parameter' 和 'Value'
            trial_row = pd.DataFrame([[""] + [trial_name] + [""] * (df.shape[1] - 2)], columns=df.columns)
            df_with_trial = pd.concat([trial_row, df], ignore_index=True)
        else:
            # 之後的資料集只保留 'Value'，不顯示 'Parameter'
            df_without_param = df.drop(columns=["Parameter"], errors='ignore')
            
            # 確保 trial_row 的列數與 df_without_param 相同
            trial_row = pd.DataFrame([[""] * (df_without_param.shape[1] - 1) + [trial_name]], columns=df_without_param.columns)
            
            # 合併 trial_row 和 df_without_param
            df_with_trial = pd.concat([trial_row, df_without_param], ignore_index=True)
        
        # 合併到 combined_df
        combined_df = pd.concat([combined_df, df_with_trial], axis=1)
    
    elif "BP" in trial_name:
        if idx == 0:
            # 第一個 BP 資料集保留 'Parameter' 和 'Value'
            trial_row = pd.DataFrame([[""] + [trial_name] + [""] * (df.shape[1] - 2)], columns=df.columns)
            df_with_trial = pd.concat([trial_row, df], ignore_index=True)
        else:
            # 之後的資料集只保留 'Value'，不顯示 'Parameter'
            df_without_param = df.drop(columns=["Parameter"], errors='ignore')
            
            # 確保 trial_row 的列數與 df_without_param 相同
            trial_row = pd.DataFrame([[""] * (df_without_param.shape[1] - 1) + [trial_name]], columns=df_without_param.columns)
            
            # 合併 trial_row 和 df_without_param
            df_with_trial = pd.concat([trial_row, df_without_param], ignore_index=True)
        
        # 合併到 combined_df
        combined_df = pd.concat([combined_df, df_with_trial], axis=1)

# 最終輸出的 Excel 檔案名稱
output_file_path = "Report_Data.xlsx"

# 直接行列轉置並儲存 Excel
with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
    if not combined_df.empty:
        combined_df.T.to_excel(writer, sheet_name="Combined_data", index=False, header=False)  # 轉置後存檔

print(f"轉置後的 Excel 檔案已儲存為: {output_file_path}")

#%% normalization of BW data 
# report for bw data
# time normalization 
def time_normalize(data, target_length):
    original_length = data.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    re_sample_data = interp1d(x_original, data, axis=0, kind='linear')(x_target)
    return re_sample_data

# Data names and keys
data_names = ["BW1", "BW2", "BW3"]

# Initialize empty lists to store the extracted values
av_ankle_bw = []
av_knee_bw = []
av_hip_bw = []
av_trunk_bw = []
av_FP_V_bw = []
av_Rft_LV_bw = []

# Process "IJ", "KJ", and "KR" sets
for group in ["IJ", "KJ", "KR"]:
    for data_name in data_names:
        # Build the key for the dictionary lookup
        key = f"{group}_Walk_{data_name} - Gait_Analysis.txt"

        # Select the appropriate dictionary
        if group == "IJ":
            results_dict = IJ_walk_results
        elif group == "KJ":
            results_dict = KJ_walk_results
        else:  # group == "KR"
            results_dict = KR_walk_results

        # Check if the key exists in the dictionary
        if key in results_dict:
            av_ankle_bw.append(results_dict[key]["av_data"]["ankle_AV"])
            av_knee_bw.append(results_dict[key]["av_data"]["knee_AV"])
            av_hip_bw.append(results_dict[key]["av_data"]["hip_AV"])
            av_trunk_bw.append(results_dict[key]["av_data"]["trunk_AV"])
            av_FP_V_bw.append(results_dict[key]["data_eventsetting"]["FP_V"])
            av_Rft_LV_bw.append(results_dict[key]["data_eventsetting"]["Rft_LV"])
        else:
            print(f"Key {key} not found in {group}_walk_results. Skipping...")
            
nr_av_ankle_bw = [time_normalize(var, 101) for var in av_ankle_bw]
nr_av_knee_bw = [time_normalize(var, 101) for var in av_knee_bw]
nr_av_hip_bw = [time_normalize(var, 101) for var in av_hip_bw]
nr_av_trunk_bw = [time_normalize(var, 101) for var in av_trunk_bw]
nr_av_Rft_LV_bw = [time_normalize(var, 101) for var in av_Rft_LV_bw]
nr_FP_V_bw = [time_normalize(var, 101) for var in av_FP_V_bw]
# mean and sd 
def compute_mean_std(var_list):
    var_array = np.vstack(var_list)  
    mean_values = np.mean(var_array, axis=0) 
    std_values = np.std(var_array, axis=0)    
    return mean_values, std_values

# compute each variable
mean_av_ankle_bw, std_av_ankle_bw = compute_mean_std(nr_av_ankle_bw)
mean_av_knee_bw, std_av_knee_bw = compute_mean_std(nr_av_knee_bw)
mean_av_hip_bw, std_av_hip_bw = compute_mean_std(nr_av_hip_bw)
mean_av_trunk_bw, std_av_trunk_bw = compute_mean_std(nr_av_trunk_bw)
mean_av_Rft_LV_bw, std_av_Rft_LV_bw = compute_mean_std(nr_av_Rft_LV_bw)
mean_FP_V_bw, std_FP_V_bw = compute_mean_std(nr_FP_V_bw)

event1 = np.where(mean_FP_V_bw[1:] > 20)[0]
if event1.size > 0:
    event1 = event1[0]  
    print(f"Touch down (event1): {event1}")

import numpy as np

variables = [nr_av_ankle_bw, nr_av_knee_bw, nr_av_hip_bw, nr_av_trunk_bw, nr_av_Rft_LV_bw, nr_FP_V_bw]
variable_names = ['av_ankle_bw', 'av_knee_bw', 'av_hip_bw', 'av_trunk_bw', 'av_Rft_LV_bw', 'av_FP_V_bw']

# Prepare a matrix with max, min, frame data, and frame numbers
result_matrix_bw = []

# Add header row
header = ["Variable", "Max", "Max Frame", "Min", "Min Frame", "Velocity at Frame 0", "Velocity at Frame 38", "Velocity at Frame 100"]
result_matrix_bw.append(header)

# Loop through each variable and its name
for var, name in zip(variables, variable_names):
    for i, sub_var in enumerate(var):
        max_val = np.max(sub_var)
        min_val = np.min(sub_var)
        max_frame = np.argmax(sub_var)  # Get frame index of max value
        min_frame = np.argmin(sub_var)  # Get frame index of min value
        frame_0_val = sub_var[0]  # 0th frame (index 0)
        frame_38_val = sub_var[37]  # 38th frame (index 37)
        frame_100_val = sub_var[100]  # 100th frame (index 100)

        # Append the results
        result_matrix_bw.append([f"{name}_{i+1}", max_val, max_frame, min_val, min_frame, frame_0_val, frame_38_val, frame_100_val])

# Convert to a numpy array for easier handling (optional)
result_matrix_bw = np.array(result_matrix_bw, dtype=object)


#%% normalization of BP data 
# report for bw data
# time normalization 
def time_normalize(data, target_length):
    original_length = data.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    re_sample_data = interp1d(x_original, data, axis=0, kind='linear')(x_target)
    return re_sample_data

# Data names and keys
data_names = ["BP1", "BP2", "BP3"]

# Initialize empty lists to store the extracted values
av_ankle_bp = []
av_knee_bp = []
av_hip_bp = []
av_trunk_bp = []
av_FP_V_bp = []
av_Rft_LV_bp = []

# Process "IJ", "KJ", and "KR" sets
for group in ["IJ", "KJ", "KR"]:
    for data_name in data_names:
        # Build the key for the dictionary lookup
        key = f"{group}_Walk_{data_name} - Gait_Analysis.txt"

        # Select the appropriate dictionary
        if group == "IJ":
            results_dict = IJ_walk_results
        elif group == "KJ":
            results_dict = KJ_walk_results
        else:  # group == "KR"
            results_dict = KR_walk_results

        # Check if the key exists in the dictionary
        if key in results_dict:
            av_ankle_bp.append(results_dict[key]["av_data"]["ankle_AV"])
            av_knee_bp.append(results_dict[key]["av_data"]["knee_AV"])
            av_hip_bp.append(results_dict[key]["av_data"]["hip_AV"])
            av_trunk_bp.append(results_dict[key]["av_data"]["trunk_AV"])
            av_FP_V_bp.append(results_dict[key]["data_eventsetting"]["FP_V"])
            av_Rft_LV_bp.append(results_dict[key]["data_eventsetting"]["Rft_LV"])
        else:
            print(f"Key {key} not found in {group}_walk_results. Skipping...")
            
nr_av_ankle_bp = [time_normalize(var, 101) for var in av_ankle_bp]
nr_av_knee_bp = [time_normalize(var, 101) for var in av_knee_bp]
nr_av_hip_bp = [time_normalize(var, 101) for var in av_hip_bp]
nr_av_trunk_bp = [time_normalize(var, 101) for var in av_trunk_bp]
nr_av_Rft_LV_bp = [time_normalize(var, 101) for var in av_Rft_LV_bp]
nr_FP_V_bp = [time_normalize(var, 101) for var in av_FP_V_bp]
# mean and sd 
def compute_mean_std(var_list):
    var_array = np.vstack(var_list)  
    mean_values = np.mean(var_array, axis=0) 
    std_values = np.std(var_array, axis=0)    
    return mean_values, std_values

# compute each variable
mean_av_ankle_bp, std_av_ankle_bp = compute_mean_std(nr_av_ankle_bp)
mean_av_knee_bp, std_av_knee_bp = compute_mean_std(nr_av_knee_bp)
mean_av_hip_bp, std_av_hip_bp = compute_mean_std(nr_av_hip_bp)
mean_av_trunk_bp, std_av_trunk_bp = compute_mean_std(nr_av_trunk_bp)
mean_av_Rft_LV_bp, std_av_Rft_LV_bp = compute_mean_std(nr_av_Rft_LV_bp)
mean_FP_V_bp, std_FP_V_bp = compute_mean_std(nr_FP_V_bp)

event1 = np.where(mean_FP_V_bp[1:] > 20)[0]
if event1.size > 0:
    event1 = event1[0]  
    print(f"Touch down (event1): {event1}")

import numpy as np

variables = [nr_av_ankle_bp, nr_av_knee_bp, nr_av_hip_bp, nr_av_trunk_bp, nr_av_Rft_LV_bp, nr_FP_V_bp]
variable_names = ['av_ankle_bp', 'av_knee_bp', 'av_hip_bp', 'av_trunk_bp', 'av_Rft_LV_bp', 'av_FP_V_bp']

# Prepare a matrix with max, min, frame data, and frame numbers
result_matrix_bp= []

# Add header row
header = ["Variable", "Max", "Max Frame", "Min", "Min Frame", "Velocity at Frame 0", "Velocity at Frame 38", "Velocity at Frame 100"]
result_matrix_bp.append(header)

# Loop through each variable and its name
for var, name in zip(variables, variable_names):
    for i, sub_var in enumerate(var):
        max_val = np.max(sub_var)
        min_val = np.min(sub_var)
        max_frame = np.argmax(sub_var)  # Get frame index of max value
        min_frame = np.argmin(sub_var)  # Get frame index of min value
        frame_0_val = sub_var[0]  # 0th frame (index 0)
        frame_38_val = sub_var[37]  # 38th frame (index 37)
        frame_100_val = sub_var[100]  # 100th frame (index 100)

        # Append the results
        result_matrix_bp.append([f"{name}_{i+1}", max_val, max_frame, min_val, min_frame, frame_0_val, frame_38_val, frame_100_val])

# Convert to a numpy array for easier handling (optional)
result_matrix_bp = np.array(result_matrix_bp, dtype=object)

#%%
# ankle AV
frame_number = np.arange(0, 101)  # Frames 1 to 101
upper_bound_bp = mean_av_ankle_bp + std_av_ankle_bp
lower_bound_bp = mean_av_ankle_bp - std_av_ankle_bp
upper_bound_bw = mean_av_ankle_bw + std_av_ankle_bw
lower_bound_bw = mean_av_ankle_bw - std_av_ankle_bw
plt.figure(figsize=(8, 6))
plt.plot(frame_number, mean_av_ankle_bp, label='Mean of Ankle BP', color='blue')
plt.fill_between(frame_number, lower_bound_bp, upper_bound_bp, color='blue', alpha=0.2, label='Mean ± Std')
plt.plot(frame_number, mean_av_ankle_bw, label='Mean of Ankle BW', color='red')
plt.fill_between(frame_number, lower_bound_bw, upper_bound_bw, color='red', alpha=0.2, label='Mean ± Std')
plt.axvline(x=0, color='orange', linestyle='--', label='First Toe Off')
plt.axvline(x=event1, color='red', linestyle='--', label='Touch Down')
plt.axvline(x=100, color='blue', linestyle='--', label='Second Toe Off')
plt.title('Ankle Angular Velocity', fontsize=16)
plt.xlabel('Time (percentage)', fontsize=12)
plt.ylabel('Velocity (cm/s)', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# knee AV
upper_bound_bw = mean_av_knee_bw + std_av_knee_bw
lower_bound_bw = mean_av_knee_bw - std_av_knee_bw
upper_bound_bp = mean_av_knee_bp + std_av_knee_bp
lower_bound_bp = mean_av_knee_bp - std_av_knee_bp
plt.figure(figsize=(8, 6))
plt.plot(frame_number, mean_av_knee_bp, label='Mean of Knee BP', color='blue')
plt.fill_between(frame_number, lower_bound_bp, upper_bound_bp, color='blue', alpha=0.2, label='Mean ± Std')
plt.plot(frame_number, mean_av_knee_bw, label='Mean of Knee BW', color='red')
plt.fill_between(frame_number, lower_bound_bw, upper_bound_bw, color='red', alpha=0.2, label='Mean ± Std')
plt.axvline(x=0, color='orange', linestyle='--', label='First Toe Off')
plt.axvline(x=event1, color='red', linestyle='--', label='Touch Down')
plt.axvline(x=100, color='blue', linestyle='--', label='Second Toe Off')
plt.title('Knee Angular Velocity ', fontsize=16)
plt.xlabel('Time (percentage)', fontsize=12)
plt.ylabel('Velocity (cm/s)', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Hip AV
upper_bound_bw = mean_av_hip_bw + std_av_hip_bw
lower_bound_bw = mean_av_hip_bw - std_av_hip_bw
upper_bound_bp = mean_av_hip_bp + std_av_hip_bp
lower_bound_bp = mean_av_hip_bp - std_av_hip_bp
plt.figure(figsize=(8, 6))
plt.plot(frame_number, mean_av_hip_bp, label='Mean of Hip BP', color='blue')
plt.fill_between(frame_number, lower_bound_bp, upper_bound_bp, color='blue', alpha=0.2, label='Mean ± Std')
plt.plot(frame_number, mean_av_hip_bw, label='Mean of Hip BW', color='red')
plt.fill_between(frame_number, lower_bound_bw, upper_bound_bw, color='red', alpha=0.2, label='Mean ± Std')
plt.axvline(x=0, color='orange', linestyle='--', label='First Toe Off')
plt.axvline(x=event1, color='red', linestyle='--', label='Touch Down')
plt.axvline(x=100, color='blue', linestyle='--', label='Second Toe Off')
plt.title('Hip Angular Velocity (BP)', fontsize=16)
plt.xlabel('Time (percentage)', fontsize=12)
plt.ylabel('Velocity (cm/s)', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Trunk AV 
upper_bound_bp = mean_av_trunk_bp + std_av_trunk_bp
lower_bound_bp = mean_av_trunk_bp - std_av_trunk_bp
upper_bound_bw = mean_av_trunk_bw + std_av_trunk_bw
lower_bound_bw = mean_av_trunk_bw - std_av_trunk_bw

plt.figure(figsize=(8, 6))
plt.plot(frame_number, mean_av_trunk_bp, label='Mean of Trunk BP', color='blue')
plt.fill_between(frame_number, lower_bound_bp, upper_bound_bp, color='blue', alpha=0.2, label='Mean ± Std')
plt.plot(frame_number, mean_av_trunk_bw, label='Mean of Trunk BW', color='red')
plt.fill_between(frame_number, lower_bound_bw, upper_bound_bw, color='red', alpha=0.2, label='Mean ± Std')
plt.axvline(x=0, color='orange', linestyle='--', label='First Toe Off')
plt.axvline(x=event1, color='red', linestyle='--', label='Touch Down')
plt.axvline(x=100, color='blue', linestyle='--', label='Second Toe Off')
plt.title('Trunk Angular Velocity', fontsize=16)
plt.xlabel('Time (percentage)', fontsize=12)
plt.ylabel('Velocity (cm/s)', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Place legend outside the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()
