import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    return filtfilt(*butter(order, cutoff / nyquist, btype=filter_type), data, axis=0)

def time_d(data, interval):
    return np.gradient(data, interval, axis=0)

def time_dd(data, interval):
    return np.gradient(np.gradient(data, interval, axis=0), interval, axis=0)

def calculate_theta(data1, data2):
    m = np.shape(data1)[0]
    theta = np.zeros(m)
    for i in range(m):
        A = data1[i, :]
        B = data2[i, :]
        nor_A = A / np.linalg.norm(A)
        nor_B = B / np.linalg.norm(B)
        R = np.array([[nor_A[0], nor_A[1]], [-nor_A[1], nor_A[0]]])
        c = np.dot(R, nor_B)
        theta[i] = np.arctan2(c[1], c[0])
    return theta

def unwrap_deg(data):
    data = data.copy()
    dp = np.diff(data)
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    dp_corr = dps - dp
    dp_corr[np.abs(dp) < np.pi] = 0
    data[1:] += np.cumsum(dp_corr)
    return data

def process(file_path, output_path):
    
    
    df = pd.read_csv(file_path, header=None)

    fs = int(float(df.iloc[2, 1]))
    sampling_interval = 1 / fs

    segment_names = ['mid_hip', 'Rhip', 'RKnee', 'RAnkle', 'Lhip', 'LKnee', 'LAnkle']
    segment_columns = [(23, 26), (26, 29), (29, 32), (32, 35), (35, 38), (38, 41), (41, 44)]
    segments = {name: df.iloc[6:, start:end].astype(float).to_numpy() for name, (start, end) in zip(segment_names, segment_columns)}

    def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
        nyquist = 0.5 * fs
        return filtfilt(*butter(order, cutoff / nyquist, btype=filter_type), data, axis=0)

    def time_d(data, interval):
        return np.gradient(data, interval, axis=0)

    def time_dd(data, interval):
        return np.gradient(np.gradient(data, interval, axis=0), interval, axis=0)

    cutoff = 6
    filtered_segments = {name: butterworth_filter(seg, cutoff, fs) for name, seg in segments.items()}
    linear_velocity = {name: time_d(filtered_segments[name], sampling_interval) for name in segment_names}
    linear_acceleration = {name: time_dd(filtered_segments[name], sampling_interval) for name in segment_names}

    mid_hip_y_pos = filtered_segments["mid_hip"][:, 1]
    threshold = np.mean(mid_hip_y_pos[:5]) - 0.05
    event1 = np.where(mid_hip_y_pos < threshold)[0][0] if np.any(mid_hip_y_pos < threshold) else None
    event2 = np.where(mid_hip_y_pos[event1:] >= threshold)[0][0] + event1 if event1 is not None and np.any(mid_hip_y_pos[event1:] >= threshold) else None

    Lhip_xy_position = filtered_segments['Lhip'][:, 0:2]
    LKnee_xy_position = filtered_segments['LKnee'][:, 0:2]
    LAnkle_xy_position = filtered_segments['LAnkle'][:, 0:2]

    Lankle_knee_line = LAnkle_xy_position - LKnee_xy_position
    Lhip_knee_line = Lhip_xy_position - LKnee_xy_position

    def calculate_theta(data1, data2):
        m = np.shape(data1)[0]
        theta = np.zeros(m)
        for i in range(m):
            A = data1[i, :]
            B = data2[i, :]
            nor_A = A / np.linalg.norm(A)
            nor_B = B / np.linalg.norm(B)
            theta[i] = np.arctan2(np.cross(nor_A, nor_B), np.dot(nor_A, nor_B))
        return theta

    Lknee_sagital_angle = calculate_theta(Lankle_knee_line, Lhip_knee_line)

    def unwrap_deg(data):
        dp = np.diff(data)
        dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
        dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
        dp_corr = dps - dp
        dp_corr[np.abs(dp) < np.pi] = 0
        data[1:] += np.cumsum(dp_corr)
        return data

    Lknee_sagital_angle = (unwrap_deg(Lknee_sagital_angle)) * (180 / np.pi)
    Lknee_sagital_angle = Lknee_sagital_angle - Lknee_sagital_angle[0]
    Lknee_sagital_angle = 180 - Lknee_sagital_angle
    max_flexion_knee_angle = np.min(Lknee_sagital_angle)



    export = {
        "Rknee_sagital_angle":  Lknee_sagital_angle[event1:event2],
        "Max_knee_flexion_angle": ([max_flexion_knee_angle] * (event2 - event1))
    }
    
    
        
    # for name in segment_names:
    #     fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    #     fig.suptitle(f'{name} Position, Velocity, and Acceleration', fontsize=16)
        
    #     # pick data from event1 to event2 
    #     position_data = filtered_segments[name][event1:event2]
    #     velocity_data = linear_velocity[name][event1:event2]
    #     acceleration_data = linear_acceleration[name][event1:event2]  
        
    #     for idx, (data, label, ylabel) in enumerate(zip(
    #         [position_data, velocity_data, acceleration_data],
    #         ['Position', 'Linear Velocity', 'Linear Acceleration'],
    #         ['Position (m)', 'Velocity (m/s)', 'Acceleration (m/s²)'])):
    
    #         axs[idx].plot(data[:, 0], label='X')
    #         axs[idx].plot(data[:, 1], label='Y')
    #         axs[idx].plot(data[:, 2], label='Z')
    #         axs[idx].set_title(f'{name} {label}')
    #         axs[idx].set_xlabel('Frame')
    #         axs[idx].set_ylabel(ylabel)
    #         axs[idx].legend(loc='upper right') 
    
    #         if event1 is not None:
    #             axs[idx].axvline(x=0, color='r', linestyle='--', label='Event 1')  # event1 在切片後為 0
    #         if event2 is not None:
    #             axs[idx].axvline(x=event2 - event1, color='g', linestyle='--', label='Event 2')
    
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    #     plt.show()

    output_df = pd.DataFrame(export)
    output_df.to_csv('/Users/kairenzheng/Desktop/squat3_opencap.csv', index=False)


# Example usage:
file_path = '/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/squat3.csv'
output_path = '/Users/kairenzheng/Desktop'
process(file_path, output_path)

#%% deal with tracker data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def calculate_theta(data1, data2):
    m = np.shape(data1)[0]
    theta = np.zeros(m)
    for i in range(m):
        A = data1[i, :]
        B = data2[i, :]
        nor_A = A / np.linalg.norm(A)
        nor_B = B / np.linalg.norm(B)
        R = np.array([[nor_A[0], nor_A[1]], [-nor_A[1], nor_A[0]]])
        c = np.dot(R, nor_B)
        theta[i] = np.arctan2(c[1], c[0])
    return theta

def unwrap_deg(data):
    data = data.copy()
    dp = np.diff(data)
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    dp_corr = dps - dp
    dp_corr[np.abs(dp) < np.pi] = 0
    data[1:] += np.cumsum(dp_corr)
    return data

file_path = '/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/squat3＿tracker.csv'



df = pd.read_csv(file_path, header=None)
# sampling interval 
sampling_interval = float(df.iloc[3, 0])  

# 計算取樣頻率
fs = 1 / sampling_interval


Lhip_xy_position = df.iloc[2:, 1:3].astype(float).to_numpy()
LKnee_xy_position = df.iloc[2:, 4:6].astype(float).to_numpy()
Lankle_xy_position = df.iloc[2:, 7:9].astype(float).to_numpy()

Lankle_knee_line = Lankle_xy_position - LKnee_xy_position
Lhip_knee_line = Lhip_xy_position - LKnee_xy_position


Lknee_sagital_angle = calculate_theta(Lhip_knee_line, Lankle_knee_line)


Lknee_sagital_angle = (unwrap_deg(Lknee_sagital_angle)) * (180 / np.pi)
Lknee_sagital_angle = Lknee_sagital_angle - Lknee_sagital_angle[0]


mid_hip_y_pos = Lhip_xy_position[:, 1]
threshold = np.mean(mid_hip_y_pos[:5]) - 0.05
event1 = np.where(mid_hip_y_pos < threshold)[0][0] if np.any(mid_hip_y_pos < threshold) else None
event2 = np.where(mid_hip_y_pos[event1:] >= threshold)[0][0] + event1 if event1 is not None and np.any(mid_hip_y_pos[event1:] >= threshold) else None

Lknee_sagital_angle = Lknee_sagital_angle[event1: event2]
Lknee_sagital_angle = 180 - Lknee_sagital_angle
max_flexion_knee_angle = np.min(Lknee_sagital_angle)


export = {
    "Rknee_sagital_angle":  Lknee_sagital_angle,
    "Max_knee_flexion_angle": ([max_flexion_knee_angle] * len(Lknee_sagital_angle))
}

output_df = pd.DataFrame(export)
output_df.to_csv('/Users/kairenzheng/Desktop/squat3_tracker.csv', index=False)

#%%  compared two data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 讀取資料
df = pd.read_csv('/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/results.csv')

# 提取膝關節彎曲角度和最大值
def extract_data(start_col):
    knee_flexion = df.iloc[:, start_col].dropna().astype(float).to_numpy()
    max_flexion = df.iloc[0, start_col + 1]
    return knee_flexion, max_flexion

opencap_cols = [1, 4, 7]
tracker_cols = [10, 13, 16]
opencap_data = [extract_data(col) for col in opencap_cols]
tracker_data = [extract_data(col) for col in tracker_cols]

# 計算 RMS 差異
rms = np.sqrt(np.mean([(o[1] - t[1]) ** 2 for o, t in zip(opencap_data, tracker_data)]))
print("rms:", rms)

# 時間正規化
def time_normalize(data, length=100):
    return interp1d(np.linspace(0, 1, len(data)), data, axis=0, kind='linear')(np.linspace(0, 1, length))

# 整理時間正規化後的資料
data_100 = [np.vstack((time_normalize(o[0]), time_normalize(t[0]))).T for o, t in zip(opencap_data, tracker_data)]

# ICC 計算
def icc_calculate(Y, icc_type):
    n, k = Y.shape
    mean_Y, SST = np.mean(Y), ((Y - np.mean(Y)) ** 2).sum()
    x, x0 = np.kron(np.eye(k), np.ones((n, 1))), np.tile(np.eye(n), (k, 1))
    X = np.hstack([x, x0])
    predicted_Y = X @ np.linalg.pinv(X.T @ X) @ X.T @ Y.flatten("F")
    SSE, MSE = ((Y.flatten("F") - predicted_Y) ** 2).sum(), ((Y.flatten("F") - predicted_Y) ** 2).sum() / ((n - 1) * (k - 1))
    MSC, MSR = (((np.mean(Y, 0) - mean_Y) ** 2).sum() * n) / (k - 1), (((np.mean(Y, 1) - mean_Y) ** 2).sum() * k) / (n - 1)

    if icc_type == "icc(2)":
        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
        return ICC1

# 計算並輸出 ICC 值
for i, data in enumerate(data_100, 1):
    print(f"ICC(2,1) for squat {i}:", icc_calculate(data, "icc(2)"))

# 繪製圖表：將每組 opencap 和 tracker 數據畫在同一張圖上
for i, data in enumerate(data_100, 1):
    plt.figure(figsize=(8, 5))
    plt.plot(data[:, 0], label='OpenCap Knee Flexion')
    plt.plot(data[:, 1], label='Tracker Knee Flexion')
    plt.title(f'Squat {i}: OpenCap vs Tracker Knee Flexion')
    plt.xlabel('Normalized Frame (%)')
    plt.ylabel('Knee Flexion Angle (degrees)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
#%% open cap wihtin subject RMS
import numpy as np

squat1_open = data_100[0][:, 1]
squat2_open = data_100[1][:, 1]
squat3_open = data_100[2][:, 1]

# Compute mean per frame
mean_per_frame = np.mean([squat1_open, squat2_open, squat3_open], axis=0)

# Compute RMS for each frame
rms_per_frame1 = np.sqrt((squat1_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff1 = np.mean(rms_per_frame1)

print("RMS per frame:", rms_per_frame1)
print("Average RMS difference:", avg_rms_diff1)

# Compute RMS for each frame
rms_per_frame2 = np.sqrt((squat2_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff2 = np.mean(rms_per_frame2)

# Compute RMS for each frame
rms_per_frame2= np.sqrt((squat2_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff2 = np.mean(rms_per_frame2)

# Compute RMS for each frame
rms_per_frame3= np.sqrt((squat3_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff3 = np.mean(rms_per_frame3)

# Compute RMS for each frame
rms_per_frame3= np.sqrt((squat3_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff3 = np.mean(rms_per_frame3)

average_rms = (avg_rms_diff1 + avg_rms_diff2 + avg_rms_diff3) / 3

#%% tracker wihtin subject RMS

import numpy as np

squat1_open = data_100[0][:, 1]
squat2_open = data_100[1][:, 1]
squat3_open = data_100[2][:, 1]

# Compute mean per frame
mean_per_frame = np.mean([squat1_open, squat2_open, squat3_open], axis=0)

# Compute RMS for each frame
rms_per_frame1 = np.sqrt((squat1_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff1 = np.mean(rms_per_frame1)

print("RMS per frame:", rms_per_frame1)
print("Average RMS difference:", avg_rms_diff1)

# Compute RMS for each frame
rms_per_frame2 = np.sqrt((squat2_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff2 = np.mean(rms_per_frame2)

# Compute RMS for each frame
rms_per_frame2= np.sqrt((squat2_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff2 = np.mean(rms_per_frame2)

# Compute RMS for each frame
rms_per_frame3= np.sqrt((squat3_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff3 = np.mean(rms_per_frame3)

# Compute RMS for each frame
rms_per_frame3= np.sqrt((squat3_open - mean_per_frame) ** 2)

# Compute the average RMS difference across all frames
avg_rms_diff3 = np.mean(rms_per_frame3)

average_rms = (avg_rms_diff1 + avg_rms_diff2 + avg_rms_diff3) / 3


    
    
    
    
    
#%% to get linear kinematics 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    return filtfilt(*butter(order, cutoff / nyquist, btype=filter_type), data, axis=0)

def time_d(data, interval):
    return np.gradient(data, interval, axis=0)

def time_dd(data, interval):
    return np.gradient(np.gradient(data, interval, axis=0), interval, axis=0)


file_path = '/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/VJ.csv'


df = pd.read_csv(file_path, header=None)

fs = int(float(df.iloc[2, 1]))
sampling_interval = 1 / fs

segment_names = ['mid_hip', 'Rhip', 'RKnee', 'RAnkle', 'Lhip', 'LKnee', 'LAnkle']
segment_columns = [(23, 26), (26, 29), (29, 32), (32, 35), (35, 38), (38, 41), (41, 44)]
segments = {name: df.iloc[6:, start:end].astype(float).to_numpy() for name, (start, end) in zip(segment_names, segment_columns)}

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    return filtfilt(*butter(order, cutoff / nyquist, btype=filter_type), data, axis=0)

def time_d(data, interval):
    return np.gradient(data, interval, axis=0)

def time_dd(data, interval):
    return np.gradient(np.gradient(data, interval, axis=0), interval, axis=0)

cutoff = 6
filtered_segments = {name: butterworth_filter(seg, cutoff, fs) for name, seg in segments.items()}
linear_velocity = {name: time_d(filtered_segments[name], sampling_interval) for name in segment_names}
linear_acceleration = {name: time_dd(filtered_segments[name], sampling_interval) for name in segment_names}


for name in segment_names:
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{name} Position, Velocity, and Acceleration', fontsize=16)
    
    # Pick data from event1 to event2
    position_data = filtered_segments[name][event1:event2]
    velocity_data = linear_velocity[name][event1:event2]
    acceleration_data = linear_acceleration[name][event1:event2]
    
    for idx, (data, label, ylabel) in enumerate(zip(
        [position_data, velocity_data, acceleration_data],
        ['Position', 'Linear Velocity', 'Linear Acceleration'],
        ['Position (m)', 'Velocity (m/s)', 'Acceleration (m/s²)'])):

        axs[idx].plot(data[:, 0], label='X')
        axs[idx].plot(data[:, 1], label='Y')
        axs[idx].plot(data[:, 2], label='Z')
        axs[idx].set_title(f'{name} {label}')
        axs[idx].set_xlabel('Frame')
        axs[idx].set_ylabel(ylabel)
        axs[idx].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


