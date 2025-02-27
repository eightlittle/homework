import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# run all functions that we need to 
def process(file_path, output_path):
    #open data
    df = pd.read_csv(file_path, header=None)
    # pick up frequency
    fs = int(float(df.iloc[2, 1]))
    # time interrval = 1 / frequenccy
    sampling_interval = 1 / fs
    # pick up points
    segment_names = ['mid_hip', 'Rhip', 'RKnee', 'RAnkle', 'Lhip', 'LKnee', 'LAnkle']
    segment_columns = [(23, 26), (26, 29), (29, 32), (32, 35), (35, 38), (38, 41), (41, 44)]
    # change variable types
    segments = {name: df.iloc[6:, start:end].astype(float).to_numpy() for name, (start, end) in zip(segment_names, segment_columns)}
    
    # run function for each points
    def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
        nyquist = 0.5 * fs
        return filtfilt(*butter(order, cutoff / nyquist, btype=filter_type), data, axis=0)

    # set up elements for functions 
    cutoff = 6
    filtered_segments = {name: butterworth_filter(seg, cutoff, fs) for name, seg in segments.items()}

    # set up events -> standard it 5 cm downward 
    mid_hip_y_pos = filtered_segments["mid_hip"][:, 1]
    threshold = np.mean(mid_hip_y_pos[:5]) - 0.05
    event1 = np.where(mid_hip_y_pos < threshold)[0][0] if np.any(mid_hip_y_pos < threshold) else None
    event2 = np.where(mid_hip_y_pos[event1:] >= threshold)[0][0] + event1 if event1 is not None and np.any(mid_hip_y_pos[event1:] >= threshold) else None
    
    # set up elements to get relative angles 
    Lhip_xy_position = filtered_segments['Lhip'][:, 0:2]
    LKnee_xy_position = filtered_segments['LKnee'][:, 0:2]
    LAnkle_xy_position = filtered_segments['LAnkle'][:, 0:2]
    
    # create two vectors
    Lankle_knee_line = LAnkle_xy_position - LKnee_xy_position
    Lhip_knee_line = Lhip_xy_position - LKnee_xy_position
    
    # get relative angles
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
    
    # check if it has gimbal lock and change unit to degree
    Lknee_sagital_angle = (unwrap_deg(Lknee_sagital_angle)) * (180 / np.pi)
    
    # set up the original angle is the first frame
    Lknee_sagital_angle = Lknee_sagital_angle - Lknee_sagital_angle[0]
    
    # get the knee angle 
    Lknee_sagital_angle = 180 - Lknee_sagital_angle
    
    # pick up the minimum knee angle 
    max_flexion_knee_angle = np.min(Lknee_sagital_angle)
    
    # export data 
    export = {
        "Rknee_sagital_angle":  Lknee_sagital_angle[event1:event2],
        "Max_knee_flexion_angle": ([max_flexion_knee_angle] * (event2 - event1))
    }
    
    output_df = pd.DataFrame(export)
    # if you need to use -> change the output file name 
    # output to csv file 
    output_df.to_csv('/Users/kairenzheng/Desktop/squat3_opencap.csv', index=False)


# Example usage: 
# if you need to use -> change the input file name 
file_path = '/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/squat3.csv'
output_path = '/Users/kairenzheng/Desktop'
process(file_path, output_path)

#%% deal with tracker data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# create function of relative angle
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

# create function for gimbal lock problem
def unwrap_deg(data):
    data = data.copy()
    dp = np.diff(data)
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    dp_corr = dps - dp
    dp_corr[np.abs(dp) < np.pi] = 0
    data[1:] += np.cumsum(dp_corr)
    return data

# if you need to use -> change the input file name 
# input file
file_path = '/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/squat3＿tracker.csv'

    #open data
df = pd.read_csv(file_path, header=None)

# get sampling interval 
sampling_interval = float(df.iloc[3, 0])  

# get frequency = 1 / sampling_interval
fs = 1 / sampling_interval

# pick up positions
Lhip_xy_position = df.iloc[2:, 1:3].astype(float).to_numpy()
LKnee_xy_position = df.iloc[2:, 4:6].astype(float).to_numpy()
Lankle_xy_position = df.iloc[2:, 7:9].astype(float).to_numpy()

# create vectors 
Lankle_knee_line = Lankle_xy_position - LKnee_xy_position
Lhip_knee_line = Lhip_xy_position - LKnee_xy_position

# get relative angles
Lknee_sagital_angle = calculate_theta(Lhip_knee_line, Lankle_knee_line)

# check if it has gimbal lock and change unit to degree
Lknee_sagital_angle = (unwrap_deg(Lknee_sagital_angle)) * (180 / np.pi)

# set up the original angle is the first frame
Lknee_sagital_angle = Lknee_sagital_angle - Lknee_sagital_angle[0]

# elements for events 
mid_hip_y_pos = Lhip_xy_position[:, 1]
threshold = np.mean(mid_hip_y_pos[:5]) - 0.05
event1 = np.where(mid_hip_y_pos < threshold)[0][0] if np.any(mid_hip_y_pos < threshold) else None
event2 = np.where(mid_hip_y_pos[event1:] >= threshold)[0][0] + event1 if event1 is not None and np.any(mid_hip_y_pos[event1:] >= threshold) else None

# cut data -> data during from event1 to event2 
Lknee_sagital_angle = Lknee_sagital_angle[event1: event2]
# find the knee flexion angle 
Lknee_sagital_angle = 180 - Lknee_sagital_angle

# find the minimum knee flexion angle 
max_flexion_knee_angle = np.min(Lknee_sagital_angle)

# export data 
export = {
    "Rknee_sagital_angle":  Lknee_sagital_angle,
    "Max_knee_flexion_angle": ([max_flexion_knee_angle] * len(Lknee_sagital_angle))
}

output_df = pd.DataFrame(export)
# output to csv file 
# if you need to use -> change the output file name 
output_df.to_csv('/Users/kairenzheng/Desktop/squat3_tracker.csv', index=False)

#%%  compared two data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# open data 
df = pd.read_csv('/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/results.csv')

# pick knee flexion data and max knee flexion angle 
def extract_data(start_col):
    knee_flexion = df.iloc[:, start_col].dropna().astype(float).to_numpy()
    max_flexion = df.iloc[0, start_col + 1]
    return knee_flexion, max_flexion

# pick up datas 
opencap_cols = [1, 4, 7]
tracker_cols = [10, 13, 16]
opencap_data = [extract_data(col) for col in opencap_cols]
tracker_data = [extract_data(col) for col in tracker_cols]

# calculate rms
rms = np.sqrt(np.mean([(o[1] - t[1]) ** 2 for o, t in zip(opencap_data, tracker_data)]))
print("rms:", rms)

# create function for time normalization -> to 100 frames 
def time_normalize(data, length=100):
    return interp1d(np.linspace(0, 1, len(data)), data, axis=0, kind='linear')(np.linspace(0, 1, length))

# run time normalization
data_100 = [np.vstack((time_normalize(o[0]), time_normalize(t[0]))).T for o, t in zip(opencap_data, tracker_data)]

# create ICC function 
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

# run ICC 
for i, data in enumerate(data_100, 1):
    print(f"ICC(2,1) for squat {i}:", icc_calculate(data, "icc(2)"))

# make graph 
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

# pick up data 
# knee flexion angle of opencap
squat1_open = data_100[0][:, 0]
squat2_open = data_100[1][:, 0]
squat3_open = data_100[2][:, 0]

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

# average rms of three trails 
average_rms = (avg_rms_diff1 + avg_rms_diff2 + avg_rms_diff3) / 3

#%% tracker wihtin subject RMS

import numpy as np

# pick up data 
# knee flexion angle of tracker
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

# average rms of three trails 
average_rms = (avg_rms_diff1 + avg_rms_diff2 + avg_rms_diff3) / 3
#%% to get linear kinematics 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# filtering -> cutoff frequency set up 6
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    return filtfilt(*butter(order, cutoff / nyquist, btype=filter_type), data, axis=0)

# time derivate 
def time_d(data, interval):
    return np.gradient(data, interval, axis=0)

# time derivate 
def time_dd(data, interval):
    return np.gradient(np.gradient(data, interval, axis=0), interval, axis=0)

# calculate relative angle 
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

# if we have gimbal lock -> use this function to fix it 
def unwrap_deg(data):
    data = data.copy()
    dp = np.diff(data)
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    dp_corr = dps - dp
    dp_corr[np.abs(dp) < np.pi] = 0
    data[1:] += np.cumsum(dp_corr)
    return data

# input data
file_path = '/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/KINE7670_homeworks/opencap_study/data_opencap_squat_trc/VJ.csv'

# run data
df = pd.read_csv(file_path, header=None)

# get frequency
fs = int(float(df.iloc[2, 1]))

# get time interval = 1 / ferquency
sampling_interval = 1 / fs

# pick up positions 
segment_names = ['mid_hip', 'Rhip', 'RKnee', 'RAnkle', 'Lhip', 'LKnee', 'LAnkle']
segment_columns = [(23, 26), (26, 29), (29, 32), (32, 35), (35, 38), (38, 41), (41, 44)]
segments = {name: df.iloc[6:, start:end].astype(float).to_numpy() for name, (start, end) in zip(segment_names, segment_columns)}


# set up elements 
cutoff = 6

# run butterworth_filter
filtered_segments = {name: butterworth_filter(seg, cutoff, fs) for name, seg in segments.items()}

# get lineart velocity 
linear_velocity = {name: time_d(filtered_segments[name], sampling_interval) for name in segment_names}

# get linear acceleration 
linear_acceleration = {name: time_dd(filtered_segments[name], sampling_interval) for name in segment_names}

# make graph
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


