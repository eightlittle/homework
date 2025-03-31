import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Dr. Weimar's lab setting
# X = bilateral axis
# Y = anterior-posterior axis
# Z = polar axis

# functions
# Butterworth filter function
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)
#%%
file_path = r'/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/AU_classes/2025 spring/KINE_7620/assignment_position_velocity_acc/winterdataset.csv'
df = pd.read_csv(file_path, header=None)

# Pick up time (starting from row 3 if that's where your data starts)
time = df.iloc[2:, 1].astype(float).reset_index(drop=True)

# Sampling information
sampling_interval = 0.0145
fs = 1 / sampling_interval  # Sampling frequency
fc = 6  # Cutoff frequency for filtering

    
r_ank_x = df.iloc[2:, 6].astype(float).to_numpy()
r_ank_y = df.iloc[2:, 7].astype(float).to_numpy()

r_knee_x = df.iloc[2:, 4].astype(float).to_numpy()
r_knee_y = df.iloc[2:, 5].astype(float).to_numpy()


r_hip_x = df.iloc[2:, 2].astype(float).to_numpy()
r_hip_y = df.iloc[2:, 3].astype(float).to_numpy()

rank_X = butterworth_filter(r_ank_x*100, fc, fs, order=4, filter_type='low')
rank_Y = butterworth_filter(r_ank_y*100, fc, fs, order=4, filter_type='low')
rknee_X = butterworth_filter(r_knee_x*100, fc, fs, order=4, filter_type='low')
rknee_Y = butterworth_filter(r_knee_y*100, fc, fs, order=4, filter_type='low')
rhip_X = butterworth_filter(r_hip_x*100, fc, fs, order=4, filter_type='low')
rhip_Y = butterworth_filter(r_hip_y*100, fc, fs, order=4, filter_type='low')

#%%
import numpy as np

# --- 角度計算函式 ---

def angular_distance(A, B):
    """計算兩個向量的夾角（適用於整個時間序列）"""
    dot_product = np.sum(A * B, axis=1)
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    cos_theta = dot_product / (norm_A * norm_B)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta)

def relative_angular_position(A, B):
    """計算相對角度（使用旋轉矩陣）"""
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    a = A / norm_A
    b = B / norm_B

    R = np.stack([
        np.stack([a[:, 0], a[:, 1]], axis=1),
        np.stack([-a[:, 1], a[:, 0]], axis=1)
    ], axis=1)

    c = np.einsum('ijk,ik->ij', R, b)
    theta = np.arctan2(c[:, 1], c[:, 0])
    return np.degrees(theta)

def TWO_triganle_method(A, B):
    """使用 atan2 計算兩向量夾角（2D 高中數學方式）"""
    theta_A = np.arctan2(A[:, 1], A[:, 0])
    theta_B = np.arctan2(B[:, 1], B[:, 0])
    theta = theta_B - theta_A
    return np.degrees(theta)

def unwrap_deg(data):
    """
    Unwrap angle data in degrees to prevent discontinuities (e.g., jumps at 180/-180 degrees).
    
    input:  data (array-like) in degrees
    output: unwrapped data in degrees (discontinuities corrected)
    """
    # Convert degrees to radians
    data_rad = np.radians(data)
    
    # Perform unwrap in radians
    dp = np.diff(data_rad)
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    dp_corr = dps - dp
    dp_corr[np.abs(dp) < np.pi] = 0
    data_rad[1:] += np.cumsum(dp_corr)
    
    # Convert back to degrees
    return np.degrees(data_rad)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


# --- 建立向量 ---

A = np.stack([rhip_X - rknee_X, rhip_Y - rknee_Y], axis=1)   # hip → knee
B = np.stack([rank_X - rknee_X, rank_Y - rknee_Y], axis=1)   # ankle → knee
C = -B  # knee → ankle，反向向量

# --- 計算角度 ---

angle_abs = angular_distance(A, B)
angle_relative = relative_angular_position(A, B)
angle_two_tri = TWO_triganle_method(A, C)

# --- 處理跳變問題 ---

# angle_abs = unwrap_deg(angle_abs)
angle_relative = unwrap_deg(angle_relative)
angle_two_tri = unwrap_deg(angle_two_tri)
angle_two_tri = angle_two_tri + 180

# 計算 RMS
rms_abs_VS_relative = rms(angle_abs - angle_relative)
rms_abs_VS_two_tri = rms(angle_abs - angle_two_tri)
rms_relative_VS_two_tri = rms(angle_relative - angle_two_tri)

print(f"RMS of absolute vs relative method  : {rms_abs_VS_relative:.2f}°")
print(f"RMS of absolute vs two triangles : {rms_abs_VS_two_tri:.2f}°")
print(f"RMS of relative vs two triangles method   : {rms_relative_VS_two_tri:.2f}°")

fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=300, sharex=True)

# Absolute angle
axs[0].plot(time[1:-1], angle_abs[1:-1], label='Absolute method', color='blue')
axs[0].set_title('Knee Angle - Absolute Method', fontsize=12)
axs[0].set_ylabel('Angle (degree)', fontsize=10)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)

# Relative angle
axs[1].plot(time[1:-1], angle_relative[1:-1], label='Relative method', color='green')
axs[1].set_title('Knee Angle - Relative Method', fontsize=12)
axs[1].set_ylabel('Angle (degree)', fontsize=10)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.7)

# Two triangles angle
axs[2].plot(time[1:-1], angle_two_tri[1:-1], label='Two Triangles method', color='orange')
axs[2].set_title('Knee Angle - Two Triangles Method', fontsize=12)
axs[2].set_xlabel('Time (sec)', fontsize=10)
axs[2].set_ylabel('Angle (degree)', fontsize=10)
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%% angular velocity and angular acceleration 
def time_d(data, sampling_interval):
    length = len(data)
    velocity = np.zeros(length)
    for i in range(length):
        if i == 0 or i == length - 1:
            velocity[i] = 0
        else:
            velocity[i] = (data[i + 1] - data[i - 1]) / (2 * sampling_interval)
    return velocity

def time_dd(data, sampling_interval):
    length = len(data)
    acceleration = np.zeros(length)
    for i in range(length):
        if i == 0 or i == length - 1:
            acceleration[i] = 0
        else:
            acceleration[i] = (data[i + 1] - 2 * data[i] + data[i - 1]) / (sampling_interval ** 2)
    return acceleration

knee_AV = time_d(angle_relative, sampling_interval)
knee_AA = time_dd(angle_relative, sampling_interval)


r_ank_LV_X = time_d(r_ank_x, sampling_interval)

#%%
# find Event1 和 Event2
# set up the standard of touch down = velocity < 1 m/s
event1 = np.where(r_ank_LV_X[1:-1] < 1)[0]
if event1.size > 0:
    event1 = event1[0] 
    print(f"Touch down (Event1): {event1}")
    
# set up the standard of touch down = velocity > 1 m/s after event1 + 3 frame 
event2 = np.where(r_ank_LV_X[event1+1: -1] > 1)[0]
if event2.size > 0:
    event2 = event2[0] + event1 
    print(f"Heel off (Event2): {event2}")


# graph position with events 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], angle_relative[1:-1], label='knee angular pos', color='blue', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.title('knee angle vs time for during gait(after filtering)', fontsize=14)
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Right knee ankle (degree)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# graph velocicty with events 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], knee_AV[1:-1], label='knee angular vel', color='blue', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.title('knee angular velocity vs time for during gait(after filtering)')
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Right knee angular velocity  (degree/s)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# graph acceleration with events 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], knee_AA[1:-1], label='knee angular acc', color='blue', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.title('acceleration vs time for during gait(after filtering)')
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Right ankle X acceleration  (m/s²)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()





