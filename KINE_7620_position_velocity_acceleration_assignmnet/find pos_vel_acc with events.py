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

#%% read data 
# Load your data
file_path = r'C:\Users\kent1\OneDrive - Auburn University\AU_classes\2025 spring\KINE_7620\assignment_position_velocity_acc\winterdataset.csv'
df = pd.read_csv(file_path, header=None)

# Pick up time (starting from row 3 if that's where your data starts)
time = df.iloc[2:, 1].astype(float).reset_index(drop=True)

# Sampling information
sampling_interval = time.iloc[1] - time.iloc[0]
fs = 1 / sampling_interval  # Sampling frequency
cutoff = 6  # Cutoff frequency for filtering

# Pick up point data - right ankle X
r_ankle_x = df.iloc[2:, 6].astype(float).reset_index(drop=True)

raw_velocity = time_d(r_ankle_x, sampling_interval)
raw_acc = time_dd(r_ankle_x, sampling_interval)
#%% filter position data - output = filtered_r_ankle_x
#  filter
filtered_r_ankle_x = butterworth_filter(r_ankle_x, cutoff, fs)

#%% calculate velocity and acceleration 
r_ank_veloicty_X = time_d(filtered_r_ankle_x, sampling_interval)
r_ank_acc_X = time_dd(filtered_r_ankle_x, sampling_interval)

#%% find events 

# find Event1 和 Event2
# set up the standard of touch down = velocity < 1 m/s
event1 = np.where(r_ank_veloicty_X[1:-1] < 0.3)[0]
if event1.size > 0:
    event1 = event1[0] 
    print(f"Touch down (Event1): {event1}")
    
# set up the standard of touch down = velocity > 1 m/s after event1 + 3 frame 
event2 = np.where(r_ank_veloicty_X[event1+1: -1] > 0.3)[0]
if event2.size > 0:
    event2 = event2[0] + event1 
    print(f"Heel off (Event2): {event2}")


# graph position with events 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], filtered_r_ankle_x[1:-1], label='Ankle X pos', color='blue', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.title('position vs time for the ankle during gait(after filtering)))', fontsize=14)
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Right Ankle X Position (m)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# graph velocicty with events 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], r_ank_veloicty_X[1:-1], label='Ank X vel', color='blue', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.title('velocity vs time for the ankle during gait')
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('right ankle X velocity  (m/s)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# graph acceleration with events 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], r_ank_acc_X[1:-1], label='Ankle X acc', color='blue', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.title('acceleration vs time for the ankle during gait')
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('right ankle X acceleration  (m/s²)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%% raw data vs filtered data

# Plot raw vs filtered signal
plt.figure(figsize=(10, 5))
plt.plot(time[1:-1], r_ankle_x[1:-1], label='Raw Data', alpha=0.6, color='blue')
plt.plot(time[1:-1], filtered_r_ankle_x[1:-1], label='Filtered Data', linewidth=2, color='red')
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.xlabel('Time (sec)')
plt.ylabel('Position - Right Ankle X (m)')
plt.title('Right Ankle X Position: Raw vs Filtered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Velocity raw vs filtered signal
plt.figure(figsize=(10, 5))
plt.plot(time[1:-1], raw_velocity[1:-1], label='Raw Data', alpha=0.6, color='blue')
plt.plot(time[1:-1], r_ank_veloicty_X[1:-1], label='Filtered Data', linewidth=2, color='red')
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.xlabel('Time (sec)')
plt.ylabel('Velocity  - Right Ankle X (m)')
plt.title('Right Ankle X Velocity: Raw vs Filtered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Acceleration raw vs filtered signal
plt.figure(figsize=(10, 5))
plt.plot(time[1:-1], raw_acc[1:-1], label='Raw Data', alpha=0.6, color='blue')
plt.plot(time[1:-1], r_ank_acc_X[1:-1], label='Filtered Data', linewidth=2, color='red')
plt.axvline(x=time[event1], color='red', linestyle='--', label=f'Touch down (Event1: the_{event1}_frame)')
plt.axvline(x=time[event2], color='green', linestyle='--', label=f'Heel off (Event2: the_{event2}_frame)')
plt.xlabel('Time (sec)')
plt.ylabel('Accekeration  - Right Ankle X (m)')
plt.title('Right Ankle X Acceleration: Raw vs Filtered')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
