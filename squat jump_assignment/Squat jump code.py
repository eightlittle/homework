import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# lab setting
# X axis = anterior posterior direction
# Y axis = vertical direction
# Z axis = bilateral direction 
# sensor set up = 100Hz
# force plate set up = 1200Hz
# system re-sample to = 240Hz 
#%% calculate impluse and GRF normalized by BW (body weight) 
# read CSV file 

# file_path = r"C:\Users\kent1\OneDrive - Auburn University\AU_classes\2025 spring\KINE_7670\CMJ data\txt files\Kathryn_AFAP_CMJ1 - CMJ_report_class_7670.txt"
def CMJ_ANA(file_path):
    # 找到 "Sample #" 所在的行數
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "Sample #" in line:
                start_row = i
                break
    
    # 
    df = pd.read_csv(file_path, delimiter="\t", skiprows=start_row)
    
    df = df.iloc[:710, :]
    
    # 
    if "Sample #" in df.columns:
        df = df.loc[~df.drop(columns=["Sample #"]).isna().all(axis=1)]
    else:
        print("警告: 'Sample #' 欄位未找到，將直接刪除整行 NaN 值")
        df.dropna(how="all", inplace=True)
    
    mass = df.iloc[0, 21]
    # pick up data
    weight = mass * 9.81
    frequency = 240
    FP_data = df.iloc[:, 1].values.astype(np.float64) 
    FP_data = np.array(FP_data)
    sample = df.iloc[:, 0]
    time = sample / frequency
    
    column_data = FP_data
    zero_indices = np.where(column_data == 0)[0]
    
    # pre-deal with the data 
    if zero_indices.size > 0:
        first_zero_index = zero_indices[0]
        last_zero_index = zero_indices[-1]
        column_data[first_zero_index:last_zero_index + 1] = 0
        FP_data = column_data
    
    # filter the GRF data 
    def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
        nyquist = 0.5 * fs  # 計算奈奎斯特頻率
        normalized_cutoff = cutoff / nyquist  # 正規化截止頻率
        b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)  # 設計濾波器
        return filtfilt(b, a, data)  # 使用雙向濾波器
    
    # set up 
    cutoff_frequency = 25 #Hz
    dt = 1 / frequency
    
    FP_data = butterworth_filter(FP_data, cutoff_frequency, frequency, order=4, filter_type='low')
    
    impluse = FP_data * dt
    GRF_BW = FP_data / weight
    
    
    #calculate knee angle, anglar velocity, and angular acc 
    
    lank_x = df["left_ank_X_pos"].values.flatten()
    lank_y = df["left_ank_Y_pos"].values.flatten()
    
    lknee_x = df["left_knee_X_pos"].values.flatten()
    lknee_y = df["left_knee_Y_pos"].values.flatten()
    
    lhip_x = df["left_hip_X_pos"].values.flatten()
    lhip_y = df["left_hip_Y_pos"].values.flatten()
    
    # interpolation -> gap filling method
    def interpolate_with_cubic(data):
        data = pd.DataFrame(data)
        data.replace(0, np.nan, inplace=True)
        data = data.interpolate(method='linear', axis=0)
        data.bfill(inplace=True)  # 使用新的 bfill 方法
        data.ffill(inplace=True)  # 使用新的 ffill 方法
        if data.isnull().values.any() or (data == 0).any().any():
            # 使用 cubic interpolation 並套用新的 bfill 和 ffill
            data = data.interpolate(method='cubic', axis=0).bfill().ffill()
        return data.values
    
    # gap filling method -> apply to each variable 
    fc = 6    # cutoff frequency
    
    # filter each variable 
    Lank_X = butterworth_filter(lank_x*100, fc, frequency, order=4, filter_type='low')
    Lank_Y = butterworth_filter(lank_y*100, fc, frequency, order=4, filter_type='low')
    Lknee_X = butterworth_filter(lknee_x*100, fc, frequency, order=4, filter_type='low')
    Lknee_Y = butterworth_filter(lknee_y*100, fc, frequency, order=4, filter_type='low')
    Lhip_X = butterworth_filter(lhip_x*100, fc, frequency, order=4, filter_type='low')
    Lhip_Y = butterworth_filter(lhip_y*100, fc, frequency, order=4, filter_type='low')
    
    # combine x and y point into a metrix for each point
    Lank = np.vstack((Lank_X, Lank_Y)).T
    Lknee = np.vstack((Lknee_X, Lknee_Y)).T
    Lhip = np.vstack((Lhip_X, Lhip_Y)).T
    
    # calcualate related angle
    n = np.shape(sample)[0]  # 取得 sample 的行數
    global_y = np.tile([0, 1], (n, 1))  # 創建 n x 2 的矩陣
    
    # create a line between two points
    
    Lknee_ankle_line = Lknee - Lank
    Lankle_knee_line = Lank - Lknee
    Lhip_knee_line = Lhip - Lknee
    Lknee_hip_line = Lknee - Lhip
    
    
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
    
    Lknee_sagital_angle = calculate_theta(Lhip_knee_line, Lankle_knee_line)
    
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
    
    Lknee_sagital_angle = unwrap_deg(Lknee_sagital_angle)
    Lknee_sagital_angle = Lknee_sagital_angle*(180/np.pi)
    
    def time_d(data, sampling_interval):
        length = len(data)
        velocity = np.zeros(length)
        for i in range(length):
            if i == 0 or i == length - 1:
                velocity[i] = 0
            else:
                velocity[i] = (data[i + 1] - data[i - 1]) / (2 * sampling_interval)
        return velocity
    
    sampling_interval = 1 / frequency
    Lknee_sagital_AV = time_d(Lknee_sagital_angle, sampling_interval)
    
    # event setting
    # event1 start point: set up GRF < 95% BW
    event1 = np.where(GRF_BW < 0.95)[0]
    if event1.size > 0:
        event1 = event1[0] 
    
    # event 2 GRF_Y back to zero: GRF_BW GRF_BW > 1.00
    event2 =  np.where(GRF_BW[event1:] > 1.00)[0]
    if event2.size > 0:
        event2 = event2[0] + event1
    
    # event 3 GRF_Y min before bottom point: GRF_BW GRF_BW minimum
    event3_range = GRF_BW[event1:event2]
    event3_idx_in_range = np.argmin(event3_range)  # index of min value in the sliced range
    event3 = event3_idx_in_range + event1  # convert to global index
    
    # # event 5 take off point: GRF_Y == minimum 
    event5 =  np.where(GRF_BW[event3:] < 0.02)[0]
    if event5.size > 0:
        event5 = event5[0] + event3
    # event5 = 291 # => ONLY FOR kj_AFAP trial
    
    # real event 4 = the bottom 
    event4 = np.argmin(Lknee_sagital_angle[event2:event5]) + event2
    
    
    # event 6  touch down : GRF_Y > 0.1 after take off 
    event6 =  np.where(GRF_BW[event5:] > 0.1)[0]
    if event6.size > 0:
        event6 = event6[0] + event5
        
    print(f"start point (Event1): {event1}")
    print(f"Min_GRF_Y (Event3): {event3}")   
    print(f"GRF_Y back to BW (Event2): {event2}") 
    print(f"bottom point (event4): {event4}")
    print(f"take off point (Event5): {event5}")
    print(f"touch down (Event6): {event6}")
        
    events = {
        "1_START": event1,
        "2_MIN_GRF_Y": event3,
        "3_Back_to_BW": event2,
        "4_Botton_point": event4,
        "5_Take_Off": event5,
        "6_Touch_Down": event6
        }
    
    # event setting
    event_labels = ['START', 'MIN_GRF_Y', 'Back to BW', 'Botton point', 'Take Off', 'Touch Down']
    # ST = the start of the jump
    # MIN_GRF_Y = minimum vertical GRF
    # Back to BW = GRF goes up to equal to the body weight
    # Botton point = lowest point of the jump 
    # Take Off = GRF equal to zero
    # Touch Down = GRF above to zero again after take off
    
    # # Force vs Time plot with Event Markers
    # plt.figure(figsize=(10, 6))
    # plt.plot(time[events["1_START"] - 100: events["6_Touch_Down"] + 200], 
    #          GRF_BW[events["1_START"] - 100: events["6_Touch_Down"] + 200])
    
    # for i, key in enumerate(events):
    #     idx = events[key]
    #     if idx is not None and idx < len(time):  
    #         plt.axvline(x=time[idx], color='r', linestyle='--')
    #         plt.text(time[idx], max(GRF_BW) * 1.03, event_labels[i], 
    #                  ha='center', va='bottom', fontsize=10, rotation=45)
    
    # plt.title('Force vs Time', fontsize=14, loc='right', pad=20)
    # plt.xlabel('Time (s)')
    # plt.ylabel('GRF_Y (Body Weight)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # # Knee angle vs Time plot with Event Markers
    # plt.figure(figsize=(10, 6))
    # plt.plot(time[events["1_START"] - 100: events["6_Touch_Down"] + 200], 
    #          Lknee_sagital_angle[events["1_START"] - 100: events["6_Touch_Down"] + 200])
    
    # for i, key in enumerate(events):
    #     idx = events[key]
    #     if idx is not None and idx < len(time):  
    #         plt.axvline(x=time[idx], color='r', linestyle='--')
    #         plt.text(time[idx], max(Lknee_sagital_angle) * 1.03, event_labels[i], 
    #                  ha='center', va='bottom', fontsize=10, rotation=45)
    
    # plt.title('Knee angle vs Time', fontsize=14, loc='right', pad=20)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Knee angle (degree)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    
    # # Knee Angular Velocity vs Time plot with Event Markers
    # plt.figure(figsize=(10, 6))
    # plt.plot(time[events["1_START"] - 100: events["6_Touch_Down"] + 200], 
    #          Lknee_sagital_AV[events["1_START"] - 100: events["6_Touch_Down"] + 200])
    
    # # 標註每個事件
    # for i, key in enumerate(events):
    #     idx = events[key]
    #     if idx is not None and idx < len(time):  # 確保索引有效
    #         plt.axvline(x=time[idx], color='r', linestyle='--')
    #         plt.text(time[idx], max(Lknee_sagital_AV) * 1.03, event_labels[i], 
    #                  ha='center', va='bottom', fontsize=10, rotation=45)
    
    # # 在0劃一條水平線
    # plt.axhline(0, color='orange', linestyle='--')
    
    # # 標註 extension 和 flexion
    # plt.text(time[events["1_START"] - 200], 0.05, 'Positive = Extension', 
    #          ha='left', va='bottom', fontsize=14, color='green')
    # plt.text(time[events["1_START"] - 200], -0.05, 'Negative = Flexion', 
    #          ha='left', va='top', fontsize=14, color='red')
    
    # # 設定標題和軸標籤
    # plt.title('Knee Angular Velocity vs Time', fontsize=14, loc='right', pad=20)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Knee Angular Velocity (degree / sec)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    
    # find outcome 
    # jump height calculated by impluse / Take-off Velocity method (same)
    # BW = body weight（N）
    BW = weight 
    g = 9.81   # m/s²
    # calculat net force
    net_force = FP_data[events["1_START"]:events["5_Take_Off"]] - BW
    
    # Integration of force = Impulse（unit：N·s）
    dt = 1 / 240
    impulse = np.trapz(net_force, dx=dt)
    
    # velocity of take off
    takeoff_velocity = impulse / mass
    
    # calculate jump height = velocity of take off square / 2 * gravity 
    jump_height = (takeoff_velocity ** 2) / (2 * g)
    
    # results 
    # kinematics
    jump_height = jump_height
    knee_fle_ang_at_bottom = Lknee_sagital_angle[events["4_Botton_point"]]
    # knee_fle_ang_at_bottom = 360 - knee_fle_ang_at_bottom # # => ONLY FOR kj_AFAP trial
    max_knee_fle_vel = np.min(Lknee_sagital_AV[events["1_START"]: events["5_Take_Off"]])
    
    # max_knee_ext_vel = np.min(Lknee_sagital_angle[events["5_Take_Off"]:events["1_START"]])
    
    # kinetics
    breaking_impluse = sum(impluse[events["3_Back_to_BW"]:events["4_Botton_point"]]) - sum(impluse[events["2_MIN_GRF_Y"]:events["3_Back_to_BW"]])
    breaking_impluse = breaking_impluse / weight
    propulsion_impluse = sum(impluse[events["4_Botton_point"]:events["5_Take_Off"]])
    propulsion_impluse = propulsion_impluse / weight
    max_GRF = np.max(GRF_BW[events["1_START"]:events["5_Take_Off"]])
    min_GRF = np.min(GRF_BW[events["1_START"]:events["5_Take_Off"]])
    
    # time 
    unweight_time = (events["4_Botton_point"] - events["1_START"]) / frequency
    breaking_time = (events["4_Botton_point"] - events["3_Back_to_BW"]) / frequency
    propulsion_time = ( events["5_Take_Off"] - events["4_Botton_point"]) / frequency
    flight_time = ( events["6_Touch_Down"] - events["5_Take_Off"]) / frequency
    total_time = (events["5_Take_Off"] - events["1_START"]) / frequency

    # export data 
    export = {
        # time related 
        "unweight_time (sec)": f"{unweight_time:.2f}", 
        "breaking_time (sec)": f"{breaking_time:.2f}", 
        "propulsion_time (sec)": f"{propulsion_time:.2f}", 
        "flight_time (sec)": f"{flight_time:.2f}",
        "total_time (sec)": f"{total_time:.2f}", 
        
        # kinetics
        "breaking_impluse (J / BW)": f"{breaking_impluse:.2f}",
        "propulsion_impluse (J / BW)": f"{propulsion_impluse:.2f}",
        "max GRF (N / BW)": f"{max_GRF:.2f}",
        "min GRF (N / BW)": f"{min_GRF:.2f}",
        
        # kinematics
        "jump height (m)": f"{jump_height:.2f}",
        "knee_fle_ang_at_bottom (degree)": f"{knee_fle_ang_at_bottom:.2f}",
        "max_knee_fle_vel(degree / s)":f"{max_knee_fle_vel:.2f}",
        # "max_knee_ext_vel (degree  s)": max_knee_ext_vel,
        
        "events": events,
        "force plate data": GRF_BW,
        "knee angle data": Lknee_sagital_angle,
        "knee angular velocity data": Lknee_sagital_AV
    }
    return export

# only for kai-jen CMJ_AFAP
def CMJ_ANA_1(file_path):

    # 找到 "Sample #" 所在的行數
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "Sample #" in line:
                start_row = i
                break
    
    # 
    df = pd.read_csv(file_path, delimiter="\t", skiprows=start_row)
    
    df = df.iloc[:710, :]
    
    # 
    if "Sample #" in df.columns:
        df = df.loc[~df.drop(columns=["Sample #"]).isna().all(axis=1)]
    else:
        print("警告: 'Sample #' 欄位未找到，將直接刪除整行 NaN 值")
        df.dropna(how="all", inplace=True)
    
    mass = df.iloc[0, 21]
    # pick up data
    weight = mass * 9.81
    frequency = 240
    FP_data = df.iloc[:, 1].values.astype(np.float64) 
    FP_data = np.array(FP_data)
    sample = df.iloc[:, 0]
    time = sample / frequency
    
    column_data = FP_data
    zero_indices = np.where(column_data == 0)[0]
    
    # pre-deal with the data 
    if zero_indices.size > 0:
        first_zero_index = zero_indices[0]
        last_zero_index = zero_indices[-1]
        column_data[first_zero_index:last_zero_index + 1] = 0
        FP_data = column_data
    
    # filter the GRF data 
    def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
        nyquist = 0.5 * fs  # 計算奈奎斯特頻率
        normalized_cutoff = cutoff / nyquist  # 正規化截止頻率
        b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)  # 設計濾波器
        return filtfilt(b, a, data)  # 使用雙向濾波器
    
    # set up 
    cutoff_frequency = 25 #Hz
    dt = 1 / frequency
    
    FP_data = butterworth_filter(FP_data, cutoff_frequency, frequency, order=4, filter_type='low')
    
    impluse = FP_data * dt
    GRF_BW = FP_data / weight
    
    
    #calculate knee angle, anglar velocity, and angular acc 
    
    lank_x = df["left_ank_X_pos"].values.flatten()
    lank_y = df["left_ank_Y_pos"].values.flatten()
    
    lknee_x = df["left_knee_X_pos"].values.flatten()
    lknee_y = df["left_knee_Y_pos"].values.flatten()
    
    lhip_x = df["left_hip_X_pos"].values.flatten()
    lhip_y = df["left_hip_Y_pos"].values.flatten()
    
    # interpolation -> gap filling method
    def interpolate_with_cubic(data):
        data = pd.DataFrame(data)
        data.replace(0, np.nan, inplace=True)
        data = data.interpolate(method='linear', axis=0)
        data.bfill(inplace=True)  # 使用新的 bfill 方法
        data.ffill(inplace=True)  # 使用新的 ffill 方法
        if data.isnull().values.any() or (data == 0).any().any():
            # 使用 cubic interpolation 並套用新的 bfill 和 ffill
            data = data.interpolate(method='cubic', axis=0).bfill().ffill()
        return data.values
    
    # gap filling method -> apply to each variable 
    fc = 6    # cutoff frequency
    
    # filter each variable 
    Lank_X = butterworth_filter(lank_x*100, fc, frequency, order=4, filter_type='low')
    Lank_Y = butterworth_filter(lank_y*100, fc, frequency, order=4, filter_type='low')
    Lknee_X = butterworth_filter(lknee_x*100, fc, frequency, order=4, filter_type='low')
    Lknee_Y = butterworth_filter(lknee_y*100, fc, frequency, order=4, filter_type='low')
    Lhip_X = butterworth_filter(lhip_x*100, fc, frequency, order=4, filter_type='low')
    Lhip_Y = butterworth_filter(lhip_y*100, fc, frequency, order=4, filter_type='low')
    
    # combine x and y point into a metrix for each point
    Lank = np.vstack((Lank_X, Lank_Y)).T
    Lknee = np.vstack((Lknee_X, Lknee_Y)).T
    Lhip = np.vstack((Lhip_X, Lhip_Y)).T
    
    # calcualate related angle
    n = np.shape(sample)[0]  # 取得 sample 的行數
    global_y = np.tile([0, 1], (n, 1))  # 創建 n x 2 的矩陣
    
    # create a line between two points
    
    Lknee_ankle_line = Lknee - Lank
    Lankle_knee_line = Lank - Lknee
    Lhip_knee_line = Lhip - Lknee
    Lknee_hip_line = Lknee - Lhip
    
    
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
    
    Lknee_sagital_angle = calculate_theta(Lankle_knee_line, Lhip_knee_line)
    
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
    
    Lknee_sagital_angle = unwrap_deg(Lknee_sagital_angle)
    Lknee_sagital_angle = Lknee_sagital_angle*(180/np.pi)
    Lknee_sagital_angle = 360 - Lknee_sagital_angle
    
    def time_d(data, sampling_interval):
        length = len(data)
        velocity = np.zeros(length)
        for i in range(length):
            if i == 0 or i == length - 1:
                velocity[i] = 0
            else:
                velocity[i] = (data[i + 1] - data[i - 1]) / (2 * sampling_interval)
        return velocity
    
    sampling_interval = 1 / frequency
    Lknee_sagital_AV = time_d(Lknee_sagital_angle, sampling_interval)
    
    # event setting
    # event1 start point: set up GRF < 95% BW
    event1 = np.where(GRF_BW < 0.95)[0]
    if event1.size > 0:
        event1 = event1[0] 
    
    # event 2 GRF_Y back to zero: GRF_BW GRF_BW > 1.00
    event2 =  np.where(GRF_BW[event1:] > 1.00)[0]
    if event2.size > 0:
        event2 = event2[0] + event1
    
    # event 3 GRF_Y min before bottom point: GRF_BW GRF_BW minimum
    event3_range = GRF_BW[event1:event2]
    event3_idx_in_range = np.argmin(event3_range)  # index of min value in the sliced range
    event3 = event3_idx_in_range + event1  # convert to global index
    
    # # event 5 take off point: GRF_Y == minimum 
    # event5 =  np.where(GRF_BW[event3:] < 0.02)[0]
    # if event5.size > 0:
    #     event5 = event5[0] + event3
    event5 = 291 # => ONLY FOR kj_AFAP trial
    
    # real event 4 = the bottom 
    event4 = np.argmin(Lknee_sagital_angle[event2:event5]) + event2
    
    
    # event 6  touch down : GRF_Y > 0.1 after take off 
    event6 =  np.where(GRF_BW[event5:] > 0.1)[0]
    if event6.size > 0:
        event6 = event6[0] + event5
        
    print(f"start point (Event1): {event1}")
    print(f"Min_GRF_Y (Event3): {event3}")   
    print(f"GRF_Y back to BW (Event2): {event2}") 
    print(f"bottom point (event4): {event4}")
    print(f"take off point (Event5): {event5}")
    print(f"touch down (Event6): {event6}")
        
    events = {
        "1_START": event1,
        "2_MIN_GRF_Y": event3,
        "3_Back_to_BW": event2,
        "4_Botton_point": event4,
        "5_Take_Off": event5,
        "6_Touch_Down": event6
        }
    
    # event setting
    event_labels = ['START', 'MIN_GRF_Y', 'Back to BW', 'Botton point', 'Take Off', 'Touch Down']
    # ST = the start of the jump
    # MIN_GRF_Y = minimum vertical GRF
    # Back to BW = GRF goes up to equal to the body weight
    # Botton point = lowest point of the jump 
    # Take Off = GRF equal to zero
    # Touch Down = GRF above to zero again after take off
    
    # # Force vs Time plot with Event Markers
    # plt.figure(figsize=(10, 6))
    # plt.plot(time[events["1_START"] - 100: events["6_Touch_Down"] + 200], 
    #          GRF_BW[events["1_START"] - 100: events["6_Touch_Down"] + 200])
    
    # for i, key in enumerate(events):
    #     idx = events[key]
    #     if idx is not None and idx < len(time):  
    #         plt.axvline(x=time[idx], color='r', linestyle='--')
    #         plt.text(time[idx], max(GRF_BW) * 1.03, event_labels[i], 
    #                  ha='center', va='bottom', fontsize=10, rotation=45)
    
    # plt.title('Force vs Time', fontsize=14, loc='right', pad=20)
    # plt.xlabel('Time (s)')
    # plt.ylabel('GRF_Y (Body Weight)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # # Knee angle vs Time plot with Event Markers
    # plt.figure(figsize=(10, 6))
    # plt.plot(time[events["1_START"] - 100: events["6_Touch_Down"] + 200], 
    #          Lknee_sagital_angle[events["1_START"] - 100: events["6_Touch_Down"] + 200])
    
    # for i, key in enumerate(events):
    #     idx = events[key]
    #     if idx is not None and idx < len(time):  
    #         plt.axvline(x=time[idx], color='r', linestyle='--')
    #         plt.text(time[idx], max(Lknee_sagital_angle) * 1.03, event_labels[i], 
    #                  ha='center', va='bottom', fontsize=10, rotation=45)
    
    # plt.title('Knee angle vs Time', fontsize=14, loc='right', pad=20)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Knee angle (degree)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    
    # # Knee Angular Velocity vs Time plot with Event Markers
    # plt.figure(figsize=(10, 6))
    # plt.plot(time[events["1_START"] - 100: events["6_Touch_Down"] + 200], 
    #          Lknee_sagital_AV[events["1_START"] - 100: events["6_Touch_Down"] + 200])
    
    # # 標註每個事件
    # for i, key in enumerate(events):
    #     idx = events[key]
    #     if idx is not None and idx < len(time):  # 確保索引有效
    #         plt.axvline(x=time[idx], color='r', linestyle='--')
    #         plt.text(time[idx], max(Lknee_sagital_AV) * 1.03, event_labels[i], 
    #                  ha='center', va='bottom', fontsize=10, rotation=45)
    
    # # 在0劃一條水平線
    # plt.axhline(0, color='orange', linestyle='--')
    
    # # 標註 extension 和 flexion
    # plt.text(time[events["1_START"] - 200], 0.05, 'Positive = Extension', 
    #          ha='left', va='bottom', fontsize=14, color='green')
    # plt.text(time[events["1_START"] - 200], -0.05, 'Negative = Flexion', 
    #          ha='left', va='top', fontsize=14, color='red')
    
    # # 設定標題和軸標籤
    # plt.title('Knee Angular Velocity vs Time', fontsize=14, loc='right', pad=20)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Knee Angular Velocity (degree / sec)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    
    # find outcome 
    # jump height calculated by impluse / Take-off Velocity method (same)
    # BW = body weight（N）
    BW = weight 
    g = 9.81   # m/s²
    # calculat net force
    net_force = FP_data[events["1_START"]:events["5_Take_Off"]] - BW
    
    # Integration of force = Impulse（unit：N·s）
    dt = 1 / 240
    impulse = np.trapz(net_force, dx=dt)
    
    # velocity of take off
    takeoff_velocity = impulse / mass
    
    # calculate jump height = velocity of take off square / 2 * gravity 
    jump_height = (takeoff_velocity ** 2) / (2 * g)
    
    # results 
    # kinematics
    jump_height = jump_height
    knee_fle_ang_at_bottom = Lknee_sagital_angle[events["4_Botton_point"]]
    # knee_fle_ang_at_bottom = 360 - knee_fle_ang_at_bottom # # => ONLY FOR kj_AFAP trial
    max_knee_fle_vel = np.min(Lknee_sagital_AV[events["1_START"]: events["5_Take_Off"]])
    
    # max_knee_ext_vel = np.min(Lknee_sagital_angle[events["5_Take_Off"]:events["1_START"]])
    
    # kinetics
    breaking_impluse = sum(impluse[events["3_Back_to_BW"]:events["4_Botton_point"]]) - sum(impluse[events["2_MIN_GRF_Y"]:events["3_Back_to_BW"]])
    breaking_impluse = breaking_impluse / weight
    propulsion_impluse = sum(impluse[events["4_Botton_point"]:events["5_Take_Off"]])
    propulsion_impluse = propulsion_impluse / weight
    max_GRF = np.max(GRF_BW[events["1_START"]:events["5_Take_Off"]])
    min_GRF = np.min(GRF_BW[events["1_START"]:events["5_Take_Off"]])
    
    # time 
    unweight_time = (events["4_Botton_point"] - events["1_START"]) / frequency
    breaking_time = (events["4_Botton_point"] - events["3_Back_to_BW"]) / frequency
    propulsion_time = ( events["5_Take_Off"] - events["4_Botton_point"]) / frequency
    total_time = (events["5_Take_Off"] - events["1_START"]) / frequency

    # export data 
    export = {
        # time related 
        "unweight_time (sec)": f"{unweight_time:.2f}", 
        "breaking_time (sec)": f"{breaking_time:.2f}", 
        "propulsion_time (sec)": f"{propulsion_time:.2f}", 
        "total_time (sec)": f"{total_time:.2f}", 
        
        # kinetics
        "breaking_impluse (J / BW)": f"{breaking_impluse:.2f}",
        "propulsion_impluse (J / BW)": f"{propulsion_impluse:.2f}",
        "max GRF (N / BW)": f"{max_GRF:.2f}",
        "min GRF (N / BW)": f"{min_GRF:.2f}",
        
        # kinematics
        "jump height (m)": f"{jump_height:.2f}",
        "knee_fle_ang_at_bottom (degree)": f"{knee_fle_ang_at_bottom:.2f}",
        "max_knee_fle_vel(degree / s)":f"{max_knee_fle_vel:.2f}",
        # "max_knee_ext_vel (degree  s)": max_knee_ext_vel,
        
        "events": events,
        "force plate data": GRF_BW,
        "knee angle data": Lknee_sagital_angle,
        "knee angular velocity data": Lknee_sagital_AV
    }
    return export

Ian_AFAP = CMJ_ANA(r'/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/AU_classes/2025 spring/KINE_7670/CMJ data/txt files/ianjump_AFAP_CMJ1 - CMJ_report_class_7670.txt')
Ian_AHAP = CMJ_ANA(r'/Users/kairenzheng/Library/CloudStorage/OneDrive-AuburnUniversity/AU_classes/2025 spring/KINE_7670/CMJ data/txt files/ianjump_AHAP_CMJ1 - CMJ_report_class_7670.txt')

#%% GRF summary 
import matplotlib.pyplot as plt

# 取得 GRF 數據
Ian_AFAP_GRF = Ian_AFAP["force plate data"]
Ian_AHAP_GRF = Ian_AHAP["force plate data"]

# 擷取事件範圍內的 GRF 數據
start_AFAP = Ian_AFAP['events']["1_START"]
touchdown_AFAP = Ian_AFAP['events']["6_Touch_Down"]
end_AFAP = min(touchdown_AFAP + 1, len(Ian_AFAP_GRF))  # 確保不超出範圍
Ian_AFAP_GRF = Ian_AFAP_GRF[start_AFAP:end_AFAP]

start_AHAP = Ian_AHAP['events']["1_START"]
touchdown_AHAP = Ian_AHAP['events']["6_Touch_Down"]
end_AHAP = min(touchdown_AHAP + 1, len(Ian_AHAP_GRF))  # 確保不超出範圍
Ian_AHAP_GRF = Ian_AHAP_GRF[start_AHAP:end_AHAP]

# 定義 AFAP 事件並確保範圍正確
events_AFAP = Ian_AFAP["events"]
events_AFAP_adjusted = {
    "event1": 1,
    "event2": events_AFAP["2_MIN_GRF_Y"] - start_AFAP,
    "event3": events_AFAP["3_Back_to_BW"] - start_AFAP,
    "event4": events_AFAP["4_Botton_point"] - start_AFAP,
    "event5": events_AFAP["5_Take_Off"] - start_AFAP,
    "event6": min(events_AFAP["6_Touch_Down"] - start_AFAP, len(Ian_AFAP_GRF) - 1)  # 確保不超出範圍
}

# 定義 AHAP 事件並確保範圍正確
events_AHAP = Ian_AHAP["events"]
events_AHAP_adjusted = {
    "event1": 1,
    "event2": events_AHAP["2_MIN_GRF_Y"] - start_AHAP,
    "event3": events_AHAP["3_Back_to_BW"] - start_AHAP,
    "event4": events_AHAP["4_Botton_point"] - start_AHAP,
    "event5": events_AHAP["5_Take_Off"] - start_AHAP,
    "event6": min(events_AHAP["6_Touch_Down"] - start_AHAP, len(Ian_AHAP_GRF) - 1)  # 確保不超出範圍
}

# 計算時間軸
time_AFAP = [t / 500 for t in range(len(Ian_AFAP_GRF))]
time_AHAP = [t / 500 for t in range(len(Ian_AHAP_GRF))]

# 調整事件標記時間
valid_events_AFAP = {k: v / 500 for k, v in events_AFAP_adjusted.items() if 0 <= v < len(Ian_AFAP_GRF)}
valid_events_AHAP = {k: v / 500 for k, v in events_AHAP_adjusted.items() if 0 <= v < len(Ian_AHAP_GRF)}

# 創建圖形
fig, ax = plt.subplots(figsize=(20, 5))

# 畫 AFAP 和 AHAP 的 GRF 曲線
line_AFAP, = ax.plot(time_AFAP, Ian_AFAP_GRF, label="AFAP GRF", color='b')
line_AHAP, = ax.plot(time_AHAP, Ian_AHAP_GRF, label="AHAP GRF", color='g')

# 畫 AFAP 事件標記
for i, (key, value) in enumerate(valid_events_AFAP.items(), start=1):
    ax.text(value, Ian_AFAP_GRF[int(value * 500)], str(i), color='red', fontsize=20, ha='center', va='bottom')

# 畫 AHAP 事件標記
for i, (key, value) in enumerate(valid_events_AHAP.items(), start=1):
    ax.text(value, Ian_AHAP_GRF[int(value * 500)], str(i), color='red', fontsize=20, ha='center', va='bottom')

# 設定圖例
legend_labels = [
    "1: START",
    "2: MIN_GRF",
    "3: Back to BW",
    "4: Bottom point",
    "5: Take off",
    "6: Touch down"
]
dummy_handles = [plt.Line2D([0], [0], color="white", lw=0)] * len(legend_labels)
ax.legend(
    [line_AFAP, line_AHAP] + dummy_handles,
    ["AFAP GRF (Blue Line)", "AHAP GRF (Green Line)"] + legend_labels,
    loc="upper right",
    fontsize=12,
    frameon=True
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("GRF(N)")
ax.set_title(" Sample 3 GRF AFAP VS AHAP")

plt.tight_layout()
plt.show()

#%% angle summary 
import matplotlib.pyplot as plt

# 取得 angle 數據
Ian_AFAP_GRF = Ian_AFAP["knee angle data"]
Ian_AHAP_GRF = Ian_AHAP["knee angle data"]

# 擷取事件範圍內的 GRF 數據
start_AFAP = Ian_AFAP['events']["1_START"]
touchdown_AFAP = Ian_AFAP['events']["5_Take_Off"]
end_AFAP = min(touchdown_AFAP + 1, len(Ian_AFAP_GRF))  # 確保不超出範圍
Ian_AFAP_GRF = Ian_AFAP_GRF[start_AFAP:end_AFAP]

start_AHAP = Ian_AHAP['events']["1_START"]
touchdown_AHAP = Ian_AHAP['events']["5_Take_Off"]
end_AHAP = min(touchdown_AHAP + 1, len(Ian_AHAP_GRF))  # 確保不超出範圍
Ian_AHAP_GRF = Ian_AHAP_GRF[start_AHAP:end_AHAP]

# 定義 AFAP 事件並確保範圍正確
events_AFAP = Ian_AFAP["events"]
events_AFAP_adjusted = {
    "event1": 1,
    "event2": events_AFAP["2_MIN_GRF_Y"] - start_AFAP,
    "event3": events_AFAP["3_Back_to_BW"] - start_AFAP,
    "event4": events_AFAP["4_Botton_point"] - start_AFAP,
    "event5": events_AFAP["5_Take_Off"] - start_AFAP,
    # "event6": min(events_AFAP["6_Touch_Down"] - start_AFAP, len(Ian_AFAP_GRF) - 1)  # 確保不超出範圍
}

# 定義 AHAP 事件並確保範圍正確
events_AHAP = Ian_AHAP["events"]
events_AHAP_adjusted = {
    "event1": 1,
    "event2": events_AHAP["2_MIN_GRF_Y"] - start_AHAP,
    "event3": events_AHAP["3_Back_to_BW"] - start_AHAP,
    "event4": events_AHAP["4_Botton_point"] - start_AHAP,
    "event5": events_AHAP["5_Take_Off"] - start_AHAP,
    # "event6": min(events_AHAP["6_Touch_Down"] - start_AHAP, len(Ian_AHAP_GRF) - 1)  # 確保不超出範圍
}

# 計算時間軸
time_AFAP = [t / 500 for t in range(len(Ian_AFAP_GRF))]
time_AHAP = [t / 500 for t in range(len(Ian_AHAP_GRF))]

# 調整事件標記時間
valid_events_AFAP = {k: v / 500 for k, v in events_AFAP_adjusted.items() if 0 <= v < len(Ian_AFAP_GRF)}
valid_events_AHAP = {k: v / 500 for k, v in events_AHAP_adjusted.items() if 0 <= v < len(Ian_AHAP_GRF)}

# 創建圖形
fig, ax = plt.subplots(figsize=(20, 5))

# 畫 AFAP 和 AHAP 的 GRF 曲線
line_AFAP, = ax.plot(time_AFAP, Ian_AFAP_GRF, label="AFAP GRF", color='b')
line_AHAP, = ax.plot(time_AHAP, Ian_AHAP_GRF, label="AHAP GRF", color='g')

# 畫 AFAP 事件標記
for i, (key, value) in enumerate(valid_events_AFAP.items(), start=1):
    ax.text(value, Ian_AFAP_GRF[int(value * 500)], str(i), color='red', fontsize=20, ha='center', va='bottom')

# 畫 AHAP 事件標記
for i, (key, value) in enumerate(valid_events_AHAP.items(), start=1):
    ax.text(value, Ian_AHAP_GRF[int(value * 500)], str(i), color='red', fontsize=20, ha='center', va='bottom')

# 設定圖例
legend_labels = [
    "1: START",
    "2: MIN_GRF",
    "3: Back to BW",
    "4: Bottom point",
    "5: Take off"
    # "6_Touch_Down"
]
dummy_handles = [plt.Line2D([0], [0], color="white", lw=0)] * len(legend_labels)
ax.legend(
    [line_AFAP, line_AHAP] + dummy_handles,
    ["AFAP knee angle (Blue Line)", "AHAP knee angle (Green Line)"] + legend_labels,
    loc="upper right",
    fontsize=12,
    frameon=True
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("knee angle (degree)")
ax.set_title(" Sample 3 knee angle AFAP VS AHAP")

plt.tight_layout()
plt.show()

#%% angualr velocity summary 
import matplotlib.pyplot as plt

# 取得 angle 數據
Ian_AFAP_GRF = Ian_AFAP["knee angular velocity data"]
Ian_AHAP_GRF = Ian_AHAP["knee angular velocity data"]

# 擷取事件範圍內的 GRF 數據
start_AFAP = Ian_AFAP['events']["1_START"]
touchdown_AFAP = Ian_AFAP['events']["5_Take_Off"]
end_AFAP = min(touchdown_AFAP + 1, len(Ian_AFAP_GRF))  # 確保不超出範圍
Ian_AFAP_GRF = Ian_AFAP_GRF[start_AFAP:end_AFAP]

start_AHAP = Ian_AHAP['events']["1_START"]
touchdown_AHAP = Ian_AHAP['events']["5_Take_Off"]
end_AHAP = min(touchdown_AHAP + 1, len(Ian_AHAP_GRF))  # 確保不超出範圍
Ian_AHAP_GRF = Ian_AHAP_GRF[start_AHAP:end_AHAP]

# 定義 AFAP 事件並確保範圍正確
events_AFAP = Ian_AFAP["events"]
events_AFAP_adjusted = {
    "event1": 1,
    "event2": events_AFAP["2_MIN_GRF_Y"] - start_AFAP,
    "event3": events_AFAP["3_Back_to_BW"] - start_AFAP,
    "event4": events_AFAP["4_Botton_point"] - start_AFAP,
    "event5": events_AFAP["5_Take_Off"] - start_AFAP,
    # "event6": min(events_AFAP["6_Touch_Down"] - start_AFAP, len(Ian_AFAP_GRF) - 1)  # 確保不超出範圍
}

# 定義 AHAP 事件並確保範圍正確
events_AHAP = Ian_AHAP["events"]
events_AHAP_adjusted = {
    "event1": 1,
    "event2": events_AHAP["2_MIN_GRF_Y"] - start_AHAP,
    "event3": events_AHAP["3_Back_to_BW"] - start_AHAP,
    "event4": events_AHAP["4_Botton_point"] - start_AHAP,
    "event5": events_AHAP["5_Take_Off"] - start_AHAP,
    # "event6": min(events_AHAP["6_Touch_Down"] - start_AHAP, len(Ian_AHAP_GRF) - 1)  # 確保不超出範圍
}

# 計算時間軸
time_AFAP = [t / 500 for t in range(len(Ian_AFAP_GRF))]
time_AHAP = [t / 500 for t in range(len(Ian_AHAP_GRF))]

# 調整事件標記時間
valid_events_AFAP = {k: v / 500 for k, v in events_AFAP_adjusted.items() if 0 <= v < len(Ian_AFAP_GRF)}
valid_events_AHAP = {k: v / 500 for k, v in events_AHAP_adjusted.items() if 0 <= v < len(Ian_AHAP_GRF)}

# 創建圖形
fig, ax = plt.subplots(figsize=(20, 5))

# 畫 AFAP 和 AHAP 的 GRF 曲線
line_AFAP, = ax.plot(time_AFAP, Ian_AFAP_GRF, label="AFAP GRF", color='b')
line_AHAP, = ax.plot(time_AHAP, Ian_AHAP_GRF, label="AHAP GRF", color='g')

# 畫 AFAP 事件標記
for i, (key, value) in enumerate(valid_events_AFAP.items(), start=1):
    ax.text(value, Ian_AFAP_GRF[int(value * 500)], str(i), color='red', fontsize=20, ha='center', va='bottom')

# 畫 AHAP 事件標記
for i, (key, value) in enumerate(valid_events_AHAP.items(), start=1):
    ax.text(value, Ian_AHAP_GRF[int(value * 500)], str(i), color='red', fontsize=20, ha='center', va='bottom')

# 設定圖例
legend_labels = [
    "1: START",
    "2: MIN_GRF",
    "3: Back to BW",
    "4: Bottom point",
    "5: Take off"
    # "6: Touch down"
]
dummy_handles = [plt.Line2D([0], [0], color="white", lw=0)] * len(legend_labels)
ax.legend(
    [line_AFAP, line_AHAP] + dummy_handles,
    ["AFAP knee angle (Blue Line)", "AHAP knee angular velocity (Green Line)"] + legend_labels,
    loc="upper right",
    fontsize=12,
    frameon=True
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("knee angular velocity (degree / s)")
ax.set_title(" Sample 3 knee angular velocity AFAP VS AHAP")

plt.tight_layout()
plt.show()