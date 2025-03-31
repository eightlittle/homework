import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# update graph
"""
KAI-JEN
0327 2025 update graphs and examples for each cutoff frequency
"""
# 讀取 CSV 檔案
file_path = r'C:\Users\kent1\OneDrive - Auburn University\AU_classes\2025 spring\KINE_7620\assignment_position_velocity_acc\winterdataset.csv'
df = pd.read_csv(file_path, header=None)

# 選取時間數據（從第 3 行開始）
time = df.iloc[2:, 1].astype(float).reset_index(drop=True)

# 計算採樣頻率
sampling_interval = time.iloc[2] - time.iloc[1]
fs = 1 / 0.0145  # 採樣頻率
cutoff = 6  # 截止頻率

# 取出右腳踝 X 軸數據
data = np.array(df.iloc[2:, 6].astype(float)).reshape(-1, 1)


def two_order_for(data, a0, a1, a2, b1, b2):
    forward_f = np.zeros_like(data)
    forward_f[:2] = data[:2]  
    for i in range(2, len(data)):
        forward_f[i] = (a0 * data[i] + a1 * data[i-1] + a2 * data[i-2]
                        - b1 * forward_f[i-1] - b2 * forward_f[i-2])
    return forward_f


def two_order_back(forward_f, a0, a1, a2, b1, b2):
    backward_f = np.zeros_like(forward_f)
    reversed_f = forward_f[::-1]  
    backward_f[:2] = reversed_f[:2]
    for i in range(2, len(reversed_f)):
        backward_f[i] = (a0 * reversed_f[i] + a1 * reversed_f[i-1] + a2 * reversed_f[i-2]
                         - b1 * backward_f[i-1] - b2 * backward_f[i-2])
    return backward_f[::-1]  


def total(fs, fc, data):
    fr = fs / fc
    omgc = np.tan(np.pi / fr)
    c = 1 + 2 * np.cos(np.pi / 4) * omgc + omgc**2
    a0 = a2 = omgc**2 / c
    a1 = 2 * a0
    b1 = 2 * (omgc**2 - 1) / c
    b2 = (1 - 2 * np.cos(np.pi / 4) * omgc + omgc**2) / c

    # 只處理單變數 data
    filtered_f = two_order_for(data[:, 0], a0, a1, a2, b1, b2).reshape(-1, 1)
    filtered_b = two_order_back(filtered_f[:, 0], a0, a1, a2, b1, b2).reshape(-1, 1)

    return filtered_f, filtered_b


def filtering(data, fs, fc_range=30):
    forward_f, backward_f = total(fs, cutoff, data)

    # 繪圖
    plt.figure(figsize=(8, 5))
    plt.plot(time, data, label="Original", color='k', linestyle='dashed', alpha=0.7)
    plt.plot(time, forward_f, label="Two Order Forward", color='b')
    plt.plot(time, backward_f, label="Two Order Backward", color='r')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    plt.title("Filtering Process")
    plt.legend()
    plt.grid()
    plt.show()

    def rms(original, filtered):
        return np.sqrt(np.mean((original - filtered) ** 2))

    # 計算不同截止頻率下的 RMS 誤差
    filtered_6 = total(fs, 6, data)[1]
    try_rms_6 = rms(data, filtered_6)
    print("RMS (cutoff = 6 Hz):", try_rms_6)

    filtered_15 = total(fs, 15, data)[1]
    try_rms_15 = rms(data, filtered_15)
    print("RMS (cutoff = 15 Hz):", try_rms_15)

    filtered_30 = total(fs, 30, data)[1]
    try_rms_30 = rms(data, filtered_30)
    print("RMS (cutoff = 30 Hz):", try_rms_30)

    # RMS analysis for multiple cutoff frequencies
    frequency = np.zeros(fc_range)
    for fc in range(1, fc_range + 1):
        filtered_data = total(fs, fc, data)[1]
        frequency[fc - 1] = rms(data, filtered_data)

    # 產生 RMS 殘差圖
    fig, ax = plt.subplots()
    ax.plot(range(1, fc_range + 1), frequency, label="RMS")
    ax.axhline(frequency[15], color='r', linestyle='--', label='Zero noise pass')

    # 直線擬合（19Hz ~ 29Hz）
    a = (frequency[28] - frequency[18]) / (29 - 19)
    b = frequency[15] - a * 16
    rms_each_frequency = np.array([a * i + b for i in range(1, fc_range + 1)])

    ax.plot(range(1, fc_range + 1), rms_each_frequency, label="Straight line")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual")
    ax.legend()

    return fig


fig = filtering(data, fs=fs)
plt.show()
