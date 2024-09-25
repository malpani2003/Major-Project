import numpy as np
import matplotlib.pyplot as plt

# Function to generate a normal ECG signal
def generate_normal_ecg(length):
    t = np.linspace(0, length, length * 100)
    ecg = 1.2 * np.sin(1.5 * np.pi * t) + 0.25 * np.sin(30 * np.pi * t) + 0.1 * np.sin(50 * np.pi * t)
    return t, ecg

# Function to generate an abnormal ECG signal (e.g., with an arrhythmia)
def generate_abnormal_ecg(length):
    t = np.linspace(0, length, length * 400)
    ecg = 1.2 * np.sin(1.5 * np.pi * t) + 0.25 * np.sin(30 * np.pi * t) + 0.1 * np.sin(50 * np.pi * t)
    # Introduce an irregularity
    ecg[int(0.3 * len(ecg)):int(0.4 * len(ecg))] += 0.5 * np.sin(2 * np.pi * t[int(0.3 * len(t)):int(0.4 * len(t))])
    return t, ecg

# Plot the ECG signals
def plot_ecg(t, ecg, title):
    plt.figure(figsize=(10, 4))
    plt.plot(t, ecg, label='ECG Signal')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

# Generate and plot normal ECG
t_normal, ecg_normal = generate_normal_ecg(10)
plot_ecg(t_normal, ecg_normal, 'Normal ECG Signal')

# Generate and plot abnormal ECG
t_abnormal, ecg_abnormal = generate_abnormal_ecg(10)
plot_ecg(t_abnormal, ecg_abnormal, 'Abnormal ECG Signal')