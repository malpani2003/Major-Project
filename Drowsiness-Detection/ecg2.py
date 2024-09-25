import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import numpy as np

# Load the electrocardiogram data
ecg = electrocardiogram()

# Print the ECG data (optional)
print(len(ecg))

# Generate time data for the x-axis
fs = 10000  # The sampling frequency of the ECG signal in Hz
segment_length_seconds = 10  # Length of the ECG segment to plot (in seconds)
segment_length = segment_length_seconds * fs

# Ensure we don't go out of bounds
max_start_index = len(ecg) - segment_length
start_index = np.random.randint(0, max_start_index)
end_index = start_index + segment_length

# Extract the segment
ecg_segment = ecg[start_index:end_index]
time_segment = np.arange(ecg_segment.size) / fs

# Plot the ECG segment
plt.figure(figsize=(12, 6))
plt.plot(time_segment, ecg_segment)
plt.title("ECG Signal Segment")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Detect R-peaks in the ECG segment
peaks, _ = find_peaks(ecg_segment, distance=fs/2.5, height=0.5)

# Mark the R-peaks on the ECG plot
plt.plot(time_segment[peaks], ecg_segment[peaks], "o")

# Calculate BPM
num_beats = len(peaks)
duration_seconds = len(ecg_segment) / fs
duration_minutes = duration_seconds / 60
bpm = num_beats / duration_minutes

print(f"Detected R-peaks: {num_beats}")
print(f"Segment start index: {start_index}")
print(f"Segment end index: {end_index}")
print(f"Duration (s): {duration_seconds:.2f}")
print(f"Beats per minute (BPM): {bpm:.2f}")

plt.show()
