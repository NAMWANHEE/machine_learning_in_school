import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math

size = 10
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
#램덤하게 신호 생성
a = np.random.randint(0, 2, size)
m = np.zeros(len(t), dtype=np.float32)
for i in range(len(t)):
    m[i] = a[math.floor(t[i])]
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
ax1.set_title('Generate random n-bit binary signal', fontproperties=zhfont1, fontsize=20)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, m, 'b')
# 주파수 설정
fc1 = 4000
fc2 = 10000
Fs = 160000  # sampling frequency
ts = np.arange(0, (100 * size) / Fs, 1 / Fs)
coherent_carrier1 = np.cos(np.dot(2 * pi * fc1, ts))
coherent_carrier2 = np.cos(np.dot(2 * pi * fc2, ts))

bfsk = m * coherent_carrier1 + (1 - m) * coherent_carrier2

# BFSK로 변조 신호
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title('BFSK modulation signal', fontproperties=zhfont1, fontsize=20)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, bfsk, 'r')

def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(y ** 2) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y

noise_bpsk = awgn(bfsk, 5)
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('BFSK modulation signal superimposed noise waveform', fontproperties=zhfont1, fontsize=15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, noise_bpsk, 'r')

[b11, a11] = signal.ellip(5, 0.5, 60, [2000 * 2 / 160000, 6000 * 2 / 160000], btype='bandpass', analog=False,
                          output='ba')

[b12, a12] = signal.ellip(5, 0.5, 60, (2000 * 2 / 160000), btype='lowpass', analog=False, output='ba')
bandpass_out1 = signal.filtfilt(b11, a11, noise_bpsk)
coherent_demod1 = bandpass_out1 * (coherent_carrier1 * 2)
lowpass_out1 = signal.filtfilt(b12, a12, coherent_demod1)
[b21, a21] = signal.ellip(5, 0.5, 60, [8000 * 2 / 160000, 12000 * 2 / 160000], btype='bandpass', analog=False,
                          output='ba')
bandpass_out2 = signal.filtfilt(b21, a21, noise_bpsk)
coherent_demod2 = bandpass_out2 * (coherent_carrier2 * 2)
lowpass_out2 = signal.filtfilt(b12, a12, coherent_demod2)

fig2 = plt.figure(figsize=(8,4))
bx1 = fig2.add_subplot(3, 1, 1)
bx1.set_title('BFSK signal after coherent demodulation, signal before sampling decision', fontproperties = zhfont1, fontsize=15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, lowpass_out1, 'r')

detection_2fsk = np.zeros(len(t), dtype=np.float32)
flag = np.zeros(size, dtype=np.float32)

for i in range(10):
    tempF = 0
    for j in range(100):
        tempF = tempF + lowpass_out1[i * 100 + j]
    if tempF > 50:
        flag[i] = 1
    else:
         flag[i] = 0
for i in range(size):
    if flag[i] == 0:
        for j in range(100):
            detection_2fsk[i * 100 + j] = 0
    else:
        for j in range(100):
            detection_2fsk[i * 100 + j] = 1

bx2 = fig2.add_subplot(3, 1, 2)
bx2.set_title('signal after BFSK signal sampling decision', fontproperties = zhfont1, fontsize=15)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, detection_2fsk, 'g')
plt.show()