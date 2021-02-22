import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math

size = 10
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
a = np.random.randint(0, 2, size)
m = np.zeros(len(t), dtype=np.float32)
for i in range(len(t)):
    m[i] = a[math.floor(t[i])]
fig = plt.figure()
# BINARY 신호 생성
ax1 = fig.add_subplot(3, 1, 1)
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
ax1.set_title('Generate random n-bit binary signal', fontproperties=zhfont1, fontsize=20)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, m, 'b')
fc = 4000 #주파수
Fs = 20 * fc  # sampling frequency
ts = np.arange(0, (100 * size) / Fs, 1 / Fs) # 1/Fs : 샘플당 시간
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4)

# BPSK 변조 신호
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title('BPSK modulation signal', fontproperties=zhfont1, fontsize=20)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, bpsk, 'r')
def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(y ** 2) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y
# AWGN 잡음
noise_bpsk = awgn(bpsk, 5)
# 잡음이 포함된 BPSK 신호
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('BPSK modulated signal superimposed noise waveform', fontproperties=zhfont1, fontsize=15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, noise_bpsk, 'r')
# passband 는 [2000,6000]
[b11, a11] = signal.ellip(5, 0.5, 60, [2000 * 2 / 80000, 6000 * 2 / 80000], btype='bandpass', analog=False, output='ba')
# Low psss filter 을 2000Hz까지 설정
[b12, a12] = signal.ellip(5, 0.5, 60, (2000 * 2 / 80000), btype='lowpass', analog=False, output='ba')
bandpass_out = signal.filtfilt(b11, a11, noise_bpsk)
coherent_demod = bandpass_out * (coherent_carrier * 2)
#low pass filter 통과한 신호 출력
lowpass_out = signal.filtfilt(b12, a12, coherent_demod)
fig2 = plt.figure()
bx1 = fig2.add_subplot(3, 1, 1)
bx1.set_title('local carrier downconversion, after low pass filter', fontproperties=zhfont1, fontsize=15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, lowpass_out, 'r')

# 원래 신호로 복조하여 출력
detection_bpsk = np.zeros(len(t), dtype=np.float32)
flag = np.zeros(size, dtype=np.float32)

for i in range(10):

    tempF = 0
    for j in range(100):
        tempF = tempF + lowpass_out[i * 100 + j]
    if tempF > 0:

        flag[i] = 1
    else:
        flag[i] = 0

for i in range(size):

    if flag[i] == 0:
        for j in range(100):
            detection_bpsk[i * 100 + j] = 0
    else:
        for j in range(100):
            detection_bpsk[i * 100 + j] = 1

bx2 = fig2.add_subplot(3, 1, 2)
bx2.set_title('signal after BPSK signal sampling decision', fontproperties = zhfont1, fontsize=15)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, detection_bpsk, 'g')
plt.show()
