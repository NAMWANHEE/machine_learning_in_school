import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math

size = 10
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
#랜덤 신호 생성
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
#주파수 설정
fc = 4000
fs = 20 * fc
ts = np.arange(0, (100 * size) / fs, 1 / fs)
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))

ook = m * np.cos(np.dot(2 * pi * fc, ts))
# BASK 변조 신호
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title('BASK modulation signal', fontproperties=zhfont1, fontsize=20)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, ook, 'r')


# AWGN 잡음 생성함수
def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(y ** 2) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y
# 잡음생성
noise_ook = awgn(ook, 5)

# BASK 변조 신호에 잡음 추가한 신호
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('BASK modulation signal superimposed noise waveform', fontproperties=zhfont1, fontsize=15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, noise_ook, 'r')

[b11, a11] = signal.ellip(5, 0.5, 60, [2000 * 2 / 80000, 6000 * 2 / 80000], btype='bandpass', analog=False, output='ba')
[b12, a12] = signal.ellip(5, 0.5, 60, (2000 * 2 / 80000), btype='lowpass', analog=False, output='ba')
bandpass_out = signal.filtfilt(b11, a11, noise_ook)
coherent_demod = bandpass_out * (coherent_carrier * 2)
lowpass_out = signal.filtfilt(b12, a12, coherent_demod)
fig2 = plt.figure(figsize=(10,4))
bx1 = fig2.add_subplot(3, 1, 1)
bx1.set_title('BFSK signal after coherent demodulation, signal before sampling decision', fontproperties=zhfont1, fontsize=15)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, lowpass_out, 'r')

#원래 신호로 복조하여 출력
detection_bpsk = np.zeros(len(t), dtype=np.float32)
flag = np.zeros(size, dtype=np.float32)

for i in range(10):
    tempF = 0
    for j in range(100):
        tempF = tempF + lowpass_out[i * 100 + j]
    if tempF > 5:
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
bx2.set_title('signal after BFSK signal sampling decision', fontproperties=zhfont1, fontsize=15)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, detection_bpsk, 'g')
plt.show()
