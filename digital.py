from numpy import sqrt
import random
import matplotlib.pyplot as plt
from scipy.special import erfc
import numpy as np
N = 100000
snrindB_range = np.arange(0, 10)
itr = len(snrindB_range)
ber = [None] * itr
ber1 = [None] * itr
tx_symbol = 0
noise = 0
ch_coeff = 0
rx_symbol = 0
det_symbol = 0
for n in range(0, itr):

    snrindB = snrindB_range[n]
    snr = 10.0 ** (snrindB / 10.0)
    noise_std = 1 / sqrt(2 * snr)
    noise_mean = 0

    no_errors = 0
    for m in range(0, N):
        tx_symbol = 2 * random.randint(0, 1) - 1
        noise = random.gauss(noise_mean, noise_std)
        rx_symbol = tx_symbol + noise
        det_symbol = 2 * (rx_symbol >= 0) - 1
        no_errors += 1 * (tx_symbol != det_symbol)

    ber[n] = no_errors / N
    print("SNR in dB:", snrindB)
    print("Numbder of errors:", no_errors)
    print("Error probability:", ber[n])
BER_theory = 0.5*erfc(np.sqrt(10**(snrindB_range/10)))
plt.semilogy(snrindB_range, ber, 'o-', label='simulation')
plt.semilogy(snrindB_range, BER_theory, '^-', label='theory')
plt.xlabel('snr(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BPSK Modulation in AWGN channel')
plt.legend()
plt.show()
plt.close()
N = 100000
snrindB_range = np.arange(0, 10)
itr = len(snrindB_range)
ber = [None] * itr
ber1 = [None] * itr
tx_symbol = 0
noise = 0
ch_coeff = 0
rx_symbol = 0
det_symbol = 0
for n in range(0, itr):

    snrindB = snrindB_range[n]
    snr = 10.0 ** (snrindB / 10.0)
    noise_std = 1 / sqrt(2 * snr)
    noise_mean = 0

    no_errors = 0
    for m in range(0, N):
        tx_symbol = 2 * random.randint(0, 1) - 1
        noise = random.gauss(noise_mean, noise_std)
        ch_coeff=sqrt(random.gauss(0, 1) ** 2 + random.gauss(0, 1) ** 2) / sqrt(2)
        rx_symbol = tx_symbol * ch_coeff + noise
        det_symbol = 2 * (rx_symbol >= 0) - 1
        no_errors += 1 * (tx_symbol != det_symbol)

    ber[n] = no_errors / N
    ber1[n] = 0.5 * (1 - (sqrt(snr / (snr + 1))))
    print("SNR in dB:", snrindB)
    print("Numbder of errors:", no_errors)
    print("Error probability:", ber[n])
    print("Error probability theoritically:", ber1[n])
plt.semilogy(snrindB_range, ber, 'o-', label='simulation')
plt.semilogy(snrindB_range, ber1, '^-', label='theory')

plt.xlabel('snr(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BPSK Modulation in Rayleigh frequency flat fading channel')
plt.legend()
plt.show()