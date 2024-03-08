import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Initialization
N = [100, 1000, 10000, 100000, 1000000]
EbN0_indB = np.arange(-2, 9, 2)
Ebn0 = 10**(EbN0_indB/10)
E = 1
times = 50
err_rate = np.zeros((len(Ebn0), len(N), times))

# Monte Carlo Simulation
for iN, n in enumerate(N):
    for Eni, eb in enumerate(Ebn0):
        sigma = E / np.sqrt(2 * eb)
        for i in range(times):
            r = np.random.rand(n)
            source = (r >= 0.5).astype(float)
            x = 1 - 2 * source
            noise = np.random.randn(n) * sigma
            y = E * x + noise
            result = (y <= 0).astype(float)
            error_num = np.sum(np.abs(result - source) > 1.e-6)
            err_rate[Eni, iN, i] = error_num / n

# Theoretical bit error rate
pe_theory = 0.5 * erfc(np.sqrt(Ebn0))

# Calculating variance
variance = np.zeros((len(Ebn0), len(N)))
for Eni in range(len(Ebn0)):
    for iN in range(len(N)):
        ber = err_rate[Eni, iN, :]
        relative_error = (ber - pe_theory[Eni]) / pe_theory[Eni]
        variance[Eni, iN] = np.var(relative_error)

# Plotting
plt.figure()
for i in range(len(N)):
    plt.semilogy(EbN0_indB, variance[:, i], label=f'{N[i]} bits')
plt.xlabel('Eb/N0 (dB)')
plt.ylabel('Error Variance')
plt.yscale('log')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()

# Second plot
BER1 = err_rate[:, 4, 0]
plt.figure()
plt.semilogy(EbN0_indB, BER1, '-ro', label='Simulation')
plt.semilogy(EbN0_indB, pe_theory, '-bs', label='Theoretical')
plt.xlabel('Eb/N0(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.legend()
plt.show()
