import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Initialization
N = 100000  # Total number of bits for each Eb/N0 in fixed bits method
EbN0_indB = np.arange(-2, 9, 2)
Ebn0 = 10 ** (EbN0_indB / 10)
E = 1
times = 10  # Number of iterations for averaging
err_rate_fixed_bits = np.zeros((len(Ebn0), times))
err_rate_fixed_errors = np.zeros((len(Ebn0), times))
target_errors = 100  # Target number of errors for the fixed error method
batch_size = 1000  # Number of bits processed in each iteration for fixed errors method

# Monte Carlo Simulation
for Eni, eb in enumerate(Ebn0):
    sigma = E / np.sqrt(2 * eb)
    for i in range(times):
        # Fixed number of bits method
        r = np.random.rand(N)
        source = (r >= 0.5).astype(float)
        x = 1 - 2 * source
        noise = np.random.randn(N) * sigma
        y = E * x + noise
        result = (y <= 0).astype(float)
        error_num = np.sum(np.abs(result - source) > 1.e-6)
        err_rate_fixed_bits[Eni, i] = error_num / N

        # Fixed number of errors method
        error_count = 0
        bit_count = 0
        while error_count < target_errors:
            r = np.random.rand(batch_size)
            source = (r >= 0.5).astype(float)
            x = 1 - 2 * source
            noise = np.random.randn(batch_size) * sigma
            y = E * x + noise
            result = (y <= 0).astype(float)
            error_count += np.sum(np.abs(result - source) > 1.e-6)
            bit_count += batch_size
        err_rate_fixed_errors[Eni, i] = error_count / bit_count

# Theoretical bit error rate
pe_theory = 0.5 * erfc(np.sqrt(Ebn0))

# Plotting
plt.figure()
plt.semilogy(EbN0_indB, np.mean(err_rate_fixed_bits, axis=1), '-ro', label='Fixed Bits')
plt.semilogy(EbN0_indB, np.mean(err_rate_fixed_errors, axis=1), '-bs', label='Fixed Errors')
plt.semilogy(EbN0_indB, pe_theory, '-g*', label='Theoretical')
plt.xlabel('Eb/N0(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.legend()
plt.show()
