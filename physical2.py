import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
N = 10000  # 每个Eb/N0值的总比特数
EbN0_indB = np.arange(-2, 9, 2)
EbN0 = 10 ** (EbN0_indB / 10)
E = 1
times = 50  # 迭代次数，用于求平均
err_rate_fixed_bits = np.zeros((len(EbN0), times))

# 蒙特卡洛仿真
for Eni, eb in enumerate(EbN0):
    sigma = E / np.sqrt(2 * eb)
    for i in range(times):
        r = np.random.rand(N)
        source = (r >= 0.5).astype(float)
        x = 1 - 2 * source
        noise = np.random.randn(N) * sigma
        y = E * x + noise
        result = (y <= 0).astype(float)
        error_num = np.sum(np.abs(result - source) > 1.e-6)
        err_rate_fixed_bits[Eni, i] = error_num / N

# 绘图
plt.figure(figsize=(10, 8))
for i in range(times):
    plt.semilogy(EbN0_indB, err_rate_fixed_bits[:, i], '-', alpha=0.3)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BER vs. Eb/N0 for Different Realizations')
plt.show()
