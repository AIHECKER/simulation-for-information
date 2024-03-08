import numpy as np
import matplotlib.pyplot as plt

# 初始化参数
EbN0_dB = np.arange(-2, 9, 2)  # Eb/N0值的dB表示
EbN0 = 10 ** (EbN0_dB / 10)    # Eb/N0的线性表示
E = 1                           # 每个比特的能量
times = 50                      # 迭代次数，用于求平均
max_errors = 100                # 每个Eb/N0值的最大错误个数
err_rate_fixed_bits = np.zeros((len(EbN0), times))
bits_transmitted = np.zeros((len(EbN0), times))  # 各个信噪比下传输的比特数

# 蒙特卡洛仿真
for Eni, eb in enumerate(EbN0):
    sigma = E / np.sqrt(2 * eb)
    for i in range(times):
        error_num = 0
        total_bits = 0
        while error_num < max_errors:
            r = np.random.rand(1000)
            source = (r >= 0.5).astype(float)
            x = 1 - 2 * source
            noise = np.random.randn(1000) * sigma
            y = E * x + noise
            result = (y <= 0).astype(float)
            error_num += np.sum(np.abs(result - source) > 1.e-6)
            total_bits += 1000
        err_rate_fixed_bits[Eni, i] = error_num / total_bits
        bits_transmitted[Eni, i] = total_bits

# 绘图
plt.figure(figsize=(10, 8))
for i in range(times):
    plt.semilogy(EbN0_dB, err_rate_fixed_bits[:, i], '-', alpha=0.3)

plt.xlabel('Eb/N0 (dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BER vs. Eb/N0 for Different Realizations')
plt.show()

# 打印各个信噪比下传输的比特数
for Eni, eb in enumerate(EbN0_dB):
    print(f"Eb/N0 = {eb} dB: Average Bits Transmitted = {np.mean(bits_transmitted[Eni, :])}")
