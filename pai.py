import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def estimate_pi(N):
    # 在正方形内随机投点
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    # 计算落在单位圆内的点的数量
    inside_circle = np.sum(x**2 + y**2 <= 1)
    # 估算π值
    pi_estimate = 4 * inside_circle / N
    return pi_estimate

# 设置不同的仿真次数
N_values = [100, 1000, 10000, 100000]
pi_estimates = []
errors = []
conf_intervals = []

for N in N_values:
    pi_estimate = estimate_pi(N)
    pi_estimates.append(pi_estimate)
    errors.append(abs(pi_estimate - np.pi))
    # 计算标准差
    std_dev = np.sqrt(pi_estimate * (4 - pi_estimate) / N)
    # 计算95%置信区间
    conf_interval = 1.96 * std_dev
    conf_intervals.append(conf_interval)

# 绘制绝对误差和置信区间随仿真次数变化的曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(N_values, errors, marker='o')
plt.xscale('log')
plt.xlabel('仿真次数')
plt.ylabel('绝对误差')
plt.title('绝对误差随仿真次数变化的曲线')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(N_values, conf_intervals, marker='o', color='r')
plt.xscale('log')
plt.xlabel('仿真次数')
plt.ylabel('95%置信区间宽度')
plt.title('95%置信区间宽度随仿真次数变化的曲线')
plt.grid(True)

plt.tight_layout()
plt.show()
# 打印最终估计的π值
print(f"最终估计的π值为: {pi_estimates[-1]}")