import numpy as np

# 加载生成的 deepsdf.npz 数据
data = np.load('C:/Users/dell/Downloads/PartSDF-main/PartSDF-main/deepsdf.npz')

pos_samples = data['pos']
neg_samples = data['neg']

print(f"正样本 (外部空气) 形状: {pos_samples.shape}") 
print(f"负样本 (内部材料) 形状: {neg_samples.shape}")

# 打印第一条数据看看长什么样
print("\n抽查第一条数据:")
print(f"X坐标: {pos_samples[0][0]:.4f}")
print(f"Y坐标: {pos_samples[0][1]:.4f}")
print(f"Z坐标: {pos_samples[0][2]:.4f}")
print(f"SDF值: {pos_samples[0][3]:.4f}")
print(f"体积V: {pos_samples[0][4]:.4f}  <-- 如果这里有值，说明你的条件数据生成大获全胜！")

# 检查包含了多少种不同的体积条件
unique_volumes = np.unique(pos_samples[:, 4])
print(f"\n这份数据中包含了 {len(unique_volumes)} 种不同的体积分数变体:")
print(unique_volumes)