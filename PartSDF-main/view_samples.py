import numpy as np
import trimesh
import os

# 1. 设置文件路径 (修改为你实际的 npz 路径)
npz_path = 'C:/Users/dell/Downloads/PartSDF-main/PartSDF-main/deepsdf.npz'
output_dir = 'C:/Users/dell/Downloads/PartSDF-main/PartSDF-main/visualizations'

os.makedirs(output_dir, exist_ok=True)

# 2. 加载数据并合并正负样本
print("正在加载数据...")
data = np.load(npz_path)
all_samples = np.concatenate([data['pos'], data['neg']], axis=0)

# 3. 找出所有不同的体积条件 v
unique_volumes = np.unique(all_samples[:, 4])
print(f"共发现 {len(unique_volumes)} 种不同的体积变体:\n{unique_volumes}\n")

# 4. 提取表面点 (SDF 绝对值非常接近 0 的点)
# 阈值设置得越小，表面越薄越精确；设置得越大，点越多但表面越厚
surface_threshold = 0.008 

for i, v in enumerate(unique_volumes):
    # a. 选出当前体积条件下的所有点
    v_mask = (all_samples[:, 4] == v)
    v_samples = all_samples[v_mask]
    
    # b. 筛选出表面点 (|sdf| < threshold)
    surface_mask = np.abs(v_samples[:, 3]) < surface_threshold
    surface_points = v_samples[surface_mask, :3] # 只取前三维 x,y,z
    
    print(f"体积 V={v:.4f} 的模型，提取到 {surface_points.shape[0]} 个表面点。")
    
    # c. 使用 trimesh 导出为 .ply 文件
    pc = trimesh.PointCloud(surface_points)
    
    # 为了在不同文件间作区分，可以给点云上个颜色 (可选)
    # 这里我们用从冷到暖的颜色区分从瘦到胖
    color_val = int(255 * (i / (len(unique_volumes) - 1)))
    pc.colors = np.array([[color_val, 0, 255 - color_val, 255]] * len(surface_points))
    
    filename = os.path.join(output_dir, f"chair_surface_v_{v:.4f}.ply")
    pc.export(filename)
    print(f"--> 已保存至: {filename}\n")

print("全部导出完成！请在 Windows 资源管理器中打开可视化文件夹查看。")