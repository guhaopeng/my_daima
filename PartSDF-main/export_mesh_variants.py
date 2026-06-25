import numpy as np
import trimesh
import igl
import os
from skimage.measure import marching_cubes

# 1. 设定路径和参数
# 请确保 mesh_path 指向你原始的那把 test_chair.obj
mesh_path = r'C:/Users/dell/Downloads/PartSDF-main/PartSDF-main/1a6f615e8b1b5ae4dbbc9440457e303e.obj' 
output_dir = r'C:/Users/dell/Downloads/PartSDF-main/PartSDF-main/visualizations_obj'
os.makedirs(output_dir, exist_ok=True)

# 你的 5 个胖瘦偏移量
offsets = [-0.04, -0.02, 0.0, 0.02, 0.04]

# Marching Cubes 的分辨率 (数值越大模型越精细，但算得越慢，64或128比较合适)
res = 96 

print(f"正在加载原始模型: {mesh_path}")
mesh = trimesh.load(mesh_path)

# 2. 在 [-1, 1] 的空间内构建密集的 3D 网格
print(f"正在构建 {res}x{res}x{res} 的密集网格...")
coords = np.linspace(-1.0, 1.0, res, dtype=np.float32)
# 注意 indexing='ij' 保证坐标轴顺序正确
grid = np.stack(np.meshgrid(coords, coords, coords, indexing='ij'), axis=-1).reshape(-1, 3)

# 3. 使用 igl 计算整个空间网格到原始 mesh 的 SDF
print("正在计算全局 SDF 场 (这可能需要十几秒)...")
# igl.signed_distance 返回 [SDF值, 最近表面点索引, 最近表面点坐标]
sdf_global = igl.signed_distance(grid, mesh.vertices, mesh.faces)[0]
sdf_global = sdf_global.reshape(res, res, res)

# 4. 循环生成 5 种不同形态的 obj
for offset in offsets:
    print(f"\n---> 正在处理偏移量: {offset}")
    # a. 对全局 SDF 进行数学偏移 (制造胖瘦)
    new_sdf = sdf_global - offset
    
    try:
        # b. 使用 Marching Cubes 提取 SDF=0 的等值面
        # spacing 设置体素间距，使得提取出的顶点在 [0, 2] 范围内
        spacing = 2.0 / (res - 1)
        verts, faces, normals, _ = marching_cubes(new_sdf, level=0.0, spacing=(spacing, spacing, spacing))
        
        # 将顶点坐标平移回 [-1, 1] 的世界坐标系
        verts = verts - 1.0
        
        # c. 保存为 OBJ 文件
        new_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        out_name = os.path.join(output_dir, f"chair_variant_offset_{offset}.obj")
        new_mesh.export(out_name)
        print(f"提取成功！已保存至: {out_name}")
        
    except ValueError as e:
        # 如果模型太瘦导致断裂消失，marching cubes 可能会找不到表面
        print(f"提取失败 (可能是因为模型太瘦导致表面消失): {e}")

print("\n全部 OBJ 导出完成！")