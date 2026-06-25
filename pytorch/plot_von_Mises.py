import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib import cm  # 导入 Matplotlib 色图模块，用于 jet 色图
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 导入 3D 多边形集合，用于绘制等值面网格
from skimage.measure import marching_cubes  # 导入 marching_cubes，用于从体数据提取等值面


def plot_von_Mises(density, von_mises, iso_level=0.5):            # 输入密度场与应力场并绘制 3D von Mises 图
    density = np.asarray(density, dtype=float)                    # 将 density 转成浮点数组，避免后续类型不一致
    von_mises = np.asarray(von_mises, dtype=float)                # 将 von_mises 转成浮点数组，保证插值与映射稳定
    if density.ndim != 3:                                         # 检查 density 是否为三维体数据
        raise ValueError("density must be a 3D array")  
    if von_mises.shape != density.shape:                          # 检查应力场与密度场尺寸是否一致
        raise ValueError("von_mises must have the same shape as density")  
    if np.min(density) > iso_level or np.max(density) < iso_level:# 检查等值面阈值是否落在数据范围内
        raise ValueError("iso_level is outside density range; no isosurface can be extracted")  

    verts, faces, _, _ = marching_cubes(density, level=iso_level)  # 提取等值面顶点与三角面片（对应 MATLAB isosurface）
    if faces.size == 0:  
        raise ValueError("No faces extracted from density at the given iso_level")  

    v_idx = np.clip(np.rint(verts).astype(int), 0, np.array(density.shape) - 1)  # 将浮点顶点坐标取整并限制到合法索引范围
    vertex_values = von_mises[v_idx[:, 0], v_idx[:, 1], v_idx[:, 2]]             # 在每个顶点位置采样 von Mises 值
    face_values = vertex_values[faces].mean(axis=1)                              # 将三角面三个顶点的应力取平均，作为该面的着色标量

    vmin = float(np.nanmin(face_values)) if np.isfinite(np.nanmin(face_values)) else 0.0  # 计算颜色映射下界，异常时回退到 0
    vmax = float(np.nanmax(face_values)) if np.isfinite(np.nanmax(face_values)) else 1.0  # 计算颜色映射上界，异常时回退到 1
    if abs(vmax - vmin) < 1e-12:  
        vmax = vmin + 1.0                                                                 # 给一个最小跨度保证颜色映射可用

    norm = plt.Normalize(vmin=vmin, vmax=vmax)                                            # 创建归一化器，将应力值映射到 [0,1]
    face_colors = cm.get_cmap("jet")(norm(face_values))                                   # 按 jet 色图生成每个面的 RGBA 颜色
    triangles = verts[faces]                                                              # 根据 face 索引取出每个三角面的三个顶点坐标

    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111, projection="3d") 
    mesh = Poly3DCollection(triangles, linewidths=0.0)                                    # 用三角面构造 3D 网格对象
    mesh.set_facecolor(face_colors)                                                       # 设置每个三角面的颜色（对应 MATLAB isocolors）
    mesh.set_edgecolor("none")  
    ax.add_collection3d(mesh)                                                             # 将网格添加到 3D 坐标轴

    ax.set_xlim(0, density.shape[0] - 1)  # 设置 x 轴显示范围为体数据第一维
    ax.set_ylim(0, density.shape[1] - 1)  # 设置 y 轴显示范围为体数据第二维
    ax.set_zlim(0, density.shape[2] - 1)  # 设置 z 轴显示范围为体数据第三维
    ax.set_box_aspect((density.shape[0], density.shape[1], density.shape[2]))  
    ax.view_init(elev=25, azim=-60)  # 设置观察视角，等价于 MATLAB view(3) 的一个常用角度
    ax.set_axis_off()  

    mappable = cm.ScalarMappable(norm=norm, cmap="jet")  # 创建颜色条映射对象
    mappable.set_array([])  # 设置空数组以激活 colorbar 显示
    fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.05)  # 添加颜色条以表示应力大小
    plt.draw()  # 立即刷新图像（对应 MATLAB drawnow）
    return fig, ax  
