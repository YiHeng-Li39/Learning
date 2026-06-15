import matplotlib.pyplot as plt
import os
import numpy as np

def reproduce_figure_3():
    print("========= 开始复现 Figure 3 (有效边界散点图) =========")
    
    # 定义输出路径
    output_dir = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取自论文 Table 3 (破产州数量, X轴)
    # 对应的 gamma_2 序列为: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    insolvent_ARO = [36, 30, 26, 25, 22, 21, 19, 16, 15, 13, 12]
    insolvent_RO1 = [36, 32, 26, 26, 23, 22, 21, 18, 15, 13, 12]
    insolvent_RO2 = [32, 28, 23, 23, 20, 20, 19, 16, 14, 12, 11]
    
    # 提取自论文 Table 4 (盈余 S/D, 乘以 1e9 转换为绝对金额, Y轴)
    surplus_ARO = np.array([-9.23, -5.88, -3.56, -1.80, -0.46, 0.55, 1.41, 2.16, 2.78, 3.32, 3.82]) * 1e9
    surplus_RO1 = np.array([-9.27, -6.50, -3.73, -0.96, 1.80, 4.57, 7.34, 10.11, 12.88, 15.64, 18.41]) * 1e9
    surplus_RO2 = np.array([-9.16, -6.41, -3.67, -0.92, 1.84, 4.60, 7.37, 10.14, 12.90, 15.67, 18.43]) * 1e9
    
    # 基准模型 (静态单点)
    insolvent_CMA, surplus_CMA = 36, -8.31e9
    insolvent_Hist, surplus_Hist = 52, -1.98e10
    
    # ================= 开始绘图 =================
    plt.figure(figsize=(10, 6))
    
    # 采用 seaborn-whitegrid 风格的浅色背景网格
    plt.grid(True, linestyle='-', alpha=0.5, color='#E0E0E0')
    plt.gca().set_facecolor('#FAFAFA')
    
    # 绘制各模型的散点
    # ARO: 蓝色圆圈
    plt.scatter(insolvent_ARO, surplus_ARO, c='#4C72B0', marker='o', s=60, label='ARO', alpha=0.9)
    # RO1: 橙色三角形
    plt.scatter(insolvent_RO1, surplus_RO1, c='#DD8452', marker='^', s=60, label='RO1', alpha=0.9)
    # RO2: 绿色正方形
    plt.scatter(insolvent_RO2, surplus_RO2, c='#55A868', marker='s', s=60, label='RO2', alpha=0.9)
    
    # CMA: 红色粗十字 (增大尺寸使其明显)
    plt.scatter(insolvent_CMA, surplus_CMA, c='#C44E52', marker='P', s=120, label='CMA', alpha=0.9)
    # Hist: 紫色菱形 (增大尺寸)
    plt.scatter(insolvent_Hist, surplus_Hist, c='#8172B2', marker='d', s=120, label='Hist', alpha=0.9)
    
    # 设置坐标轴标签与细节
    plt.xlabel('Number of Insolvent States', fontsize=12)
    plt.ylabel('Total Surplus (or Deficit) ($)', fontsize=12)
    
    # 科学计数法格式化 Y 轴
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # 调整图例
    plt.legend(loc='upper right', frameon=True, framealpha=1, edgecolor='black', fontsize=10)
    
    # 设置坐标轴显示范围，精准对齐原图视觉
    plt.xlim(8, 54)
    plt.ylim(-2.2e10, 2.05e10)
    
    # 移除顶部和右侧的边框线使图表更加整洁（类似 seaborn 默认风格）
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    output_path = os.path.join(output_dir, "Reproduced_Figure_3_Efficient_Frontier.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3 已完美生成并保存至: {output_path}")

if __name__ == "__main__":
    reproduce_figure_3()