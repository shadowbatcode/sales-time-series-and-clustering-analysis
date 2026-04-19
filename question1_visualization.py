"""
论文级蔬菜销售数据可视化脚本 - 优化版
根据数据特点选择最合适的可视化方式
每个图最多4个子图，强调信息的有效传达
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import networkx as nx
from scipy.stats import probplot, gaussian_kde, zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 设置论文级图表风格
#plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.1)

class PaperVisualizer:
    """论文级可视化类"""
    
    def __init__(self, data_file='图表数据.pkl'):
        """初始化"""
        self.load_data(data_file)
        # 论文标准图表尺寸
        self.single_col_width = 3.5  # 单栏宽度（英寸）
        self.double_col_width = 7.0  # 双栏宽度（英寸）
        
    def load_data(self, data_file):
        """加载数据"""
        try:
            with open(data_file, 'rb') as f:
                self.chart_data = pickle.load(f)
            print(f"✓ 成功加载数据文件: {data_file}")
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            self.chart_data = {}
    
    def figure1_distribution_analysis(self, save=True, dpi=300):
        """
        图1：销量分布特征分析
        (a) 分组小提琴图对比多品类分布
        (b) 异常值检测散点图
        (c) 分布形态分类图
        (d) 累积分布函数对比
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.35)
        
        # 准备数据
        corr_data = self.chart_data.get('correlation', {})
        if 'category_matrix' not in corr_data:
            print("缺少必要数据")
            return
        
        cat_matrix = corr_data['category_matrix']
        
        # (a) 分组小提琴图 - 展示完整分布形态
        ax1 = fig.add_subplot(gs[0, :])  # 跨两列
        
        # 选择销量前8的品类
        top_cats = cat_matrix.sum().nlargest(8).index
        
        # 准备数据
        plot_data = []
        positions = []
        colors = []
        color_palette = sns.color_palette("Set2", len(top_cats))
        
        for i, cat in enumerate(top_cats):
            # 获取非零数据
            data = cat_matrix[cat][cat_matrix[cat] > 0]
            if len(data) > 10:
                plot_data.append(data.values)
                positions.append(i)
                colors.append(color_palette[i])
        
        # 创建小提琴图
        parts = ax1.violinplot(plot_data, positions=positions, 
                               widths=0.7, showmeans=True, showmedians=True)
        
        # 自定义颜色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        
        # 添加箱线图覆盖
        bp = ax1.boxplot(plot_data, positions=positions, widths=0.3,
                         patch_artist=True, showfliers=False,
                         boxprops=dict(facecolor='white', alpha=0.3),
                         medianprops=dict(color='red', linewidth=2),
                         meanprops=dict(marker='o', markerfacecolor='blue', 
                                      markeredgecolor='blue', markersize=6))
        
        # 添加统计标注
        for i, (pos, data) in enumerate(zip(positions, plot_data)):
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax1.text(pos, ax1.get_ylim()[1]*0.95, 
                    f'μ={mean_val:.1f}\nm={median_val:.1f}',
                    ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels([cat[:10] for cat in top_cats], rotation=45, ha='right')
        ax1.set_xlabel('品类', fontsize=11)
        ax1.set_ylabel('日销量 (千克)', fontsize=11)
        ax1.set_title('(a) 主要品类销量分布形态对比', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        ax1.plot([], [], 'o', color='blue', label='均值')
        ax1.plot([], [], '-', color='red', linewidth=2, label='中位数')
        ax1.legend(loc='upper right', fontsize=9)
        
        # (b) 异常值检测 - Z-score vs IQR方法对比
        ax2 = fig.add_subplot(gs[1, 0])
        
        # 选择一个典型品类进行异常值分析
        example_cat = top_cats[0]
        data = cat_matrix[example_cat].values
        
        # 计算Z-score
        z_scores = np.abs(zscore(data))
        
        # 计算IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        # 异常值判定
        z_outliers = z_scores > 3
        iqr_outliers = (data < (Q1 - 1.5*IQR)) | (data > (Q3 + 1.5*IQR))
        
        # 绘制散点图
        time_index = range(len(data))
        
        # 正常点
        normal_mask = ~(z_outliers | iqr_outliers)
        ax2.scatter(np.array(time_index)[normal_mask], data[normal_mask], 
                   c='gray', s=10, alpha=0.5, label='正常值')
        
        # Z-score异常值
        z_only = z_outliers & ~iqr_outliers
        if z_only.any():
            ax2.scatter(np.array(time_index)[z_only], data[z_only], 
                       c='orange', s=50, marker='^', label='仅Z-score异常', 
                       edgecolors='black', linewidth=1)
        
        # IQR异常值
        iqr_only = iqr_outliers & ~z_outliers
        if iqr_only.any():
            ax2.scatter(np.array(time_index)[iqr_only], data[iqr_only], 
                       c='blue', s=50, marker='s', label='仅IQR异常',
                       edgecolors='black', linewidth=1)
        
        # 两种方法都判定为异常
        both_outliers = z_outliers & iqr_outliers
        if both_outliers.any():
            ax2.scatter(np.array(time_index)[both_outliers], data[both_outliers], 
                       c='red', s=80, marker='*', label='双重异常',
                       edgecolors='black', linewidth=1.5)
        
        # 添加阈值线
        ax2.axhline(y=Q3 + 1.5*IQR, color='blue', linestyle='--', 
                   alpha=0.5, label='IQR上界')
        ax2.axhline(y=Q1 - 1.5*IQR, color='blue', linestyle='--', 
                   alpha=0.5, label='IQR下界')
        
        ax2.set_xlabel('时间索引', fontsize=11)
        ax2.set_ylabel('日销量 (千克)', fontsize=11)
        ax2.set_title(f'(b) {example_cat} 异常值检测', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        
        # (c) 分布形态分类 - 基于偏度和峰度
        ax3 = fig.add_subplot(gs[1, 1])
        
        # 计算所有品类的偏度和峰度
        stats_data = []
        for cat in cat_matrix.columns:
            data = cat_matrix[cat]
            if len(data[data > 0]) > 30:
                stats_data.append({
                    'category': cat,
                    'skewness': data.skew(),
                    'kurtosis': data.kurtosis(),
                    'cv': data.std() / data.mean() if data.mean() > 0 else 0,
                    'mean': data.mean()
                })
        
        stats_df = pd.DataFrame(stats_data)
        
        # 定义分布类型
        def classify_distribution(skew, kurt):
            if abs(skew) < 0.5 and abs(kurt) < 1:
                return '近似正态', 'green'
            elif skew > 1:
                return '右偏', 'orange'
            elif skew < -1:
                return '左偏', 'blue'
            elif kurt > 3:
                return '尖峰', 'red'
            elif kurt < -1:
                return '平峰', 'purple'
            else:
                return '其他', 'gray'
        
        # 分类并绘制
        for _, row in stats_df.iterrows():
            dist_type, color = classify_distribution(row['skewness'], row['kurtosis'])
            size = 50 + row['mean'] * 2  # 点大小表示销量
            ax3.scatter(row['skewness'], row['kurtosis'], 
                       s=size, c=color, alpha=0.6, 
                       edgecolors='black', linewidth=0.5)
        
        # 添加分区背景
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # 添加分区标注
        ax3.text(2, 5, '右偏\n尖峰', fontsize=9, alpha=0.5, ha='center')
        ax3.text(-2, 5, '左偏\n尖峰', fontsize=9, alpha=0.5, ha='center')
        ax3.text(2, -2, '右偏\n平峰', fontsize=9, alpha=0.5, ha='center')
        ax3.text(-2, -2, '左偏\n平峰', fontsize=9, alpha=0.5, ha='center')
        ax3.text(0, 0, '正态', fontsize=9, alpha=0.5, ha='center', 
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
        
        ax3.set_xlabel('偏度 (Skewness)', fontsize=11)
        ax3.set_ylabel('峰度 (Kurtosis)', fontsize=11)
        ax3.set_title('(c) 品类分布形态分类图', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.6, label='近似正态'),
            Patch(facecolor='orange', alpha=0.6, label='右偏'),
            Patch(facecolor='blue', alpha=0.6, label='左偏'),
            Patch(facecolor='red', alpha=0.6, label='尖峰'),
            Patch(facecolor='purple', alpha=0.6, label='平峰')
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.suptitle('图1：蔬菜品类销量分布特征分析', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            plt.savefig('Fig1_分布特征分析.png', dpi=dpi, bbox_inches='tight')
            print(f"✓ 已保存: Fig1_分布特征分析.png")
        
        plt.show()
        return fig
    
    def figure2_temporal_patterns(self, save=True, dpi=300):
        """
        图2：时间模式分析
        (a) STL分解展示
        (b) 周内模式热力图
        (c) 月度趋势变化
        (d) 自相关瀑布图
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ts_data = self.chart_data.get('time_series', {})
        corr_data = self.chart_data.get('correlation', {})
        
        if not ts_data or 'category_matrix' not in corr_data:
            print("缺少时间序列数据")
            return
        
        cat_matrix = corr_data['category_matrix']
        
        # (a) STL分解 - 使用子图展示各成分
        if ts_data.get('stl_trend') is not None:
            ax1 = fig.add_subplot(gs[:, 0])  # 跨两行
            
            # 创建4个子轴
            ax1.axis('off')
            
            # 原始序列
            ax1_1 = fig.add_subplot(4, 2, 1)
            series = ts_data['series']
            ax1_1.plot(series, color='black', linewidth=0.8)
            ax1_1.set_ylabel('原始销量', fontsize=9)
            ax1_1.set_xticklabels([])
            ax1_1.grid(True, alpha=0.3)
            
            # 趋势
            ax1_2 = fig.add_subplot(4, 2, 3)
            trend = ts_data['stl_trend']
            ax1_2.plot(trend, color='red', linewidth=1.5)
            ax1_2.set_ylabel('长度趋势', fontsize=9)
            ax1_2.set_xticklabels([])
            ax1_2.grid(True, alpha=0.3)
            
            # 季节性
            ax1_3 = fig.add_subplot(4, 2, 5)
            seasonal = ts_data['stl_seasonal']
            ax1_3.plot(seasonal[:100], color='green', linewidth=1)
            ax1_3.set_ylabel('季节模式', fontsize=9)
            ax1_3.set_xticklabels([])
            ax1_3.grid(True, alpha=0.3)
            
            # 残差
            ax1_4 = fig.add_subplot(4, 2, 7)
            resid = ts_data['stl_resid']
            ax1_4.plot(resid, color='blue', linewidth=0.5, alpha=0.7)
            ax1_4.set_ylabel('随机波动', fontsize=9)
            ax1_4.set_xlabel('时间索引', fontsize=9)
            ax1_4.grid(True, alpha=0.3)
            
            # 添加总标题
            ax1_1.set_title('(a) 蔬菜销量趋势分解（STL）', fontsize=12, fontweight='bold')
        
        # (b) 周内销售模式热力图
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 准备数据：计算每个品类每个星期几的平均销量
        weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        hours = list(range(24))  # 如果有小时数据
        
        # 创建星期-品类矩阵
        top_cats = cat_matrix.sum().nlargest(15).index
        weekly_pattern = np.zeros((7, len(top_cats)))
        
        for i, cat in enumerate(top_cats):
            for wd in range(7):
                mask = pd.to_datetime(cat_matrix.index).weekday == wd
                weekly_pattern[wd, i] = cat_matrix.loc[mask, cat].mean()
        
        # 按行归一化（展示相对模式）        
        weekly_pattern_norm = weekly_pattern / weekly_pattern.sum(axis=0)

        # 用 seaborn 绘制热力图
        im = sns.heatmap(weekly_pattern_norm,
                    cmap="YlGnBu",  # 更自然的配色
                    xticklabels=[cat[:8] for cat in top_cats],
                    yticklabels=weekdays,
                    annot=True, fmt=".2f",  # 自动标注数值
                    cbar_kws={'label': '相对销量占比'},
                    ax=ax2, linewidths=0.3, linecolor='white')
        
        ax2.set_title('(b) 品类周内销售模式', fontsize=12, fontweight='bold')
        ax2.set_xlabel('品类', fontsize=10)
        ax2.set_ylabel('星期', fontsize=10)
        
        # (c) 月度趋势分析
        ax3 = fig.add_subplot(gs[1, 1])
        
        # 计算月度聚合
        top_cats_monthly = cat_matrix[top_cats[:5]]  # 选择前5个品类
        
        # 按月聚合
        monthly_data = []
        for cat in top_cats_monthly.columns:
            monthly = cat_matrix[cat].resample('M').agg(['mean', 'std'])
            monthly_data.append(monthly)
        
        # 绘制月度趋势
        colors = sns.color_palette("Set2", 5)  # 柔和的调色盘
        for i, (cat, monthly) in enumerate(zip(top_cats_monthly.columns, monthly_data)):
            ax3.plot(monthly.index, monthly['mean'],
                     color=colors[i], linewidth=2, label=cat[:10])
            ax3.fill_between(monthly.index,
                             monthly['mean'] - monthly['std'],
                             monthly['mean'] + monthly['std'],
                             color=colors[i], alpha=0.15)

        # ===== 优化横轴格式 =====
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y年%m月'))  # 中文形式
        # ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))  # 英文缩写，如 Aug-2020

        # 显示每 3 个月一个标签，避免过密
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

        ax3.set_title('(c) 主要品类月度趋势', fontsize=12, fontweight='bold')
        ax3.set_xlabel('时间', fontsize=11)
        ax3.set_ylabel('月均销量 (千克)', fontsize=11)
        ax3.legend(loc='best', fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        
        if save:
            plt.savefig('Fig2_时间模式分析.png', dpi=dpi, bbox_inches='tight')
            print(f"✓ 已保存: Fig2_时间模式分析.png")
        
        plt.show()
        return fig
    
    def figure3_correlation_analysis(self, save=True, dpi=300):
        """
        图3：相关性分析
        (a) 聚类热力图
        (b) 散点矩阵（前4个品类）
        (c) 网络中心性分析
        (d) 平行坐标图
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        corr_data = self.chart_data.get('correlation', {})
        if not corr_data:
            print("缺少相关性数据")
            return
        
        spearman_corr = corr_data.get('spearman_corr', pd.DataFrame())
        cat_matrix = corr_data.get('category_matrix', pd.DataFrame())
        
        # (a) 层次聚类热力图
        ax1 = fig.add_subplot(gs[0, 0])
        
        # 选择前20个品类
        n_cats = min(20, len(spearman_corr))
        corr_subset = spearman_corr.iloc[:n_cats, :n_cats]
        
        # 层次聚类
        from scipy.spatial.distance import squareform
        distance_matrix = 1 - abs(corr_subset)
        np.fill_diagonal(distance_matrix.values, 0)
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # 获取聚类顺序
        from scipy.cluster.hierarchy import dendrogram
        dendro = dendrogram(linkage_matrix, no_plot=True)
        cluster_order = dendro['leaves']
        
        # 重排矩阵
        clustered_corr = corr_subset.iloc[cluster_order, cluster_order]
        
        # 绘制热力图
        mask = np.triu(np.ones_like(clustered_corr), k=1)
        sns.heatmap(clustered_corr, mask=mask, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "相关系数"},
                   vmin=-1, vmax=1, ax=ax1, 
                   xticklabels=[x[:8] for x in clustered_corr.columns],
                   yticklabels=[y[:8] for y in clustered_corr.index])
        
        ax1.set_title('(a) 品类相关性层次聚类', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)
        
        # (b) 散点矩阵 - 展示前4个高相关品类
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 找出相关性最高的4个品类
        corr_values = []
        for i in range(len(corr_subset)):
            for j in range(i+1, len(corr_subset)):
                corr_values.append((abs(corr_subset.iloc[i, j]), 
                                  corr_subset.index[i], 
                                  corr_subset.index[j]))
        corr_values.sort(reverse=True)
        
        # 获取相关性最高的品类
        selected_cats = set()
        for _, cat1, cat2 in corr_values[:6]:
            selected_cats.add(cat1)
            selected_cats.add(cat2)
            if len(selected_cats) >= 4:
                break
        selected_cats = list(selected_cats)[:4]
        
        # 创建散点矩阵
        from pandas.plotting import scatter_matrix
        scatter_data = cat_matrix[selected_cats].iloc[::5, :]  # 采样以提高性能
        
        # 清除当前子图并创建新的
        ax2.remove()
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 手动创建2x2散点矩阵
        for i in range(2):
            for j in range(2):
                ax_sub = fig.add_subplot(4, 4, 4+i*4+j+1)
                if i == j:
                    # 对角线上显示分布
                    ax_sub.hist(scatter_data[selected_cats[i]], bins=20, 
                              color='skyblue', edgecolor='black', alpha=0.7)
                    if i == 0:
                        ax_sub.set_title(selected_cats[i][:8], fontsize=8)
                else:
                    # 非对角线显示散点图
                    ax_sub.scatter(scatter_data[selected_cats[j]], 
                                 scatter_data[selected_cats[i]],
                                 alpha=0.5, s=10)
                    # 添加趋势线
                    z = np.polyfit(scatter_data[selected_cats[j]], 
                                 scatter_data[selected_cats[i]], 1)
                    p = np.poly1d(z)
                    ax_sub.plot(scatter_data[selected_cats[j]], 
                              p(scatter_data[selected_cats[j]]),
                              "r--", alpha=0.5, linewidth=1)
                
                if j == 0:
                    ax_sub.set_ylabel(selected_cats[i][:8], fontsize=8)
                if i == 1:
                    ax_sub.set_xlabel(selected_cats[j][:8], fontsize=8)
                
                ax_sub.tick_params(labelsize=6)
        
        # 添加总标题
        fig.text(0.55, 0.72, '(b) 高相关品类散点矩阵', 
                fontsize=12, fontweight='bold', ha='center')
        
        # (c) 网络分析 - 度分布和中心性
        ax3 = fig.add_subplot(gs[1, 0])
        
        # 构建网络
        G = nx.Graph()
        threshold = 0.3
        
        for i in range(len(corr_subset)):
            for j in range(i+1, len(corr_subset)):
                if abs(corr_subset.iloc[i, j]) > threshold:
                    G.add_edge(corr_subset.index[i], corr_subset.index[j],
                             weight=abs(corr_subset.iloc[i, j]))
        
        if len(G.nodes()) > 0:
            # 计算中心性指标
            degree_cent = nx.degree_centrality(G)
            between_cent = nx.betweenness_centrality(G)
            
            # 准备数据
            nodes = list(G.nodes())[:10]  # 显示前10个
            x = np.arange(len(nodes))
            
            # 创建双轴图
            ax3_twin = ax3.twinx()
            
            # 度中心性（条形图）
            bars = ax3.bar(x - 0.2, [degree_cent[n] for n in nodes], 
                          0.4, label='度中心性', color='blue', alpha=0.7)
            
            # 介数中心性（线图）
            line = ax3_twin.plot(x, [between_cent[n] for n in nodes], 
                               'ro-', label='介数中心性', linewidth=2, markersize=8)
            
            ax3.set_xlabel('品类', fontsize=11)
            ax3.set_ylabel('度中心性', fontsize=11, color='blue')
            ax3_twin.set_ylabel('介数中心性', fontsize=11, color='red')
            ax3.set_title('(c) 网络中心性分析', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels([n[:8] for n in nodes], rotation=45, ha='right')
            
            # 合并图例
            lines, labels = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)
            
            ax3.grid(True, alpha=0.3)
        
        # (d) 平行坐标图 - 展示品类特征模式
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 准备数据：选择几个代表性品类
        selected_cats_parallel = cat_matrix.columns[:8]
        
        # 计算每个品类的统计特征
        features_data = []
        for cat in selected_cats_parallel:
            data = cat_matrix[cat]
            features_data.append({
                '品类': cat[:10],
                '均值': data.mean(),
                '标准差': data.std(),
                'CV': data.std()/data.mean() if data.mean() > 0 else 0,
                '最大值': data.max(),
                '零占比': (data == 0).mean()
            })
        
        features_df = pd.DataFrame(features_data)
        
        # 归一化数据
        numeric_cols = ['均值', '标准差', 'CV', '最大值', '零占比']
        features_norm = features_df.copy()
        for col in numeric_cols:
            max_val = features_df[col].max()
            min_val = features_df[col].min()
            if max_val > min_val:
                features_norm[col] = (features_df[col] - min_val) / (max_val - min_val)
        
        # 绘制平行坐标
        colors_parallel = sns.color_palette("husl", len(features_norm))
        
        for i, row in features_norm.iterrows():
            values = [row[col] for col in numeric_cols]
            ax4.plot(range(len(numeric_cols)), values, 'o-', 
                    color=colors_parallel[i], alpha=0.7, 
                    linewidth=2, markersize=6, label=row['品类'])
        
        ax4.set_xticks(range(len(numeric_cols)))
        ax4.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax4.set_ylabel('归一化值', fontsize=11)
        ax4.set_title('(d) 品类特征平行坐标图', fontsize=12, fontweight='bold')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim([-0.05, 1.05])
        
        plt.suptitle('图3：品类相关性与特征分析', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            plt.savefig('Fig3_相关性分析.png', dpi=dpi, bbox_inches='tight')
            print(f"✓ 已保存: Fig3_相关性分析.png")
        
        plt.show()
        return fig
    
    def figure4_clustering_insights(self, save=True, dpi=300):
        """
        图4：聚类洞察分析
        (a) 聚类树状图
        (b) 轮廓分析图
        (c) 聚类特征条形图对比
        (d) 业务解释桑基图
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        cluster_data = self.chart_data.get('clustering', {})
        if not cluster_data:
            print("缺少聚类数据")
            return
        
        # (a) 层次聚类树状图
        ax1 = fig.add_subplot(gs[0, 0])
        
        features = cluster_data.get('features', np.array([]))
        if len(features) > 0:
            # 计算层次聚类
            linkage_matrix = linkage(features[:50], method='ward')  # 限制样本数
            
            # 绘制树状图
            dendro = dendrogram(linkage_matrix, ax=ax1, 
                              orientation='top',
                              labels=None,
                              distance_sort='descending',
                              show_leaf_counts=True,
                              leaf_font_size=8)
            
            ax1.set_title('(a) 单品层次聚类树状图', fontsize=12, fontweight='bold')
            ax1.set_xlabel('样本索引', fontsize=11)
            ax1.set_ylabel('Ward距离', fontsize=11)
            
            # 添加阈值线
            threshold = 0.7 * max(linkage_matrix[:, 2])
            ax1.axhline(y=threshold, color='red', linestyle='--', 
                       label=f'截断阈值={threshold:.2f}')
            ax1.legend(loc='upper right', fontsize=9)
        
        # (b) 轮廓分析
        ax2 = fig.add_subplot(gs[0, 1])
        
        best_labels = cluster_data.get('best_labels', [])
        if len(best_labels) > 0 and len(features) > 0:
            from sklearn.metrics import silhouette_samples
            
            silhouette_vals = silhouette_samples(features, best_labels)
            
            y_lower = 10
            colors_sil = sns.color_palette("husl", len(np.unique(best_labels)))
            
            for i, cluster in enumerate(np.unique(best_labels)):
                # 获取该聚类的轮廓值
                cluster_silhouette_vals = silhouette_vals[best_labels == cluster]
                cluster_silhouette_vals.sort()
                
                size_cluster = cluster_silhouette_vals.shape[0]
                y_upper = y_lower + size_cluster
                
                ax2.fill_betweenx(np.arange(y_lower, y_upper),
                                 0, cluster_silhouette_vals,
                                 facecolor=colors_sil[i], alpha=0.7)
                
                # 标注聚类标签
                ax2.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster),
                        fontsize=9)
                
                y_lower = y_upper + 10
            
            # 添加平均线
            avg_score = silhouette_vals.mean()
            ax2.axvline(x=avg_score, color='red', linestyle='--',
                       label=f'平均值={avg_score:.3f}')
            
            ax2.set_title('(b) 聚类轮廓分析', fontsize=12, fontweight='bold')
            ax2.set_xlabel('轮廓系数', fontsize=11)
            ax2.set_ylabel('聚类样本', fontsize=11)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.set_xlim([-0.1, 1])
            ax2.grid(True, alpha=0.3, axis='x')
        
        # (c) 聚类特征对比 - 分组条形图
        ax3 = fig.add_subplot(gs[1, 0])
        
        profiles = cluster_data.get('profiles', pd.DataFrame())
        cluster_labels_map = cluster_data.get('cluster_labels', {})
        
        if len(profiles) > 0:
            # 选择关键特征
            key_features = ['日均销量', 'CV', '零占比', '爆发度']
            
            # 准备数据
            n_clusters = len(profiles)
            x = np.arange(len(key_features))
            width = 0.8 / n_clusters
            
            colors_bar = sns.color_palette("Set2", n_clusters)
            
            for i, cluster_id in enumerate(profiles.index):
                values = []
                for feat in key_features:
                    if (feat, 'mean') in profiles.columns:
                        val = profiles.loc[cluster_id, (feat, 'mean')]
                        # 归一化
                        max_val = profiles[(feat, 'mean')].max()
                        if max_val > 0:
                            values.append(val / max_val)
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                
                cluster_name = cluster_labels_map.get(cluster_id, f'聚类{cluster_id}')
                bars = ax3.bar(x + i*width, values, width, 
                              label=cluster_name, color=colors_bar[i], alpha=0.8)
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    if val > 0.1:
                        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                               f'{val:.2f}', ha='center', fontsize=8)
            
            ax3.set_xlabel('特征', fontsize=11)
            ax3.set_ylabel('归一化值', fontsize=11)
            ax3.set_title('(c) 聚类特征对比', fontsize=12, fontweight='bold')
            ax3.set_xticks(x + width * (n_clusters-1) / 2)
            ax3.set_xticklabels(key_features)
            ax3.legend(loc='upper left', fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim([0, 1.2])
        
        # (d) 聚类分布饼图
        ax4 = fig.add_subplot(gs[1, 1])
        
        item_features_df = cluster_data.get('item_features_df', pd.DataFrame())
        if len(item_features_df) > 0 and '聚类标签' in item_features_df.columns:
            # 统计每个聚类的数量和销量
            cluster_stats = item_features_df.groupby('聚类标签').agg({
                '单品编码': 'count',
                '总销量': 'sum'
            }).rename(columns={'单品编码': '单品数'})
            
            # 创建嵌套饼图
            colors_pie = sns.color_palette("Set3", len(cluster_stats))
            
            # 外圈：单品数量
            wedges, texts, autotexts = ax4.pie(cluster_stats['单品数'], 
                                               labels=cluster_stats.index,
                                               colors=colors_pie,
                                               autopct='%1.1f%%',
                                               startangle=90,
                                               radius=1,
                                               wedgeprops=dict(width=0.3, edgecolor='white'),
                                               textprops={'fontsize': 9})
            
            # 内圈：销量占比
            ax4.pie(cluster_stats['总销量'],
                   colors=[c + (0.6,) for c in colors_pie],  # 透明度
                   startangle=90,
                   radius=0.7,
                   wedgeprops=dict(width=0.3, edgecolor='white'))
            
            # 添加图例
            ax4.legend(wedges, [f'{label}\n({cluster_stats.loc[label, "单品数"]}个)' 
                               for label in cluster_stats.index],
                      loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=9)
            
            ax4.set_title('(d) 聚类分布（外:数量 内:销量）', 
                         fontsize=12, fontweight='bold')
        
        plt.suptitle('图4：单品聚类洞察分析', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            plt.savefig('Fig4_聚类洞察.png', dpi=dpi, bbox_inches='tight')
            print(f"✓ 已保存: Fig4_聚类洞察.png")
        
        plt.show()
        return fig
    
    def generate_all_figures(self, dpi=300):
        """生成所有论文图表"""
        print("\n" + "="*50)
        print("生成论文级可视化图表")
        print("="*50)
        
        figures = {}
        
        # 生成各图
        print("\n生成图1：分布特征分析...")
        try:
            figures['fig1'] = self.figure1_distribution_analysis(save=True, dpi=dpi)
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
        
        print("\n生成图2：时间模式分析...")
        try:
            figures['fig2'] = self.figure2_temporal_patterns(save=True, dpi=dpi)
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
        
        print("\n生成图3：相关性分析...")
        try:
            figures['fig3'] = self.figure3_correlation_analysis(save=True, dpi=dpi)
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
        
        print("\n生成图4：聚类洞察...")
        try:
            figures['fig4'] = self.figure4_clustering_insights(save=True, dpi=dpi)
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
        
        print("\n" + "="*50)
        print("图表生成完成！")
        print("="*50)
        
        return figures


def main():
    """主函数"""
    print("="*60)
    print("论文级蔬菜销售数据可视化系统")
    print("="*60)
    
    # 创建可视化器
    visualizer = PaperVisualizer('图表数据.pkl')
    
    while True:
        print("\n请选择操作：")
        print("1. 生成图1：分布特征分析")
        print("2. 生成图2：时间模式分析")
        print("3. 生成图3：相关性分析")
        print("4. 生成图4：聚类洞察")
        print("5. 生成所有图表（推荐）")
        print("6. 生成高DPI图表（用于印刷）")
        print("0. 退出")
        
        choice = input("\n请输入选项 (0-6): ").strip()
        
        if choice == '0':
            print("感谢使用！")
            break
        elif choice == '1':
            visualizer.figure1_distribution_analysis()
        elif choice == '2':
            visualizer.figure2_temporal_patterns()
        elif choice == '3':
            visualizer.figure3_correlation_analysis()
        elif choice == '4':
            visualizer.figure4_clustering_insights()
        elif choice == '5':
            visualizer.generate_all_figures(dpi=300)
        elif choice == '6':
            dpi = input("请输入DPI值（默认600）: ").strip()
            dpi = int(dpi) if dpi else 600
            visualizer.generate_all_figures(dpi=dpi)
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()
