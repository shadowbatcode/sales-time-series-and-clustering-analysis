import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === 1. 设置中文字体 ===
# Windows 示例
# rcParams['font.sans-serif'] = ['SimHei']  
# Mac 示例
rcParams['font.sans-serif'] = ['SimHei']  
rcParams['axes.unicode_minus'] = False

# === 2. 读取数据 ===
df = pd.read_excel("sales.xlsx")  # 或 pd.read_excel("sales.xlsx")
df["销售日期_dt"] = pd.to_datetime(df["销售日期_dt"])

# === 3. 数据聚合（按商品+类别+日期） ===
df_agg = df.groupby(["单品名称", "分类名称", "销售日期_dt"]).agg({
    "销量(千克)": "sum",
    "销售额": "sum",
    "是否打折销售": "max"
}).reset_index()

# === 4. 创建图谱 ===
G = nx.DiGraph()

for _, row in df_agg.iterrows():
    product = row["单品名称"]
    category = row["分类名称"]
    date = str(row["销售日期_dt"].date())

    # 添加节点
    G.add_node(product, type="商品", size=row["销量(千克)"])
    G.add_node(category, type="类别")
    G.add_node(date, type="日期")

    # 添加边
    G.add_edge(product, category, relation="属于")
    G.add_edge(product, date, relation="销售于", weight=row["销量(千克)"])

# === 5. 布局与绘图 ===
plt.figure(figsize=(18, 12))
pos = nx.spring_layout(G, k=0.5, seed=42)  # 紧凑布局

# 设置节点颜色和大小
node_colors = []
node_sizes = []
for n, attr in G.nodes(data=True):
    if attr["type"] == "商品":
        node_colors.append("orange")
        node_sizes.append(attr.get("size", 10)*20)
    elif attr["type"] == "类别":
        node_colors.append("lightblue")
        node_sizes.append(300)
    else:  # 日期
        node_colors.append("lightgreen")
        node_sizes.append(200)

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

# 绘制边，宽度映射销量
edge_weights = [G[u][v].get("weight", 1) for u,v in G.edges()]
nx.draw_networkx_edges(G, pos, width=[max(0.5, w/10) for w in edge_weights], alpha=0.6, arrows=False)

# 绘制标签（只标商品和类别，避免日期节点太多）
labels = {n: n for n, attr in G.nodes(data=True) if attr["type"] in ["商品","类别"]}
nx.draw_networkx_labels(G, pos, labels, font_size=10)

plt.axis("off")
plt.title("销售知识图谱（商品-类别-日期）", fontsize=18)
plt.tight_layout()
plt.savefig("sales_graph_chinese.png", dpi=300)
plt.show()
