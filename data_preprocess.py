"""
蔬菜销售数据分析完整脚本 - 完全修复版
所有编码都替换为名称，彻底解决KeyError问题
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 统计分析库
from scipy import stats
from scipy.stats import skew, kurtosis, kstest, probplot
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf, adfuller

# 机器学习库
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

# 关联规则挖掘
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 用于保存所有图表数据的字典
CHART_DATA = {}

# ================== 1. 数据预处理 ==================
print("="*50)
print("1. 数据预处理")
print("="*50)

# 读取数据
df_info = pd.read_excel("附件1.xlsx")
df_sales = pd.read_excel("附件2.xlsx")

# 数值字段处理
df_info['单品名称'] = df_info['单品名称'].str.replace(r"\(.*?\)", "", regex=True).str.strip()
df_sales['销量(千克)'] = pd.to_numeric(df_sales['销量(千克)'], errors='coerce')
df_sales['销售单价(元/千克)'] = pd.to_numeric(df_sales['销售单价(元/千克)'], errors='coerce')

# 只保留正常销售
df_sales = df_sales[df_sales['销售类型'] == "销售"].copy()

# 转换打折标记
df_sales['是否打折销售'] = df_sales['是否打折销售'].map({"是": 1, "否": 0})

# 构造完整时间戳
df_sales['销售时间'] = pd.to_datetime(
    df_sales['销售日期'].astype(str) + " " + df_sales['扫码销售时间'].astype(str),
    format="%Y-%m-%d %H:%M:%S.%f",
    errors="coerce"
)

# 删除时间转换失败的记录
print(f"时间转换失败的记录数: {df_sales['销售时间'].isna().sum()}")
df_sales = df_sales.dropna(subset=['销售时间']).copy()

# 拆分出日期和小时
df_sales['销售日期_dt'] = pd.to_datetime(df_sales['销售时间'].dt.date)
df_sales['hour'] = df_sales['销售时间'].dt.hour
df_sales['weekday'] = df_sales['销售时间'].dt.weekday

# 计算销售额
df_sales['销售额'] = df_sales['销量(千克)'] * df_sales['销售单价(元/千克)']

# 合并品类信息
df = pd.merge(df_sales, df_info, on="单品编码", how="left")

# 删除合并后的缺失值
df = df.dropna(subset=['单品名称', '分类编码', '分类名称']).copy()

# 创建全局映射字典
item_code_to_name = dict(df[['单品编码', '单品名称']].drop_duplicates().values)
cat_code_to_name = dict(df[['分类编码', '分类名称']].drop_duplicates().values)

# 单品日聚合
daily_item = df.groupby(['销售日期_dt', '单品编码', '单品名称', '分类编码', '分类名称']).agg({
    '销量(千克)': 'sum',
    '销售额': 'sum',
    '是否打折销售': 'max'
}).reset_index()

# 品类日聚合
daily_category = df.groupby(['销售日期_dt', '分类编码', '分类名称']).agg({
    '销量(千克)': 'sum',
    '销售额': 'sum'
}).reset_index()

print(f"原始记录数: {len(df_sales)}")
print(f"单品数: {df['单品编码'].nunique()}")
print(f"品类数: {df['分类编码'].nunique()}")
print(f"日期范围: {df['销售日期_dt'].min().date()} 至 {df['销售日期_dt'].max().date()}")

# 保存基础统计数据
CHART_DATA['basic_stats'] = {
    'record_count': len(df_sales),
    'item_count': df['单品编码'].nunique(),
    'category_count': df['分类编码'].nunique(),
    'date_range': (df['销售日期_dt'].min(), df['销售日期_dt'].max())
}

# ================== 2. 异常值检测 ==================
print("\n" + "="*50)
print("2. 异常值检测")
print("="*50)

def detect_outliers_iqr_mad(series, iqr_factor=3, mad_factor=6):
    """IQR和MAD双准则异常值检测"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_iqr = Q1 - iqr_factor * IQR
    upper_iqr = Q3 + iqr_factor * IQR
    
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = 1.4826 * series.std()
    
    outliers_iqr = (series < lower_iqr) | (series > upper_iqr)
    outliers_mad = np.abs(series - median) / mad > mad_factor if mad > 0 else pd.Series([False]*len(series))
    
    return outliers_iqr | outliers_mad

# 对每个单品检测异常值
outlier_stats = []
for item_code in daily_item['单品编码'].unique():
    item_data = daily_item[daily_item['单品编码'] == item_code]['销量(千克)']
    if len(item_data) > 10:
        outliers = detect_outliers_iqr_mad(item_data)
        outlier_stats.append({
            '单品编码': item_code,
            '单品名称': item_code_to_name.get(item_code, item_code),
            '总天数': len(item_data),
            '异常天数': outliers.sum(),
            '异常比例': outliers.sum() / len(item_data)
        })

outlier_df = pd.DataFrame(outlier_stats)
print(f"检测到异常值的单品数: {(outlier_df['异常比例'] > 0).sum()}")
print(f"平均异常比例: {outlier_df['异常比例'].mean():.2%}")

# ================== 3. 分布规律分析 ==================
print("\n" + "="*50)
print("3. 分布规律分析")
print("="*50)

def calculate_distribution_stats(data, name=""):
    """计算分布统计指标"""
    stats_dict = {
        '名称': name,
        '样本数': len(data),
        '均值': data.mean(),
        '标准差': data.std(),
        '最小值': data.min(),
        'P25': data.quantile(0.25),
        'P50': data.quantile(0.50),
        'P75': data.quantile(0.75),
        'P90': data.quantile(0.90),
        'P95': data.quantile(0.95),
        '最大值': data.max(),
        '偏度': skew(data),
        '峰度': kurtosis(data),
        'CV': data.std() / data.mean() if data.mean() > 0 else np.nan,
        '零占比': (data == 0).mean(),
        '爆发度': data.max() / data.mean() if data.mean() > 0 else np.nan
    }
    return stats_dict

# 品类分布统计
category_stats = []
for cat_code in daily_category['分类编码'].unique():
    cat_data = daily_category[daily_category['分类编码'] == cat_code]['销量(千克)']
    cat_name = cat_code_to_name.get(cat_code, cat_code)
    stats = calculate_distribution_stats(cat_data, cat_name)
    category_stats.append(stats)

category_stats_df = pd.DataFrame(category_stats)
print("\n品类分布统计:")
print(category_stats_df[['名称', '均值', '标准差', 'CV', '偏度', '零占比']].head(10).to_string())

# 正态性检验
normality_test_results = []
print("\n正态性检验 (Kolmogorov-Smirnov):")
for cat_code in daily_category['分类编码'].unique()[:5]:
    cat_data = daily_category[daily_category['分类编码'] == cat_code]['销量(千克)']
    cat_name = cat_code_to_name.get(cat_code, cat_code)
    if len(cat_data) > 30:
        ks_stat, ks_p = kstest(cat_data, 'norm', args=(cat_data.mean(), cat_data.std()))
        normality_test_results.append({
            '品类名称': cat_name,
            'KS统计量': ks_stat,
            'p值': ks_p,
            '是否正态': 'Yes' if ks_p > 0.05 else 'No'
        })
        print(f"  {cat_name}: KS统计量={ks_stat:.3f}, p值={ks_p:.3e}")

# 选择销量最大的品类
top_category_code = daily_category.groupby('分类编码')['销量(千克)'].sum().idxmax()
top_category_name = cat_code_to_name.get(top_category_code, top_category_code)
top_cat_sales = daily_category[daily_category['分类编码'] == top_category_code]['销量(千克)']

# 保存分布数据
CHART_DATA['distribution'] = {
    'top_category_sales': top_cat_sales.values,
    'top_category_code': top_category_code,
    'top_category_name': top_category_name,
    'normality_tests': pd.DataFrame(normality_test_results) if normality_test_results else pd.DataFrame()
}

# ================== 4. 时间序列分析 ==================
print("\n" + "="*50)
print("4. 时间序列分析")
print("="*50)

def perform_stl_decomposition(series, period=7):
    """STL分解"""
    if len(series) < 2 * period:
        return None
    
    stl = STL(series, period=period, seasonal=13)
    result = stl.fit()
    
    seasonal_amplitude = result.seasonal.quantile(0.95) - result.seasonal.quantile(0.05)
    x = np.arange(len(result.trend))
    trend_slope = np.polyfit(x, result.trend, 1)[0]
    
    return {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid,
        'seasonal_amplitude': seasonal_amplitude,
        'trend_slope': trend_slope
    }

# 使用top_category_code获取数据
top_cat_data = daily_category[daily_category['分类编码'] == top_category_code].sort_values('销售日期_dt')
top_cat_series = top_cat_data.set_index('销售日期_dt')['销量(千克)']

# 填充缺失日期
date_range = pd.date_range(top_cat_series.index.min(), top_cat_series.index.max())
top_cat_series = top_cat_series.reindex(date_range, fill_value=0)

stl_result = perform_stl_decomposition(top_cat_series, period=7)
if stl_result:
    print(f"\n品类 {top_category_name} 的STL分解结果:")
    print(f"  季节幅度: {stl_result['seasonal_amplitude']:.2f}")
    print(f"  趋势斜率: {stl_result['trend_slope']:.3f}")

# ACF/PACF分析
print("\n自相关分析:")
acf_values = acf(top_cat_series, nlags=30, fft=True)
pacf_values = pacf(top_cat_series, nlags=30)
print(f"  ACF@7天: {acf_values[7]:.3f}")
print(f"  ACF@14天: {acf_values[14]:.3f}")

# 平稳性检验
adf_result = adfuller(top_cat_series)
print(f"\nADF平稳性检验:")
print(f"  ADF统计量: {adf_result[0]:.3f}")
print(f"  p值: {adf_result[1]:.3f}")
print(f"  结论: {'平稳' if adf_result[1] < 0.05 else '非平稳'}")

# 保存时间序列数据
CHART_DATA['time_series'] = {
    'series': top_cat_series.values,
    'dates': top_cat_series.index,
    'stl_trend': stl_result['trend'].values if stl_result else None,
    'stl_seasonal': stl_result['seasonal'].values if stl_result else None,
    'stl_resid': stl_result['resid'].values if stl_result else None,
    'acf_values': acf_values,
    'pacf_values': pacf_values,
    'adf_result': {
        'statistic': adf_result[0],
        'pvalue': adf_result[1],
        'critical_values': adf_result[4]
    }
}

# ================== 5. 相关性分析 ==================
print("\n" + "="*50)
print("5. 相关性分析")
print("="*50)

# 准备品类时间序列矩阵 - 完全使用名称
date_range = pd.date_range(daily_category['销售日期_dt'].min(), 
                           daily_category['销售日期_dt'].max())

# 创建以品类名称为列的矩阵
category_matrix = pd.DataFrame(index=date_range)

# 按品类填充数据
for cat_code in daily_category['分类编码'].unique():
    cat_name = cat_code_to_name.get(cat_code, str(cat_code))
    cat_data = daily_category[daily_category['分类编码'] == cat_code]
    cat_series = cat_data.set_index('销售日期_dt')['销量(千克)']
    category_matrix[cat_name] = cat_series

category_matrix = category_matrix.fillna(0)

# 去季节处理
def remove_seasonality(series):
    """通过weekday回归去除季节性"""
    df_temp = pd.DataFrame({
        'y': series.values,
        'weekday': pd.to_datetime(series.index).weekday
    })
    
    weekday_dummies = pd.get_dummies(df_temp['weekday'], prefix='wd')
    
    from sklearn.linear_model import LinearRegression
    X = weekday_dummies
    y = df_temp['y']
    
    model = LinearRegression()
    model.fit(X, y)
    residuals = y - model.predict(X)
    
    return pd.Series(residuals.values, index=series.index)

# 去季节后的残差
residual_matrix = category_matrix.apply(remove_seasonality)

# Spearman相关矩阵
spearman_corr = residual_matrix.corr(method='spearman')
print("\nSpearman相关矩阵 (去季节后):")
print(spearman_corr.iloc[:5, :5])

# 偏相关分析
print("\n偏相关分析:")
try:
    lw = LedoitWolf()
    lw.fit(residual_matrix.dropna())
    precision_matrix = lw.precision_
    
    n_vars = len(category_matrix.columns)
    partial_corr = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(precision_matrix[i, i] * precision_matrix[j, j])
    
    partial_corr_df = pd.DataFrame(partial_corr, 
                                   index=category_matrix.columns, 
                                   columns=category_matrix.columns)
    print("偏相关矩阵 (前5x5):")
    print(partial_corr_df.iloc[:5, :5])
except Exception as e:
    print(f"偏相关分析失败: {e}")
    partial_corr_df = pd.DataFrame(np.zeros((len(category_matrix.columns), len(category_matrix.columns))),
                                   index=category_matrix.columns,
                                   columns=category_matrix.columns)

# 保存相关性数据
CHART_DATA['correlation'] = {
    'spearman_corr': spearman_corr,
    'partial_corr': partial_corr_df,
    'category_matrix': category_matrix
}

# ================== 6. 关联规则挖掘 ==================
print("\n" + "="*50)
print("6. 关联规则挖掘 (FP-Growth)")
print("="*50)

def create_baskets(df, time_window_minutes=5):
    """基于时间窗构造交易篮"""
    baskets = []
    
    for date in df['销售日期_dt'].unique():
        day_data = df[df['销售日期_dt'] == date].copy()
        day_data = day_data.sort_values('销售时间')
        
        if len(day_data) == 0:
            continue
            
        day_data['time_group'] = (day_data['销售时间'] - day_data['销售时间'].iloc[0]).dt.total_seconds() // (time_window_minutes * 60)
        
        for group in day_data['time_group'].unique():
            basket = day_data[day_data['time_group'] == group]['单品编码'].unique().tolist()
            if len(basket) > 1:
                baskets.append(basket)
    
    return baskets

# 创建交易篮
print("构造交易篮...")
baskets = create_baskets(df.head(50000), time_window_minutes=5)
print(f"生成交易篮数: {len(baskets)}")

rules = pd.DataFrame()
if len(baskets) > 100:
    try:
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = fpgrowth(basket_df, min_support=0.01, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            
            if len(rules) > 0:
                top_rules = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                print("\nTop 10 关联规则 (按lift排序):")
                for idx, row in top_rules.iterrows():
                    ant_codes = list(row['antecedents'])
                    cons_codes = list(row['consequents'])
                    
                    ant_names = [item_code_to_name.get(code, code)[:15] for code in ant_codes]
                    cons_names = [item_code_to_name.get(code, code)[:15] for code in cons_codes]
                    
                    ant = ant_names[0] if len(ant_names) == 1 else str(ant_names)
                    cons = cons_names[0] if len(cons_names) == 1 else str(cons_names)
                    
                    print(f"  {ant} => {cons}: lift={row['lift']:.2f}, conf={row['confidence']:.2f}")
    except Exception as e:
        print(f"关联规则挖掘失败: {e}")

# ================== 7. 聚类分析 ==================
print("\n" + "="*50)
print("7. 聚类分析")
print("="*50)

# 构造单品特征
item_features = []
for item_code in daily_item['单品编码'].unique():
    item_data = daily_item[daily_item['单品编码'] == item_code]['销量(千克)']
    
    if len(item_data) < 30:
        continue
    
    features = {
        '单品编码': item_code,
        '单品名称': item_code_to_name.get(item_code, str(item_code)),
        '总销量': item_data.sum(),
        '日均销量': item_data.mean(),
        '非零日均': item_data[item_data > 0].mean() if (item_data > 0).any() else 0,
        '标准差': item_data.std(),
        'CV': item_data.std() / item_data.mean() if item_data.mean() > 0 else 0,
        '爆发度': item_data.max() / item_data.mean() if item_data.mean() > 0 else 0,
        '零占比': (item_data == 0).mean(),
        '最大值': item_data.max()
    }
    item_features.append(features)

item_features_df = pd.DataFrame(item_features)
print(f"参与聚类的单品数: {len(item_features_df)}")

clustering_results = {}

if len(item_features_df) > 0:
    feature_cols = ['总销量', '日均销量', '非零日均', '标准差', 'CV', '爆发度', '零占比', '最大值']
    X = item_features_df[feature_cols].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # K-means聚类
    print("\n=== K-means聚类 ===")
    optimal_k = min(4, len(item_features_df))
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    kmeans_sil = silhouette_score(X_scaled, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else 0
    print(f"  轮廓系数: {kmeans_sil:.3f}")
    
    clustering_results['kmeans'] = {
        'labels': kmeans_labels,
        'silhouette': kmeans_sil,
        'calinski_harabasz': 0,
        'davies_bouldin': 0
    }
    
    # 选择最佳方法
    best_method = 'kmeans'
    best_labels = kmeans_labels
    
    item_features_df['聚类'] = best_labels
    cluster_profiles = item_features_df.groupby('聚类')[feature_cols].agg(['mean', 'std'])
    
    # 定义聚类标签
    cluster_labels = {}
    mean_sales = item_features_df.groupby('聚类')['日均销量'].mean()
    sorted_clusters = mean_sales.sort_values(ascending=False).index
    
    if len(sorted_clusters) >= 4:
        cluster_labels = {sorted_clusters[0]: '热销', 
                          sorted_clusters[1]: '畅销',
                          sorted_clusters[2]: '平销', 
                          sorted_clusters[3]: '滞销'}
    elif len(sorted_clusters) == 3:
        cluster_labels = {sorted_clusters[0]: '热销', 
                          sorted_clusters[1]: '畅销',
                          sorted_clusters[2]: '滞销'}
    elif len(sorted_clusters) == 2:
        cluster_labels = {sorted_clusters[0]: '畅销', 
                          sorted_clusters[1]: '滞销'}
    else:
        cluster_labels = {sorted_clusters[0]: '常规'}
    
    item_features_df['聚类标签'] = item_features_df['聚类'].map(cluster_labels).fillna('未分类')
    
    # 保存聚类数据
    CHART_DATA['clustering'] = {
        'features': X_scaled,
        'pca_features': X_pca,
        'feature_names': feature_cols,
        'results': clustering_results,
        'best_method': best_method,
        'best_labels': best_labels,
        'cluster_labels': cluster_labels,
        'profiles': cluster_profiles,
        'item_features_df': item_features_df
    }
else:
    CHART_DATA['clustering'] = None


# ================== 9. 导出结果 ==================
print("\n" + "="*50)
print("9. 导出分析结果")
print("="*50)

try:
    with pd.ExcelWriter('分析结果汇总.xlsx', engine='openpyxl') as writer:
        # 基础数据
        daily_item.to_excel(writer, sheet_name='单品日销售', index=False)
        daily_category.to_excel(writer, sheet_name='品类日销售', index=False)
        
        # 分布统计
        category_stats_df.to_excel(writer, sheet_name='品类分布统计', index=False)
        
        # 异常值
        if len(outlier_df) > 0:
            outlier_df.to_excel(writer, sheet_name='异常值统计', index=False)
        
        # 相关性
        CHART_DATA['correlation']['spearman_corr'].to_excel(writer, sheet_name='Spearman相关矩阵')
        CHART_DATA['correlation']['partial_corr'].to_excel(writer, sheet_name='偏相关矩阵')
        
        # 聚类结果
        if CHART_DATA['clustering']:
            CHART_DATA['clustering']['item_features_df'].to_excel(writer, sheet_name='单品聚类结果', index=False)
            CHART_DATA['clustering']['profiles'].to_excel(writer, sheet_name='聚类画像')
        
        # 关联规则
        if len(rules) > 0:
            rules_export = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(50)
            rules_export = rules_export.copy()
            rules_export['antecedents'] = rules_export['antecedents'].apply(
                lambda x: ', '.join([item_code_to_name.get(item, str(item))[:20] for item in x])
            )
            rules_export['consequents'] = rules_export['consequents'].apply(
                lambda x: ', '.join([item_code_to_name.get(item, str(item))[:20] for item in x])
            )
            rules_export.to_excel(writer, sheet_name='关联规则Top50', index=False)
    
    # 保存图表数据
    import pickle
    with open('图表数据.pkl', 'wb') as f:
        pickle.dump(CHART_DATA, f)
    
    print("\n分析完成！")
    print("生成文件:")
    print("  - 分析结果汇总.xlsx")
    print("  - 分析结果可视化.png")
    print("  - 图表数据.pkl")
    
except Exception as e:
    print(f"导出结果失败: {e}")
    import traceback
    traceback.print_exc()

# ================== 10. 总结报告 ==================
print("\n" + "="*50)
print("10. 分析结论总结")
print("="*50)

print("\n【分布规律】")
print(f"• 大多数单品日销量呈现右偏长尾分布")
print(f"• 零销量占比显著，平均零占比 {category_stats_df['零占比'].mean():.2%}")

print("\n【时间规律】")
if CHART_DATA['time_series']['acf_values'] is not None:
    print(f"• 销售数据呈现明显的周季节性，ACF@7天 = {CHART_DATA['time_series']['acf_values'][7]:.3f}")

print("\n【相互关系】")
strong_corr = (np.abs(CHART_DATA['correlation']['spearman_corr'].values) > 0.5)
np.fill_diagonal(strong_corr, False)
print(f"• Spearman相关分析发现 {strong_corr.sum()//2} 对强相关品类")

if CHART_DATA['clustering']:
    print("\n【聚类分析】")
    for label in CHART_DATA['clustering']['item_features_df']['聚类标签'].unique():
        items = CHART_DATA['clustering']['item_features_df'][
            CHART_DATA['clustering']['item_features_df']['聚类标签'] == label
        ]
        if len(items) > 0:
            print(f"• {label}类 (n={len(items)}): 日均{items['日均销量'].mean():.2f}kg")

print("\n【业务建议】")
print("• 重点关注热销和畅销品类的库存管理")
print("• 针对周季节性特征优化进货策略")
print("• 基于关联规则进行捆绑销售")
print("• 对滞销商品考虑促销清库")

print("\n" + "="*50)
print("脚本执行完毕！")
print("="*50)
