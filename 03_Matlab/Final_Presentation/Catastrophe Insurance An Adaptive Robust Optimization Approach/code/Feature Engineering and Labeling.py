import pandas as pd
import numpy as np
import os

# 严格过滤为 52 个辖区
VALID_52_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'PR', 'VI' # Puerto Rico and U.S. Virgin Islands
]

def feature_engineering(df_master):
    print("开始第二阶段：特征工程与目标值构建...")
    
    # 1. 严格过滤为 52 个辖区
    df = df_master[df_master['state'].isin(VALID_52_STATES)].copy()
    df.sort_values(by=['state', 'year'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 2. 构建特征：过去 1-5 年的年度损失 (Lag features)
    print("正在构建时序滞后特征 (Lag 1 to 5)...")
    for lag in range(1, 6):
        df[f'loss_lag_{lag}'] = df.groupby('state')['annual_loss'].shift(lag)
    
    # 用 0 填充由于 shift 产生的早期年份的 NaN
    lag_cols = [f'loss_lag_{i}' for i in range(1, 6)]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    # 3. 计算训练集 (1975-2011) 的阈值界限
    train_data = df[df['year'] <= 2011]
    thresholds = {
        '90th': np.percentile(train_data['annual_loss'], 90),
        '95th': np.percentile(train_data['annual_loss'], 95),
        '99th': np.percentile(train_data['annual_loss'], 99)
    }
    print(f"根据训练集计算的损失阈值 (Thresholds):\n"
          f"90%: ${thresholds['90th']:,.0f}\n"
          f"95%: ${thresholds['95th']:,.0f}\n"
          f"99%: ${thresholds['99th']:,.0f}")
    
    # 4. 构建前瞻性目标值 (Forward-looking targets)
    K_horizons = [3, 5, 10]
    
    print("正在构建未来 K 年的洪灾分类标签 (0/1)...")
    for k in K_horizons:
        for p_name, threshold in thresholds.items():
            col_name = f'target_{k}yr_{p_name}'
            
            # 使用[::-1]配合rolling反向计算未来K年的最大损失，检查是否超阈值
            future_max_loss = df.groupby('state')['annual_loss'].shift(-1).iloc[::-1].groupby(df['state']).rolling(window=k, min_periods=1).max().iloc[::-1].reset_index(level=0, drop=True)
            
            df[col_name] = (future_max_loss >= threshold).astype(int)
            
            # 由于在最后几年缺乏未来数据，我们将未知年份的 target 设为 NaN
            mask_unknown = df['year'] > (2022 - k)
            df.loc[mask_unknown, col_name] = np.nan
            
    print("特征工程完成！")
    return df, thresholds

if __name__ == "__main__":
    # 定义数据的输入输出路径 (请确保与你的本地路径一致)
    input_csv = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\data\processed_master_data.csv"
    output_csv = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\data\ml_features_data.csv"
    
    if not os.path.exists(input_csv):
        print(f"找不到文件: {input_csv}")
        print("请确保在执行第一段清洗代码时，去掉了末尾 df_master.to_csv(...) 的注释并成功运行了它！")
    else:
        print(f"成功读取数据: {input_csv}")
        df_master = pd.read_csv(input_csv)
        
        # 运行特征工程
        df_ml, thresholds_dict = feature_engineering(df_master)
        
        # 检查当前数据结构
        print("\n生成的数据集示例 (最后5行):")
        print(df_ml[['state', 'year', 'annual_loss', 'loss_lag_1', 'target_3yr_90th']].tail())
        
        # 保存带有 ML 特征和标签的数据，供第三阶段使用
        df_ml.to_csv(output_csv, index=False)
        print(f"\n特征工程数据已成功保存至: {output_csv}")