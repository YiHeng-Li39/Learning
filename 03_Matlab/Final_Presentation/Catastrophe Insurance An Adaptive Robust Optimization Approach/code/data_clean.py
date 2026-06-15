import pandas as pd
import numpy as np

def load_and_preprocess_data():
    # 1. 定义本地文件路径
    claims_path = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\data\FimaNfipClaimsV2.csv"
    policies_path = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\data\FimaNfipPoliciesV2.csv"

    print("正在加载数据集，这可能需要一些时间...")
    
    # 2. 加载与处理索赔数据 (Claims)
    # 提取所需列: dateOfLoss, state, amountPaidOnBuildingClaim
    usecols_claims = ['dateOfLoss', 'state', 'amountPaidOnBuildingClaim']
    df_claims = pd.read_csv(claims_path, usecols=usecols_claims)
    
    # 剔除缺乏索赔金额的数据
    df_claims = df_claims.dropna(subset=['amountPaidOnBuildingClaim'])
    
    # 剔除指定的 4 个非主权/数据不足地区
    excluded_states = ['MP', 'AS', 'GU', 'DC']
    df_claims = df_claims[~df_claims['state'].isin(excluded_states)]
    
    # 提取年份并过滤时间范围 (1975 - 2022)
    df_claims['year'] = pd.to_datetime(df_claims['dateOfLoss'], errors='coerce').dt.year
    df_claims = df_claims[(df_claims['year'] >= 1975) & (df_claims['year'] <= 2022)]
    
    # 聚合：按“州”和“年份”计算年度总损失
    annual_claims = df_claims.groupby(['state', 'year'])['amountPaidOnBuildingClaim'].sum().reset_index()
    annual_claims.rename(columns={'amountPaidOnBuildingClaim': 'annual_loss'}, inplace=True)
    
    # 3. 加载与处理保单保费数据 (Policies)
    # 提取所需列: propertyState, policyTerminationDate, totalInsurancePremiumOfThePolicy
    usecols_policies = ['propertyState', 'policyTerminationDate', 'totalInsurancePremiumOfThePolicy']
    df_policies = pd.read_csv(policies_path, usecols=usecols_policies)
    
    # 剔除指定地区
    df_policies = df_policies[~df_policies['propertyState'].isin(excluded_states)]
    
    # 提取年份并过滤时间范围 (2009 - 2022，注意论文中提及保单数据涵盖2009-2022)
    df_policies['year'] = pd.to_datetime(df_policies['policyTerminationDate'], errors='coerce').dt.year
    df_policies = df_policies[(df_policies['year'] >= 2009) & (df_policies['year'] <= 2022)]
    
    # 聚合：按“州”和“年份”计算年度总保费和保单数量
    annual_policies = df_policies.groupby(['propertyState', 'year']).agg(
        total_premium=('totalInsurancePremiumOfThePolicy', 'sum'),
        policy_count=('totalInsurancePremiumOfThePolicy', 'count') # 用于后续推导平均保费和需求阻尼
    ).reset_index()
    annual_policies.rename(columns={'propertyState': 'state'}, inplace=True)

    # 4. 构建面板数据并进行线性插值 (Linear Interpolation)
    # 获取清洗后剩余的 52 个辖区
    valid_states = annual_claims['state'].unique()
    all_years = list(range(1975, 2023))
    
    # 创建完整的 State-Year 面板框架
    multi_index = pd.MultiIndex.from_product([valid_states, all_years], names=['state', 'year'])
    df_panel = pd.DataFrame(index=multi_index).reset_index()
    
    # 合并索赔数据
    df_merged = pd.merge(df_panel, annual_claims, on=['state', 'year'], how='left')
    
    # 合并保单数据
    df_merged = pd.merge(df_merged, annual_policies, on=['state', 'year'], how='left')
    
    # 分州进行缺失值线性插值
    print("正在执行状态级别的缺失值线性插值...")
    df_merged['annual_loss'] = df_merged.groupby('state')['annual_loss'].apply(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    ).reset_index(level=0, drop=True)
    
    # 对于插值后仍然缺失的早期年份损失，填充为 0
    df_merged['annual_loss'] = df_merged['annual_loss'].fillna(0)
    
    # 5. 划分数据集
    df_train = df_merged[df_merged['year'] <= 2012].copy()
    df_test = df_merged[df_merged['year'] >= 2013].copy()
    
    print(f"数据清洗完毕！")
    print(f"包含的辖区数量: {len(valid_states)} (预期: 52)")
    print(f"训练集规模 (1975-2012): {df_train.shape[0]} 条")
    print(f"测试集规模 (2013-2022): {df_test.shape[0]} 条")
    
    return df_merged, df_train, df_test

# 运行预处理
if __name__ == "__main__":
    df_master, df_train, df_test = load_and_preprocess_data()
    # 此时您可以将 df_master 保存为中间 csv，方便后续调试
    df_master.to_csv("processed_master_data.csv", index=False)