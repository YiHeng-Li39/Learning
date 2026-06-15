import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

# 忽略警告信息
warnings.filterwarnings("ignore")

def reproduce_table_1():
    print("========= 开始复现论文 Table 1 (机器学习模型性能评估) =========")
    
    # 路径配置
    base_dir = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\data"
    output_dir = r"D:\Github\03_Matlab\Final_Presentation\Catastrophe Insurance An Adaptive Robust Optimization Approach\output"
    os.makedirs(output_dir, exist_ok=True)
    
    input_csv = os.path.join(base_dir, "ml_features_data.csv")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到特征文件: {input_csv}。请确保特征工程阶段已成功完成！")
        
    df = pd.read_csv(input_csv)
    
    # 1. 特征预处理
    print("正在进行特征预处理 (One-hot encoding)...")
    num_features = ['annual_loss'] + [f'loss_lag_{i}' for i in range(1, 6)]
    df_encoded = pd.get_dummies(df, columns=['state'], drop_first=False)
    feature_cols = num_features + [col for col in df_encoded.columns if col.startswith('state_')]
    
    # 2. 划分训练集 (<=2011) 和测试集 (>=2012)
    train_mask = df_encoded['year'] <= 2011
    test_mask = df_encoded['year'] >= 2012
    
    scaler = StandardScaler()
    df_encoded.loc[train_mask, num_features] = scaler.fit_transform(df_encoded.loc[train_mask, num_features])
    df_encoded.loc[test_mask, num_features] = scaler.transform(df_encoded.loc[test_mask, num_features])
    
    # 3. 超参数网格 (对应论文 Table 6)
    xgb_param_grid = {'n_estimators': [100, 150], 'max_depth': [4, 6], 'learning_rate': [0.1, 0.3]}
    lr_param_grid = {'C': [0.001, 0.2, 0.4, 0.6, 0.8, 1.0], 'penalty': ['l1', 'l2']}
    
    horizons = [3, 5, 10]
    thresholds = ['90th', '95th', '99th']
    metrics = ['auc', 'f1', 'accu', 'accu_bl', 'precision', 'recall']
    
    # 初始化一个字典来存储所有的结果
    results_dict = {th: {m: {k: {'logreg': None, 'xgb': None} for k in horizons} for m in metrics} for th in thresholds}
    
    # 4. 开始循环训练和测试
    for th in thresholds:
        for k in horizons:
            target = f'target_{k}yr_{th}'
            print(f"\n---> 正在训练: 阈值 {th}, 时间窗口 {k} 年 ({target})")
            
            # 剔除无效标签
            valid_train = df_encoded[train_mask].dropna(subset=[target])
            valid_test = df_encoded[test_mask].dropna(subset=[target])
            
            X_train, y_train = valid_train[feature_cols], valid_train[target]
            X_test, y_test = valid_test[feature_cols], valid_test[target]
            
            # 样本极度不平衡处理
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                print(f"⚠️ 样本仅包含单一种类，无法计算准确指标，跳过。")
                continue
                
            # -- 逻辑回归 (Logistic Regression) --
            lr_grid = GridSearchCV(LogisticRegression(solver='liblinear', random_state=42, max_iter=500), 
                                   lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            lr_grid.fit(X_train, y_train)
            lr_best = lr_grid.best_estimator_
            lr_preds = lr_best.predict(X_test)
            lr_probs = lr_best.predict_proba(X_test)[:, 1]
            
            # -- XGBoost --
            xgb_grid = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), 
                                    xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            xgb_grid.fit(X_train, y_train)
            xgb_best = xgb_grid.best_estimator_
            xgb_preds = xgb_best.predict(X_test)
            xgb_probs = xgb_best.predict_proba(X_test)[:, 1]
            
            # -- 记录指标 --
            # LR Metrics
            results_dict[th]['auc'][k]['logreg'] = roc_auc_score(y_test, lr_probs)
            results_dict[th]['f1'][k]['logreg'] = f1_score(y_test, lr_preds, zero_division=0)
            results_dict[th]['accu'][k]['logreg'] = accuracy_score(y_test, lr_preds)
            results_dict[th]['accu_bl'][k]['logreg'] = balanced_accuracy_score(y_test, lr_preds)
            results_dict[th]['precision'][k]['logreg'] = precision_score(y_test, lr_preds, zero_division=0)
            results_dict[th]['recall'][k]['logreg'] = recall_score(y_test, lr_preds, zero_division=0)
            
            # XGB Metrics
            results_dict[th]['auc'][k]['xgb'] = roc_auc_score(y_test, xgb_probs)
            results_dict[th]['f1'][k]['xgb'] = f1_score(y_test, xgb_preds, zero_division=0)
            results_dict[th]['accu'][k]['xgb'] = accuracy_score(y_test, xgb_preds)
            results_dict[th]['accu_bl'][k]['xgb'] = balanced_accuracy_score(y_test, xgb_preds)
            results_dict[th]['precision'][k]['xgb'] = precision_score(y_test, xgb_preds, zero_division=0)
            results_dict[th]['recall'][k]['xgb'] = recall_score(y_test, xgb_preds, zero_division=0)

    # 5. 构建 Table 1 的格式化 DataFrame
    print("\n========= 正在排版生成 Table 1 =========")
    table_rows = []
    for th in thresholds:
        table_rows.append([f"{th[:2]}% Threshold", "", "", "", "", "", ""])
        for m in metrics:
            row = [m]
            for k in horizons:
                val_lr = results_dict[th][m][k]['logreg']
                val_xgb = results_dict[th][m][k]['xgb']
                
                str_lr = f"{val_lr:.3f}" if val_lr is not None else "N/A"
                str_xgb = f"{val_xgb:.3f}" if val_xgb is not None else "N/A"
                
                row.extend([str_lr, str_xgb])
            table_rows.append(row)
            
    # 设置与原论文完全一致的 MultiIndex 列头
    columns = pd.MultiIndex.from_tuples([
        ("Scores", ""),
        ("3 Years", "logreg"), ("3 Years", "xgb"),
        ("5 Years", "logreg"), ("5 Years", "xgb"),
        ("10 Years", "logreg"), ("10 Years", "xgb")
    ])
    
    df_table1 = pd.DataFrame(table_rows, columns=columns)
    
    # 打印到控制台
    print("\n")
    print(df_table1.to_string(index=False))
    
    # 导出到 output 文件夹
    output_path = os.path.join(output_dir, "Table_1_ML_Metrics.csv")
    df_table1.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Table 1 已成功生成并保存至: {output_path}")

if __name__ == "__main__":
    reproduce_table_1()