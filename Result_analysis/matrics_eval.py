import pandas as pd
import numpy as np

# 讀取 CSV 檔案（請替換為你的實際檔案路徑）
file_path = 'LMaUnet_Large.csv'  # 更新為 predictions.csv
# 定義 probability 閾值（與範例一致）
threshold = 0.40
csv_name = file_path.split('.')[0]  # 提取檔案名（不含副檔名），例如 'predictions'

try:
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
except Exception as e:
    print(f"讀取檔案時出錯: {e}")
    print("請檢查檔案路徑、格式或編碼（嘗試 'latin1' 或 'utf-8-sig'）")
    exit()

# 檢查並列印實際欄位名稱
print("CSV 檔案的欄位名稱:")
print(df.columns.tolist())

# 確認 'nodule_type' 是否存在
if 'nodule_type' not in df.columns:
    print("\n錯誤：'nodule_type' 欄位不存在！")
    print("可能的欄位名稱（請檢查是否有拼寫錯誤或多餘空格）：")
    for col in df.columns:
        if 'nodule' in col.lower():
            print(f"- {col}")
    print("請確認 CSV 檔案中的欄位名稱，並修改程式中的 'nodule_type' 為正確名稱")
    exit()



# 清理數據：將 probability < threshold 替換為 -1，然後將 -1 替換為 NaN
df.loc[df['probability'] < threshold, 'probability'] = -1
df['probability'] = df['probability'].replace(-1, np.nan)

# 定義 TP、FN、FP
def classify_detections(df, threshold):
    # TP: is_gt = True 且 probability >= threshold
    tp = df[(df['is_gt'] == True) & (df['probability'] >= threshold)]
    
    # FN: is_gt = True 且 probability 是 NaN（即原來的 -1 或新替換的 -1）
    fn = df[(df['is_gt'] == True) & (df['probability'].isna())]
    
    # FP: is_gt = False 且 probability >= threshold
    fp = df[(df['is_gt'] == False) & (df['probability'] >= threshold)]
    
    return tp, fn, fp

# 計算 recall、precision 和 F1 score
def calculate_metrics(tp_count, fn_count, fp_count):
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'TP': tp_count,
        'FN': fn_count,
        'FP': fp_count,
        'Recall': recall,
        'Precision': precision,
        'F1': f1
    }

# 按 nodule_type 分組並計算指標
nodule_types = df['nodule_type'].unique()
results = {}

for nodule_type in nodule_types:
    df_type = df[df['nodule_type'] == nodule_type]
    tp, fn, fp = classify_detections(df_type, threshold)
    metrics = calculate_metrics(len(tp), len(fn), len(fp))
    results[nodule_type] = metrics

# 計算 suspicious_combined（probably_suspicious + suspicious）
suspicious_types = ['probably_suspicious', 'suspicious']
df_suspicious_combined = df[df['nodule_type'].isin(suspicious_types)]
tp_susp, fn_susp, fp_susp = classify_detections(df_suspicious_combined, threshold)
results['suspicious_combined'] = calculate_metrics(len(tp_susp), len(fn_susp), len(fp_susp))

# 計算總和（all）
all_tp = sum([results[n]['TP'] for n in nodule_types])
all_fn = sum([results[n]['FN'] for n in nodule_types])
all_fp = sum([results[n]['FP'] for n in nodule_types])
all_metrics = calculate_metrics(all_tp, all_fn, all_fp)

# 計算 series-based recall（假設 seriesuid 為唯一病例標識）
series_gt = df[df['is_gt'] == True]['seriesuid'].unique()
series_detected = df[(df['is_gt'] == False) & (df['probability'] >= threshold)]['seriesuid'].unique()
series_tp = len(np.intersect1d(series_gt, series_detected))
series_fn = len(series_gt) - series_tp
series_recall = series_tp / (series_tp + series_fn) if (series_tp + series_fn) > 0 else 0

# 同時輸出到終端機和續寫 {csvname_log}.txt
output_file = f"{csv_name}_log.txt"
with open(output_file, 'a', encoding='utf-8') as f:
    # 輸出到終端機和檔案
    output_content = [f"Prob threshold: {threshold} eval.py:234"]
    max_type_length = max(len(str(n)) for n in list(nodule_types) + ['suspicious_combined', 'all', 'Recall(series_based)']) + 5  # 確保有足夠的空間
    
    for nodule_type in sorted(nodule_types):
        metrics = results[nodule_type]
        line = f"{nodule_type:<{max_type_length}}: Recall={metrics['Recall']:.3f}, Precision={metrics['Precision']:.3f}, F1={metrics['F1']:.3f}, TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}"
        output_content.append(line)
    
    line = f"{'suspicious_combined':<{max_type_length}}: Recall={results['suspicious_combined']['Recall']:.3f}, Precision={results['suspicious_combined']['Precision']:.3f}, F1={results['suspicious_combined']['F1']:.3f}, TP={results['suspicious_combined']['TP']}, FP={results['suspicious_combined']['FP']}, FN={results['suspicious_combined']['FN']}"
    output_content.append(line)
    
    line = f"{'all':<{max_type_length}}: Recall={all_metrics['Recall']:.3f}, Precision={all_metrics['Precision']:.3f}, F1={all_metrics['F1']:.3f}, TP={all_metrics['TP']}, FP={all_metrics['FP']}, FN={all_metrics['FN']}"
    output_content.append(line)
    
    line = f"{'Recall(series_based)':<{max_type_length}}: {series_recall:.3f} eval.py:286"
    output_content.append(line)
    
    # 同時輸出到終端機和檔案
    for line in output_content:
        print(line)
        f.write(line + '\n')

print(f"\nResults saved to '{output_file}'")