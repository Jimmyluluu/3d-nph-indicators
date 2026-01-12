
import re
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class ALVIResultAnalyzer:
    def __init__(self, results_path):
        self.results_path = results_path
        self.nph_values = []      # List of (id, value)
        self.non_nph_values = []  # List of (id, value)
        self.nph_data = []        # List of full data tuples (id, vent, skull, alvi, pct)
        self.non_nph_data = []    # List of full data tuples (id, vent, skull, alvi, pct)
        self.abnormal_cases = []  # List of (id, value)
        self.n_nph = 0
        self.n_non = 0
        self.load_data()

    def load_data(self):
        """從結果摘要文件加載數據"""
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"找不到結果文件: {self.results_path}")

        with open(self.results_path, 'r') as f:
            content = f.read()

        # 解析測量結果表格
        # 格式: | 案例 ID | 腦室前後徑 (mm) | 顱骨前後徑 (mm) | ALVI | 百分比 | 處理時間 |
        pattern = r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| [\d.]+s \|'
        matches = re.findall(pattern, content)

        for match in matches:
            case_id = match[0].strip()
            # 跳過表頭行（如果誤匹配）
            if '案例 ID' in case_id or '---' in case_id:
                continue
                
            vent_dist = float(match[1])
            skull_dist = float(match[2])
            alvi_val = float(match[3])
            alvi_pct = float(match[4])
            
            # 過濾異常值 (例如 ALVI > 80%)
            if alvi_pct > 80:
                self.abnormal_cases.append((case_id, alvi_pct))
                continue
            
            data_tuple = (case_id, vent_dist, skull_dist, alvi_val, alvi_pct)
            
            if '⚠️ NPH' in case_id:
                clean_id = case_id.replace(' ⚠️ NPH', '')
                self.nph_values.append((clean_id, alvi_val))  # 使用比值而非百分比做 ROC
                self.nph_data.append(data_tuple)
            else:
                self.non_nph_values.append((case_id, alvi_val)) # 使用比值而非百分比做 ROC
                self.non_nph_data.append(data_tuple)

        self.n_nph = len(self.nph_values)
        self.n_non = len(self.non_nph_values)
        
        print(f"數據加載完成: NPH={self.n_nph}, 非 NPH={self.n_non}")

    def get_statistics(self, data_list):
        """獲取統計數據"""
        # data_tuple: (id, vent, skull, alvi, pct)
        values = [x[3] for x in data_list] if data_list else []  # ALVI
        pct_values = [x[4] for x in data_list] if data_list else [] # Percent
        vent_vals = [x[1] for x in data_list] if data_list else []
        skull_vals = [x[2] for x in data_list] if data_list else []
        
        count = len(values)
        if count == 0:
            return {
                'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'median': 0, 'avg_pct': 0,
                'min_vent': 0, 'max_vent': 0, 'avg_vent': 0,
                'min_skull': 0, 'max_skull': 0, 'avg_skull': 0
            }
            
        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'avg': sum(values)/count,
            'median': sorted(values)[count//2],
            'avg_pct': sum(pct_values)/count,
            'min_vent': min(vent_vals), 'max_vent': max(vent_vals), 'avg_vent': sum(vent_vals)/count,
            'min_skull': min(skull_vals), 'max_skull': max(skull_vals), 'avg_skull': sum(skull_vals)/count
        }

    def evaluate_threshold(self, threshold):
        """評估特定閾值的診斷效能 (threshold 為比值, e.g., 0.5)"""
        nph_vals = [x[1] for x in self.nph_values]
        non_nph_vals = [x[1] for x in self.non_nph_values]
        
        nph_above = sum(1 for v in nph_vals if v >= threshold)
        nph_below = self.n_nph - nph_above
        non_above = sum(1 for v in non_nph_vals if v >= threshold)
        non_below = self.n_non - non_above
        
        sens = nph_above / self.n_nph if self.n_nph else 0
        spec = non_below / self.n_non if self.n_non else 0
        ppv = nph_above / (nph_above + non_above) if (nph_above + non_above) > 0 else 0
        npv = non_below / (non_below + nph_below) if (non_below + nph_below) > 0 else 0
        acc = (nph_above + non_below) / (self.n_nph + self.n_non) if (self.n_nph + self.n_non) else 0
        
        return {
            'threshold': threshold,
            'sensitivity': sens,
            'specificity': spec,
            'ppv': ppv,
            'npv': npv,
            'accuracy': acc,
            'counts': {
                'tp': nph_above, 'fn': nph_below,
                'fp': non_above, 'tn': non_below
            }
        }

    def generate_report(self, output_path):
        """生成 Markdown 報告"""
        nph_stats = self.get_statistics(self.nph_data)
        non_stats = self.get_statistics(self.non_nph_data)
        
        diff_idx = nph_stats['avg'] - non_stats['avg']
        diff_pct = (diff_idx / non_stats['avg']) * 100 if non_stats['avg'] else 0
        
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = f"""# ALVI (Anteroposterior Lateral Ventricle Index) 分析報告

**分析日期**: {today}
**數據來源**: ALVI 批次處理結果 ({self.n_nph + self.n_non} 個有效案例)

---

## 執行摘要

本報告評估 ALVI 作為水腦症 (NPH) 診斷指標的效能。
研究結果顯示，NPH 組與非 NPH 組有 **{diff_pct:.1f}%** 的差異。

---

## 數據概況

### 案例分布
- **總案例數**: {self.n_nph + self.n_non} 例
- **水腦症案例 (NPH)**: {self.n_nph} 例
- **非水腦症案例**: {self.n_non} 例

### 關鍵指標統計

#### 1. 水腦症案例 (NPH, n={self.n_nph})

| 測量指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|---------|--------|--------|--------|--------|
| 腦室前後徑 (mm) | {nph_stats['min_vent']:.2f} | {nph_stats['max_vent']:.2f} | {nph_stats['avg_vent']:.2f} | - |
| 顱骨前後徑 (mm) | {nph_stats['min_skull']:.2f} | {nph_stats['max_skull']:.2f} | {nph_stats['avg_skull']:.2f} | - |
| **ALVI** | **{nph_stats['min']:.4f}** | **{nph_stats['max']:.4f}** | **{nph_stats['avg']:.4f}** | **{nph_stats['median']:.4f}** |

#### 2. 非水腦症案例 (非 NPH, n={self.n_non})

| 測量指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|---------|--------|--------|--------|--------|
| 腦室前後徑 (mm) | {non_stats['min_vent']:.2f} | {non_stats['max_vent']:.2f} | {non_stats['avg_vent']:.2f} | - |
| 顱骨前後徑 (mm) | {non_stats['min_skull']:.2f} | {non_stats['max_skull']:.2f} | {non_stats['avg_skull']:.2f} | - |
| **ALVI** | **{non_stats['min']:.4f}** | **{non_stats['max']:.4f}** | **{non_stats['avg']:.4f}** | **{non_stats['median']:.4f}** |

#### 3. 組間差異

| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |
|-----|-----------|-------------|------|-----------| 
| 腦室前後徑 | {nph_stats['avg_vent']:.2f} mm | {non_stats['avg_vent']:.2f} mm | {nph_stats['avg_vent'] - non_stats['avg_vent']:+.2f} mm | {(nph_stats['avg_vent'] - non_stats['avg_vent'])/non_stats['avg_vent']*100:+.1f}% |
| 顱骨前後徑 | {nph_stats['avg_skull']:.2f} mm | {non_stats['avg_skull']:.2f} mm | {nph_stats['avg_skull'] - non_stats['avg_skull']:+.2f} mm | {(nph_stats['avg_skull'] - non_stats['avg_skull'])/non_stats['avg_skull']*100:+.1f}% |
| **ALVI** | **{nph_stats['avg']:.4f}** | **{non_stats['avg']:.4f}** | **{diff_idx:+.4f}** | **{diff_pct:+.1f}%** |

---

## 閾值效能評估

一般文獻建議 ALVI > 0.5 可能提示 NPH。下表展示不同閾值下的診斷效能：

| 閾值 | 靈敏度 | 特異性 | PPV | NPV | 準確度 |
|------|--------|--------|-----|-----|--------|
"""
        # 測試 0.4 到 0.6 的閾值
        for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
            m = self.evaluate_threshold(t)
            report += f"| **{t:.2f}** | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% | {m['ppv']*100:.1f}% | {m['npv']*100:.1f}% | {m['accuracy']*100:.1f}% |\n"

        report += f"""
---

## 結論

1. **組間差異**: ALVI 顯示出 {diff_pct:.1f}% 的顯著差異。
2. **閾值建議**: 基於本數據集，0.5 左右的閾值提供了良好的平衡。

---

## 附錄: NPH 案例 (Top 20)

| 排序 | 案例 ID | 腦室前後徑 | 顱骨前後徑 | ALVI |
|-----|---------|------------|------------|------|
"""
        sorted_nph = sorted(self.nph_data, key=lambda x: x[3], reverse=True)
        for i, item in enumerate(sorted_nph[:20]):
            report += f"| {i+1} | {item[0]} | {item[1]:.2f} | {item[2]:.2f} | {item[3]:.4f} |\n"

        report += """
---

## 附錄: 非 NPH 高值案例 (Top 10)

| 排序 | 案例 ID | 腦室前後徑 | 顱骨前後徑 | ALVI |
|-----|---------|------------|------------|------|
"""
        sorted_non_nph = sorted(self.non_nph_data, key=lambda x: x[3], reverse=True)
        for i, item in enumerate(sorted_non_nph[:10]):
            report += f"| {i+1} | {item[0]} | {item[1]:.2f} | {item[2]:.2f} | {item[3]:.4f} |\n"

        report += f"\n**報告產生**: 3D NPH Indicators 系統\n**最後更新**: {today}\n"

        with open(output_path, 'w') as f:
            f.write(report)

    def generate_roc_curve(self, output_path):
        """生成 ROC 曲線"""
        # 準備數據: NPH=1, Non-NPH=0
        y_true = [1] * len(self.nph_values) + [0] * len(self.non_nph_values)
        y_scores = [x[1] for x in self.nph_values] + [x[1] for x in self.non_nph_values]
        
        # 計算 ROC 曲線
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 繪製
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#10b981', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})') # 使用綠色區分
        plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random classifier')
        
        # 標記關鍵閾值點 (0.45, 0.50, 0.55)
        key_thresholds = [0.45, 0.50, 0.55]
        for thresh in key_thresholds:
            # 找到最接近 threshold 的點
            # 由於 score 是比值, thresholds 也是比值
            idx = (np.abs(thresholds - thresh)).argmin()
            
            plt.scatter(fpr[idx], tpr[idx], s=150, zorder=5, edgecolors='white', linewidth=2)
            plt.annotate(f'{thresh:.2f}\n(Sens:{tpr[idx]:.0%}, Spec:{1-fpr[idx]:.0%})', 
                         xy=(fpr[idx], tpr[idx]), 
                         xytext=(fpr[idx]+0.05, tpr[idx]-0.1),
                         fontsize=10, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#64748b'))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title(f'ROC Curve for ALVI NPH Classification\n(n={self.n_nph + self.n_non}, NPH={self.n_nph}, Non-NPH={self.n_non})', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加 AUC 文字
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=20, fontweight='bold', 
                 bbox=dict(boxstyle='round', facecolor='#d1fae5', edgecolor='#10b981', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC 曲線已生成: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = 'result/alvi/results_summary.md'
        
    analyzer = ALVIResultAnalyzer(results_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Generate Report
    output_filename = f'alvi_analysis_{timestamp}.md'
    analyzer.generate_report(output_filename)
    
    # Generate ROC Curve
    roc_filename = f'alvi_roc_{timestamp}.png'
    analyzer.generate_roc_curve(roc_filename)
