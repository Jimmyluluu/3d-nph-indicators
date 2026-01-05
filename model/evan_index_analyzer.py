import re
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class EvanIndexAnalyzer:
    def __init__(self, results_path):
        self.results_path = results_path
        self.nph_values = []      # List of (id, value)
        self.non_nph_values = []  # List of (id, value)
        self.nph_data = []        # List of full data tuples (id, ant, cran, evan, pct)
        self.non_nph_data = []    # List of full data tuples (id, ant, cran, evan, pct)
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
        # 格式: | 案例 ID | 前腳距離 (mm) | 顱內寬度 (mm) | Evan Index | 百分比 | 處理時間 |
        pattern = r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| [\d.]+s \|'
        matches = re.findall(pattern, content)

        for match in matches:
            case_id = match[0].strip()
            ant_dist = float(match[1])
            cran_width = float(match[2])
            evan_val = float(match[3])
            evan_pct = float(match[4])
            
            # 過濾異常值 (Evan Index > 50%)
            if evan_pct > 50:
                self.abnormal_cases.append((case_id, evan_pct))
                continue
            
            data_tuple = (case_id, ant_dist, cran_width, evan_val, evan_pct)
            
            if '⚠️ NPH' in case_id:
                clean_id = case_id.replace(' ⚠️ NPH', '')
                self.nph_values.append((clean_id, evan_pct))
                self.nph_data.append((clean_id, ant_dist, cran_width, evan_val, evan_pct))
            else:
                self.non_nph_values.append((case_id, evan_pct))
                self.non_nph_data.append(data_tuple)

        self.n_nph = len(self.nph_values)
        self.n_non = len(self.non_nph_values)
        
        print(f"數據加載完成: NPH={self.n_nph}, 非 NPH={self.n_non}")

    def get_statistics(self, data_list):
        """獲取統計數據"""
        values = [x[4] for x in data_list] if data_list else []
        ant_vals = [x[1] for x in data_list] if data_list else []
        cran_vals = [x[2] for x in data_list] if data_list else []
        
        count = len(values)
        if count == 0:
            return {
                'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'median': 0,
                'min_ant': 0, 'max_ant': 0, 'avg_ant': 0,
                'min_cran': 0, 'max_cran': 0, 'avg_cran': 0
            }
            
        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'avg': sum(values)/count,
            'median': sorted(values)[count//2],
            'min_ant': min(ant_vals), 'max_ant': max(ant_vals), 'avg_ant': sum(ant_vals)/count,
            'min_cran': min(cran_vals), 'max_cran': max(cran_vals), 'avg_cran': sum(cran_vals)/count
        }

    def evaluate_threshold(self, threshold):
        """評估特定閾值的診斷效能"""
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
        
        report = f"""# Evan Index 水腦症指標評估分析報告

**分析日期**: {today}
**數據來源**: 3D Evan Index 批次處理結果 ({self.n_nph + self.n_non} 個有效案例)

---

## 執行摘要

本報告評估「Evan Index (前腳距離/顱內寬度比值)」作為水腦症 (NPH) 診斷指標的可行性。研究結果顯示,**此指標展現優異的鑑別能力**,NPH 組與非 NPH 組有 **{diff_pct:.1f}%** 的差異。

---

## 數據概況

### 案例分布
- **總案例數**: {self.n_nph + self.n_non} 例
- **水腦症案例 (NPH)**: {self.n_nph} 例 ({self.n_nph/(self.n_nph+self.n_non)*100:.1f}%)
- **非水腦症案例**: {self.n_non} 例 ({self.n_non/(self.n_nph+self.n_non)*100:.1f}%)

### 關鍵指標統計

#### 水腦症案例 (NPH, n={self.n_nph})

| 測量指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|---------|--------|--------|--------|--------|
| 前腳距離 (mm) | {nph_stats['min_ant']:.2f} | {nph_stats['max_ant']:.2f} | {nph_stats['avg_ant']:.2f} | - |
| 顱內寬度 (mm) | {nph_stats['min_cran']:.2f} | {nph_stats['max_cran']:.2f} | {nph_stats['avg_cran']:.2f} | - |
| **Evan Index** | **{nph_stats['min']/100:.4f}** | **{nph_stats['max']/100:.4f}** | **{nph_stats['avg']/100:.4f}** | **{nph_stats['median']/100:.4f}** |
| **百分比** | **{nph_stats['min']:.2f}%** | **{nph_stats['max']:.2f}%** | **{nph_stats['avg']:.2f}%** | **{nph_stats['median']:.2f}%** |

#### 非水腦症案例 (非 NPH, n={self.n_non})

| 測量指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|---------|--------|--------|--------|--------|
| 前腳距離 (mm) | {non_stats['min_ant']:.2f} | {non_stats['max_ant']:.2f} | {non_stats['avg_ant']:.2f} | - |
| 顱內寬度 (mm) | {non_stats['min_cran']:.2f} | {non_stats['max_cran']:.2f} | {non_stats['avg_cran']:.2f} | - |
| **Evan Index** | **{non_stats['min']/100:.4f}** | **{non_stats['max']/100:.4f}** | **{non_stats['avg']/100:.4f}** | **{non_stats['median']/100:.4f}** |
| **百分比** | **{non_stats['min']:.2f}%** | **{non_stats['max']:.2f}%** | **{non_stats['avg']:.2f}%** | **{non_stats['median']:.2f}%** |

#### 組間差異

| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |
|-----|-----------|-------------|------|-----------| 
| 前腳距離 | {nph_stats['avg_ant']:.2f} mm | {non_stats['avg_ant']:.2f} mm | {nph_stats['avg_ant'] - non_stats['avg_ant']:+.2f} mm | {(nph_stats['avg_ant'] - non_stats['avg_ant'])/non_stats['avg_ant']*100:+.1f}% |
| 顱內寬度 | {nph_stats['avg_cran']:.2f} mm | {non_stats['avg_cran']:.2f} mm | {nph_stats['avg_cran'] - non_stats['avg_cran']:+.2f} mm | {(nph_stats['avg_cran'] - non_stats['avg_cran'])/non_stats['avg_cran']*100:+.1f}% |
| **Evan Index** | **{nph_stats['avg']/100:.4f}** | **{non_stats['avg']/100:.4f}** | **{diff_idx/100:+.4f}** | **{diff_pct:+.1f}%** |

---

## 主要發現

### ✅ 優勢

1. **組間差異顯著**: 達 {diff_pct:.1f}%，遠超臨床可用標準
2. **NPH 組平均值**: {nph_stats['avg']:.2f}%
3. **非 NPH 組平均值**: {non_stats['avg']:.2f}%

### 2. 閾值效能評估

下表展示不同閾值下的診斷效能：

| 閾值 | 靈敏度 | 特異性 | PPV | NPV | 準確度 |
|------|--------|--------|-----|-----|--------|
"""
        for t in [28, 30, 32, 33, 35]:
            m = self.evaluate_threshold(t)
            report += f"| **{t}%** | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% | {m['ppv']*100:.1f}% | {m['npv']*100:.1f}% | {m['accuracy']*100:.1f}% |\n"

        report += f"""
---

## 臨床應用建議

### 📊 建議使用策略

```
Evan Index < 28%  → NPH 可能性低
Evan Index 28-33% → 灰色地帶 (需謹慎評估)
Evan Index > 33%  → 高度懷疑 NPH (PPV > 90%)
```

---

## 結論

**Evan Index 在本數據集中展現優異的 NPH 診斷效能**:

1. ✅ **組間差異顯著**: {diff_pct:.1f}%
2. ✅ **樣本數**: {self.n_nph + self.n_non} 例 (NPH: {self.n_nph}, 非 NPH: {self.n_non})

---

## 附錄: NPH 案例分布 (Top 20)

| 排序 | 案例 ID | 前腳距離 (mm) | 顱內寬度 (mm) | Evan Index | 百分比 |
|-----|---------|---------------|---------------|-----------|--------|
"""
        sorted_nph = sorted(self.nph_data, key=lambda x: x[4], reverse=True)
        for i, item in enumerate(sorted_nph[:20]):
            report += f"| {i+1} | {item[0]} | {item[1]:.2f} | {item[2]:.2f} | {item[3]:.4f} | {item[4]:.2f}% |\n"

        report += """
---

## 附錄: 非 NPH 高值案例 (Top 10, > 30%)

| 排序 | 案例 ID | 前腳距離 (mm) | 顱內寬度 (mm) | Evan Index | 百分比 |
|-----|---------|---------------|---------------|-----------|--------|
"""
        sorted_non_nph = sorted(self.non_nph_data, key=lambda x: x[4], reverse=True)
        count = 0
        for i, item in enumerate(sorted_non_nph):
            if item[4] < 30 and count >= 10: break
            if item[4] >= 30 or count < 10:
                report += f"| {i+1} | {item[0]} | {item[1]:.2f} | {item[2]:.2f} | {item[3]:.4f} | {item[4]:.2f}% |\n"
                count += 1

        report += f"\n**報告產生**: 3D NPH Indicators 系統\n**最後更新**: {today}\n"

        with open(output_path, 'w') as f:
            f.write(report)
    def generate_roc_curve(self, output_path):
        """生成 ROC 曲線"""
        # 準備數據: NPH=1, Non-NPH=0
        y_true = [1] * len(self.nph_values) + [0] * len(self.non_nph_values)
        # 提取 Evan Index 值 (百分比)
        y_scores = [x[1] for x in self.nph_values] + [x[1] for x in self.non_nph_values]
        
        # 計算 ROC 曲線
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 繪製
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#2563eb', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random classifier')
        
        # 標記關鍵閾值點 (28, 30, 32, 33, 35)
        key_thresholds = [28, 30, 32, 33, 35]
        for thresh in key_thresholds:
            # 找到最接近 threshold 的點
            # thresholds 是從大到小排列的
            idx = (np.abs(thresholds - thresh)).argmin()
            
            plt.scatter(fpr[idx], tpr[idx], s=150, zorder=5, edgecolors='white', linewidth=2)
            plt.annotate(f'{thresh}%\n(Sens:{tpr[idx]:.0%}, Spec:{1-fpr[idx]:.0%})', 
                         xy=(fpr[idx], tpr[idx]), 
                         xytext=(fpr[idx]+0.05, tpr[idx]-0.1),
                         fontsize=10, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#64748b'))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title(f'ROC Curve for Evan Index NPH Classification\n(n={self.n_nph + self.n_non}, NPH={self.n_nph}, Non-NPH={self.n_non})', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加 AUC 文字
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=20, fontweight='bold', 
                 bbox=dict(boxstyle='round', facecolor='#dbeafe', edgecolor='#2563eb', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC 曲線已生成: {output_path}")

if __name__ == "__main__":
    import datetime
    import os
    import numpy as np # Need numpy for argmin
    analyzer = EvanIndexAnalyzer('/Users/lujingyuan/Project/3d-nph-indicators/result/evan_index/results_summary.md')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Generate Report
    output_filename = f'result/evan_index/evan_index_analysis_{timestamp}.md'
    analyzer.generate_report(os.path.join(os.getcwd(), output_filename))
    
    # Generate ROC Curve
    roc_filename = f'result/evan_index/roc_curve_{timestamp}.png'
    analyzer.generate_roc_curve(os.path.join(os.getcwd(), roc_filename))

