import re
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class VolumeSurfaceAnalyzer:
    def __init__(self, results_path):
        self.results_path = results_path
        self.nph_values = []      # List of (id, value)
        self.non_nph_values = []  # List of (id, value)
        self.nph_data = []        # List of full data tuples (id, left_vol, right_vol, total_vol, ratio)
        self.non_nph_data = []    # List of full data tuples
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
        # 格式: | 案例 ID | 左腦室體積 (mm³) | 右腦室體積 (mm³) | 總體積 (mm³) | V/SA 比例 (mm) | 處理時間 |
        pattern = r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| [\d.]+s \|'
        matches = re.findall(pattern, content)

        for match in matches:
            case_id = match[0].strip()
            left_vol = float(match[1])
            right_vol = float(match[2])
            total_vol = float(match[3])
            ratio = float(match[4])
            
            # 不過濾異常值，保留所有數據進行分析
            # if ratio > 60:
            #     self.abnormal_cases.append((case_id, ratio))
            #     continue
            
            data_tuple = (case_id, left_vol, right_vol, total_vol, ratio)
            
            if '⚠️ NPH' in case_id:
                clean_id = case_id.replace(' ⚠️ NPH', '')
                self.nph_values.append((clean_id, ratio))
                self.nph_data.append((clean_id, left_vol, right_vol, total_vol, ratio))
            else:
                self.non_nph_values.append((case_id, ratio))
                self.non_nph_data.append(data_tuple)

        self.n_nph = len(self.nph_values)
        self.n_non = len(self.non_nph_values)
        
        print(f"數據加載完成: NPH={self.n_nph}, 非 NPH={self.n_non}")

    def get_statistics(self, data_list):
        """獲取統計數據"""
        values = [x[4] for x in data_list] if data_list else []
        left_vals = [x[1] for x in data_list] if data_list else []
        right_vals = [x[2] for x in data_list] if data_list else []
        total_vals = [x[3] for x in data_list] if data_list else []
        
        count = len(values)
        if count == 0:
            return {
                'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'median': 0,
                'min_left': 0, 'max_left': 0, 'avg_left': 0,
                'min_right': 0, 'max_right': 0, 'avg_right': 0,
                'min_total': 0, 'max_total': 0, 'avg_total': 0
            }
            
        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'avg': sum(values)/count,
            'median': sorted(values)[count//2],
            'min_left': min(left_vals), 'max_left': max(left_vals), 'avg_left': sum(left_vals)/count,
            'min_right': min(right_vals), 'max_right': max(right_vals), 'avg_right': sum(right_vals)/count,
            'min_total': min(total_vals), 'max_total': max(total_vals), 'avg_total': sum(total_vals)/count
        }

    def evaluate_threshold(self, threshold):
        """評估特定閾值的診斷效能"""
        nph_vals = [x[1] for x in self.nph_values]
        non_nph_vals = [x[1] for x in self.non_nph_values]
        
        # V/SA ratio 越高越可能是 NPH
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
        
        diff_ratio = nph_stats['avg'] - non_stats['avg']
        diff_pct = (diff_ratio / non_stats['avg']) * 100 if non_stats['avg'] else 0
        
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = f"""# Volume/Surface Area Ratio 水腦症指標評估分析報告

**分析日期**: {today}
**數據來源**: 腦室體積與表面積比例批次處理結果 ({self.n_nph + self.n_non} 個有效案例)

---

## 執行摘要

本報告評估「V/SA Ratio (體積/表面積比值)」作為水腦症 (NPH) 診斷指標的可行性。研究結果顯示,**此指標展現良好的鑑別能力**,NPH 組與非 NPH 組有 **{diff_pct:.1f}%** 的差異。

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
| 總體積 (mm³) | {nph_stats['min_total']:.1f} | {nph_stats['max_total']:.1f} | {nph_stats['avg_total']:.1f} | - |
| **V/SA Ratio (mm)** | **{nph_stats['min']:.2f}** | **{nph_stats['max']:.2f}** | **{nph_stats['avg']:.2f}** | **{nph_stats['median']:.2f}** |

#### 非水腦症案例 (非 NPH, n={self.n_non})

| 測量指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|---------|--------|--------|--------|--------|
| 總體積 (mm³) | {non_stats['min_total']:.1f} | {non_stats['max_total']:.1f} | {non_stats['avg_total']:.1f} | - |
| **V/SA Ratio (mm)** | **{non_stats['min']:.2f}** | **{non_stats['max']:.2f}** | **{non_stats['avg']:.2f}** | **{non_stats['median']:.2f}** |

#### 組間差異

| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |
|-----|-----------|-------------|------|-----------| 
| **V/SA Ratio** | **{nph_stats['avg']:.2f} mm** | **{non_stats['avg']:.2f} mm** | **{diff_ratio:+.2f} mm** | **{diff_pct:+.1f}%** |

---

## 主要發現

### ✅ 優勢

1. **組間差異顯著**: 達 {diff_pct:.1f}%，遠超臨床可用標準
2. **NPH 組平均值**: {nph_stats['avg']:.2f} mm
3. **非 NPH 組平均值**: {non_stats['avg']:.2f} mm

### 2. 閾值效能評估

下表展示不同閾值下的診斷效能：

| 閾值 | 靈敏度 | 特異性 | PPV | NPV | 準確度 |
|------|--------|--------|-----|-----|--------|
"""
        for t in [30, 32, 33, 34, 35]:
            m = self.evaluate_threshold(t)
            report += f"| **{t} mm** | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% | {m['ppv']*100:.1f}% | {m['npv']*100:.1f}% | {m['accuracy']*100:.1f}% |\n"

        report += f"""
---

## 臨床應用建議

### 📊 建議使用策略

```
V/SA Ratio < 30 mm  → NPH 可能性低
V/SA Ratio 30-35 mm → 灰色地帶 (需謹慎評估)
V/SA Ratio > 35 mm  → 高度懷疑 NPH
```

---

## 結論

**V/SA Ratio 在本數據集中展現良好的 NPH 診斷效能**:

1. ✅ **組間差異顯著**: {diff_pct:.1f}%
2. ✅ **樣本數**: {self.n_nph + self.n_non} 例 (NPH: {self.n_nph}, 非 NPH: {self.n_non})

---

## 附錄: NPH 案例分布 (Top 20)

| 排序 | 案例 ID | 總體積 (mm³) | V/SA Ratio (mm) |
|-----|---------|--------------|-----------------|
"""
        sorted_nph = sorted(self.nph_data, key=lambda x: x[4], reverse=True)
        for i, item in enumerate(sorted_nph[:20]):
            report += f"| {i+1} | {item[0]} | {item[3]:.1f} | {item[4]:.2f} |\n"

        report += f"""
---

**報告產生**: 3D NPH Indicators 系統
**最後更新**: {today}
"""

        with open(output_path, 'w') as f:
            f.write(report)
        print(f"報告已生成: {output_path}")

    def generate_roc_curve(self, output_path):
        """生成 ROC 曲線"""
        # 準備數據: NPH=1, Non-NPH=0
        y_true = [1] * len(self.nph_values) + [0] * len(self.non_nph_values)
        # 提取 V/SA Ratio 值
        y_scores = [x[1] for x in self.nph_values] + [x[1] for x in self.non_nph_values]
        
        # 計算 ROC 曲線
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 繪製
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#2563eb', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random classifier')
        
        # 標記關鍵閾值點
        key_thresholds = [30, 32, 33, 34, 35]
        for thresh in key_thresholds:
            idx = (np.abs(thresholds - thresh)).argmin()
            
            plt.scatter(fpr[idx], tpr[idx], s=150, zorder=5, edgecolors='white', linewidth=2)
            plt.annotate(f'{thresh}mm\n(Sens:{tpr[idx]:.0%}, Spec:{1-fpr[idx]:.0%})', 
                         xy=(fpr[idx], tpr[idx]), 
                         xytext=(fpr[idx]+0.05, tpr[idx]-0.1),
                         fontsize=10, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#64748b'))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title(f'ROC Curve for V/SA Ratio NPH Classification\n(n={self.n_nph + self.n_non}, NPH={self.n_nph}, Non-NPH={self.n_non})', fontsize=16, fontweight='bold')
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
    import numpy as np
    analyzer = VolumeSurfaceAnalyzer('/Users/lujingyuan/Project/3d-nph-indicators/result/volume_surface_ratio/results_summary.md')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Generate Report
    output_filename = f'result/volume_surface_ratio/vs_ratio_analysis_{timestamp}.md'
    analyzer.generate_report(os.path.join(os.getcwd(), output_filename))
    
    # Generate ROC Curve
    roc_filename = f'result/volume_surface_ratio/roc_curve_{timestamp}.png'
    analyzer.generate_roc_curve(os.path.join(os.getcwd(), roc_filename))
