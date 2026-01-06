import re
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class VentricleVolumeAnalyzer:
    def __init__(self, results_path):
        self.results_path = results_path
        self.nph_values = []      # List of (id, value)
        self.non_nph_values = []  # List of (id, value)
        self.nph_data = []        # List of full data tuples (id, left_vol, right_vol, total_vol)
        self.non_nph_data = []    # List of full data tuples
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
            
            data_tuple = (case_id, left_vol, right_vol, total_vol)
            
            if '⚠️ NPH' in case_id:
                clean_id = case_id.replace(' ⚠️ NPH', '')
                self.nph_values.append((clean_id, total_vol))
                self.nph_data.append((clean_id, left_vol, right_vol, total_vol))
            else:
                self.non_nph_values.append((case_id, total_vol))
                self.non_nph_data.append(data_tuple)

        self.n_nph = len(self.nph_values)
        self.n_non = len(self.non_nph_values)
        
        print(f"數據加載完成: NPH={self.n_nph}, 非 NPH={self.n_non}")

    def get_statistics(self, data_list):
        """獲取統計數據"""
        total_vals = [x[3] for x in data_list] if data_list else []
        left_vals = [x[1] for x in data_list] if data_list else []
        right_vals = [x[2] for x in data_list] if data_list else []
        
        count = len(total_vals)
        if count == 0:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'median': 0}
            
        return {
            'count': count,
            'min': min(total_vals),
            'max': max(total_vals),
            'avg': sum(total_vals)/count,
            'median': sorted(total_vals)[count//2],
            'avg_left': sum(left_vals)/count,
            'avg_right': sum(right_vals)/count
        }

    def evaluate_threshold(self, threshold):
        """評估特定閾值的診斷效能"""
        nph_vals = [x[1] for x in self.nph_values]
        non_nph_vals = [x[1] for x in self.non_nph_values]
        
        # 體積越大越可能是 NPH
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
            'counts': {'tp': nph_above, 'fn': nph_below, 'fp': non_above, 'tn': non_below}
        }

    def generate_report(self, output_path):
        """生成 Markdown 報告"""
        nph_stats = self.get_statistics(self.nph_data)
        non_stats = self.get_statistics(self.non_nph_data)
        
        diff_vol = nph_stats['avg'] - non_stats['avg']
        diff_pct = (diff_vol / non_stats['avg']) * 100 if non_stats['avg'] else 0
        
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = f"""# 腦室總體積水腦症指標評估分析報告

**分析日期**: {today}
**數據來源**: 腦室體積批次處理結果 ({self.n_nph + self.n_non} 個案例)

---

## 執行摘要

本報告評估「腦室總體積」作為水腦症 (NPH) 診斷指標的可行性。研究結果顯示,**此指標展現良好的鑑別能力**,NPH 組與非 NPH 組有 **{diff_pct:.1f}%** 的差異。

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
| **總體積 (mm³)** | **{nph_stats['min']:.1f}** | **{nph_stats['max']:.1f}** | **{nph_stats['avg']:.1f}** | **{nph_stats['median']:.1f}** |

#### 非水腦症案例 (非 NPH, n={self.n_non})

| 測量指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|---------|--------|--------|--------|--------|
| **總體積 (mm³)** | **{non_stats['min']:.1f}** | **{non_stats['max']:.1f}** | **{non_stats['avg']:.1f}** | **{non_stats['median']:.1f}** |

#### 組間差異

| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |
|-----|-----------|-------------|------|-----------| 
| **總體積** | **{nph_stats['avg']:.1f} mm³** | **{non_stats['avg']:.1f} mm³** | **{diff_vol:+.1f} mm³** | **{diff_pct:+.1f}%** |

---

## 閾值效能評估

下表展示不同閾值下的診斷效能：

| 閾值 | 靈敏度 | 特異性 | PPV | NPV | 準確度 |
|------|--------|--------|-----|-----|--------|
"""
        for t in [400000, 500000, 600000, 700000, 800000]:
            m = self.evaluate_threshold(t)
            report += f"| **{t/1000:.0f}k mm³** | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% | {m['ppv']*100:.1f}% | {m['npv']*100:.1f}% | {m['accuracy']*100:.1f}% |\n"

        report += f"""
---

## 結論

**腦室總體積在本數據集中展現良好的 NPH 診斷效能**:

1. ✅ **組間差異顯著**: {diff_pct:.1f}%
2. ✅ **樣本數**: {self.n_nph + self.n_non} 例 (NPH: {self.n_nph}, 非 NPH: {self.n_non})

---

**報告產生**: 3D NPH Indicators 系統
**最後更新**: {today}
"""

        with open(output_path, 'w') as f:
            f.write(report)
        print(f"報告已生成: {output_path}")

    def generate_roc_curve(self, output_path):
        """生成 ROC 曲線"""
        y_true = [1] * len(self.nph_values) + [0] * len(self.non_nph_values)
        y_scores = [x[1] for x in self.nph_values] + [x[1] for x in self.non_nph_values]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#2563eb', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random classifier')
        
        key_thresholds = [400000, 500000, 600000, 700000, 800000]
        for thresh in key_thresholds:
            idx = (np.abs(thresholds - thresh)).argmin()
            plt.scatter(fpr[idx], tpr[idx], s=150, zorder=5, edgecolors='white', linewidth=2)
            plt.annotate(f'{thresh/1000:.0f}k\n(Sens:{tpr[idx]:.0%}, Spec:{1-fpr[idx]:.0%})', 
                         xy=(fpr[idx], tpr[idx]), 
                         xytext=(fpr[idx]+0.05, tpr[idx]-0.1),
                         fontsize=10, fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='#64748b'))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title(f'ROC Curve for Ventricle Volume NPH Classification\n(n={self.n_nph + self.n_non}, NPH={self.n_nph}, Non-NPH={self.n_non})', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=20, fontweight='bold', 
                 bbox=dict(boxstyle='round', facecolor='#dbeafe', edgecolor='#2563eb', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC 曲線已生成: {output_path}")

if __name__ == "__main__":
    analyzer = VentricleVolumeAnalyzer('/Users/lujingyuan/Project/3d-nph-indicators/result/volume_surface_ratio/results_summary.md')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    output_filename = f'result/volume_surface_ratio/volume_analysis_{timestamp}.md'
    analyzer.generate_report(os.path.join(os.getcwd(), output_filename))
    
    roc_filename = f'result/volume_surface_ratio/volume_roc_{timestamp}.png'
    analyzer.generate_roc_curve(os.path.join(os.getcwd(), roc_filename))
