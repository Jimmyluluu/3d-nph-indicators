#!/usr/bin/env python3
"""
通用結果分析器
支援所有 NPH 指標的批次處理結果分析、統計、報告生成和 ROC 曲線
"""

import re
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# 指標配置定義
INDICATOR_CONFIGS = {
    'evan_index': {
        'name': 'Evan Index',
        'full_name': '3D Evan Index',
        'pattern': r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| [\d.]+s \|',
        'fields': ['case_id', 'anterior_distance', 'cranial_width', 'ratio', 'percent'],
        'primary_field': 'percent',  # 用於 ROC 曲線的欄位
        'threshold': 30.0,  # 診斷閾值（百分比）
        'threshold_range': [25.0, 27.5, 30.0, 32.5, 35.0],  # 閾值掃描範圍
        'outlier_threshold': 50.0,  # 異常值過濾閾值
        'unit': '%',
        'report_title': '3D Evan Index 分析報告',
    },
    'alvi': {
        'name': 'ALVI',
        'full_name': 'Anteroposterior Lateral Ventricle Index',
        'pattern': r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| [\d.]+s \|',
        'fields': ['case_id', 'ventricle_ap_diameter', 'skull_ap_diameter', 'ratio', 'percent'],
        'primary_field': 'ratio',  # ALVI 使用比值做 ROC
        'threshold': 0.5,  # 診斷閾值（比值）
        'threshold_range': [0.40, 0.45, 0.50, 0.55, 0.60],  # 閾值掃描範圍
        'outlier_threshold': 0.8,  # 異常值過濾閾值（比值）
        'unit': '',
        'report_title': 'ALVI 分析報告',
    },
    'volume_surface_ratio': {
        'name': 'V/SA Ratio',
        'full_name': 'Volume to Surface Area Ratio',
        'pattern': r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| [\d.]+s \|',
        'fields': ['case_id', 'left_volume', 'right_volume', 'total_volume', 'ratio'],
        'primary_field': 'ratio',
        'threshold': None,  # 無固定閾值
        'outlier_threshold': None,
        'unit': 'mm',
        'report_title': '體積/表面積比例分析報告',
    },
    'surface_area': {
        'name': 'Surface Area',
        'full_name': 'Ventricle Surface Area',
        'pattern': r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| [\d.]+s \|',
        'fields': ['case_id', 'left_area', 'right_area', 'total_area'],
        'primary_field': 'total_area',
        'threshold': None,
        'outlier_threshold': None,
        'unit': 'mm²',
        'report_title': '腦室表面積分析報告',
    },
    'ventricle_volume': {
        'name': 'Volume',
        'full_name': 'Ventricle Volume',
        'pattern': r'\| ([^\|]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| [\d.]+s \|',
        'fields': ['case_id', 'left_volume', 'right_volume', 'total_volume'],
        'primary_field': 'total_volume',
        'threshold': None,
        'outlier_threshold': None,
        'unit': 'mm³',
        'report_title': '腦室體積分析報告',
    },
}


class BaseResultAnalyzer:
    """通用結果分析器基礎類別"""
    
    def __init__(self, results_path, indicator_type):
        """
        初始化分析器
        
        Args:
            results_path: 結果摘要文件路徑
            indicator_type: 指標類型 ('evan_index', 'alvi', 'volume_surface_ratio', etc.)
        """
        if indicator_type not in INDICATOR_CONFIGS:
            raise ValueError(f"不支援的指標類型: {indicator_type}. 可用類型: {list(INDICATOR_CONFIGS.keys())}")
        
        self.results_path = results_path
        self.indicator_type = indicator_type
        self.config = INDICATOR_CONFIGS[indicator_type]
        
        self.nph_values = []      # List of (id, value)
        self.non_nph_values = []  # List of (id, value)
        self.nph_data = []        # List of full data tuples
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
        
        pattern = self.config['pattern']
        matches = re.findall(pattern, content)
        
        for match in matches:
            case_id = match[0].strip()
            
            # 跳過表頭行
            if '案例 ID' in case_id or '---' in case_id:
                continue
            
            # 解析數據
            data_dict = {}
            for i, field in enumerate(self.config['fields']):
                if i == 0:  # case_id
                    data_dict[field] = case_id
                else:
                    data_dict[field] = float(match[i])
            
            # 獲取主要欄位值（用於 ROC 和過濾）
            primary_value = data_dict[self.config['primary_field']]
            
            # 過濾異常值
            outlier_threshold = self.config.get('outlier_threshold')
            if outlier_threshold is not None and primary_value > outlier_threshold:
                self.abnormal_cases.append((case_id, primary_value))
                continue
            
            # 分類 NPH / 非 NPH
            if '⚠️ NPH' in case_id:
                clean_id = case_id.replace(' ⚠️ NPH', '')
                self.nph_values.append((clean_id, primary_value))
                self.nph_data.append(data_dict)
            else:
                self.non_nph_values.append((case_id, primary_value))
                self.non_nph_data.append(data_dict)
        
        self.n_nph = len(self.nph_values)
        self.n_non = len(self.non_nph_values)
        
        print(f"數據加載完成: NPH={self.n_nph}, 非 NPH={self.n_non}")
    
    def get_statistics(self, data_list):
        """獲取統計數據"""
        if not data_list:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        
        primary_field = self.config['primary_field']
        values = [d[primary_field] for d in data_list]
        
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
        }
        
        # 添加其他欄位的統計（如果有）
        for field in self.config['fields'][1:]:  # 跳過 case_id
            if field != primary_field:
                field_values = [d[field] for d in data_list]
                stats[f'min_{field}'] = min(field_values)
                stats[f'max_{field}'] = max(field_values)
                stats[f'avg_{field}'] = sum(field_values) / len(field_values)
        
        return stats
    
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
            'tp': nph_above,
            'tn': non_below,
            'fp': non_above,
            'fn': nph_below
        }
    
    def generate_roc_curve(self, output_path):
        """生成 ROC 曲線"""
        # 準備數據
        y_true = [1] * self.n_nph + [0] * self.n_non
        y_scores = [x[1] for x in self.nph_values] + [x[1] for x in self.non_nph_values]
        
        # 計算 ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # 繪製
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#10b981', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#94a3b8', lw=2, linestyle='--', label='Random classifier')
        
        # 標記關鍵閾值點（如果有定義閾值範圍）
        threshold_range = self.config.get('threshold_range')
        if threshold_range and len(threshold_range) >= 3:
            # 取中間 3 個閾值作為關鍵點
            mid_idx = len(threshold_range) // 2
            key_thresholds = threshold_range[mid_idx-1:mid_idx+2]
            
            for thresh in key_thresholds:
                # 找到最接近 threshold 的點
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
        plt.title(f'ROC Curve for {self.config["full_name"]} NPH Classification\n(n={self.n_nph + self.n_non}, NPH={self.n_nph}, Non-NPH={self.n_non})', 
                  fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加 AUC 文字框
        plt.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=20, fontweight='bold', 
                 bbox=dict(boxstyle='round', facecolor='#d1fae5', edgecolor='#10b981', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ROC 曲線已生成: {output_path}")
        return roc_auc
    
    def generate_report(self, output_path):
        """生成分析報告（Markdown 格式）"""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # 計算統計數據
        nph_stats = self.get_statistics(self.nph_data)
        non_nph_stats = self.get_statistics(self.non_nph_data)
        all_stats = self.get_statistics(self.nph_data + self.non_nph_data)
        
        # 計算組間差異
        diff = nph_stats['avg'] - non_nph_stats['avg']
        diff_pct = (diff / non_nph_stats['avg'] * 100) if non_nph_stats['avg'] != 0 else 0
        
        # 評估閾值（如果有定義）
        threshold_eval = None
        if self.config.get('threshold') is not None:
            threshold_eval = self.evaluate_threshold(self.config['threshold'])
        
        # 生成報告
        report = f"""# {self.config['report_title']}

**分析日期**: {today}
**數據來源**: 批次處理結果 ({self.n_nph + self.n_non} 個有效案例)

---

## 執行摘要

本報告評估 {self.config['full_name']} 作為水腦症 (NPH) 診斷指標的效能。
研究結果顯示，NPH 組與非 NPH 組有 **{abs(diff_pct):.1f}%** 的差異。

---

## 數據概況

- **總案例數**: {self.n_nph + self.n_non}
- **NPH 案例**: {self.n_nph}
- **非 NPH 案例**: {self.n_non}
"""
        
        if self.abnormal_cases:
            report += f"- **異常值過濾**: {len(self.abnormal_cases)} 例\n"
        
        report += f"""
---

## 統計分析

### 整體統計

| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|------|--------|--------|--------|--------|
| {self.config['name']} | {all_stats['min']:.4f} | {all_stats['max']:.4f} | {all_stats['avg']:.4f} | {all_stats['median']:.4f} |

### NPH 組統計 (n={nph_stats['count']})

| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|------|--------|--------|--------|--------|
| {self.config['name']} | {nph_stats['min']:.4f} | {nph_stats['max']:.4f} | {nph_stats['avg']:.4f} | {nph_stats['median']:.4f} |

### 非 NPH 組統計 (n={non_nph_stats['count']})

| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |
|------|--------|--------|--------|--------|
| {self.config['name']} | {non_nph_stats['min']:.4f} | {non_nph_stats['max']:.4f} | {non_nph_stats['avg']:.4f} | {non_nph_stats['median']:.4f} |

### 組間差異

- **NPH 平均值**: {nph_stats['avg']:.4f} {self.config['unit']}
- **非 NPH 平均值**: {non_nph_stats['avg']:.4f} {self.config['unit']}
- **差異**: {diff:+.4f} {self.config['unit']} ({diff_pct:+.1f}%)

---
"""
        
        # 閾值評估（如果有）
        if self.config.get('threshold') is not None:
            # 閾值區間掃描
            threshold_range = self.config.get('threshold_range', [self.config['threshold']])
            
            report += f"""
## 閾值評估

### 閾值區間掃描

| 閾值 | 靈敏度 | 特異度 | PPV | NPV | 準確度 |
|------|--------|--------|-----|-----|--------|
"""
            for t in threshold_range:
                m = self.evaluate_threshold(t)
                report += f"| **{t:.2f}** | {m['sensitivity']*100:.1f}% | {m['specificity']*100:.1f}% | {m['ppv']*100:.1f}% | {m['npv']*100:.1f}% | {m['accuracy']*100:.1f}% |\n"
            
            # 推薦閾值的詳細評估
            threshold_eval = self.evaluate_threshold(self.config['threshold'])
            report += f"""

### 推薦閾值詳細評估

使用閾值 **{threshold_eval['threshold']}** {self.config['unit']}:

| 指標 | 數值 |
|------|------|
| 靈敏度 (Sensitivity) | {threshold_eval['sensitivity']:.1%} |
| 特異度 (Specificity) | {threshold_eval['specificity']:.1%} |
| 陽性預測值 (PPV) | {threshold_eval['ppv']:.1%} |
| 陰性預測值 (NPV) | {threshold_eval['npv']:.1%} |
| 準確度 (Accuracy) | {threshold_eval['accuracy']:.1%} |

**混淆矩陣**:

|  | 預測 NPH | 預測非 NPH |
|---|----------|------------|
| **實際 NPH** | {threshold_eval['tp']} (TP) | {threshold_eval['fn']} (FN) |
| **實際非 NPH** | {threshold_eval['fp']} (FP) | {threshold_eval['tn']} (TN) |

---
"""
        
        # Top 案例列表
        primary_field = self.config['primary_field']
        
        # NPH Top 20
        sorted_nph = sorted(self.nph_data, key=lambda x: x[primary_field], reverse=True)
        report += f"""
## 附錄: NPH 案例 (Top 20)

| 排序 | 案例 ID | {self.config['name']} |
|-----|---------|----------|
"""
        for i, item in enumerate(sorted_nph[:20]):
            report += f"| {i+1} | {item['case_id']} | {item[primary_field]:.4f} |\n"
        
        # 非 NPH Top 10
        sorted_non_nph = sorted(self.non_nph_data, key=lambda x: x[primary_field], reverse=True)
        report += f"""

## 附錄: 非 NPH 高值案例 (Top 10)

| 排序 | 案例 ID | {self.config['name']} |
|-----|---------|----------|
"""
        for i, item in enumerate(sorted_non_nph[:10]):
            report += f"| {i+1} | {item['case_id']} | {item[primary_field]:.4f} |\n"
        
        report += """
---
"""
        
        report += f"""
## 結論

{self.config['full_name']} 在本研究中顯示 NPH 組與非 NPH 組之間存在 {abs(diff_pct):.1f}% 的差異。
"""
        
        if threshold_eval:
            report += f"使用閾值 {threshold_eval['threshold']} {self.config['unit']} 時，靈敏度為 {threshold_eval['sensitivity']:.1%}，特異度為 {threshold_eval['specificity']:.1%}。\n"
        
        report += """
---

*本報告由自動化分析系統生成*
"""
        
        # 寫入檔案
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析報告已生成: {output_path}")


# 工廠函數
def create_analyzer(indicator_type, results_path):
    """
    創建指定類型的結果分析器
    
    Args:
        indicator_type: 指標類型
        results_path: 結果摘要文件路徑
    
    Returns:
        BaseResultAnalyzer 實例
    """
    return BaseResultAnalyzer(results_path, indicator_type)


# 主程式入口（用於獨立執行）
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python result_analyzer.py <indicator_type> <results_path>")
        print(f"可用的指標類型: {list(INDICATOR_CONFIGS.keys())}")
        sys.exit(1)
    
    indicator_type = sys.argv[1]
    results_path = sys.argv[2]
    
    try:
        analyzer = create_analyzer(indicator_type, results_path)
        
        # 生成報告和 ROC 曲線
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        report_path = f"{indicator_type}_analysis_{timestamp}.md"
        roc_path = f"{indicator_type}_roc_{timestamp}.png"
        
        analyzer.generate_report(report_path)
        analyzer.generate_roc_curve(roc_path)
        
        print(f"\n✓ 分析完成！")
        print(f"  報告: {report_path}")
        print(f"  ROC 曲線: {roc_path}")
        
    except Exception as e:
        print(f"錯誤: {e}")
        sys.exit(1)
