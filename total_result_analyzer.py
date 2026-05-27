#!/usr/bin/env python3
"""
總合 ROC 曲線分析
將所有 NPH 指標的 ROC 曲線繪製在同一張圖上

啟用指標：
  - ALVI
  - Evan Index
  - Ventricle Volume
  - Surface Area（從 V/SA Ratio 與 Volume 反推）
  - Volume / Surface Area Ratio
  - Callosal Angle
"""

import re
import os
import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 確保可以 import model/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.result_analyzer import BaseResultAnalyzer, INDICATOR_CONFIGS


# ──────────────────────────────────────────────────────────────
# 路徑設定
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'result')

_VS_PATH = os.path.join(RESULT_DIR, 'volume_surface_ratio', 'results_summary.md')

# (key, results_path) — 'surface_area_derived' 由自訂 analyzer 處理
INDICATORS_TO_LOAD = [
    ('evan_index',           os.path.join(RESULT_DIR, 'evan_index', 'results_summary.md')),
    ('alvi',                 os.path.join(RESULT_DIR, 'alvi',       'results_summary.md')),
    ('ventricle_volume',     _VS_PATH),
    ('surface_area_derived', _VS_PATH),   # 特殊：反推 surface area = total_volume / ratio
    ('volume_surface_ratio', _VS_PATH),
    ('callosal_angle',       os.path.join(RESULT_DIR, 'callosal_angle', 'results_summary.md')),
]

# 各指標顯示設定
INDICATOR_STYLES = {
    'evan_index':           {'label': 'Evan Index',   'color': '#3b82f6'},
    'alvi':                 {'label': 'ALVI',          'color': '#ef4444'},
    'ventricle_volume':     {'label': 'Volume',        'color': '#f59e0b'},
    'surface_area_derived': {'label': 'Surface Area',  'color': '#8b5cf6'},
    'volume_surface_ratio': {'label': 'V/SA Ratio',    'color': '#10b981'},
    'callosal_angle':       {'label': 'Callosal Angle','color': '#ec4899'},
}


# ──────────────────────────────────────────────────────────────
# Surface Area 衍生分析器
# 從同一份報表的 total_volume 和 V/SA ratio 反推 surface area
# ──────────────────────────────────────────────────────────────
class SurfaceAreaDerivedAnalyzer:
    """從 Volume 與 V/SA Ratio 反推表面積的輕量分析器"""

    def __init__(self, results_path: str):
        self.results_path = results_path
        self.config = {
            'name': 'Surface Area (Derived)',
            'full_name': 'Ventricle Surface Area (Derived)',
            'direction': 'up',   # 體積越大 → 表面積越大 → 越可能 NPH
            'primary_field': 'surface_area',
            'unit': 'mm²',
        }
        self.nph_values = []
        self.non_nph_values = []
        self.n_nph = 0
        self.n_non = 0
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"找不到結果文件: {self.results_path}")

        with open(self.results_path, 'r') as f:
            content = f.read()

        # 使用 volume_surface_ratio 的 pattern 讀取
        pattern = INDICATOR_CONFIGS['volume_surface_ratio']['pattern']
        matches = re.findall(pattern, content)

        for match in matches:
            case_id = match[0].strip()
            if '案例 ID' in case_id or '---' in case_id:
                continue
            try:
                total_volume = float(match[3])
                ratio = float(match[4])
                if ratio <= 0:
                    continue
                # surface_area = total_volume / (V/SA ratio)
                surface_area = total_volume / ratio
            except (ValueError, IndexError):
                continue

            if '⚠️ NPH' in case_id:
                clean_id = case_id.replace(' ⚠️ NPH', '')
                self.nph_values.append((clean_id, surface_area))
            else:
                self.non_nph_values.append((case_id, surface_area))

        self.n_nph = len(self.nph_values)
        self.n_non = len(self.non_nph_values)
        print(f"  Surface Area (Derived): NPH={self.n_nph}, 非 NPH={self.n_non}")


# ──────────────────────────────────────────────────────────────
# 工廠函數
# ──────────────────────────────────────────────────────────────
def _build_analyzer(key: str, path: str):
    """依 key 建立對應の分析器實例"""
    if key == 'surface_area_derived':
        return SurfaceAreaDerivedAnalyzer(path)
    return BaseResultAnalyzer(path, key)


# ──────────────────────────────────────────────────────────────
# ROC 計算
# ──────────────────────────────────────────────────────────────
def _compute_roc(analyzer):
    """
    回傳 (fpr, tpr, thresholds_actual, roc_auc)。
    若指標方向為 'down'（數值越小越可能 NPH），則對分數取負後再跑 ROC。
    """
    y_true = [1] * analyzer.n_nph + [0] * analyzer.n_non
    y_scores = [v for _, v in analyzer.nph_values] + [v for _, v in analyzer.non_nph_values]

    direction = analyzer.config.get('direction', 'up')
    scores_for_roc = [-s for s in y_scores] if direction == 'down' else y_scores

    fpr, tpr, thresholds_roc = roc_curve(y_true, scores_for_roc)
    roc_auc = auc(fpr, tpr)

    # 還原閾值到原始量綱
    thresholds_actual = -thresholds_roc if direction == 'down' else thresholds_roc
    return fpr, tpr, thresholds_actual, roc_auc


def _find_youden_idx(fpr, tpr) -> int:
    """Youden's J = TPR - FPR，找最大值的索引（最佳轉折點）"""
    return int(np.argmax(tpr - fpr))


# ──────────────────────────────────────────────────────────────
# 標注偏移 — 讓五個標注不相互遮擋
# 依 (fpr, tpr) 位置動態決定文字框方向，再加上 stagger
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# 主繪圖函數
# ──────────────────────────────────────────────────────────────
def generate_combined_roc(output_path: str | None = None) -> str:
    """
    產生所有指標的合圖 ROC 曲線；最佳轉折點（Youden's J）只印在終端，不標注於圖上。

    Args:
        output_path: 輸出 PNG 路徑；若 None 則自動命名存至 result/

    Returns:
        儲存的檔案路徑
    """
    print("=" * 60)
    print("載入各指標數據…")
    print("=" * 60)

    analyzers = {}
    for key, path in INDICATORS_TO_LOAD:
        if key not in INDICATOR_STYLES:
            continue
        try:
            analyzers[key] = _build_analyzer(key, path)
        except Exception as exc:
            print(f"  [WARN] 無法載入 {key}: {exc}")

    if not analyzers:
        raise RuntimeError("無法載入任何指標數據，請確認 result/ 目錄結構正確")

    # ── 建立圖表 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))

    # 對角線（隨機分類器）
    ax.plot([0, 1], [0, 1],
            color='#94a3b8', lw=1.8, linestyle='--',
            label='Random Classifier', zorder=1)

    print("\n計算 ROC 曲線…")

    for key, analyzer in analyzers.items():
        style = INDICATOR_STYLES[key]

        try:
            fpr, tpr, thresholds, roc_auc = _compute_roc(analyzer)
        except Exception as exc:
            print(f"  [WARN] 無法計算 {key} ROC: {exc}")
            continue

        # ── 繪曲線 ──
        ax.plot(fpr, tpr,
                color=style['color'],
                lw=2.5,
                label=f"{style['label']}  (AUC = {roc_auc:.3f})",
                zorder=2)

        # ── 最佳轉折點（Youden's J）— 印出，不標注在圖上 ──
        opt_idx = _find_youden_idx(fpr, tpr)
        opt_fpr = float(fpr[opt_idx])
        opt_tpr = float(tpr[opt_idx])
        opt_thresh = float(thresholds[opt_idx])
        unit = analyzer.config.get('unit', '')

        n_total = analyzer.n_nph + analyzer.n_non
        print(f"  ✓ {style['label']:14s}  AUC={roc_auc:.3f}"
              f"  轉折點=(FPR={opt_fpr:.3f}, TPR={opt_tpr:.3f})"
              f"  閾值={opt_thresh:.3f}{unit}  Sens={opt_tpr:.0%}  Spec={1-opt_fpr:.0%}"
              f"  n={n_total} (NPH={analyzer.n_nph})")

    # ── 圖表裝飾 ──────────────────────────────────────────────
    ax.set_xlim([0.0, 1.0]) # type: ignore
    ax.set_ylim([0.0, 1.05]) # type: ignore
    ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=13)
    ax.set_title(
        'Combined ROC Curves — NPH Indicators\n'
        'ALVI  ·  Evan Index  ·  Volume  ·  Surface Area  ·  V/SA Ratio  ·  Callosal Angle',
        fontsize=14, fontweight='bold', pad=14,
    )
    ax.legend(loc='lower right', fontsize=11, framealpha=0.92)
    ax.grid(True, alpha=0.25, zorder=0)

    plt.tight_layout()

    # ── 儲存 ──────────────────────────────────────────────────
    if output_path is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        output_path = os.path.join(RESULT_DIR, f'combined_roc_{timestamp}.png')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ 合圖已儲存：{output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    out = generate_combined_roc()
    print(f"\n完成！輸出：{out}")
