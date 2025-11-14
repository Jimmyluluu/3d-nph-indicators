#!/usr/bin/env python3
"""
報表產生模組 - 產生 NPH 指標批次處理的 Markdown 報表
"""

from pathlib import Path
from datetime import datetime


def format_time(seconds):
    """
    格式化時間顯示

    Args:
        seconds: 秒數

    Returns:
        str: 格式化後的時間字串
    """
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} 分鐘"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} 小時"


def load_nph_list(nph_file="nph-list.txt"):
    """
    讀取 NPH 案例列表

    Args:
        nph_file: NPH 列表檔案路徑

    Returns:
        set: NPH 案例 ID 集合
    """
    nph_file_path = Path(nph_file)
    if not nph_file_path.exists():
        # 如果找不到檔案，回傳預設列表
        return {
            "000235496D", "001612043H", "000152785B",
            "000072318C", "data_5", "001149210H",
            "000087554H", "000137208D", "000096384I", "000206288G"
        }

    nph_cases = set()
    with open(nph_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            case_id = line.strip()
            if case_id:
                nph_cases.add(case_id)
    return nph_cases


# 指標配置：定義不同指標類型的欄位映射和顯示名稱
INDICATOR_CONFIGS = {
    'centroid_ratio': {
        'title': '腦室質心距離比值批次處理報表',
        'distance_field': 'ventricle_distance_mm',
        'distance_label': '腦室距離 (mm)',
        'ratio_field': 'ratio',
        'ratio_label': '比值',
        'ratio_percent_field': 'ratio_percent',
        'footer': '3D NPH Indicators'
    },
    'evan_index': {
        'title': '3D Evan Index 批次處理報表',
        'distance_field': 'anterior_horn_distance_mm',
        'distance_label': '前腳距離 (mm)',
        'ratio_field': 'evan_index',
        'ratio_label': 'Evan Index',
        'ratio_percent_field': 'evan_index_percent',
        'footer': '3D Evan Index Calculator'
    }
}


def generate_markdown_report(results, output_path, total_time, success_count, error_count, indicator_type='centroid_ratio'):
    """
    統一的 Markdown 報表生成函數

    Args:
        results: 處理結果列表
        output_path: 輸出檔案路徑
        total_time: 總處理時間（秒）
        success_count: 成功案例數
        error_count: 失敗案例數
        indicator_type: 指標類型 ('centroid_ratio' 或 'evan_index')

    Raises:
        ValueError: 當指標類型不支援時
    """
    # 取得指標配置
    if indicator_type not in INDICATOR_CONFIGS:
        raise ValueError(f"不支援的指標類型: {indicator_type}。可用的類型: {list(INDICATOR_CONFIGS.keys())}")

    config = INDICATOR_CONFIGS[indicator_type]
    nph_cases = load_nph_list()

    with open(output_path, 'w', encoding='utf-8') as f:
        # 報表標題
        f.write(f"# {config['title']}\n\n")
        f.write(f"**處理時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 摘要統計
        f.write("## 處理摘要\n\n")
        f.write(f"- **總案例數**: {len(results)}\n")
        f.write(f"- **成功**: {success_count} 個\n")
        f.write(f"- **失敗**: {error_count} 個\n")
        f.write(f"- **成功率**: {success_count/len(results)*100:.1f}%\n")
        f.write(f"- **總耗時**: {format_time(total_time)}\n")
        f.write(f"- **平均每案例**: {format_time(total_time/len(results))}\n\n")

        # 成功案例表格
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            f.write("## 測量結果\n\n")
            f.write(f"| 案例 ID | {config['distance_label']} | 顱內寬度 (mm) | {config['ratio_label']} | 百分比 | 處理時間 |\n")
            f.write("|---------|---------------|---------------|------|--------|----------|\n")

            for result in successful_results:
                case_id = result.get('case_id', 'N/A')
                distance = result.get(config['distance_field'], 0)
                width = result.get('cranial_width_mm', 0)
                ratio = result.get(config['ratio_field'], 0)
                percent = result.get(config['ratio_percent_field'], 0)
                time_str = result.get('processing_time', 'N/A')

                if case_id in nph_cases:
                    case_id_display = f"{case_id} ⚠️ NPH"
                else:
                    case_id_display = case_id

                f.write(f"| {case_id_display} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {time_str} |\n")

            # 統計資訊
            distances = [r[config['distance_field']] for r in successful_results]
            widths = [r['cranial_width_mm'] for r in successful_results]
            ratios = [r[config['ratio_field']] for r in successful_results]

            f.write("\n### 統計數據（全部案例）\n\n")
            f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            f.write(f"| {config['distance_label']} | {min(distances):.2f} | {max(distances):.2f} | {sum(distances)/len(distances):.2f} | {sorted(distances)[len(distances)//2]:.2f} |\n")
            f.write(f"| 顱內寬度 (mm) | {min(widths):.2f} | {max(widths):.2f} | {sum(widths)/len(widths):.2f} | {sorted(widths)[len(widths)//2]:.2f} |\n")
            f.write(f"| {config['ratio_label']} | {min(ratios):.4f} | {max(ratios):.4f} | {sum(ratios)/len(ratios):.4f} | {sorted(ratios)[len(ratios)//2]:.4f} |\n")

            # NPH 和非 NPH 分組統計
            nph_results = [r for r in successful_results if r.get('case_id') in nph_cases]
            non_nph_results = [r for r in successful_results if r.get('case_id') not in nph_cases]

            if nph_results:
                f.write(f"\n### NPH 案例統計 (n={len(nph_results)})\n\n")
                nph_distances = [r[config['distance_field']] for r in nph_results]
                nph_widths = [r['cranial_width_mm'] for r in nph_results]
                nph_ratios = [r[config['ratio_field']] for r in nph_results]

                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| {config['distance_label']} | {min(nph_distances):.2f} | {max(nph_distances):.2f} | {sum(nph_distances)/len(nph_distances):.2f} | {sorted(nph_distances)[len(nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(nph_widths):.2f} | {max(nph_widths):.2f} | {sum(nph_widths)/len(nph_widths):.2f} | {sorted(nph_widths)[len(nph_widths)//2]:.2f} |\n")
                f.write(f"| {config['ratio_label']} | {min(nph_ratios):.4f} | {max(nph_ratios):.4f} | {sum(nph_ratios)/len(nph_ratios):.4f} | {sorted(nph_ratios)[len(nph_ratios)//2]:.4f} |\n")

            if non_nph_results:
                f.write(f"\n### 非 NPH 案例統計 (n={len(non_nph_results)})\n\n")
                non_nph_distances = [r[config['distance_field']] for r in non_nph_results]
                non_nph_widths = [r['cranial_width_mm'] for r in non_nph_results]
                non_nph_ratios = [r[config['ratio_field']] for r in non_nph_results]

                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| {config['distance_label']} | {min(non_nph_distances):.2f} | {max(non_nph_distances):.2f} | {sum(non_nph_distances)/len(non_nph_distances):.2f} | {sorted(non_nph_distances)[len(non_nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(non_nph_widths):.2f} | {max(non_nph_widths):.2f} | {sum(non_nph_widths)/len(non_nph_widths):.2f} | {sorted(non_nph_widths)[len(non_nph_widths)//2]:.2f} |\n")
                f.write(f"| {config['ratio_label']} | {min(non_nph_ratios):.4f} | {max(non_nph_ratios):.4f} | {sum(non_nph_ratios)/len(non_nph_ratios):.4f} | {sorted(non_nph_ratios)[len(non_nph_ratios)//2]:.4f} |\n")

            # 組間差異
            if nph_results and non_nph_results:
                f.write("\n### 組間差異\n\n")
                f.write("| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |\n")
                f.write("|-----|-----------|-------------|------|-----------|\n")

                nph_dist_mean = sum(nph_distances) / len(nph_distances)
                non_nph_dist_mean = sum(non_nph_distances) / len(non_nph_distances)
                dist_diff = nph_dist_mean - non_nph_dist_mean
                dist_pct = (dist_diff / non_nph_dist_mean) * 100
                f.write(f"| {config['distance_label']} | {nph_dist_mean:.2f} mm | {non_nph_dist_mean:.2f} mm | {dist_diff:+.2f} mm | {dist_pct:+.1f}% |\n")

                nph_width_mean = sum(nph_widths) / len(nph_widths)
                non_nph_width_mean = sum(non_nph_widths) / len(non_nph_widths)
                width_diff = nph_width_mean - non_nph_width_mean
                width_pct = (width_diff / non_nph_width_mean) * 100
                f.write(f"| 顱內寬度 | {nph_width_mean:.2f} mm | {non_nph_width_mean:.2f} mm | {width_diff:+.2f} mm | {width_pct:+.1f}% |\n")

                nph_ratio_mean = sum(nph_ratios) / len(nph_ratios)
                non_nph_ratio_mean = sum(non_nph_ratios) / len(non_nph_ratios)
                ratio_diff = nph_ratio_mean - non_nph_ratio_mean
                ratio_pct = (ratio_diff / non_nph_ratio_mean) * 100
                f.write(f"| **{config['ratio_label']}** | **{nph_ratio_mean:.4f}** | **{non_nph_ratio_mean:.4f}** | **{ratio_diff:+.4f}** | **{ratio_pct:+.1f}%** |\n")

            # NPH 案例詳細列表
            if nph_results:
                f.write("\n## NPH 案例詳細數據\n\n")
                f.write(f"| 案例 ID | {config['distance_label']} | 顱內寬度 (mm) | {config['ratio_label']} | 百分比 | 排序 |\n")
                f.write("|---------|---------------|---------------|------|--------|------|\n")

                nph_sorted = sorted(nph_results, key=lambda x: x[config['ratio_field']], reverse=True)
                for i, result in enumerate(nph_sorted, 1):
                    case_id = result.get('case_id', 'N/A')
                    distance = result.get(config['distance_field'], 0)
                    width = result.get('cranial_width_mm', 0)
                    ratio = result.get(config['ratio_field'], 0)
                    percent = result.get(config['ratio_percent_field'], 0)

                    rank_note = ""
                    if i == 1:
                        rank_note = " (最高)"
                    elif i == len(nph_sorted):
                        rank_note = " (最低)"

                    f.write(f"| {case_id} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {i}{rank_note} |\n")

        # 失敗案例
        failed_results = [r for r in results if r.get('status') == 'error']
        if failed_results:
            f.write("\n## 失敗案例\n\n")
            f.write("| 案例 ID | 錯誤類型 | 錯誤訊息 |\n")
            f.write("|---------|----------|----------|\n")

            for result in failed_results:
                case_id = result.get('case_id', 'N/A')
                error_type = result.get('error_type', 'Unknown')
                error_msg = result.get('error_message', 'N/A')
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                f.write(f"| {case_id} | {error_type} | {error_msg} |\n")

        f.write(f"\n---\n\n*由 {config['footer']} 自動產生*\n")
