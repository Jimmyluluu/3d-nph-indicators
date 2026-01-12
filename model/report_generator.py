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
    },
    'alvi': {
        'title': 'ALVI 批次處理報表',
        'distance_field': 'ventricle_ap_diameter_mm',
        'distance_label': '腦室前後徑 (mm)',
        'ratio_field': 'alvi',
        'ratio_label': 'ALVI',
        'ratio_percent_field': 'alvi_percent',
        'additional_field': 'skull_ap_diameter_mm',
        'additional_label': '顱骨前後徑 (mm)',
        'footer': 'ALVI Calculator'
    },
    'volume_surface_ratio': {
        'title': '腦室體積與表面積比例批次處理報表',
        'left_volume_field': 'left_volume',
        'left_volume_label': '左腦室體積 (mm³)',
        'right_volume_field': 'right_volume',
        'right_volume_label': '右腦室體積 (mm³)',
        'total_volume_field': 'total_volume',
        'total_volume_label': '總體積 (mm³)',
        'left_area_field': 'left_surface_area',
        'left_area_label': '左腦室表面積 (mm²)',
        'right_area_field': 'right_surface_area',
        'right_area_label': '右腦室表面積 (mm²)',
        'total_area_field': 'total_surface_area',
        'total_area_label': '總表面積 (mm²)',
        'total_ratio_field': 'total_ratio',
        'total_ratio_label': 'V/SA 比例 (mm)',
        'footer': 'Volume-to-Surface Ratio Calculator'
    }
}


def generate_markdown_report(results, output_path, total_time, success_count, error_count,
                             indicator_type='centroid_ratio', use_is_nph_field=False):
    """
    統一的 Markdown 報表生成函數

    Args:
        results: 處理結果列表
        output_path: 輸出檔案路徑
        total_time: 總處理時間（秒）
        success_count: 成功案例數
        error_count: 失敗案例數
        indicator_type: 指標類型 ('centroid_ratio' 或 'evan_index')
        use_is_nph_field: 是否使用結果中的 is_nph 欄位來判斷 NPH 分類
                          (True: 使用 is_nph 欄位, False: 使用 nph-list.txt)

    Raises:
        ValueError: 當指標類型不支援時
    """
    # 取得指標配置
    if indicator_type not in INDICATOR_CONFIGS:
        raise ValueError(f"不支援的指標類型: {indicator_type}。可用的類型: {list(INDICATOR_CONFIGS.keys())}")

    config = INDICATOR_CONFIGS[indicator_type]

    # 決定如何判斷 NPH 分類
    if use_is_nph_field:
        # 新模式：從結果的 is_nph 欄位判斷
        def is_nph_case(result):
            return result.get('is_nph', False)
    else:
        # 舊模式：從 nph-list.txt 判斷
        nph_cases = load_nph_list()
        def is_nph_case(result):
            return result.get('case_id') in nph_cases

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

            # 根據指標類型決定表格格式
            if indicator_type == "volume_surface_ratio":
                # 體積表面積比例專用表格格式（簡化版，只顯示總體比例）
                f.write(f"| 案例 ID | {config['left_volume_label']} | {config['right_volume_label']} | {config['total_volume_label']} | {config['total_ratio_label']} | 處理時間 |\n")
                f.write("|---------|-------------------|--------------------|----------------|----------------|----------|\n")

                for result in successful_results:
                    case_id = result.get('case_id', 'N/A')
                    left_volume = result.get(config['left_volume_field'], 0)
                    right_volume = result.get(config['right_volume_field'], 0)
                    total_volume = result.get(config['total_volume_field'], 0)
                    total_ratio = result.get(config['total_ratio_field'], 0)
                    time_str = result.get('processing_time', 'N/A')

                    if is_nph_case(result):
                        case_id_display = f"{case_id} ⚠️ NPH"
                    else:
                        case_id_display = case_id

                    f.write(f"| {case_id_display} | {left_volume:.1f} | {right_volume:.1f} | {total_volume:.1f} | {total_ratio:.4f} | {time_str} |\n")

            else:
                # 原有的 distance/ratio 格式 (centroid_ratio, evan_index, alvi)
                # ALVI 使用 skull_ap_diameter_mm 而非 cranial_width_mm
                width_label = config.get('additional_label', '顱內寬度 (mm)')
                
                f.write(f"| 案例 ID | {config['distance_label']} | {width_label} | {config['ratio_label']} | 百分比 | 處理時間 |\n")
                f.write("|---------|---------------|---------------|------|--------|----------|\n")

                for result in successful_results:
                    case_id = result.get('case_id', 'N/A')
                    distance = result.get(config['distance_field'], 0)
                    
                    # 根據指標類型選擇正確的寬度欄位
                    if indicator_type == 'alvi':
                        width = result.get(config.get('additional_field', 'skull_ap_diameter_mm'), 0)
                    else:
                        width = result.get('cranial_width_mm', 0)
                    
                    ratio = result.get(config['ratio_field'], 0)
                    percent = result.get(config['ratio_percent_field'], 0)
                    time_str = result.get('processing_time', 'N/A')

                    if is_nph_case(result):
                        case_id_display = f"{case_id} ⚠️ NPH"
                    else:
                        case_id_display = case_id

                    f.write(f"| {case_id_display} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {time_str} |\n")

            # 統計資訊
            if indicator_type == "volume_surface_ratio":
                # 體積表面積比例統計（簡化版）
                left_volumes = [r[config['left_volume_field']] for r in successful_results]
                right_volumes = [r[config['right_volume_field']] for r in successful_results]
                total_volumes = [r[config['total_volume_field']] for r in successful_results]
                total_ratios = [r[config['total_ratio_field']] for r in successful_results]

                f.write("\n### 統計數據（全部案例）\n\n")
                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| {config['left_volume_label']} | {min(left_volumes):.1f} | {max(left_volumes):.1f} | {sum(left_volumes)/len(left_volumes):.1f} | {sorted(left_volumes)[len(left_volumes)//2]:.1f} |\n")
                f.write(f"| {config['right_volume_label']} | {min(right_volumes):.1f} | {max(right_volumes):.1f} | {sum(right_volumes)/len(right_volumes):.1f} | {sorted(right_volumes)[len(right_volumes)//2]:.1f} |\n")
                f.write(f"| {config['total_volume_label']} | {min(total_volumes):.1f} | {max(total_volumes):.1f} | {sum(total_volumes)/len(total_volumes):.1f} | {sorted(total_volumes)[len(total_volumes)//2]:.1f} |\n")
                f.write(f"| {config['total_ratio_label']} | {min(total_ratios):.4f} | {max(total_ratios):.4f} | {sum(total_ratios)/len(total_ratios):.4f} | {sorted(total_ratios)[len(total_ratios)//2]:.4f} |\n")

            else:
                # 原有的 distance/ratio 統計
                distances = [r[config['distance_field']] for r in successful_results]
                
                # 根據指標類型選擇正確的寬度欄位
                if indicator_type == 'alvi':
                    widths = [r.get(config.get('additional_field', 'skull_ap_diameter_mm'), 0) for r in successful_results]
                    width_label = config.get('additional_label', '顱骨前後徑 (mm)')
                else:
                    widths = [r['cranial_width_mm'] for r in successful_results]
                    width_label = '顱內寬度 (mm)'
                
                ratios = [r[config['ratio_field']] for r in successful_results]

                f.write("\n### 統計數據（全部案例）\n\n")
                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| {config['distance_label']} | {min(distances):.2f} | {max(distances):.2f} | {sum(distances)/len(distances):.2f} | {sorted(distances)[len(distances)//2]:.2f} |\n")
                f.write(f"| {width_label} | {min(widths):.2f} | {max(widths):.2f} | {sum(widths)/len(widths):.2f} | {sorted(widths)[len(widths)//2]:.2f} |\n")
                f.write(f"| {config['ratio_label']} | {min(ratios):.4f} | {max(ratios):.4f} | {sum(ratios)/len(ratios):.4f} | {sorted(ratios)[len(ratios)//2]:.4f} |\n")

            # NPH 和非 NPH 分組統計
            nph_results = [r for r in successful_results if is_nph_case(r)]
            non_nph_results = [r for r in successful_results if not is_nph_case(r)]

            if nph_results:
                f.write(f"\n### NPH 案例統計 (n={len(nph_results)})\n\n")

                if indicator_type == "volume_surface_ratio":
                    nph_total_volumes = [r[config['total_volume_field']] for r in nph_results]
                    nph_total_ratios = [r[config['total_ratio_field']] for r in nph_results]

                    f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                    f.write("|------|--------|--------|--------|--------|\n")
                    f.write(f"| {config['total_volume_label']} | {min(nph_total_volumes):.1f} | {max(nph_total_volumes):.1f} | {sum(nph_total_volumes)/len(nph_total_volumes):.1f} | {sorted(nph_total_volumes)[len(nph_total_volumes)//2]:.1f} |\n")
                    f.write(f"| {config['total_ratio_label']} | {min(nph_total_ratios):.4f} | {max(nph_total_ratios):.4f} | {sum(nph_total_ratios)/len(nph_total_ratios):.4f} | {sorted(nph_total_ratios)[len(nph_total_ratios)//2]:.4f} |\n")
                else:
                    # 原有的 distance/ratio 統計 (centroid_ratio, evan_index, alvi)
                    nph_distances = [r[config['distance_field']] for r in nph_results]
                    
                    # 根據指標類型選擇正確的寬度欄位
                    if indicator_type == 'alvi':
                        nph_widths = [r.get(config.get('additional_field', 'skull_ap_diameter_mm'), 0) for r in nph_results]
                        width_label = config.get('additional_label', '顱骨前後徑 (mm)')
                    else:
                        nph_widths = [r['cranial_width_mm'] for r in nph_results]
                        width_label = '顱內寬度 (mm)'
                    
                    nph_ratios = [r[config['ratio_field']] for r in nph_results]

                    f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                    f.write("|------|--------|--------|--------|--------|\n")
                    f.write(f"| {config['distance_label']} | {min(nph_distances):.2f} | {max(nph_distances):.2f} | {sum(nph_distances)/len(nph_distances):.2f} | {sorted(nph_distances)[len(nph_distances)//2]:.2f} |\n")
                    f.write(f"| {width_label} | {min(nph_widths):.2f} | {max(nph_widths):.2f} | {sum(nph_widths)/len(nph_widths):.2f} | {sorted(nph_widths)[len(nph_widths)//2]:.2f} |\n")
                    f.write(f"| {config['ratio_label']} | {min(nph_ratios):.4f} | {max(nph_ratios):.4f} | {sum(nph_ratios)/len(nph_ratios):.4f} | {sorted(nph_ratios)[len(nph_ratios)//2]:.4f} |\n")

            if non_nph_results:
                f.write(f"\n### 非 NPH 案例統計 (n={len(non_nph_results)})\n\n")

                if indicator_type == "volume_surface_ratio":
                    non_nph_total_volumes = [r[config['total_volume_field']] for r in non_nph_results]
                    non_nph_total_ratios = [r[config['total_ratio_field']] for r in non_nph_results]

                    f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                    f.write("|------|--------|--------|--------|--------|\n")
                    f.write(f"| {config['total_volume_label']} | {min(non_nph_total_volumes):.1f} | {max(non_nph_total_volumes):.1f} | {sum(non_nph_total_volumes)/len(non_nph_total_volumes):.1f} | {sorted(non_nph_total_volumes)[len(non_nph_total_volumes)//2]:.1f} |\n")
                    f.write(f"| {config['total_ratio_label']} | {min(non_nph_total_ratios):.4f} | {max(non_nph_total_ratios):.4f} | {sum(non_nph_total_ratios)/len(non_nph_total_ratios):.4f} | {sorted(non_nph_total_ratios)[len(non_nph_total_ratios)//2]:.4f} |\n")
                else:
                    # 原有的 distance/ratio 統計 (centroid_ratio, evan_index, alvi)
                    non_nph_distances = [r[config['distance_field']] for r in non_nph_results]
                    
                    # 根據指標類型選擇正確的寬度欄位
                    if indicator_type == 'alvi':
                        non_nph_widths = [r.get(config.get('additional_field', 'skull_ap_diameter_mm'), 0) for r in non_nph_results]
                        width_label = config.get('additional_label', '顱骨前後徑 (mm)')
                    else:
                        non_nph_widths = [r['cranial_width_mm'] for r in non_nph_results]
                        width_label = '顱內寬度 (mm)'
                    
                    non_nph_ratios = [r[config['ratio_field']] for r in non_nph_results]

                    f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                    f.write("|------|--------|--------|--------|--------|\n")
                    f.write(f"| {config['distance_label']} | {min(non_nph_distances):.2f} | {max(non_nph_distances):.2f} | {sum(non_nph_distances)/len(non_nph_distances):.2f} | {sorted(non_nph_distances)[len(non_nph_distances)//2]:.2f} |\n")
                    f.write(f"| {width_label} | {min(non_nph_widths):.2f} | {max(non_nph_widths):.2f} | {sum(non_nph_widths)/len(non_nph_widths):.2f} | {sorted(non_nph_widths)[len(non_nph_widths)//2]:.2f} |\n")
                    f.write(f"| {config['ratio_label']} | {min(non_nph_ratios):.4f} | {max(non_nph_ratios):.4f} | {sum(non_nph_ratios)/len(non_nph_ratios):.4f} | {sorted(non_nph_ratios)[len(non_nph_ratios)//2]:.4f} |\n")

            # 組間差異
            if nph_results and non_nph_results:
                f.write("\n### 組間差異\n\n")
                f.write("| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |\n")
                f.write("|-----|-----------|-------------|------|-----------|\n")

                if indicator_type == "volume_surface_ratio":
                    # 體積表面積比例的組間差異計算（簡化版）
                    nph_total_volumes = [r[config['total_volume_field']] for r in nph_results]
                    nph_total_ratios = [r[config['total_ratio_field']] for r in nph_results]
                    non_nph_total_volumes = [r[config['total_volume_field']] for r in non_nph_results]
                    non_nph_total_ratios = [r[config['total_ratio_field']] for r in non_nph_results]

                    # 總體積差異
                    nph_vol_mean = sum(nph_total_volumes) / len(nph_total_volumes)
                    non_nph_vol_mean = sum(non_nph_total_volumes) / len(non_nph_total_volumes)
                    vol_diff = nph_vol_mean - non_nph_vol_mean
                    vol_pct = (vol_diff / non_nph_vol_mean) * 100
                    f.write(f"| {config['total_volume_label']} | {nph_vol_mean:.1f} | {non_nph_vol_mean:.1f} | {vol_diff:+.1f} | {vol_pct:+.1f}% |\n")

                    # V/SA 比例差異
                    nph_ratio_mean = sum(nph_total_ratios) / len(nph_total_ratios)
                    non_nph_ratio_mean = sum(non_nph_total_ratios) / len(non_nph_total_ratios)
                    ratio_diff = nph_ratio_mean - non_nph_ratio_mean
                    ratio_pct = (ratio_diff / non_nph_ratio_mean) * 100
                    f.write(f"| **{config['total_ratio_label']}** | **{nph_ratio_mean:.4f}** | **{non_nph_ratio_mean:.4f}** | **{ratio_diff:+.4f}** | **{ratio_pct:+.1f}%** |\n")
                else:
                    # 原有的 distance/ratio 組間差異計算
                    nph_distances = [r[config['distance_field']] for r in nph_results]
                    non_nph_distances = [r[config['distance_field']] for r in non_nph_results]
                    
                    # 根據指標類型選擇正確的寬度欄位
                    if indicator_type == 'alvi':
                        nph_widths = [r.get(config.get('additional_field', 'skull_ap_diameter_mm'), 0) for r in nph_results]
                        non_nph_widths = [r.get(config.get('additional_field', 'skull_ap_diameter_mm'), 0) for r in non_nph_results]
                        width_label = config.get('additional_label', '顱骨前後徑')
                    else:
                        nph_widths = [r['cranial_width_mm'] for r in nph_results]
                        non_nph_widths = [r['cranial_width_mm'] for r in non_nph_results]
                        width_label = '顱內寬度'
                    
                    nph_ratios = [r[config['ratio_field']] for r in nph_results]
                    non_nph_ratios = [r[config['ratio_field']] for r in non_nph_results]

                    nph_dist_mean = sum(nph_distances) / len(nph_distances)
                    non_nph_dist_mean = sum(non_nph_distances) / len(non_nph_distances)
                    dist_diff = nph_dist_mean - non_nph_dist_mean
                    dist_pct = (dist_diff / non_nph_dist_mean) * 100
                    f.write(f"| {config['distance_label']} | {nph_dist_mean:.2f} mm | {non_nph_dist_mean:.2f} mm | {dist_diff:+.2f} mm | {dist_pct:+.1f}% |\n")

                    nph_width_mean = sum(nph_widths) / len(nph_widths)
                    non_nph_width_mean = sum(non_nph_widths) / len(non_nph_widths)
                    width_diff = nph_width_mean - non_nph_width_mean
                    width_pct = (width_diff / non_nph_width_mean) * 100
                    f.write(f"| {width_label} | {nph_width_mean:.2f} mm | {non_nph_width_mean:.2f} mm | {width_diff:+.2f} mm | {width_pct:+.1f}% |\n")

                    nph_ratio_mean = sum(nph_ratios) / len(nph_ratios)
                    non_nph_ratio_mean = sum(non_nph_ratios) / len(non_nph_ratios)
                    ratio_diff = nph_ratio_mean - non_nph_ratio_mean
                    ratio_pct = (ratio_diff / non_nph_ratio_mean) * 100
                    f.write(f"| **{config['ratio_label']}** | **{nph_ratio_mean:.4f}** | **{non_nph_ratio_mean:.4f}** | **{ratio_diff:+.4f}** | **{ratio_pct:+.1f}%** |\n")

            # NPH 案例詳細列表
            if nph_results:
                f.write("\n## NPH 案例詳細數據\n\n")

                if indicator_type == "volume_surface_ratio":
                    f.write(f"| 案例 ID | {config['total_volume_label']} | {config['total_area_label']} | {config['total_ratio_label']} | 排序 |\n")
                    f.write("|---------|----------------|----------------|----------------|------|\n")

                    nph_sorted = sorted(nph_results, key=lambda x: x[config['total_ratio_field']], reverse=True)
                    for i, result in enumerate(nph_sorted, 1):
                        case_id = result.get('case_id', 'N/A')
                        total_volume = result.get(config['total_volume_field'], 0)
                        total_area = result.get(config['total_area_field'], 0)
                        total_ratio = result.get(config['total_ratio_field'], 0)

                        rank_note = ""
                        if i == 1:
                            rank_note = " (最高)"
                        elif i == len(nph_sorted):
                            rank_note = " (最低)"

                        f.write(f"| {case_id} | {total_volume:.1f} | {total_area:.1f} | {total_ratio:.4f} | {i}{rank_note} |\n")
                else:
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
