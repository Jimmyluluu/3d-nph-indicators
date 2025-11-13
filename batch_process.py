#!/usr/bin/env python3
"""
批次處理腳本 - 統一處理不同 NPH 指標
支援：
1. centroid_ratio - 腦室質心距離/顱內寬度比值
2. evan_index - 腦室前腳最大距離/顱內寬度比值（3D Evan Index）
"""

import time
import argparse
import nibabel as nib
from pathlib import Path
from datetime import datetime
from model.calculation import (
    load_ventricle_pair,
    calculate_centroid_distance,
    calculate_cranial_width,
    calculate_ventricle_to_cranial_ratio,
    calculate_3d_evan_index
)
from model.visualization import (
    visualize_ventricle_distance,
    visualize_3d_evan_index,
    print_measurement_summary,
    print_evan_index_summary
)
from model.data_export import DataExporter, ProcessLogger


def scan_data_directory(base_dir, skip_not_ok=True):
    """
    掃描資料目錄，找出所有有效的案例資料夾

    Args:
        base_dir: 基礎資料目錄路徑
        skip_not_ok: 是否跳過標記為 _not_ok 的資料夾

    Returns:
        list: 案例資料夾路徑列表
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"資料目錄不存在: {base_dir}")

    # 取得所有子目錄
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    # 過濾掉隱藏檔案（._ 開頭）和 _not_ok 標記的資料夾
    valid_dirs = []
    for d in all_dirs:
        # 跳過隱藏目錄
        if d.name.startswith('.'):
            continue

        # 如果需要跳過 _not_ok
        if skip_not_ok and '_not_ok' in d.name:
            continue

        # 檢查是否包含必要的檔案（兩種命名模式）
        # 模式 1: 標準命名
        required_files_pattern1 = [
            d / "Ventricle_L.nii.gz",
            d / "Ventricle_R.nii.gz",
            d / "original.nii.gz"
        ]

        # 模式 2: data_ 開頭的命名
        if d.name.startswith('data_'):
            data_num = d.name.replace('data_', '')
            required_files_pattern2 = [
                d / f"mask_Ventricle_L_{data_num}.nii.gz",
                d / f"mask_Ventricle_R_{data_num}.nii.gz",
                d / f"original_{data_num}.nii.gz"
            ]
        else:
            required_files_pattern2 = []

        # 檢查任一模式是否存在
        if all(f.exists() for f in required_files_pattern1):
            valid_dirs.append(d)
        elif required_files_pattern2 and all(f.exists() for f in required_files_pattern2):
            valid_dirs.append(d)

    return sorted(valid_dirs)


def process_case_indicator_ratio(data_dir, output_image_path, show_plot=False, verbose=True):
    """
    處理單一案例 - 質心距離比值指標

    Args:
        data_dir: 資料目錄路徑
        output_image_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        verbose: 是否顯示詳細資訊

    Returns:
        dict: 包含所有測量結果的字典
    """
    try:
        data_path = Path(data_dir)
        case_name = data_path.name

        # 找檔案（支援兩種命名模式）
        left_path = data_path / "Ventricle_L.nii.gz"
        right_path = data_path / "Ventricle_R.nii.gz"
        original_path = data_path / "original.nii.gz"

        if case_name.startswith('data_'):
            data_num = case_name.replace('data_', '')
            left_path_alt = data_path / f"mask_Ventricle_L_{data_num}.nii.gz"
            right_path_alt = data_path / f"mask_Ventricle_R_{data_num}.nii.gz"
            original_path_alt = data_path / f"original_{data_num}.nii.gz"

            if left_path_alt.exists():
                left_path = left_path_alt
                right_path = right_path_alt
                original_path = original_path_alt

        # 載入腦室影像
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 計算質心距離
        distance_mm, left_centroid, right_centroid, voxel_size = calculate_centroid_distance(
            left_vent, right_vent
        )

        # 計算顱內寬度
        original_brain = nib.load(str(original_path))
        cranial_width, left_point, right_point, slice_idx = calculate_cranial_width(original_brain)

        # 計算比值
        ratio = calculate_ventricle_to_cranial_ratio(distance_mm, cranial_width)

        # 輸出摘要
        if verbose:
            print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size,
                                     cranial_width_mm=cranial_width, ratio=ratio)

        # 視覺化
        cranial_width_data = (cranial_width, left_point, right_point, slice_idx)
        visualize_ventricle_distance(
            left_vent, right_vent,
            left_centroid, right_centroid,
            distance_mm,
            output_path=str(output_image_path),
            show_plot=show_plot,
            original_path=str(original_path),
            cranial_width_data=cranial_width_data,
            ratio=ratio
        )

        # 返回成功結果
        return {
            'status': 'success',
            'ventricle_distance_mm': distance_mm,
            'cranial_width_mm': cranial_width,
            'ratio': ratio,
            'ratio_percent': ratio * 100,
            'left_centroid': list(left_centroid),
            'right_centroid': list(right_centroid),
            'voxel_size': list(voxel_size),
            'cranial_width_endpoints': {
                'left': list(left_point),
                'right': list(right_point),
                'slice_index': slice_idx
            }
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }


def process_case_evan_index(data_dir, output_image_path, show_plot=False, verbose=True,
                            z_range=(0.4, 0.6), y_percentile=40):
    """
    處理單一案例 - 3D Evan Index

    Args:
        data_dir: 資料目錄路徑
        output_image_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        verbose: 是否顯示詳細資訊
        z_range: Z 軸切面範圍
        y_percentile: Y 軸前方百分位數

    Returns:
        dict: 包含所有測量結果的字典
    """
    try:
        data_path = Path(data_dir)
        case_name = data_path.name

        # 找檔案（支援兩種命名模式）
        left_path = data_path / "Ventricle_L.nii.gz"
        right_path = data_path / "Ventricle_R.nii.gz"
        original_path = data_path / "original.nii.gz"

        if case_name.startswith('data_'):
            data_num = case_name.replace('data_', '')
            left_path_alt = data_path / f"mask_Ventricle_L_{data_num}.nii.gz"
            right_path_alt = data_path / f"mask_Ventricle_R_{data_num}.nii.gz"
            original_path_alt = data_path / f"original_{data_num}.nii.gz"

            if left_path_alt.exists():
                left_path = left_path_alt
                right_path = right_path_alt
                original_path = original_path_alt

        # 載入腦室影像
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 載入原始影像
        original_img = nib.load(str(original_path))

        # 計算 3D Evan Index
        evan_data = calculate_3d_evan_index(
            left_vent, right_vent, original_img,
            z_range=z_range, y_percentile=y_percentile, verbose=verbose
        )

        # 輸出摘要
        if verbose:
            print_evan_index_summary(evan_data)

        # 視覺化
        visualize_3d_evan_index(
            left_vent, right_vent,
            str(original_path),
            evan_data,
            output_path=str(output_image_path),
            show_plot=show_plot,
            z_range=z_range,
            y_percentile=y_percentile
        )

        # 返回成功結果
        return {
            'status': 'success',
            'anterior_horn_distance_mm': evan_data['anterior_horn_distance_mm'],
            'cranial_width_mm': evan_data['cranial_width_mm'],
            'evan_index': evan_data['evan_index'],
            'evan_index_percent': evan_data['evan_index_percent'],
            'anterior_horn_endpoints': evan_data['anterior_horn_endpoints'],
            'cranial_width_endpoints': evan_data['cranial_width_endpoints'],
            'anterior_horn_points_count': evan_data['anterior_horn_points_count'],
            'voxel_size': evan_data['voxel_size']
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }


def format_time(seconds):
    """格式化時間顯示"""
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
        NPH 案例 ID 集合
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


def generate_markdown_report_indicator_ratio(results, output_path, total_time, success_count, error_count):
    """產生質心距離比值的 Markdown 報表"""
    nph_cases = load_nph_list()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 腦室質心距離比值批次處理報表\n\n")
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
            f.write("| 案例 ID | 腦室距離 (mm) | 顱內寬度 (mm) | 比值 | 百分比 | 處理時間 |\n")
            f.write("|---------|---------------|---------------|------|--------|----------|\n")

            for result in successful_results:
                case_id = result.get('case_id', 'N/A')
                distance = result.get('ventricle_distance_mm', 0)
                width = result.get('cranial_width_mm', 0)
                ratio = result.get('ratio', 0)
                percent = result.get('ratio_percent', 0)
                time_str = result.get('processing_time', 'N/A')

                if case_id in nph_cases:
                    case_id_display = f"{case_id} ⚠️ NPH"
                else:
                    case_id_display = case_id

                f.write(f"| {case_id_display} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {time_str} |\n")

            # 統計資訊
            distances = [r['ventricle_distance_mm'] for r in successful_results]
            widths = [r['cranial_width_mm'] for r in successful_results]
            ratios = [r['ratio'] for r in successful_results]

            f.write("\n### 統計數據（全部案例）\n\n")
            f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            f.write(f"| 腦室距離 (mm) | {min(distances):.2f} | {max(distances):.2f} | {sum(distances)/len(distances):.2f} | {sorted(distances)[len(distances)//2]:.2f} |\n")
            f.write(f"| 顱內寬度 (mm) | {min(widths):.2f} | {max(widths):.2f} | {sum(widths)/len(widths):.2f} | {sorted(widths)[len(widths)//2]:.2f} |\n")
            f.write(f"| 比值 | {min(ratios):.4f} | {max(ratios):.4f} | {sum(ratios)/len(ratios):.4f} | {sorted(ratios)[len(ratios)//2]:.4f} |\n")

            # NPH 和非 NPH 分組統計
            nph_results = [r for r in successful_results if r.get('case_id') in nph_cases]
            non_nph_results = [r for r in successful_results if r.get('case_id') not in nph_cases]

            if nph_results:
                f.write(f"\n### NPH 案例統計 (n={len(nph_results)})\n\n")
                nph_distances = [r['ventricle_distance_mm'] for r in nph_results]
                nph_widths = [r['cranial_width_mm'] for r in nph_results]
                nph_ratios = [r['ratio'] for r in nph_results]

                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| 腦室距離 (mm) | {min(nph_distances):.2f} | {max(nph_distances):.2f} | {sum(nph_distances)/len(nph_distances):.2f} | {sorted(nph_distances)[len(nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(nph_widths):.2f} | {max(nph_widths):.2f} | {sum(nph_widths)/len(nph_widths):.2f} | {sorted(nph_widths)[len(nph_widths)//2]:.2f} |\n")
                f.write(f"| 比值 | {min(nph_ratios):.4f} | {max(nph_ratios):.4f} | {sum(nph_ratios)/len(nph_ratios):.4f} | {sorted(nph_ratios)[len(nph_ratios)//2]:.4f} |\n")

            if non_nph_results:
                f.write(f"\n### 非 NPH 案例統計 (n={len(non_nph_results)})\n\n")
                non_nph_distances = [r['ventricle_distance_mm'] for r in non_nph_results]
                non_nph_widths = [r['cranial_width_mm'] for r in non_nph_results]
                non_nph_ratios = [r['ratio'] for r in non_nph_results]

                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| 腦室距離 (mm) | {min(non_nph_distances):.2f} | {max(non_nph_distances):.2f} | {sum(non_nph_distances)/len(non_nph_distances):.2f} | {sorted(non_nph_distances)[len(non_nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(non_nph_widths):.2f} | {max(non_nph_widths):.2f} | {sum(non_nph_widths)/len(non_nph_widths):.2f} | {sorted(non_nph_widths)[len(non_nph_widths)//2]:.2f} |\n")
                f.write(f"| 比值 | {min(non_nph_ratios):.4f} | {max(non_nph_ratios):.4f} | {sum(non_nph_ratios)/len(non_nph_ratios):.4f} | {sorted(non_nph_ratios)[len(non_nph_ratios)//2]:.4f} |\n")

            # 組間差異
            if nph_results and non_nph_results:
                f.write("\n### 組間差異\n\n")
                f.write("| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |\n")
                f.write("|-----|-----------|-------------|------|-----------|\n")

                nph_dist_mean = sum(nph_distances) / len(nph_distances)
                non_nph_dist_mean = sum(non_nph_distances) / len(non_nph_distances)
                dist_diff = nph_dist_mean - non_nph_dist_mean
                dist_pct = (dist_diff / non_nph_dist_mean) * 100
                f.write(f"| 腦室距離 | {nph_dist_mean:.2f} mm | {non_nph_dist_mean:.2f} mm | {dist_diff:+.2f} mm | {dist_pct:+.1f}% |\n")

                nph_width_mean = sum(nph_widths) / len(nph_widths)
                non_nph_width_mean = sum(non_nph_widths) / len(non_nph_widths)
                width_diff = nph_width_mean - non_nph_width_mean
                width_pct = (width_diff / non_nph_width_mean) * 100
                f.write(f"| 顱內寬度 | {nph_width_mean:.2f} mm | {non_nph_width_mean:.2f} mm | {width_diff:+.2f} mm | {width_pct:+.1f}% |\n")

                nph_ratio_mean = sum(nph_ratios) / len(nph_ratios)
                non_nph_ratio_mean = sum(non_nph_ratios) / len(non_nph_ratios)
                ratio_diff = nph_ratio_mean - non_nph_ratio_mean
                ratio_pct = (ratio_diff / non_nph_ratio_mean) * 100
                f.write(f"| **比值** | **{nph_ratio_mean:.4f}** | **{non_nph_ratio_mean:.4f}** | **{ratio_diff:+.4f}** | **{ratio_pct:+.1f}%** |\n")

            # NPH 案例詳細列表
            if nph_results:
                f.write("\n## NPH 案例詳細數據\n\n")
                f.write("| 案例 ID | 腦室距離 (mm) | 顱內寬度 (mm) | 比值 | 百分比 | 排序 |\n")
                f.write("|---------|---------------|---------------|------|--------|------|\n")

                nph_sorted = sorted(nph_results, key=lambda x: x['ratio'], reverse=True)
                for i, result in enumerate(nph_sorted, 1):
                    case_id = result.get('case_id', 'N/A')
                    distance = result.get('ventricle_distance_mm', 0)
                    width = result.get('cranial_width_mm', 0)
                    ratio = result.get('ratio', 0)
                    percent = result.get('ratio_percent', 0)

                    rank_note = ""
                    if i == 1:
                        rank_note = " (最高)"
                    elif i == len(nph_sorted):
                        rank_note = " (最低)"

                    f.write(f"| {case_id} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {i}{rank_note} |\n")

            # 非 NPH 高比值案例
            non_nph_high = [r for r in non_nph_results if r['ratio'] > 0.25]
            if non_nph_high:
                f.write(f"\n## 非 NPH 高比值案例 (> 25%, n={len(non_nph_high)})\n\n")
                f.write("| 案例 ID | 腦室距離 (mm) | 顱內寬度 (mm) | 比值 | 百分比 | 建議 |\n")
                f.write("|---------|---------------|---------------|------|--------|------|\n")

                non_nph_high_sorted = sorted(non_nph_high, key=lambda x: x['ratio'], reverse=True)
                for result in non_nph_high_sorted:
                    case_id = result.get('case_id', 'N/A')
                    distance = result.get('ventricle_distance_mm', 0)
                    width = result.get('cranial_width_mm', 0)
                    ratio = result.get('ratio', 0)
                    percent = result.get('ratio_percent', 0)

                    suggestion = "建議密切追蹤"
                    if ratio >= 0.28:
                        suggestion = "**建議重新評估**"

                    f.write(f"| {case_id} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {suggestion} |\n")

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

        f.write("\n---\n\n*由 3D NPH Indicators 自動產生*\n")


def generate_markdown_report_evan_index(results, output_path, total_time, success_count, error_count):
    """產生 3D Evan Index 的 Markdown 報表"""
    nph_cases = load_nph_list()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 3D Evan Index 批次處理報表\n\n")
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
            f.write("| 案例 ID | 前腳距離 (mm) | 顱內寬度 (mm) | Evan Index | 百分比 | 處理時間 |\n")
            f.write("|---------|---------------|---------------|------------|--------|----------|\n")

            for result in successful_results:
                case_id = result.get('case_id', 'N/A')
                distance = result.get('anterior_horn_distance_mm', 0)
                width = result.get('cranial_width_mm', 0)
                evan_index = result.get('evan_index', 0)
                percent = result.get('evan_index_percent', 0)
                time_str = result.get('processing_time', 'N/A')

                if case_id in nph_cases:
                    case_id_display = f"{case_id} ⚠️ NPH"
                else:
                    case_id_display = case_id

                f.write(f"| {case_id_display} | {distance:.2f} | {width:.2f} | {evan_index:.4f} | {percent:.2f}% | {time_str} |\n")

            # 統計資訊
            distances = [r['anterior_horn_distance_mm'] for r in successful_results]
            widths = [r['cranial_width_mm'] for r in successful_results]
            evan_indices = [r['evan_index'] for r in successful_results]

            f.write("\n### 統計數據（全部案例）\n\n")
            f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            f.write(f"| 前腳距離 (mm) | {min(distances):.2f} | {max(distances):.2f} | {sum(distances)/len(distances):.2f} | {sorted(distances)[len(distances)//2]:.2f} |\n")
            f.write(f"| 顱內寬度 (mm) | {min(widths):.2f} | {max(widths):.2f} | {sum(widths)/len(widths):.2f} | {sorted(widths)[len(widths)//2]:.2f} |\n")
            f.write(f"| Evan Index | {min(evan_indices):.4f} | {max(evan_indices):.4f} | {sum(evan_indices)/len(evan_indices):.4f} | {sorted(evan_indices)[len(evan_indices)//2]:.4f} |\n")

            # NPH 和非 NPH 分組統計
            nph_results = [r for r in successful_results if r.get('case_id') in nph_cases]
            non_nph_results = [r for r in successful_results if r.get('case_id') not in nph_cases]

            if nph_results:
                f.write(f"\n### NPH 案例統計 (n={len(nph_results)})\n\n")
                nph_distances = [r['anterior_horn_distance_mm'] for r in nph_results]
                nph_widths = [r['cranial_width_mm'] for r in nph_results]
                nph_indices = [r['evan_index'] for r in nph_results]

                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| 前腳距離 (mm) | {min(nph_distances):.2f} | {max(nph_distances):.2f} | {sum(nph_distances)/len(nph_distances):.2f} | {sorted(nph_distances)[len(nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(nph_widths):.2f} | {max(nph_widths):.2f} | {sum(nph_widths)/len(nph_widths):.2f} | {sorted(nph_widths)[len(nph_widths)//2]:.2f} |\n")
                f.write(f"| Evan Index | {min(nph_indices):.4f} | {max(nph_indices):.4f} | {sum(nph_indices)/len(nph_indices):.4f} | {sorted(nph_indices)[len(nph_indices)//2]:.4f} |\n")

            if non_nph_results:
                f.write(f"\n### 非 NPH 案例統計 (n={len(non_nph_results)})\n\n")
                non_nph_distances = [r['anterior_horn_distance_mm'] for r in non_nph_results]
                non_nph_widths = [r['cranial_width_mm'] for r in non_nph_results]
                non_nph_indices = [r['evan_index'] for r in non_nph_results]

                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| 前腳距離 (mm) | {min(non_nph_distances):.2f} | {max(non_nph_distances):.2f} | {sum(non_nph_distances)/len(non_nph_distances):.2f} | {sorted(non_nph_distances)[len(non_nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(non_nph_widths):.2f} | {max(non_nph_widths):.2f} | {sum(non_nph_widths)/len(non_nph_widths):.2f} | {sorted(non_nph_widths)[len(non_nph_widths)//2]:.2f} |\n")
                f.write(f"| Evan Index | {min(non_nph_indices):.4f} | {max(non_nph_indices):.4f} | {sum(non_nph_indices)/len(non_nph_indices):.4f} | {sorted(non_nph_indices)[len(non_nph_indices)//2]:.4f} |\n")

            # 組間差異
            if nph_results and non_nph_results:
                f.write("\n### 組間差異\n\n")
                f.write("| 指標 | NPH 平均值 | 非 NPH 平均值 | 差異 | 差異百分比 |\n")
                f.write("|-----|-----------|-------------|------|-----------|\n")

                nph_dist_mean = sum(nph_distances) / len(nph_distances)
                non_nph_dist_mean = sum(non_nph_distances) / len(non_nph_distances)
                dist_diff = nph_dist_mean - non_nph_dist_mean
                dist_pct = (dist_diff / non_nph_dist_mean) * 100
                f.write(f"| 前腳距離 | {nph_dist_mean:.2f} mm | {non_nph_dist_mean:.2f} mm | {dist_diff:+.2f} mm | {dist_pct:+.1f}% |\n")

                nph_width_mean = sum(nph_widths) / len(nph_widths)
                non_nph_width_mean = sum(non_nph_widths) / len(non_nph_widths)
                width_diff = nph_width_mean - non_nph_width_mean
                width_pct = (width_diff / non_nph_width_mean) * 100
                f.write(f"| 顱內寬度 | {nph_width_mean:.2f} mm | {non_nph_width_mean:.2f} mm | {width_diff:+.2f} mm | {width_pct:+.1f}% |\n")

                nph_index_mean = sum(nph_indices) / len(nph_indices)
                non_nph_index_mean = sum(non_nph_indices) / len(non_nph_indices)
                index_diff = nph_index_mean - non_nph_index_mean
                index_pct = (index_diff / non_nph_index_mean) * 100
                f.write(f"| **Evan Index** | **{nph_index_mean:.4f}** | **{non_nph_index_mean:.4f}** | **{index_diff:+.4f}** | **{index_pct:+.1f}%** |\n")

            # NPH 案例詳細列表
            if nph_results:
                f.write("\n## NPH 案例詳細數據\n\n")
                f.write("| 案例 ID | 前腳距離 (mm) | 顱內寬度 (mm) | Evan Index | 百分比 | 排序 |\n")
                f.write("|---------|---------------|---------------|------------|--------|------|\n")

                nph_sorted = sorted(nph_results, key=lambda x: x['evan_index'], reverse=True)
                for i, result in enumerate(nph_sorted, 1):
                    case_id = result.get('case_id', 'N/A')
                    distance = result.get('anterior_horn_distance_mm', 0)
                    width = result.get('cranial_width_mm', 0)
                    evan_index = result.get('evan_index', 0)
                    percent = result.get('evan_index_percent', 0)

                    rank_note = ""
                    if i == 1:
                        rank_note = " (最高)"
                    elif i == len(nph_sorted):
                        rank_note = " (最低)"

                    f.write(f"| {case_id} | {distance:.2f} | {width:.2f} | {evan_index:.4f} | {percent:.2f}% | {i}{rank_note} |\n")

            # 非 NPH 高 Evan Index 案例
            non_nph_high = [r for r in non_nph_results if r['evan_index'] > 0.30]
            if non_nph_high:
                f.write(f"\n## 非 NPH 高 Evan Index 案例 (> 30%, n={len(non_nph_high)})\n\n")
                f.write("| 案例 ID | 前腳距離 (mm) | 顱內寬度 (mm) | Evan Index | 百分比 | 建議 |\n")
                f.write("|---------|---------------|---------------|------------|--------|------|\n")

                non_nph_high_sorted = sorted(non_nph_high, key=lambda x: x['evan_index'], reverse=True)
                for result in non_nph_high_sorted:
                    case_id = result.get('case_id', 'N/A')
                    distance = result.get('anterior_horn_distance_mm', 0)
                    width = result.get('cranial_width_mm', 0)
                    evan_index = result.get('evan_index', 0)
                    percent = result.get('evan_index_percent', 0)

                    suggestion = "建議密切追蹤"
                    if evan_index >= 0.35:
                        suggestion = "**建議重新評估**"

                    f.write(f"| {case_id} | {distance:.2f} | {width:.2f} | {evan_index:.4f} | {percent:.2f}% | {suggestion} |\n")

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

        f.write("\n---\n\n*由 3D Evan Index Calculator 自動產生*\n")


def batch_process(data_dir, indicator_type="centroid_ratio", skip_not_ok=True,
                  z_range=(0.4, 0.6), y_percentile=40):
    """
    批次處理所有案例

    Args:
        data_dir: 資料目錄路徑
        indicator_type: 指標類型 ("centroid_ratio" 或 "evan_index")
        skip_not_ok: 是否跳過標記為 _not_ok 的資料夾
        z_range: Z 軸切面範圍（僅用於 evan_index）
        y_percentile: Y 軸前方百分位數（僅用於 evan_index）
    """
    # 設定輸出目錄
    output_dir = f"result/{indicator_type}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "processing.log"

    # 選擇處理函數和報表生成函數
    if indicator_type == "centroid_ratio":
        process_func = process_case_indicator_ratio
        report_func = generate_markdown_report_indicator_ratio
        indicator_name = "腦室質心距離比值"
    elif indicator_type == "evan_index":
        process_func = lambda d, o, s, v: process_case_evan_index(d, o, s, v, z_range, y_percentile)
        report_func = generate_markdown_report_evan_index
        indicator_name = "3D Evan Index"
    else:
        raise ValueError(f"不支援的指標類型: {indicator_type}")

    with ProcessLogger(log_file) as logger:
        logger.info(f"開始批次處理 - {indicator_name}")
        logger.info(f"資料目錄: {data_dir}")
        logger.info(f"輸出目錄: {output_dir}")
        logger.info(f"跳過 _not_ok: {skip_not_ok}")

        if indicator_type == "evan_index":
            logger.info(f"前腳定義 - Z 範圍: {z_range}, Y 百分位: {y_percentile}")

        # 掃描資料目錄
        try:
            case_dirs = scan_data_directory(data_dir, skip_not_ok=skip_not_ok)
            total_cases = len(case_dirs)

            if total_cases == 0:
                logger.warning("沒有找到有效的案例資料夾！")
                return

            logger.info(f"找到 {total_cases} 個有效案例")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"掃描資料目錄失敗: {str(e)}", e)
            return

        # 處理統計
        results = []
        success_count = 0
        error_count = 0
        start_time = time.time()

        # 逐一處理每個案例
        for i, case_dir in enumerate(case_dirs, 1):
            case_id = case_dir.name
            case_start_time = time.time()

            logger.info(f"\n[{i}/{total_cases}] 處理案例: {case_id}")

            try:
                # 為每個案例建立獨立資料夾
                case_output_dir = output_path / case_id
                case_output_dir.mkdir(exist_ok=True)

                output_image_path = case_output_dir / f"{case_id}.png"

                # 處理案例
                result = process_func(
                    str(case_dir),
                    str(output_image_path),
                    show_plot=False,
                    verbose=False
                )

                processing_time = time.time() - case_start_time
                result['processing_time'] = f"{processing_time:.2f}s"
                result['case_id'] = case_id

                # 記錄結果
                if result['status'] == 'success':
                    success_count += 1
                    logger.success(f"  ✓ 成功")

                    if indicator_type == "centroid_ratio":
                        logger.info(f"     腦室距離: {result['ventricle_distance_mm']:.2f} mm")
                        logger.info(f"     顱內寬度: {result['cranial_width_mm']:.2f} mm")
                        logger.info(f"     比值: {result['ratio']:.4f} ({result['ratio_percent']:.2f}%)")
                    else:  # evan_index
                        logger.info(f"     前腳距離: {result['anterior_horn_distance_mm']:.2f} mm")
                        logger.info(f"     顱內寬度: {result['cranial_width_mm']:.2f} mm")
                        logger.info(f"     Evan Index: {result['evan_index']:.4f} ({result['evan_index_percent']:.2f}%)")

                    logger.info(f"     處理時間: {processing_time:.1f}s")
                else:
                    error_count += 1
                    logger.error(f"  ✗ 失敗 - {result.get('error_message', '未知錯誤')}", None)

                results.append(result)

                # 顯示進度
                elapsed_time = time.time() - start_time
                avg_time_per_case = elapsed_time / i
                remaining_cases = total_cases - i
                estimated_remaining = avg_time_per_case * remaining_cases

                logger.info(
                    f"  進度: {i}/{total_cases} ({i/total_cases*100:.1f}%) | "
                    f"成功: {success_count} | 失敗: {error_count} | "
                    f"預估剩餘: {format_time(estimated_remaining)}"
                )

            except Exception as e:
                error_count += 1
                processing_time = time.time() - case_start_time
                logger.error(f"  處理案例時發生例外: {str(e)}", e)

                results.append({
                    'case_id': case_id,
                    'status': 'error',
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'processing_time': f"{processing_time:.2f}s"
                })

        # 處理完成
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("批次處理完成！")
        logger.info(f"總共處理: {total_cases} 個案例")
        logger.info(f"成功: {success_count} 個")
        logger.info(f"失敗: {error_count} 個")
        logger.info(f"總耗時: {format_time(total_time)}")
        logger.info(f"平均每案例: {format_time(total_time/total_cases)}")

        # 產生 Markdown 報表
        logger.info("\n產生 Markdown 報表...")
        md_path = output_path / "results_summary.md"
        report_func(results, md_path, total_time, success_count, error_count)
        logger.success(f"Markdown 報表已儲存: {md_path}")

        logger.info("\n結果檔案位置:")
        logger.info(f"  案例資料夾: {output_path}/<case_id>/")
        logger.info(f"    - <case_id>.png (圖片)")
        logger.info(f"    - <case_id>.html (互動式圖表)")
        logger.info(f"  報表: {md_path}")
        logger.info(f"  日誌: {log_file}")
        logger.info("=" * 70)


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='3D NPH 指標批次處理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
指標類型說明:
  centroid_ratio  - 腦室質心距離/顱內寬度比值（預設）
  evan_index      - 腦室前腳最大距離/顱內寬度比值（3D Evan Index）

使用範例:
  python batch_process.py --type centroid_ratio
  python batch_process.py --type evan_index
  python batch_process.py --type evan_index --z-range 0.3 0.7 --y-percentile 50
        """
    )

    parser.add_argument(
        '--type', '-t',
        choices=['centroid_ratio', 'evan_index'],
        default='centroid_ratio',
        help='指標類型（預設: centroid_ratio）'
    )

    parser.add_argument(
        '--data-dir', '-d',
        default='/Volumes/Kuro醬の1TSSD/標記好的資料',
        help='資料目錄路徑'
    )

    parser.add_argument(
        '--skip-not-ok',
        action='store_true',
        default=True,
        help='跳過標記為 _not_ok 的資料夾（預設: True）'
    )

    parser.add_argument(
        '--z-range',
        nargs=2,
        type=float,
        default=[0.4, 0.6],
        metavar=('MIN', 'MAX'),
        help='Z 軸切面範圍（僅用於 evan_index，預設: 0.4 0.6）'
    )

    parser.add_argument(
        '--y-percentile',
        type=int,
        default=40,
        help='Y 軸前方百分位數（僅用於 evan_index，預設: 40）'
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"3D NPH 指標批次處理工具")
    print("=" * 70)
    print(f"指標類型: {args.type}")
    print(f"資料目錄: {args.data_dir}")
    print(f"輸出目錄: result/{args.type}")
    print(f"跳過 _not_ok: {'是' if args.skip_not_ok else '否'}")

    if args.type == 'evan_index':
        print(f"前腳定義 - Z 範圍: {args.z_range[0]*100}%-{args.z_range[1]*100}%, Y 百分位: {args.y_percentile}%")

    print("=" * 70)
    print()

    # 執行批次處理
    batch_process(
        data_dir=args.data_dir,
        indicator_type=args.type,
        skip_not_ok=args.skip_not_ok,
        z_range=tuple(args.z_range),
        y_percentile=args.y_percentile

    )


if __name__ == "__main__":
    main()
