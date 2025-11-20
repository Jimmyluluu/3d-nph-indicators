#!/usr/bin/env python3
"""
單案例處理模組
負責處理個別案例的指標計算
"""

from pathlib import Path
from model.calculation import (
    load_ventricle_pair,
    load_original_image,
    calculate_centroid_distance,
    calculate_cranial_width,
    calculate_ventricle_to_cranial_ratio,
    calculate_3d_evan_index,
    calculate_surface_area
)
from model.visualization import (
    visualize_ventricle_distance,
    visualize_3d_evan_index,
    print_measurement_summary,
    print_evan_index_summary,
    print_surface_area_summary
)


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

        # 載入腦室影像（自動拉正到 RAS+ 方向）
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 載入原始影像（自動拉正到 RAS+ 方向）
        original_brain = load_original_image(str(original_path), verbose=verbose)

        # 計算質心距離
        distance_mm, left_centroid, right_centroid, voxel_size = calculate_centroid_distance(
            left_vent, right_vent
        )

        # 計算顱內寬度
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
            original_img=original_brain,
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
                            z_range=(0.3, 0.9), y_percentile=4):
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

        # 載入腦室影像（自動拉正到 RAS+ 方向）
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 載入原始影像（自動拉正到 RAS+ 方向）
        original_img = load_original_image(str(original_path), verbose=verbose)

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
            original_img,
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


def process_case_surface_area(data_dir, output_image_path, show_plot=False, verbose=True):
    """
    處理單一案例 - 腦室表面積

    Args:
        data_dir (str): 資料目錄路徑
        output_image_path (str): 輸出圖片路徑
        show_plot (bool): 是否顯示互動式圖表
        verbose (bool): 是否顯示詳細資訊

    Returns:
        dict: 包含所有測量結果的字典
    """
    try:
        data_path = Path(data_dir)
        case_name = data_path.name

        # 找檔案
        left_path = data_path / "Ventricle_L.nii.gz"
        right_path = data_path / "Ventricle_R.nii.gz"
        
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(f"在 {data_dir} 中找不到腦室檔案 Ventricle_L.nii.gz 或 Ventricle_R.nii.gz")

        # 載入腦室影像
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 計算表面積
        surface_data = calculate_surface_area(
            left_vent, right_vent,
            verbose=verbose
        )

        # 輸出摘要
        if verbose:
            print_surface_area_summary(surface_data)

        # 返回成功結果（純計算，不需要視覺化）
        return {
            'status': 'success',
            'left_surface_area': surface_data['left_surface_area'],
            'right_surface_area': surface_data['right_surface_area'],
            'total_surface_area': surface_data['total_surface_area']
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }

