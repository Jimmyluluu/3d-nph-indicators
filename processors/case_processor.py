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
    calculate_ventricle_to_cranial_ratio,
    calculate_surface_area,
    calculate_volume_surface_ratio
)
from model.evan_analyzer import (
    calculate_3d_evan_index,
    calculate_cranial_width
)
from model.alvi_analyzer import calculate_alvi
from model.visualization import (
    visualize_ventricle_distance,
    visualize_3d_evan_index,
    visualize_volume_surface_ratio,
    visualize_alvi,
)

from processors.printers import (
    print_measurement_summary,
    print_evan_index_summary,
    print_surface_area_summary,
)


def find_case_files(data_dir, require_original=True, require_falx=False):
    """
    尋找案例資料夾中的腦室和原始影像檔案路徑
    
    支援兩種命名模式：
    - 標準命名: Ventricle_L.nii.gz, Ventricle_R.nii.gz, original.nii.gz, falx.nii.gz
    - data_ 命名: mask_Ventricle_L_{num}.nii.gz, mask_Ventricle_R_{num}.nii.gz, 
                  original_{num}.nii.gz, mask_Falx_{num}.nii.gz
    
    Args:
        data_dir: 資料目錄路徑
        require_original: 是否需要原始影像檔案
        require_falx: 是否需要 Falx 檔案
        
    Returns:
        dict: 包含檔案路徑的字典
            - 'left_path': 左腦室檔案路徑
            - 'right_path': 右腦室檔案路徑
            - 'original_path': 原始影像檔案路徑 (如果 require_original=True)
            - 'falx_path': Falx 檔案路徑 (如果 require_falx=True)
            
    Raises:
        FileNotFoundError: 如果找不到必要的檔案
    """
    data_path = Path(data_dir)
    case_name = data_path.name
    
    # 預設使用標準命名
    left_path = data_path / "Ventricle_L.nii.gz"
    right_path = data_path / "Ventricle_R.nii.gz"
    original_path = data_path / "original.nii.gz"
    falx_path = data_path / "falx.nii.gz"
    
    # 檢查是否為 data_ 開頭的命名模式
    if case_name.startswith('data_'):
        data_num = case_name.replace('data_', '')
        left_path_alt = data_path / f"mask_Ventricle_L_{data_num}.nii.gz"
        right_path_alt = data_path / f"mask_Ventricle_R_{data_num}.nii.gz"
        original_path_alt = data_path / f"original_{data_num}.nii.gz"
        falx_path_alt = data_path / f"mask_Falx_{data_num}.nii.gz"
        
        if left_path_alt.exists():
            left_path = left_path_alt
            right_path = right_path_alt
            original_path = original_path_alt
            falx_path = falx_path_alt
    
    # 驗證檔案存在
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"在 {data_dir} 中找不到腦室檔案")
    
    if require_original and not original_path.exists():
        raise FileNotFoundError(f"在 {data_dir} 中找不到原始影像檔案")
    
    if require_falx and not falx_path.exists():
        raise FileNotFoundError(f"在 {data_dir} 中找不到 Falx 檔案")
    
    result = {
        'left_path': left_path,
        'right_path': right_path,
    }
    
    if require_original:
        result['original_path'] = original_path
    
    if require_falx:
        result['falx_path'] = falx_path
    
    return result


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
        # 使用統一的檔案路徑查找函數
        files = find_case_files(data_dir, require_original=True)
        left_path = files['left_path']
        right_path = files['right_path']
        original_path = files['original_path']

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


def process_case_evan_index(data_dir, output_image_path, show_plot=False, verbose=True):
    """
    處理單一案例 - 3D Evan Index

    使用 Falx（大腦鐮）作為中線參考平面計算前腳距離（如果有 Falx 檔案）。

    Args:
        data_dir: 資料目錄路徑
        output_image_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        verbose: 是否顯示詳細資訊

    Returns:
        dict: 包含所有測量結果的字典
    """
    try:
        # 嘗試查找包含 Falx 的檔案
        try:
            files = find_case_files(data_dir, require_original=True, require_falx=True)
            has_falx = True
        except FileNotFoundError:
            # 如果找不到 Falx，退回到只載入腦室和原始影像
            files = find_case_files(data_dir, require_original=True, require_falx=False)
            has_falx = False
            if verbose:
                print("  ⚠️ 找不到 Falx 檔案，使用傳統質心方法")

        left_path = files['left_path']
        right_path = files['right_path']
        original_path = files['original_path']

        # 載入腦室影像（自動拉正到 RAS+ 方向）
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 載入原始影像（自動拉正到 RAS+ 方向）
        original_img = load_original_image(str(original_path), verbose=verbose)

        # 載入 Falx 影像（如果有的話）
        falx_img = None
        if has_falx:
            from model.calculation import load_falx_image
            falx_img = load_falx_image(str(files['falx_path']), verbose=verbose)

        # 計算 3D Evan Index（使用 Falx-based 或傳統方法）
        evan_data = calculate_3d_evan_index(
            left_vent, right_vent, original_img,
            falx_img=falx_img,
            verbose=verbose
        )

        # 輸出摘要
        if verbose:
            print_evan_index_summary(evan_data)

        # 視覺化（傳遞 Falx 平面資訊）
        falx_plane = evan_data.get('falx_plane', None)
        visualize_3d_evan_index(
            left_vent, right_vent,
            original_img,
            evan_data,
            output_path=str(output_image_path),
            show_plot=show_plot,
            falx_img=falx_img,
            falx_plane=falx_plane
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
            'voxel_size': evan_data['voxel_size'],
            'method': evan_data.get('method', 'centroid')
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
        # 使用統一的檔案路徑查找函數（不需要原始影像）
        files = find_case_files(data_dir, require_original=False)
        left_path = files['left_path']
        right_path = files['right_path']

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


def process_case_volume_surface_ratio(data_dir, output_image_path, show_plot=False, verbose=True):
    """
    處理單一案例 - 體積與表面積比例

    Args:
        data_dir (str): 資料目錄路徑
        output_image_path (str): 輸出圖片路徑
        show_plot (bool): 是否顯示互動式圖表
        verbose (bool): 是否顯示詳細資訊

    Returns:
        dict: 包含所有測量結果的字典
    """
    try:
        # 使用統一的檔案路徑查找函數（不需要原始影像）
        files = find_case_files(data_dir, require_original=False)
        left_path = files['left_path']
        right_path = files['right_path']

        # 載入腦室影像（自動拉正到 RAS+ 方向）
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 計算體積與表面積比例（已更新為只計算總體比例）
        ratio_data = calculate_volume_surface_ratio(
            left_vent, right_vent, verbose=verbose
        )

        # 視覺化（傳影像物件，不傳路徑）
        visualize_volume_surface_ratio(
            left_vent, right_vent,
            ratio_data,
            output_path=str(output_image_path),
            show_plot=show_plot
        )

        # 返回成功結果（已簡化，只保留總體比例）
        return {
            'status': 'success',
            'left_volume': ratio_data['left_volume'],
            'right_volume': ratio_data['right_volume'],
            'total_volume': ratio_data['total_volume'],
            'left_surface_area': ratio_data['left_surface_area'],
            'right_surface_area': ratio_data['right_surface_area'],
            'total_surface_area': ratio_data['total_surface_area'],
            'total_ratio': ratio_data['total_ratio']
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }


def process_case_alvi(data_dir, output_image_path, show_plot=False, verbose=True):
    """
    處理單一案例 - ALVI (Anteroposterior Lateral Ventricle Index)

    Args:
        data_dir: 資料目錄路徑
        output_image_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        verbose: 是否顯示詳細資訊

    Returns:
        dict: 包含所有測量結果的字典
    """
    try:
        # 使用統一的檔案路徑查找函數 (需要 Falx)
        files = find_case_files(data_dir, require_original=True, require_falx=True)
        left_path = files['left_path']
        right_path = files['right_path']
        original_path = files['original_path']
        falx_path = files['falx_path']

        # 載入腦室影像（自動拉正到 RAS+ 方向）
        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )

        # 載入原始影像（自動拉正到 RAS+ 方向）
        original_img = load_original_image(str(original_path), verbose=verbose)

        # 載入 Falx 影像
        from model.calculation import load_falx_image
        falx_img = load_falx_image(str(falx_path), verbose=verbose)

        # 計算 ALVI
        alvi_data = calculate_alvi(
            left_vent, right_vent, original_img,
            falx_img=falx_img,
            verbose=verbose
        )

        # 視覺化
        visualize_alvi(
            left_vent, right_vent, original_img,
            alvi_data,
            output_path=str(output_image_path),
            show_plot=show_plot
        )

        # 返回成功結果
        return {
            'status': 'success',
            'ventricle_ap_diameter_mm': alvi_data['ventricle_ap_diameter_mm'],
            'skull_ap_diameter_mm': alvi_data['skull_ap_diameter_mm'],
            'alvi': alvi_data['alvi'],
            'alvi_percent': alvi_data['alvi_percent'],
            'ventricle_endpoints': alvi_data['ventricle_endpoints'],
            'skull_endpoints': alvi_data['skull_endpoints'],
            'z_range': list(alvi_data['z_range']),
            'voxel_size': list(alvi_data['voxel_size'])
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }
