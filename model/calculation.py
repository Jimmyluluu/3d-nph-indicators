#!/usr/bin/env python3
"""
腦室分析工具
計算腦室質心距離
"""

import numpy as np
import nibabel as nib
from model.image_processing import get_image_data, get_voxel_size, reorient_image, extract_surface_mesh

# 從新模組導入體積表面積計算函數
from model.cal_volume_surface import calculate_surface_area, calculate_volume_smooth, calculate_volume_surface_ratio

def load_ventricle_pair(left_path, right_path, verbose=True):
    """
    載入左右腦室影像並驗證座標系統一致性（自動拉正到 RAS+ 方向）

    Args:
        left_path: 左腦室檔案路徑
        right_path: 右腦室檔案路徑
        verbose: 是否顯示驗證資訊

    Returns:
        tuple: (左腦室影像, 右腦室影像)

    Raises:
        ValueError: 如果座標系統不一致
    """

    # 載入影像並自動拉正到 RAS+ 方向
    left_img, left_orig_ornt, left_new_ornt = reorient_image(left_path, verbose=False)
    right_img, right_orig_ornt, right_new_ornt = reorient_image(right_path, verbose=False)

    # 輸出載入資訊（如果 verbose=True）
    if verbose:
        from processors.printers import print_ventricle_loading_info
        print_ventricle_loading_info(
            left_path, left_img, left_orig_ornt, left_new_ornt,
            right_path, right_img, right_orig_ornt, right_new_ornt,
            left_img.shape == right_img.shape, verbose
        )

    # 檢查體素間距是否相同
    left_voxel = left_img.header.get_zooms()[:3]
    right_voxel = right_img.header.get_zooms()[:3]

    if not np.allclose(left_voxel, right_voxel):
        raise ValueError(f"體素間距不一致！左: {left_voxel}, 右: {right_voxel}")

    
    return left_img, right_img


def load_original_image(original_path, verbose=True):
    """
    載入原始腦部影像並自動拉正到 RAS+ 方向

    此函數作為統一的原始影像載入入口,確保所有模組使用的影像都在相同座標系統

    Args:
        original_path: 原始影像檔案路徑
        verbose: 是否顯示載入資訊

    Returns:
        nibabel.Nifti1Image: 已拉正到 RAS+ 方向的影像物件
    """

    # 載入影像並自動拉正到 RAS+ 方向
    original_img, orig_ornt, new_ornt = reorient_image(original_path, verbose=False)

    # 輸出載入資訊（如果 verbose=True）
    if verbose:
        from processors.printers import print_original_image_loading_info
        print_original_image_loading_info(original_path, original_img, orig_ornt, new_ornt, verbose)

    return original_img


def calculate_centroid_3d(image, return_physical=True):
    """
    計算3D影像的質心（重心）座標

    Args:
        image: nibabel影像物件
        return_physical: 是否返回物理座標（世界座標），否則返回體素座標

    Returns:
        tuple: (x, y, z) 質心座標
            - 如果 return_physical=True: 物理座標（mm）
            - 如果 return_physical=False: 體素座標
    """
    # 取得資料
    data = get_image_data(image)

    # 找出所有非零體素的座標
    coords = np.argwhere(data > 0)

    if len(coords) == 0:
        raise ValueError("影像中沒有非零體素！")

    # 取得對應的強度值
    values = data[coords[:, 0], coords[:, 1], coords[:, 2]]

    # 計算加權質心（使用強度值作為權重）- 體素座標
    centroid_voxel = np.average(coords, axis=0, weights=values)

    if return_physical and hasattr(image, 'affine'):
        # 轉換到物理空間（世界座標）
        # 添加齊次座標 (x, y, z, 1)
        centroid_homogeneous = np.append(centroid_voxel, 1)
        # 使用 affine 矩陣轉換
        centroid_physical = image.affine @ centroid_homogeneous
        # 返回 (x, y, z)，去掉齊次座標
        return tuple(centroid_physical[:3])
    else:
        return tuple(centroid_voxel)


def calculate_centroid_distance(left_ventricle, right_ventricle):
    """
    計算左右腦室質心之間的3D歐式距離（在物理空間中）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件

    Returns:
        tuple: (距離(mm), 左質心座標(mm), 右質心座標(mm), 體素間距)
    """
    # 計算兩個質心的物理座標（世界座標，單位：mm）
    left_centroid_physical = calculate_centroid_3d(left_ventricle, return_physical=True)
    right_centroid_physical = calculate_centroid_3d(right_ventricle, return_physical=True)

    # 取得體素間距
    voxel_size = get_voxel_size(left_ventricle)

    # 計算物理空間中的歐式距離（mm）
    diff = np.array(left_centroid_physical) - np.array(right_centroid_physical)
    distance_mm = np.linalg.norm(diff)

    return distance_mm, left_centroid_physical, right_centroid_physical, voxel_size


def calculate_cranial_width(original_img, verbose=True):
    """
    計算顱內橫向最大寬度（在每個切面上計算，取最大值）

    Args:
        original_img: 原始腦部影像物件

    Returns:
        tuple: (最大寬度(mm), 左端點座標(mm), 右端點座標(mm), 切面編號)
    """
    # 取得資料
    data = get_image_data(original_img)
    voxel_size = get_voxel_size(original_img)

    max_width = 0
    max_slice_idx = 0
    left_point_voxel = None
    right_point_voxel = None

    # 輸出計算資訊（如果 verbose=True）
    if verbose:
        from processors.printers import print_cranial_width_calculation_info
        print_cranial_width_calculation_info(verbose)

    # 遍歷每個 Z 切面
    for z in range(data.shape[2]):
        slice_data = data[:, :, z]

        # 找到該切面的所有非零點
        nonzero_points = np.argwhere(slice_data > 0)

        if len(nonzero_points) == 0:
            continue

        # 找到 X 座標的最小值和最大值
        x_coords = nonzero_points[:, 0]
        x_min_idx = np.argmin(x_coords)
        x_max_idx = np.argmax(x_coords)

        x_min = x_coords[x_min_idx]
        x_max = x_coords[x_max_idx]

        # 計算該切面的寬度（體素單位）
        width_voxels = x_max - x_min

        # 轉換為物理距離（mm）
        width_mm = width_voxels * voxel_size[0]

        # 更新最大寬度
        if width_mm > max_width:
            max_width = width_mm
            max_slice_idx = z
            # 記錄左右端點（體素座標）
            y_left = nonzero_points[x_min_idx, 1]
            y_right = nonzero_points[x_max_idx, 1]
            left_point_voxel = np.array([x_min, y_left, z])
            right_point_voxel = np.array([x_max, y_right, z])

    # 將左右端點轉換到物理座標
    if left_point_voxel is not None:
        left_homogeneous = np.append(left_point_voxel, 1)
        right_homogeneous = np.append(right_point_voxel, 1)

        left_point_physical = (original_img.affine @ left_homogeneous)[:3]
        right_point_physical = (original_img.affine @ right_homogeneous)[:3]

        return max_width, tuple(left_point_physical), tuple(right_point_physical), max_slice_idx
    else:
        raise ValueError("無法在影像中找到非零體素！")


def calculate_ventricle_to_cranial_ratio(ventricle_distance_mm, cranial_width_mm):
    """
    計算腦室質心距離與顱內寬度的比值

    Args:
        ventricle_distance_mm: 腦室質心距離(mm)
        cranial_width_mm: 顱內橫向最大寬度(mm)

    Returns:
        float: 比值（腦室距離/顱內寬度）
    """
    if cranial_width_mm == 0:
        raise ValueError("顱內寬度不能為零！")

    ratio = ventricle_distance_mm / cranial_width_mm
    return ratio


def calculate_anterior_horn_max_distance(left_ventricle, right_ventricle, verbose=True):
    """
    計算左右腦室前腳之間的最大距離（3D Evan Index）

    前腳定義：
    1. Z 軸（高度）：取質心以更上方 (Superior) 的區域 (Z >= Centroid_Z)
    2. Y 軸（前後）：取質心與最前端距離的 20% 處開始 (Y >= Centroid + (Max - Centroid) * 0.2)
       這能更積極地排除腦室體部，專注於前腳區域。

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        verbose: 是否顯示計算過程資訊

    Returns:
        tuple: (最大距離(mm), 左側端點座標(mm), 右側端點座標(mm), 左側前腳點數, 右側前腳點數)
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    # 輸出計算資訊（如果 verbose=True）
    if verbose:
        print(f"  前腳篩選範圍: 質心以上，且從質心往最前端推進 20%")

    # 找出所有非零體素的座標（體素空間）
    left_coords_voxel = np.argwhere(left_data > 0)
    right_coords_voxel = np.argwhere(right_data > 0)

    if len(left_coords_voxel) == 0 or len(right_coords_voxel) == 0:
        raise ValueError("左側或右側腦室沒有非零體素！")

    # 1. 計算左右腦室質心 (體素座標)
    left_centroid = calculate_centroid_3d(left_ventricle, return_physical=False)
    right_centroid = calculate_centroid_3d(right_ventricle, return_physical=False)

    # 2. 計算 Y 軸篩選閾值 (質心往最前端推進 20%)
    # 左腦室
    left_y_max = np.max(left_coords_voxel[:, 1])
    left_y_threshold = left_centroid[1] + (left_y_max - left_centroid[1]) * 0.2
    
    # 右腦室
    right_y_max = np.max(right_coords_voxel[:, 1])
    right_y_threshold = right_centroid[1] + (right_y_max - right_centroid[1]) * 0.2

    if verbose:
        print(f"  左腦室: 質心Y={left_centroid[1]:.1f}, 最前Y={left_y_max}, 閾值Y={left_y_threshold:.1f}")
        print(f"  右腦室: 質心Y={right_centroid[1]:.1f}, 最前Y={right_y_max}, 閾值Y={right_y_threshold:.1f}")

    # 3. 篩選前腳區域
    # 左腦室 (Z >= Centroid_Z 且 Y >= Threshold_Y)
    left_mask = (left_coords_voxel[:, 2] >= left_centroid[2]) & (left_coords_voxel[:, 1] >= left_y_threshold)
    left_anterior = left_coords_voxel[left_mask]

    # 右腦室 (Z >= Centroid_Z 且 Y >= Threshold_Y)
    right_mask = (right_coords_voxel[:, 2] >= right_centroid[2]) & (right_coords_voxel[:, 1] >= right_y_threshold)
    right_anterior = right_coords_voxel[right_mask]

    if verbose:
        print(f"  篩選後點數: 左 {len(left_anterior)} / {len(left_coords_voxel)}, 右 {len(right_anterior)} / {len(right_coords_voxel)}")

    if len(left_anterior) == 0 or len(right_anterior) == 0:
        raise ValueError(f"在篩選區域沒有找到前腳點！請檢查影像是否異常或閾值過高。")

    
    # 將體素座標轉換為物理座標
    left_anterior_homogeneous = np.column_stack([left_anterior, np.ones(len(left_anterior))])
    right_anterior_homogeneous = np.column_stack([right_anterior, np.ones(len(right_anterior))])

    left_anterior_physical = (left_ventricle.affine @ left_anterior_homogeneous.T).T[:, :3]
    right_anterior_physical = (right_ventricle.affine @ right_anterior_homogeneous.T).T[:, :3]

    # 計算左右前腳之間所有點對的距離，找出最大值
    max_distance = 0
    left_max_point = None
    right_max_point = None

    # 採用優化策略：不需要計算所有點對，可以用採樣或其他優化方法
    # 這裡為了準確性，我們計算所有點對（對於前腳點雲，數量應該是可控的）

    # 如果點太多，進行降採樣
    max_points = 5000
    if len(left_anterior_physical) > max_points:
        indices = np.random.choice(len(left_anterior_physical), max_points, replace=False)
        left_anterior_physical = left_anterior_physical[indices]
        # 輸出降採樣資訊（如果 verbose=True）
        if verbose:
            from processors.printers import print_sampling_info
            print_sampling_info("左側", len(left_anterior_physical), max_points, verbose)

    if len(right_anterior_physical) > max_points:
        indices = np.random.choice(len(right_anterior_physical), max_points, replace=False)
        right_anterior_physical = right_anterior_physical[indices]
        # 輸出降採樣資訊（如果 verbose=True）
        if verbose:
            from processors.printers import print_sampling_info
            print_sampling_info("右側", len(right_anterior_physical), max_points, verbose)

    # 計算所有點對的距離
    for left_point in left_anterior_physical:
        # 計算該左側點到所有右側點的距離
        distances = np.linalg.norm(right_anterior_physical - left_point, axis=1)
        max_idx = np.argmax(distances)

        if distances[max_idx] > max_distance:
            max_distance = distances[max_idx]
            left_max_point = left_point
            right_max_point = right_anterior_physical[max_idx]

    # 輸出結果資訊（如果 verbose=True）
    if verbose:
        from processors.printers import print_anterior_horn_distance_info
        # 計算原始點數（可能在降採樣前）
        original_left_count = len(left_anterior)
        original_right_count = len(right_anterior)
        # 這裡同樣做容錯處理，因為 printer 可能還期待 z_range
        try:
            print_anterior_horn_distance_info(None, 0,
                                              original_left_count, original_right_count,
                                              max_distance, left_max_point, right_max_point, verbose)
        except TypeError:
             print(f"  計算完成: 最大距離 {max_distance:.2f} mm")

    return max_distance, tuple(left_max_point), tuple(right_max_point), len(left_anterior), len(right_anterior)


def calculate_3d_evan_index(left_ventricle, right_ventricle, original_img, verbose=True):
    """
    計算 3D Evan Index（腦室前腳最大距離與顱內寬度的比值）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        original_img: 原始腦部影像物件
        verbose: 是否顯示計算過程資訊

    Returns:
        dict: 包含所有測量結果的字典
            {
                'anterior_horn_distance_mm': float,
                'cranial_width_mm': float,
                'evan_index': float,
                'evan_index_percent': float,
                'anterior_horn_endpoints': {
                    'left': tuple,
                    'right': tuple
                },
                'cranial_width_endpoints': {
                    'left': tuple,
                    'right': tuple,
                    'slice_index': int
                },
                'anterior_horn_points_count': {
                    'left': int,
                    'right': int
                },
                'voxel_size': tuple
            }
    """
    # 計算前腳最大距離
    anterior_distance, left_endpoint, right_endpoint, left_count, right_count = \
        calculate_anterior_horn_max_distance(left_ventricle, right_ventricle, verbose=verbose)

    # 計算顱內寬度
    cranial_width, cranial_left, cranial_right, cranial_slice = \
        calculate_cranial_width(original_img)

    # 計算 Evan Index
    evan_index = anterior_distance / cranial_width
    evan_index_percent = evan_index * 100

    # 取得體素間距
    voxel_size = get_voxel_size(left_ventricle)

    # 輸出 Evan Index 計算結果（如果 verbose=True）
    if verbose:
        from processors.printers import print_evan_index_results
        print_evan_index_results(anterior_distance, cranial_width, evan_index, evan_index_percent, verbose)

    return {
        'anterior_horn_distance_mm': anterior_distance,
        'cranial_width_mm': cranial_width,
        'evan_index': evan_index,
        'evan_index_percent': evan_index_percent,
        'anterior_horn_endpoints': {
            'left': left_endpoint,
            'right': right_endpoint
        },
        'cranial_width_endpoints': {
            'left': cranial_left,
            'right': cranial_right,
            'slice_index': cranial_slice
        },
        'anterior_horn_points_count': {
            'left': left_count,
            'right': right_count
        },
        'voxel_size': voxel_size
    }