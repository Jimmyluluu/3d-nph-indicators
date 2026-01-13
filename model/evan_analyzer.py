#!/usr/bin/env python3
"""
3D Evan Index 分析模組
計算腦室前腳距離、顱內寬度及 Evan Index
"""

import numpy as np
from model.image_processing import get_image_data, get_voxel_size, convert_voxel_to_physical
from model.calculation import fit_falx_plane, calculate_centroid_3d, filter_points_by_falx_side


def calculate_anterior_horn_distance_with_falx(left_ventricle, right_ventricle, falx_plane, verbose=True):
    """
    使用 Falx 平面計算前腳橫向距離

    計算流程:
    1. 合併左右腦室點雲
    2. Y 軸篩選: 使用腦室質心往前推進 70% 處開始（取前 30% 區域）
    3. Z 軸過濾: 排除最低 15%
    4. 使用 Falx 平面分左右側，取各側最大 X 距離相加

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        falx_plane: fit_falx_plane() 返回的平面參數（用於 X 軸分側）
        verbose: 是否顯示計算過程

    Returns:
        tuple: (距離(mm), 左側端點座標, 右側端點座標, 左側點數, 右側點數)
    """
    # 取得 Falx 平面參數（用於 X 軸分側）
    A, B, C, D = falx_plane['A'], falx_plane['B'], falx_plane['C'], falx_plane['D']
    norm = np.sqrt(A**2 + B**2 + C**2)

    # 合併左右腦室點雲
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    left_coords_voxel = np.argwhere(left_data > 0)
    right_coords_voxel = np.argwhere(right_data > 0)

    # 轉換為物理座標
    left_homogeneous = np.column_stack([left_coords_voxel, np.ones(len(left_coords_voxel))])
    left_points = (left_ventricle.affine @ left_homogeneous.T).T[:, :3]

    right_homogeneous = np.column_stack([right_coords_voxel, np.ones(len(right_coords_voxel))])
    right_points = (right_ventricle.affine @ right_homogeneous.T).T[:, :3]

    # 使用 Falx 平面過濾跨中線的點 (去噪)
    if verbose:
        print("  正在過濾跨中線點...")
    left_points = filter_points_by_falx_side(left_points, falx_plane, 'left', verbose=verbose)
    right_points = filter_points_by_falx_side(right_points, falx_plane, 'right', verbose=verbose)

    # 合併所有腦室點
    all_points = np.vstack([left_points, right_points])
    
    # 計算腦室質心（用於 Y 軸篩選）
    left_centroid = np.mean(left_points, axis=0)
    right_centroid = np.mean(right_points, axis=0)
    
    # 計算合併後的 Y 軸範圍
    all_y_max = np.max(all_points[:, 1])
    avg_centroid_y = (left_centroid[1] + right_centroid[1]) / 2

    if verbose:
        print(f"  前腳篩選範圍: 基於腦室質心的前 30%")

    # 階段 1: Y 軸 ROI 區域篩選（使用腦室質心，與傳統方法一致）
    # 取質心往前推進 70% 處開始（只取前 30% 區域）
    y_threshold = avg_centroid_y + 0.7 * (all_y_max - avg_centroid_y)
    y_mask = all_points[:, 1] >= y_threshold
    anterior_points = all_points[y_mask]

    if verbose:
        print(f"  第一階段 Y 軸篩選: {len(anterior_points)} / {len(all_points)} 點")

    if len(anterior_points) == 0:
        raise ValueError("Y 軸篩選後沒有點！請檢查影像。")

    # 階段 2: Z 軸雜訊過濾（排除最低 15%）
    z_p15 = np.percentile(anterior_points[:, 2], 15)
    z_mask = anterior_points[:, 2] >= z_p15
    filtered_points = anterior_points[z_mask]

    if verbose:
        print(f"  第二階段 Z 軸過濾 (> 15%): {len(filtered_points)} 點")

    if len(filtered_points) == 0:
        raise ValueError("Z 軸過濾後沒有點！")

    # 計算點到 Falx 平面的 signed distance
    # d = (Ax + By + Cz + D) / sqrt(A² + B² + C²)
    signed_distances = (A * filtered_points[:, 0] + 
                       B * filtered_points[:, 1] + 
                       C * filtered_points[:, 2] + D) / norm

    # 分左右側（正距離 = 右側，負距離 = 左側）
    left_side_mask = signed_distances < 0
    right_side_mask = signed_distances > 0

    left_side_points = filtered_points[left_side_mask]
    right_side_points = filtered_points[right_side_mask]
    
    # Fallback 1: 如果用 signed distance 分不出左右，改用 Falx 中心 X 座標
    if len(left_side_points) == 0 or len(right_side_points) == 0:
        if verbose:
            print(f"  ⚠️ Signed distance 分側失敗，改用 Falx 中心 X 座標分側")
        falx_center_x = falx_plane['center'][0]
        left_side_mask = filtered_points[:, 0] < falx_center_x
        right_side_mask = filtered_points[:, 0] >= falx_center_x
        left_side_points = filtered_points[left_side_mask]
        right_side_points = filtered_points[right_side_mask]
    
    # Fallback 2: 如果 Falx 中心分側仍失敗，用 X 座標中位數分側（一定能成功）
    if len(left_side_points) == 0 or len(right_side_points) == 0:
        if verbose:
            print(f"  ⚠️ Falx 中心分側失敗，改用 X 座標中位數分側")
        x_median = np.median(filtered_points[:, 0])
        left_side_mask = filtered_points[:, 0] < x_median
        right_side_mask = filtered_points[:, 0] >= x_median
        left_side_points = filtered_points[left_side_mask]
        right_side_points = filtered_points[right_side_mask]
    
    if len(left_side_points) == 0 or len(right_side_points) == 0:
        raise ValueError("左側或右側沒有點！（所有點 X 座標相同）")
    
    # 重新計算距離（用 X 座標與中心的距離）
    center_x = falx_plane['center'][0]
    left_side_distances = np.abs(filtered_points[left_side_mask][:, 0] - center_x)
    right_side_distances = np.abs(filtered_points[right_side_mask][:, 0] - center_x)

    # 找各側最大距離的點
    left_max_idx = np.argmax(left_side_distances)
    right_max_idx = np.argmax(right_side_distances)

    left_max_distance = left_side_distances[left_max_idx]
    right_max_distance = right_side_distances[right_max_idx]

    left_extreme_point = left_side_points[left_max_idx]
    right_extreme_point = right_side_points[right_max_idx]

    # 總距離 = 左側最大距離 + 右側最大距離
    total_distance = left_max_distance + right_max_distance

    if verbose:
        print(f"  左側最大距離: {left_max_distance:.2f} mm")
        print(f"  右側最大距離: {right_max_distance:.2f} mm")
        print(f"  前腳總寬度: {total_distance:.2f} mm")

    return (total_distance, 
            tuple(left_extreme_point), 
            tuple(right_extreme_point),
            len(left_side_points),
            len(right_side_points))


def calculate_anterior_horn_max_distance(left_ventricle, right_ventricle, verbose=True):
    """
    計算左右腦室前腳之間的橫向距離（3D Evan Index）

    測量方式：
    1. 篩選前腳區域（質心以上、前 20%）
    2. 左前腳取最左邊（X 最小）的點
    3. 右前腳取最右邊（X 最大）的點
    4. 計算這兩點之間的距離

    前腳篩選條件：
    - Y 軸（前後）：取質心往最前端推進 70% 處開始（只取前 30% 區域）
    - Z 軸（高度）：過濾最低 15% 的異常值（保留 15-100% 範圍）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        verbose: 是否顯示計算過程資訊

    Returns:
        tuple: (距離(mm), 左側端點座標(mm), 右側端點座標(mm), 左側前腳點數, 右側前腳點數)
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    # 輸出計算資訊（如果 verbose=True）
    if verbose:
        print(f"  前腳篩選範圍: Y 軸前 30% 區域")

    # 找出所有非零體素的座標（體素空間）
    left_coords_voxel = np.argwhere(left_data > 0)
    right_coords_voxel = np.argwhere(right_data > 0)

    if len(left_coords_voxel) == 0 or len(right_coords_voxel) == 0:
        raise ValueError("左側或右側腦室沒有非零體素！")

    # 1. 計算左右腦室質心 (體素座標)
    left_centroid = calculate_centroid_3d(left_ventricle, return_physical=False)
    right_centroid = calculate_centroid_3d(right_ventricle, return_physical=False)

    # 2. 計算 Y 軸篩選閾值（質心往最前端推進 70%，取前 30% 區域）
    left_y_max = np.max(left_coords_voxel[:, 1])
    left_y_threshold = left_centroid[1] + (left_y_max - left_centroid[1]) * 0.7
    
    right_y_max = np.max(right_coords_voxel[:, 1])
    right_y_threshold = right_centroid[1] + (right_y_max - right_centroid[1]) * 0.7
    
    # 3. 第一階段：Y 軸篩選出前角區域
    left_y_mask = left_coords_voxel[:, 1] >= left_y_threshold
    left_anterior_y = left_coords_voxel[left_y_mask]
    
    right_y_mask = right_coords_voxel[:, 1] >= right_y_threshold
    right_anterior_y = right_coords_voxel[right_y_mask]
    
    # 4. 第二階段：Z 軸百分位數過濾（排除下方極端異常點）
    # 使用 15-100%：只過濾最低 15%（異常通常在下方），上方不過濾
    left_z_p15 = np.percentile(left_anterior_y[:, 2], 15)
    right_z_p15 = np.percentile(right_anterior_y[:, 2], 15)
    
    # 過濾 Z 軸下方異常值
    left_z_mask = left_anterior_y[:, 2] >= left_z_p15
    left_anterior = left_anterior_y[left_z_mask]
    
    right_z_mask = right_anterior_y[:, 2] >= right_z_p15
    right_anterior = right_anterior_y[right_z_mask]
    
    if verbose:
        print(f"  第一階段 Y 軸篩選（前 30%）: 左 {len(left_anterior_y)}, 右 {len(right_anterior_y)}")
        print(f"  第二階段 Z 軸過濾（> 15%）: 左 {len(left_anterior)}, 右 {len(right_anterior)}")

    
    # 將體素座標轉換為物理座標
    left_anterior_homogeneous = np.column_stack([left_anterior, np.ones(len(left_anterior))])
    right_anterior_homogeneous = np.column_stack([right_anterior, np.ones(len(right_anterior))])

    left_anterior_physical = (left_ventricle.affine @ left_anterior_homogeneous.T).T[:, :3]
    right_anterior_physical = (right_ventricle.affine @ right_anterior_homogeneous.T).T[:, :3]

    # 找左前腳最左邊（X 最小）的點，右前腳最右邊（X 最大）的點
    # 這符合傳統 Evan Index 的測量方式
    
    # 在物理座標中，X 軸是左右方向（RAS+ 座標系）
    # 左腦室在右側（X 較大），右腦室在左側（X 較小）
    # 但我們要找的是：左前腳最外側（X 最小）和右前腳最外側（X 最大）
    
    # 左前腳：找 X 最小的點（最左側）
    left_x_min_idx = np.argmin(left_anterior_physical[:, 0])
    left_extreme_point = left_anterior_physical[left_x_min_idx]
    
    # 右前腳：找 X 最大的點（最右側）
    right_x_max_idx = np.argmax(right_anterior_physical[:, 0])
    right_extreme_point = right_anterior_physical[right_x_max_idx]
    
    # 計算兩個極值點之間的距離
    distance = np.linalg.norm(right_extreme_point - left_extreme_point)

    # 輸出結果資訊（如果 verbose=True）
    if verbose:
        print(f"  左前腳最左點: ({left_extreme_point[0]:.2f}, {left_extreme_point[1]:.2f}, {left_extreme_point[2]:.2f}) mm")
        print(f"  右前腳最右點: ({right_extreme_point[0]:.2f}, {right_extreme_point[1]:.2f}, {right_extreme_point[2]:.2f}) mm")
        print(f"  前腳橫向距離: {distance:.2f} mm")

    return distance, tuple(left_extreme_point), tuple(right_extreme_point), len(left_anterior), len(right_anterior)


def calculate_cranial_width(original_img, falx_plane=None, verbose=True):
    """
    計算顱內橫向最大寬度

    如果提供 falx_plane，則使用 Falx 平面作為中線參考，計算左右兩側到平面的最大距離之和。
    否則使用傳統方法（在每個切面上計算 X 軸寬度，取最大值）。

    Args:
        original_img: 原始腦部影像物件
        falx_plane: Falx 平面參數（可選）
        verbose: 是否顯示計算資訊

    Returns:
        tuple: (最大寬度(mm), 左端點座標(mm), 右端點座標(mm), 切面編號或None)
    """
    # 取得資料
    data = get_image_data(original_img)
    voxel_size = get_voxel_size(original_img)

    # 輸出計算資訊（如果 verbose=True）
    if verbose:
        from processors.printers import print_cranial_width_calculation_info
        print_cranial_width_calculation_info(verbose)

    # 如果有 Falx 平面，使用 Falx-based 方法
    if falx_plane is not None:
        if verbose:
            print("  使用 Falx 平面計算顱內寬度")
        
        # 取得 Falx 平面參數
        A, B, C, D = falx_plane['A'], falx_plane['B'], falx_plane['C'], falx_plane['D']
        norm = np.sqrt(A**2 + B**2 + C**2)
        
        # 取得所有腦部非零點
        brain_coords_voxel = np.argwhere(data > 0)
        
        if len(brain_coords_voxel) == 0:
            raise ValueError("無法在影像中找到非零體素！")
        
        # 轉換為物理座標
        homogeneous = np.column_stack([brain_coords_voxel, np.ones(len(brain_coords_voxel))])
        brain_points = (original_img.affine @ homogeneous.T).T[:, :3]
        
        # 計算每個點到 Falx 平面的 signed distance
        signed_distances = (A * brain_points[:, 0] + 
                           B * brain_points[:, 1] + 
                           C * brain_points[:, 2] + D) / norm
        
        # 分左右側
        left_mask = signed_distances < 0
        right_mask = signed_distances > 0
        
        if not np.any(left_mask) or not np.any(right_mask):
            # Fallback 到傳統方法
            if verbose:
                print("  ⚠️ Falx 分側失敗，退回傳統方法")
            return calculate_cranial_width(original_img, falx_plane=None, verbose=False)
        
        left_distances = np.abs(signed_distances[left_mask])
        right_distances = signed_distances[right_mask]
        
        # 找左右側最大距離的點
        left_max_idx = np.argmax(left_distances)
        right_max_idx = np.argmax(right_distances)
        
        left_max_distance = left_distances[left_max_idx]
        right_max_distance = right_distances[right_max_idx]
        
        # 取得對應的點座標
        left_point_physical = brain_points[left_mask][left_max_idx]
        right_point_physical = brain_points[right_mask][right_max_idx]
        
        # 總寬度 = 左側最大距離 + 右側最大距離
        max_width = left_max_distance + right_max_distance
        
        if verbose:
            print(f"  左側最大距離: {left_max_distance:.2f} mm")
            print(f"  右側最大距離: {right_max_distance:.2f} mm")
            print(f"  顱內總寬度: {max_width:.2f} mm")
        
        return max_width, tuple(left_point_physical), tuple(right_point_physical), None

    # 傳統方法：遍歷每個 Z 切面
    max_width = 0
    max_slice_idx = 0
    left_point_voxel = None
    right_point_voxel = None

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


def calculate_3d_evan_index(left_ventricle, right_ventricle, original_img, falx_img=None, verbose=True):
    """
    計算 3D Evan Index（腦室前腳距離與顱內寬度的比值）

    預設使用 Falx-based 方法（如果提供 falx_img）。

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        original_img: 原始腦部影像物件
        falx_img: Falx（大腦鐮）mask 影像物件，用於中線定位
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
                'voxel_size': tuple,
                'method': str  # 'falx' or 'centroid'
            }
    """
    # 根據是否有 Falx 選擇計算方法
    if falx_img is not None:
        # 使用 Falx-based 方法
        if verbose:
            print("  使用 Falx-based 方法計算前腳距離")
        
        # 擬合 Falx 平面
        falx_plane = fit_falx_plane(falx_img, verbose=verbose)
        
        # 計算前腳距離
        anterior_distance, left_endpoint, right_endpoint, left_count, right_count = \
            calculate_anterior_horn_distance_with_falx(
                left_ventricle, right_ventricle, falx_plane, verbose=verbose
            )
        method = 'falx'
    else:
        # 使用傳統質心方法
        if verbose:
            print("  使用傳統質心方法計算前腳距離")
        anterior_distance, left_endpoint, right_endpoint, left_count, right_count = \
            calculate_anterior_horn_max_distance(left_ventricle, right_ventricle, verbose=verbose)
        method = 'centroid'
        falx_plane = None  # 沒有 Falx 平面

    # 計算顱內寬度（如果有 Falx 平面則使用之）
    cranial_width, cranial_left, cranial_right, cranial_slice = \
        calculate_cranial_width(original_img, falx_plane=falx_plane, verbose=verbose)

    # 計算 Evan Index
    evan_index = anterior_distance / cranial_width
    evan_index_percent = evan_index * 100

    # 取得體素間距
    voxel_size = get_voxel_size(left_ventricle)

    # 輸出 Evan Index 計算結果（如果 verbose=True）
    if verbose:
        from processors.printers import print_evan_index_results
        print_evan_index_results(anterior_distance, cranial_width, evan_index, evan_index_percent, verbose)

    # 建立返回結果
    result = {
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
        'voxel_size': voxel_size,
        'method': method
    }
    
    # 如果使用 Falx 方法，加入平面資訊
    if method == 'falx':
        result['falx_plane'] = falx_plane
    
    return result
