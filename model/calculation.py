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


def load_falx_image(falx_path, verbose=True):
    """
    載入 Falx（大腦鐮）mask 並自動拉正到 RAS+ 方向

    Args:
        falx_path: Falx mask 檔案路徑
        verbose: 是否顯示載入資訊

    Returns:
        nibabel.Nifti1Image: 已拉正到 RAS+ 方向的影像物件
    """
    # 載入影像並自動拉正到 RAS+ 方向
    falx_img, orig_ornt, new_ornt = reorient_image(falx_path, verbose=False)

    if verbose:
        print(f"  Falx 影像已載入: {falx_path}")
        print(f"    形狀: {falx_img.shape}, 體素: {falx_img.header.get_zooms()[:3]}")

    return falx_img


def fit_falx_plane(falx_img, verbose=True):
    """
    使用 Marching Cubes 提取平滑表面，並用 SVD 擬合 Falx（大腦鐮）平面

    Args:
        falx_img: Falx mask 影像物件（已拉正到 RAS+）
        verbose: 是否顯示擬合資訊

    Returns:
        dict: 平面參數
            - 'normal': 法向量 (A, B, C)
            - 'center': 中心點座標
            - 'A', 'B', 'C', 'D': 平面方程式參數 (Ax + By + Cz + D = 0)
            - 'y_range': (Y_min, Y_max) Falx 的 Y 軸範圍
            - 'surface_vertices': 平滑後的表面頂點
            - 'surface_faces': 表面三角面
    """
    # 使用 Marching Cubes 提取平滑表面
    falx_mesh = extract_surface_mesh(falx_img, level=0.5, verbose=False)
    falx_points = falx_mesh['vertices_physical']
    
    if len(falx_points) == 0:
        raise ValueError("Falx mask 沒有提取到表面頂點！")

    # 使用 SVD 擬合平面（等效於 PCA）
    # 1. 計算中心點
    center = np.mean(falx_points, axis=0)
    
    # 2. 中心化數據
    centered_points = falx_points - center
    
    # 3. SVD 分解：U, S, Vt = svd(centered_points)
    # Vt 的最後一行對應最小奇異值，即為法向量
    _, _, Vt = np.linalg.svd(centered_points, full_matrices=False)
    
    # 法向量 = 最小奇異值對應的向量（Vt 的最後一行）
    normal = Vt[2]

    # 確保法向量指向 X 正方向（左右方向）
    if normal[0] < 0:
        normal = -normal

    # 平面方程: Ax + By + Cz + D = 0
    # D = -(A*x0 + B*y0 + C*z0)
    A, B, C = normal
    D = -np.dot(normal, center)

    # Y 軸範圍
    y_min = np.min(falx_points[:, 1])
    y_max = np.max(falx_points[:, 1])

    if verbose:
        print(f"  Falx 平面擬合完成 (Marching Cubes + SVD):")
        print(f"    表面頂點數: {len(falx_points)}")
        print(f"    法向量: ({A:.4f}, {B:.4f}, {C:.4f})")
        print(f"    中心點: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) mm")
        print(f"    Y 軸範圍: {y_min:.2f} - {y_max:.2f} mm")

    return {
        'normal': normal,
        'center': center,
        'A': A, 'B': B, 'C': C, 'D': D,
        'y_range': (y_min, y_max),
        'falx_points': falx_points,
        'surface_vertices': falx_mesh['vertices_physical'],
        'surface_faces': falx_mesh['faces']
    }


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