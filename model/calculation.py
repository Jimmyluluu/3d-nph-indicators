#!/usr/bin/env python3
"""
腦室分析工具
計算腦室質心距離
"""

import numpy as np
import nibabel as nib
from model.image_processing import get_image_data, get_voxel_size, reorient_image, extract_surface_mesh

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

    if verbose:
        print(f"\n載入左腦室: {left_path}")
        print(f"  影像形狀: {left_img.shape}")
        print(f"  原始方向: {left_orig_ornt} → 標準方向: {left_new_ornt}")
        print(f"  體素間距: {left_img.header.get_zooms()[:3]}")

        print(f"\n載入右腦室: {right_path}")
        print(f"  影像形狀: {right_img.shape}")
        print(f"  原始方向: {right_orig_ornt} → 標準方向: {right_new_ornt}")
        print(f"  體素間距: {right_img.header.get_zooms()[:3]}")

    # 檢查體素間距是否相同
    left_voxel = left_img.header.get_zooms()[:3]
    right_voxel = right_img.header.get_zooms()[:3]

    if not np.allclose(left_voxel, right_voxel):
        raise ValueError(f"體素間距不一致！左: {left_voxel}, 右: {right_voxel}")

    if verbose:
        if left_img.shape != right_img.shape:
            print("\n⚠ 注意：影像形狀不同（可能經過裁剪）")
            print("  將使用 affine 矩陣轉換到物理空間進行計算")
        else:
            print("\n✓ 影像形狀相同")

        print("✓ 座標系統驗證通過！將在物理空間中計算。")

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

    if verbose:
        print(f"\n載入原始影像: {original_path}")
        print(f"  影像形狀: {original_img.shape}")
        print(f"  原始方向: {orig_ornt} → 標準方向: {new_ornt}")
        print(f"  體素間距: {original_img.header.get_zooms()[:3]}")

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


def calculate_cranial_width(original_img):
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

    print(f"\n計算顱內橫向最大寬度...")

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


def calculate_anterior_horn_max_distance(left_ventricle, right_ventricle, z_range=(0.3, 0.9), y_percentile=4, verbose=True):
    """
    計算左右腦室前腳之間的最大距離（3D Evan Index）

    前腳定義：結合 Z 軸（頂部區域）和 Y 軸（前方區域）來篩選前腳點雲

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        z_range: Z 軸切面範圍（tuple），例如 (0.3, 0.9) 表示取 30%-90% 的上方區域
        y_percentile: Y 軸前方百分位數，例如 4 表示取前 4% 的前方點
        verbose: 是否顯示計算過程資訊

    Returns:
        tuple: (最大距離(mm), 左側端點座標(mm), 右側端點座標(mm), 左側前腳點數, 右側前腳點數)
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    if verbose:
        print(f"\n計算腦室前腳最大距離...")
        print(f"  前腳定義：Z 軸範圍 {z_range[0]*100}%-{z_range[1]*100}%，Y 軸前 {y_percentile}%")

    # 找出所有非零體素的座標（體素空間）
    left_coords_voxel = np.argwhere(left_data > 0)
    right_coords_voxel = np.argwhere(right_data > 0)

    if len(left_coords_voxel) == 0 or len(right_coords_voxel) == 0:
        raise ValueError("左側或右側腦室沒有非零體素！")

    # 篩選前腳區域 - 使用 Z 軸範圍
    z_min = int(left_data.shape[2] * z_range[0])
    z_max = int(left_data.shape[2] * z_range[1])

    left_anterior = left_coords_voxel[(left_coords_voxel[:, 2] >= z_min) & (left_coords_voxel[:, 2] <= z_max)]
    right_anterior = right_coords_voxel[(right_coords_voxel[:, 2] >= z_min) & (right_coords_voxel[:, 2] <= z_max)]

    if len(left_anterior) == 0 or len(right_anterior) == 0:
        raise ValueError(f"在 Z 軸範圍 {z_range} 內沒有找到前腳點！請調整 z_range 參數。")

    # 進一步篩選 - 使用 Y 軸前方區域
    # 注意：在 RAS 座標系統中，Y 軸從後（Posterior）到前（Anterior）
    # 較大的 Y 值表示前方，所以我們要取前 y_percentile% 的較大值
    left_y_threshold = np.percentile(left_anterior[:, 1], 100 - y_percentile)
    right_y_threshold = np.percentile(right_anterior[:, 1], 100 - y_percentile)

    left_anterior = left_anterior[left_anterior[:, 1] >= left_y_threshold]
    right_anterior = right_anterior[right_anterior[:, 1] >= right_y_threshold]

    if len(left_anterior) == 0 or len(right_anterior) == 0:
        raise ValueError(f"在 Y 軸前 {y_percentile}% 區域內沒有找到前腳點！請調整 y_percentile 參數。")

    if verbose:
        print(f"  左側前腳點數：{len(left_anterior)}")
        print(f"  右側前腳點數：{len(right_anterior)}")
        print(f"  正在計算最大距離...")

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
        if verbose:
            print(f"  左側點雲降採樣至 {max_points} 點")

    if len(right_anterior_physical) > max_points:
        indices = np.random.choice(len(right_anterior_physical), max_points, replace=False)
        right_anterior_physical = right_anterior_physical[indices]
        if verbose:
            print(f"  右側點雲降採樣至 {max_points} 點")

    # 計算所有點對的距離
    for left_point in left_anterior_physical:
        # 計算該左側點到所有右側點的距離
        distances = np.linalg.norm(right_anterior_physical - left_point, axis=1)
        max_idx = np.argmax(distances)

        if distances[max_idx] > max_distance:
            max_distance = distances[max_idx]
            left_max_point = left_point
            right_max_point = right_anterior_physical[max_idx]

    if verbose:
        print(f"  ✓ 前腳最大距離：{max_distance:.2f} mm")
        print(f"  左側端點：({left_max_point[0]:.2f}, {left_max_point[1]:.2f}, {left_max_point[2]:.2f})")
        print(f"  右側端點：({right_max_point[0]:.2f}, {right_max_point[1]:.2f}, {right_max_point[2]:.2f})")

    return max_distance, tuple(left_max_point), tuple(right_max_point), len(left_anterior), len(right_anterior)


def calculate_3d_evan_index(left_ventricle, right_ventricle, original_img, z_range=(0.3, 0.9), y_percentile=4, verbose=True):
    """
    計算 3D Evan Index（腦室前腳最大距離與顱內寬度的比值）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        original_img: 原始腦部影像物件
        z_range: Z 軸切面範圍（tuple），例如 (0.3, 0.9) 表示取 30%-90% 的上方區域
        y_percentile: Y 軸前方百分位數，例如 4 表示取前 4% 的前方點
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
        calculate_anterior_horn_max_distance(left_ventricle, right_ventricle, z_range, y_percentile, verbose)

    # 計算顱內寬度
    cranial_width, cranial_left, cranial_right, cranial_slice = \
        calculate_cranial_width(original_img)

    # 計算 Evan Index
    evan_index = anterior_distance / cranial_width
    evan_index_percent = evan_index * 100

    # 取得體素間距
    voxel_size = get_voxel_size(left_ventricle)

    if verbose:
        print(f"\n3D Evan Index 計算結果：")
        print(f"  前腳最大距離：{anterior_distance:.2f} mm")
        print(f"  顱內寬度：{cranial_width:.2f} mm")
        print(f"  Evan Index：{evan_index:.4f} ({evan_index_percent:.2f}%)")

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


def calculate_surface_area(left_ventricle, right_ventricle, verbose=True):
    """
    計算左右腦室的表面積（純計算，不包含視覺化資料）。

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        verbose (bool): 是否顯示計算過程資訊

    Returns:
        dict: 包含表面積計算結果的字典
    """
    from skimage.measure import mesh_surface_area

    def _get_surface_area(image_obj, name):
        if verbose:
            print(f"\n計算 {name} 表面積...")

        # 使用統一的表面提取函數 (Marching Cubes 已經提供平滑表面)
        mesh_result = extract_surface_mesh(image_obj, level=0.5, verbose=verbose)

        # 取得體素座標的頂點和面
        vertices_voxel = mesh_result['vertices_voxel']
        faces = mesh_result['faces']

        # 使用 Marching Cubes 結果計算表面積
        surface_area = mesh_surface_area(vertices_voxel, faces)

        if verbose:
            print(f"  - 計算表面積: {surface_area:.2f} mm^2")

        return surface_area

    left_area = _get_surface_area(left_ventricle, "左腦室")
    right_area = _get_surface_area(right_ventricle, "右腦室")

    total_surface_area = left_area + right_area

    if verbose:
        print("\n表面積計算總結:")
        print(f"  左腦室表面積: {left_area:.2f} mm^2")
        print(f"  右腦室表面積: {right_area:.2f} mm^2")
        print(f"  總表面積: {total_surface_area:.2f} mm^2")

    return {
        'left_surface_area': left_area,
        'right_surface_area': right_area,
        'total_surface_area': total_surface_area
    }