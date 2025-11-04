#!/usr/bin/env python3
"""
腦室分析工具
計算腦室質心距離
"""

import numpy as np
import nibabel as nib
from model.reorient import get_image_data, get_voxel_size


def load_ventricle_pair(left_path, right_path, verbose=True):
    """
    載入左右腦室影像並驗證座標系統一致性

    Args:
        left_path: 左腦室檔案路徑
        right_path: 右腦室檔案路徑
        verbose: 是否顯示驗證資訊

    Returns:
        tuple: (左腦室影像, 右腦室影像)

    Raises:
        ValueError: 如果座標系統不一致
    """
    # 載入影像
    left_img = nib.load(left_path)
    right_img = nib.load(right_path)

    if verbose:
        print(f"\n載入左腦室: {left_path}")
        print(f"  影像形狀: {left_img.shape}")
        print(f"  方向: {nib.aff2axcodes(left_img.affine)}")
        print(f"  體素間距: {left_img.header.get_zooms()[:3]}")

        print(f"\n載入右腦室: {right_path}")
        print(f"  影像形狀: {right_img.shape}")
        print(f"  方向: {nib.aff2axcodes(right_img.affine)}")
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