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


def calculate_signed_distances(points, falx_plane):
    """
    計算點到 Falx 平面的有向距離
    
    Args:
        points: (N, 3) 點雲座標 (物理空間)
        falx_plane: Falx 平面參數
        
    Returns:
        np.array: (N,) 有向距離
            - 正值: 表示在法向量指向的一側 (通常設為右側)
            - 負值: 表示在另一側 (通常設為左側)
    """
    A, B, C, D = falx_plane['A'], falx_plane['B'], falx_plane['C'], falx_plane['D']
    norm = np.sqrt(A**2 + B**2 + C**2)
    
    # 距離 = (Ax + By + Cz + D) / sqrt(A^2 + B^2 + C^2)
    distances = (A * points[:, 0] + B * points[:, 1] + 
                C * points[:, 2] + D) / norm
                
    return distances


def filter_points_by_falx_side(points, falx_plane, side, verbose=True):
    """
    使用 Falx 平面過濾腦室點 (去除跨越中線的雜訊)
    
    Args:
        points: (N, 3) 點雲座標
        falx_plane: Falx 平面參數
        side: 'left' (左側/負距離) 或 'right' (右側/正距離)
        verbose: 是否顯示過濾資訊
        
    Returns:
        np.array: 過濾後的點雲
    """
    if len(points) == 0:
        return points
        
    distances = calculate_signed_distances(points, falx_plane)
    
    if side == 'left':
        # 左側應該是負距離
        mask = distances < 0
    elif side == 'right':
        # 右側應該是正距離
        mask = distances > 0
    else:
        raise ValueError(f"無效的側別: {side} (必須是 'left' 或 'right')")
        
    filtered_points = points[mask]
    
    if verbose:
        print(f"  {side}側 Falx 過濾: {len(points)} -> {len(filtered_points)} 點 (移除 {len(points) - len(filtered_points)} 點)")
        
    return filtered_points