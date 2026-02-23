#!/usr/bin/env python3
"""
ALVI (Anteroposterior Lateral Ventricle Index) 計算模組
使用 3D + Falx 特徵方法
"""

import numpy as np
from scipy.ndimage import label
from model.image_processing import get_image_data, convert_voxel_to_physical
from model.calculation import fit_falx_plane, filter_points_by_falx_side, project_points_to_plane, find_max_diameter_convex_hull


def get_largest_connected_component(data):
    """保留且僅保留最大的連通區域 (去除噪聲)"""
    labeled_array, num_features = label(data > 0)
    if num_features == 0:
        return data
    
    # 計算每個區域的大小
    sizes = np.bincount(labeled_array.ravel())
    # 0 是背景，跳過
    sizes[0] = 0
    
    # 找出最大區域的標籤
    max_label = sizes.argmax()
    
    # 建立只包含最大區域的 mask
    cleaned_data = np.zeros_like(data)
    cleaned_data[labeled_array == max_label] = data[labeled_array == max_label]
    
    return cleaned_data


def calculate_ventricle_ap_diameter(left_vent, right_vent, falx_img, z_range_percent=(0.3, 0.7), verbose=True):
    """
    計算腦室體部的前後徑 (使用 Falx 平面投影 + Convex Hull 方法)
    
    分別計算左右腦室在平行於 Falx 平面上的最長徑，然後取最大值。
    
    Args:
        left_vent: 左腦室影像物件
        right_vent: 右腦室影像物件
        falx_img: Falx 影像物件 (必要,用於定義中線和投影平面)
        z_range_percent: Z 軸篩選範圍 (預設 30%-70% 為體部)
        verbose: 是否顯示計算過程
    
    Returns:
        dict: 前後徑計算結果
    """
    if verbose:
        print("計算腦室前後徑 (Falx 投影 + Convex Hull - 取左右最大值)...")
    
    if falx_img is None:
        raise ValueError("必須提供 Falx 影像以計算腦室前後徑!")
    
    # Step 1: 取得 Falx 平面
    falx_plane = fit_falx_plane(falx_img, verbose=False)
    
    # Step 2: 取得左右腦室點雲並進行基本過濾
    def get_and_filter_ventricle_points(vent_img, side):
        data_raw = get_image_data(vent_img)
        data = get_largest_connected_component(data_raw)
        coords = np.argwhere(data > 0)
        points = convert_voxel_to_physical(coords, vent_img.affine)
        # 過濾跨越中線的點
        points = filter_points_by_falx_side(points, falx_plane, side, verbose=False)
        return points

    left_points = get_and_filter_ventricle_points(left_vent, 'left')
    right_points = get_and_filter_ventricle_points(right_vent, 'right')
    
    if verbose:
        print(f"  左腦室點數 (去噪過濾後): {len(left_points)}")
        print(f"  右腦室點數 (去噪過濾後): {len(right_points)}")
    
    # Step 3: 定義單側計算邏輯 (投影 -> 過濾異常 -> Convex Hull)
    def calculate_single_ventricle_diameter(points, name):
        """計算單側腦室的投影最長徑"""
        if len(points) == 0:
             return {
                'diameter': 0.0,
                'anterior': (0,0,0),
                'posterior': (0,0,0),
                'body_count': 0,
                'z_range': (0,0)
            }

        # 1. 篩選 Z 軸體部範圍
        z_p30 = np.percentile(points[:, 2], z_range_percent[0] * 100)
        z_p70 = np.percentile(points[:, 2], z_range_percent[1] * 100)
        body_mask = (points[:, 2] >= z_p30) & (points[:, 2] <= z_p70)
        body_points = points[body_mask]
        
        if len(body_points) < 10: # 點數太少無法計算
            if verbose: print(f"  ⚠️ {name} 篩選體部後點數不足 ({len(body_points)})")
            return {
                'diameter': 0.0, 'anterior': (0,0,0), 'posterior': (0,0,0),
                'body_count': len(body_points), 'z_range': (z_p30, z_p70)
            }
        
        # 2. 投影到 Falx 平面 (從側面壓扁)
        projected_points = project_points_to_plane(body_points, falx_plane)
        
        # 3. 排除異常點 (基於投影後的長軸方向)
        # 利用 PCA 找出投影點的分佈主軸以便進行百分位數過濾
        centroid = np.mean(projected_points, axis=0)
        centered = projected_points - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_axis = Vt[0] # 投影平面內的最長軸
        
        proj_1d = centered @ principal_axis
        p_min_val = np.percentile(proj_1d, 0.5)
        p_max_val = np.percentile(proj_1d, 99.5)
        
        # 只保留中間 99% 的點
        outlier_mask = (proj_1d >= p_min_val) & (proj_1d <= p_max_val)
        filtered_points = projected_points[outlier_mask]
        
        # 4. 使用 Convex Hull 找出過濾後點集的最長徑
        diameter, p1, p2 = find_max_diameter_convex_hull(filtered_points)
        
        # 5. 定義前後端點 (Y 座標較大者為 Anterior, RAS+ 方向)
        if p1[1] > p2[1]:
            ant_pt, post_pt = p1, p2
        else:
            ant_pt, post_pt = p2, p1
        
        return {
            'diameter': diameter,
            'anterior': ant_pt,
            'posterior': post_pt,
            'body_count': len(body_points),
            'z_range': (z_p30, z_p70)
        }
    
    # 計算左右腦室
    left_result = calculate_single_ventricle_diameter(left_points, "左腦室")
    right_result = calculate_single_ventricle_diameter(right_points, "右腦室")
    
    # Step 3: 取最大直徑
    if left_result['diameter'] > right_result['diameter']:
        max_diameter = left_result['diameter']
        final_result = left_result
        chosen_side = "左腦室"
    else:
        max_diameter = right_result['diameter']
        final_result = right_result
        chosen_side = "右腦室"
    
    if verbose:
        print(f"  左腦室前後徑: {left_result['diameter']:.2f} mm (Z範圍: {left_result['z_range'][0]:.1f}-{left_result['z_range'][1]:.1f})")
        print(f"  右腦室前後徑: {right_result['diameter']:.2f} mm (Z範圍: {right_result['z_range'][0]:.1f}-{right_result['z_range'][1]:.1f})")
        print(f"  最終選取: {chosen_side} (最大值: {max_diameter:.2f} mm)")
    
    return {
        'diameter_mm': max_diameter,
        'left_diameter_mm': left_result['diameter'],
        'right_diameter_mm': right_result['diameter'],
        'anterior_point': tuple(final_result['anterior']),
        'posterior_point': tuple(final_result['posterior']),
        'z_range': final_result['z_range'],  # 使用最大側的 Z 軸範圍
        'body_points_count': left_result['body_count'] + right_result['body_count']
    }


def calculate_skull_ap_diameter(original_img, z_range, falx_plane, verbose=True):
    """
    計算顱骨內前後徑 (在 Falx 平面上的 Y 軸最大距離)
    
    Args:
        original_img: 原始腦部影像物件
        z_range: (z_min, z_max) Z 軸範圍
        falx_plane: Falx 平面參數 (必要)
        verbose: 是否顯示計算過程
    
    Returns:
        dict: {
            'diameter_mm': float,       # 前後徑 (mm)
            'anterior_point': tuple,    # 前端點座標
            'posterior_point': tuple,   # 後端點座標
            'points_count': int         # 點數
        }
    """
    if verbose:
        print("計算顱骨內前後徑 (在 Falx 平面上測量)...")
    
    if falx_plane is None:
        raise ValueError("必須提供 Falx 平面參數以確保在中線上測量顱內前後徑!")
    
    # Step 1: 取得原始影像非零點
    data = get_image_data(original_img)
    coords_voxel = np.argwhere(data > 0)
    
    # Step 2: 轉換為物理座標
    points = convert_voxel_to_physical(coords_voxel, original_img.affine)
    
    if verbose:
        print(f"  總點數: {len(points)}")
    
    # Step 3: 篩選相同 Z 軸範圍
    z_min, z_max = z_range
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered_points = points[z_mask]
    
    if verbose:
        print(f"  Z 軸範圍: {z_min:.2f} - {z_max:.2f} mm")
        print(f"  範圍內點數: {len(filtered_points)}")
    
    if len(filtered_points) == 0:
        raise ValueError("Z 軸範圍內沒有點!")
    
    # Step 4: 篩選接近 Falx 平面的點
    A, B, C, D = falx_plane['A'], falx_plane['B'], falx_plane['C'], falx_plane['D']
    norm = np.sqrt(A**2 + B**2 + C**2)
    
    # 計算每個點到 Falx 平面的距離
    distances = np.abs(A * filtered_points[:, 0] + 
                      B * filtered_points[:, 1] + 
                      C * filtered_points[:, 2] + D) / norm
    
    # 顯示距離分布 (debug)
    if verbose:
        print(f"  距離 Falx 平面的統計:")
        print(f"    最小距離: {np.min(distances):.2f} mm")
        print(f"    最大距離: {np.max(distances):.2f} mm")
        print(f"    平均距離: {np.mean(distances):.2f} mm")
        print(f"    中位數距離: {np.median(distances):.2f} mm")
    
    # 嘗試不同的距離閾值,從 3mm 開始,如果沒有點就逐步放寬到 5mm
    distance_thresholds = [3.0, 5.0]
    filtered_points_result = None
    used_threshold = None
    
    for distance_threshold in distance_thresholds:
        near_falx_mask = distances <= distance_threshold
        temp_filtered = filtered_points[near_falx_mask]
        
        if len(temp_filtered) > 0:
            filtered_points_result = temp_filtered
            used_threshold = distance_threshold
            break
    
    if filtered_points_result is None:
        raise ValueError(f"即使放寬到 ±{distance_thresholds[-1]}mm,在 Falx 平面附近仍然沒有點!")
    
    filtered_points = filtered_points_result
    
    if verbose:
        print(f"  使用 Falx 平面作為中線 (±{used_threshold}mm)")
        print(f"  接近 Falx 平面的點數: {len(filtered_points)}")
    
    # Step 5: 計算 Y 軸最大距離
    y_min_idx = np.argmin(filtered_points[:, 1])
    y_max_idx = np.argmax(filtered_points[:, 1])
    
    y_min = filtered_points[y_min_idx, 1]
    y_max = filtered_points[y_max_idx, 1]
    diameter = y_max - y_min
    
    # Step 6: 取得前後端點座標
    anterior_point = filtered_points[y_max_idx]  # Y 最大 = 最前方
    posterior_point = filtered_points[y_min_idx]  # Y 最小 = 最後方
    
    if verbose:
        print(f"  顱骨內前後徑: {diameter:.2f} mm")
        print(f"  前端點: ({anterior_point[0]:.2f}, {anterior_point[1]:.2f}, {anterior_point[2]:.2f})")
        print(f"  後端點: ({posterior_point[0]:.2f}, {posterior_point[1]:.2f}, {posterior_point[2]:.2f})")
    
    return {
        'diameter_mm': diameter,
        'anterior_point': tuple(anterior_point),
        'posterior_point': tuple(posterior_point),
        'points_count': len(filtered_points)
    }


def calculate_alvi(left_vent, right_vent, original_img, falx_img, verbose=True):
    """
    計算 ALVI (Anteroposterior Lateral Ventricle Index)
    
    ALVI = 側腦室前後徑 / 顱骨內前後徑
    - 正常值: < 0.5
    - NPH 診斷閾值: > 0.5
    
    Args:
        left_vent: 左腦室影像物件 (已拉正到 RAS+)
        right_vent: 右腦室影像物件 (已拉正到 RAS+)
        original_img: 原始腦部影像物件 (已拉正到 RAS+)
        falx_img: Falx 影像 (必要,用於定義中線和測量顱內徑)
        verbose: 是否顯示計算過程
    
    Returns:
        dict: {
            'ventricle_ap_diameter_mm': float,     # 腦室前後徑 (左右最大值)
            'left_diameter_mm': float,             # 左腦室前後徑
            'right_diameter_mm': float,            # 右腦室前後徑
            'skull_ap_diameter_mm': float,         # 顱骨前後徑
            'alvi': float,                         # ALVI 比值
            'alvi_percent': float,                 # ALVI 百分比
            'ventricle_endpoints': {...},          # 腦室端點
            'skull_endpoints': {...},              # 顱骨端點
            'z_range': tuple,                      # Z 軸範圍
            'voxel_size': tuple                    # 體素間距
        }
    """
    if verbose:
        print("\n" + "=" * 70)
        print("開始計算 ALVI (Anteroposterior Lateral Ventricle Index)")
        print("=" * 70)
    
    # 0. 擬合 Falx 平面 (必要)
    if falx_img is None:
        raise ValueError("必須提供 Falx 影像以計算 ALVI!")
    
    falx_plane = fit_falx_plane(falx_img, verbose=verbose)
    
    # 1. 計算腦室前後徑 (PCA 方法)
    vent_result = calculate_ventricle_ap_diameter(left_vent, right_vent, falx_img=falx_img, verbose=verbose)
    ventricle_ap = vent_result['diameter_mm']
    z_range = vent_result['z_range']
    
    if verbose:
        print("\n" + "-" * 70)
    
    # 2. 計算顱骨內前後徑 (在相同 Z 範圍,使用 Falx 平面)
    skull_result = calculate_skull_ap_diameter(original_img, z_range, falx_plane=falx_plane, verbose=verbose)
    skull_ap = skull_result['diameter_mm']
    
    # 3. 計算 ALVI
    alvi = ventricle_ap / skull_ap
    alvi_percent = alvi * 100
    
    # 取得體素間距
    voxel_size = left_vent.header.get_zooms()[:3]
    
    if verbose:
        print("\n" + "-" * 70)
        print("ALVI 計算結果:")
        print(f"  腦室前後徑: {ventricle_ap:.2f} mm")
        print(f"  顱骨前後徑: {skull_ap:.2f} mm")
        print(f"  ALVI: {alvi:.4f} ({alvi_percent:.2f}%)")
        
        if alvi > 0.5:
            print(f"  ⚠️  ALVI > 0.5, 提示腦室擴大 (可能為 NPH)")
        else:
            print(f"  ✓  ALVI < 0.5, 正常範圍")
        
        print("=" * 70 + "\n")
    
    return {
        'ventricle_ap_diameter_mm': ventricle_ap,
        'left_diameter_mm': vent_result.get('left_diameter_mm'),
        'right_diameter_mm': vent_result.get('right_diameter_mm'),
        'skull_ap_diameter_mm': skull_ap,
        'alvi': alvi,
        'alvi_percent': alvi_percent,
        'ventricle_endpoints': {
            'anterior': vent_result['anterior_point'],
            'posterior': vent_result['posterior_point']
        },
        'skull_endpoints': {
            'anterior': skull_result['anterior_point'],
            'posterior': skull_result['posterior_point']
        },
        'z_range': z_range,
        'voxel_size': voxel_size
    }
