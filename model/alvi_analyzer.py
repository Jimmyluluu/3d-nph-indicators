#!/usr/bin/env python3
"""
ALVI (Anteroposterior Lateral Ventricle Index) 計算模組
使用 3D + Falx 特徵方法
"""

import numpy as np
from scipy.ndimage import label
from model.image_processing import get_image_data, convert_voxel_to_physical
from model.calculation import fit_falx_plane


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


def calculate_ventricle_ap_diameter(left_vent, right_vent, falx_img=None, z_range_percent=(0.3, 0.7), verbose=True):
    """
    計算腦室體部的前後徑 (使用 PCA 方法)
    
    分別計算左右腦室的長徑,然後取最大值 (代表最嚴重的擴大程度)
    
    Args:
        left_vent: 左腦室影像物件
        right_vent: 右腦室影像物件
        falx_img: Falx 影像物件 (用於定義中線，可選)
        z_range_percent: Z 軸篩選範圍 (預設 30%-70% 為體部)
        verbose: 是否顯示計算過程
    
    Returns:
        dict: {
            'diameter_mm': float,           # 最大前後徑 (mm)
            'left_diameter_mm': float,      # 左腦室前後徑
            'right_diameter_mm': float,     # 右腦室前後徑
            'anterior_point': tuple,        # 前端點座標
            'posterior_point': tuple,       # 後端點座標
            'z_range': tuple,               # 使用的 Z 軸範圍 (最大側的範圍)
            'body_points_count': int        # 體部總點數
        }
    """
    if verbose:
        print("計算腦室前後徑 (PCA 方法 - 取左右最大值)...")
    
    # Step 1: 取得左右腦室非零點 (先進行連通區域過濾，去除非主體的噪聲島)
    left_data_raw = get_image_data(left_vent)
    left_data = get_largest_connected_component(left_data_raw)
    left_coords = np.argwhere(left_data > 0)
    left_points = convert_voxel_to_physical(left_coords, left_vent.affine)
    
    right_data_raw = get_image_data(right_vent)
    right_data = get_largest_connected_component(right_data_raw)
    right_coords = np.argwhere(right_data > 0)
    right_points = convert_voxel_to_physical(right_coords, right_vent.affine)
    
    if verbose:
        print(f"  左腦室點數 (去噪後): {len(left_points)}")
        print(f"  右腦室點數 (去噪後): {len(right_points)}")
    
    # Step 1.5: 過濾跨越中線的錯誤標記點
    if falx_img is not None:
        # 使用 Falx 平面作為中線
        try:
            falx_plane = fit_falx_plane(falx_img, verbose=False)
            A, B, C, D = falx_plane['A'], falx_plane['B'], falx_plane['C'], falx_plane['D']
            
            # 計算每個點到 Falx 平面的有向距離
            # 距離 = (Ax + By + Cz + D) / sqrt(A^2 + B^2 + C^2)
            # 正值表示在法向量指向的一側（右側），負值表示在另一側（左側）
            norm = np.sqrt(A**2 + B**2 + C**2)
            
            left_distances = (A * left_points[:, 0] + B * left_points[:, 1] + 
                            C * left_points[:, 2] + D) / norm
            right_distances = (A * right_points[:, 0] + B * right_points[:, 1] + 
                             C * right_points[:, 2] + D) / norm
            
            # 左腦室應該在左側（負距離），右腦室應該在右側（正距離）
            left_points_filtered = left_points[left_distances < 0]
            right_points_filtered = right_points[right_distances > 0]
            
            left_points = left_points_filtered
            right_points = right_points_filtered
            
            if verbose:
                print(f"  使用 Falx 平面作為中線")
                print(f"  左腦室點數 (過濾跨線點後): {len(left_points)}")
                print(f"  右腦室點數 (過濾跨線點後): {len(right_points)}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Falx 平面擬合失敗: {e}，使用中位數方法")
            # Fallback to median method
            all_points = np.vstack([left_points, right_points])
            midline_x = np.median(all_points[:, 0])
            left_points = left_points[left_points[:, 0] < midline_x]
            right_points = right_points[right_points[:, 0] > midline_x]
            if verbose:
                print(f"  中線 X 座標: {midline_x:.2f} mm")
                print(f"  左腦室點數 (過濾跨線點後): {len(left_points)}")
                print(f"  右腦室點數 (過濾跨線點後): {len(right_points)}")
    else:
        # 沒有 Falx 影像時，使用中位數方法
        all_points = np.vstack([left_points, right_points])
        midline_x = np.median(all_points[:, 0])
        left_points = left_points[left_points[:, 0] < midline_x]
        right_points = right_points[right_points[:, 0] > midline_x]
        if verbose:
            print(f"  使用中位數方法 (無 Falx)")
            print(f"  中線 X 座標: {midline_x:.2f} mm")
            print(f"  左腦室點數 (過濾跨線點後): {len(left_points)}")
            print(f"  右腦室點數 (過濾跨線點後): {len(right_points)}")
    
    # Step 2: 分別對左右腦室計算 (包含獨立的 Z 軸範圍篩選)
    def calculate_single_ventricle_diameter(points, name):
        """計算單側腦室的長徑 (獨立計算 Z 軸範圍)"""
        if len(points) == 0:
             return {
                'diameter': 0.0,
                'anterior': (0,0,0),
                'posterior': (0,0,0),
                'body_count': 0,
                'z_range': (0,0)
            }

        # 該側的 Z 軸範圍
        z_p30 = np.percentile(points[:, 2], z_range_percent[0] * 100)
        z_p70 = np.percentile(points[:, 2], z_range_percent[1] * 100)
        
        # 篩選體部
        body_mask = (points[:, 2] >= z_p30) & (points[:, 2] <= z_p70)
        body_points = points[body_mask]
        
        if len(body_points) == 0:
            if verbose: print(f"  ⚠️ {name} 篩選體部後沒有點")
            return {
                'diameter': 0.0,
                'anterior': (0,0,0),
                'posterior': (0,0,0),
                'body_count': 0,
                'z_range': (z_p30, z_p70)
            }
        
        # PCA - 使用 SVD 找主軸方向
        centroid = np.mean(body_points, axis=0)
        centered = body_points - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_axis = Vt[0]  # 第一主成分 = 腦室長軸方向
        
        # 投影到主軸
        projections = centered @ principal_axis
        
        # 排除異常點: 使用百分位數 (0.5% - 99.5%)
        # 這可以去除分割結果中某些遠離本體的噪聲點 (例如單個像素誤差)
        p_min_val = np.percentile(projections, 0.5)
        p_max_val = np.percentile(projections, 99.5)
        
        # 計算長徑 (基於去噪後的範圍, 但為了避免過度縮短, 若去噪後差距過大可考慮用 0-100%)
        # 既然用戶明確要求去掉 "明顯有問題的點", 99% 的信賴區間是合理的
        diameter = p_max_val - p_min_val
        
        # 找前後端點座標 (最接近 percentile 的實際點)
        # 用 argmin 找絕對差最小的索引
        anterior_idx = np.argmin(np.abs(projections - p_max_val))
        posterior_idx = np.argmin(np.abs(projections - p_min_val))
        
        anterior_point = body_points[anterior_idx]
        posterior_point = body_points[posterior_idx]
        
        return {
            'diameter': diameter,
            'anterior': anterior_point,
            'posterior': posterior_point,
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


def calculate_skull_ap_diameter(original_img, z_range, verbose=True):
    """
    計算顱骨內前後徑 (在指定 Z 範圍內的 Y 軸最大距離)
    
    Args:
        original_img: 原始腦部影像物件
        z_range: (z_min, z_max) Z 軸範圍
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
        print("計算顱骨內前後徑 (Y 軸最大距離)...")
    
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
    
    # Step 4: 計算 Y 軸最大距離
    y_min_idx = np.argmin(filtered_points[:, 1])
    y_max_idx = np.argmax(filtered_points[:, 1])
    
    y_min = filtered_points[y_min_idx, 1]
    y_max = filtered_points[y_max_idx, 1]
    diameter = y_max - y_min
    
    # Step 5: 取得前後端點座標
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


def calculate_alvi(left_vent, right_vent, original_img, falx_img=None, verbose=True):
    """
    計算 ALVI (Anteroposterior Lateral Ventricle Index)
    
    ALVI = 側腦室前後徑 / 顱骨內前後徑
    - 正常值: < 0.5
    - NPH 診斷閾值: > 0.5
    
    Args:
        left_vent: 左腦室影像物件 (已拉正到 RAS+)
        right_vent: 右腦室影像物件 (已拉正到 RAS+)
        original_img: 原始腦部影像物件 (已拉正到 RAS+)
        falx_img: Falx 影像 (用於定義中線過濾，可選)
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
    
    # 1. 計算腦室前後徑 (PCA 方法)
    vent_result = calculate_ventricle_ap_diameter(left_vent, right_vent, falx_img=falx_img, verbose=verbose)
    ventricle_ap = vent_result['diameter_mm']
    z_range = vent_result['z_range']
    
    if verbose:
        print("\n" + "-" * 70)
    
    # 2. 計算顱骨內前後徑 (在相同 Z 範圍)
    skull_result = calculate_skull_ap_diameter(original_img, z_range, verbose=verbose)
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
