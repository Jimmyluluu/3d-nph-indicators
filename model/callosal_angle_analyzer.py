#!/usr/bin/env python3
"""
Callosal Angle (胼胝體角) 計算模組
結合三腦室中點、Falx 平面和 APVI 前後徑來建立測量平面並計算角度
"""

import numpy as np
from scipy.ndimage import label
from model.image_processing import get_image_data, convert_voxel_to_physical
from model.calculation import (
    fit_falx_plane, filter_points_by_falx_side, 
    project_points_to_plane, calculate_centroid_3d
)
from model.alvi_analyzer import calculate_ventricle_ap_diameter, get_largest_connected_component

def build_coronal_plane(falx_plane, ap_vector, centroid, verbose=True):
    """
    結合 Falx 法向量與 AP 方向向量，建立鉛垂平面（冠狀面）

    Args:
        falx_plane: Falx 平面參數 (包含 'normal')
        ap_vector: 腦室最長前後徑方向向量 (從 posterior 到 anterior)
        centroid: 三腦室中點座標 (用來做平面的基準點)
        verbose: 是否顯示計算過程

    Returns:
        dict: 平面參數
            - 'normal': 法向量 (A, B, C)
            - 'A', 'B', 'C', 'D': 平面方程式參數 (Ax + By + Cz + D = 0)
    """
    if verbose:
        print("  建立 Callosal Angle 計算用的鉛垂平面...")
        
    # 1. 取得 Falx 平面的法向量 (指向左或右，即 X 方向的主分量)
    falx_normal = falx_plane['normal'] 
    
    # 2. AP 向量 (指向前，即 Y 方向)
    ap_vector = np.array(ap_vector)
    ap_vector = ap_vector / np.linalg.norm(ap_vector)
    
    # 3. 確保 falx_normal 與 ap_vector 正交
    # (如果 ap_vector 與 Falx 不完全平行，可以用 Gram-Schmidt 過程使其正交於 Falx 法向量)
    ap_vector_proj = ap_vector - np.dot(ap_vector, falx_normal) * falx_normal
    if np.linalg.norm(ap_vector_proj) > 1e-6:
        ap_vector = ap_vector_proj / np.linalg.norm(ap_vector_proj)
        
    # 4. 計算上方向向量 (Z 方向) = AP 向量 × Falx 法向量 (外積)
    # 取決於方向的設定，這裡我們只需要找一個正交的 Z 向量即可
    up_vector = np.cross(ap_vector, falx_normal)
    up_vector = up_vector / np.linalg.norm(up_vector)
    # 確保 up_vector 是指向上方 (Z 分量 > 0)
    if up_vector[2] < 0:
        up_vector = -up_vector
        
    # 5. 建立「鉛垂平面」，這個平面要包含 AP 向量和 up 向量
    # 所以這個平面的法向量 就是 Falx 的法向量的變體？
    # 等等，如果平面是用來「橫切兩個側腦室」，那就是冠狀面 (Coronal Plane)
    # 冠狀面的法向量應該是指向前後 (Y方向)。
    # 也就是說，這個冠狀面垂直於 AP 向量。
    coronal_normal = ap_vector
    
    # 確保法向量指向前方 (Y > 0)
    if coronal_normal[1] < 0:
        coronal_normal = -coronal_normal
        
    A, B, C = coronal_normal
    
    # D = -(A*x0 + B*y0 + C*z0)
    # 平面必須通過三腦室的質心 (centroid)
    D = -np.dot(coronal_normal, centroid)
    
    if verbose:
        print(f"    三腦室中點: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) mm")
        print(f"    AP 方向向量: ({ap_vector[0]:.2f}, {ap_vector[1]:.2f}, {ap_vector[2]:.2f})")
        print(f"    冠狀面法向量: ({A:.4f}, {B:.4f}, {C:.4f})")
        
    return {
        'normal': coronal_normal,
        'A': A, 'B': B, 'C': C, 'D': D,
        'center': centroid
    }

def remove_outliers_iqr(points, k=1.5):
    """
    用 IQR 方法去除欲群點，對 X/Y/Z 三軸分別自評估并公等交集。

    Args:
        points: (N, 3) 物理座標點群
        k: IQR 倍數，越大容忍越寬，預設 1.5

    Returns:
        np.array: 遞除極端值後的點群
    """
    if len(points) == 0:
        return points
    mask = np.ones(len(points), dtype=bool)
    for axis in range(3):
        col = points[:, axis]
        q1, q3 = np.percentile(col, 25), np.percentile(col, 75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask &= (col >= lower) & (col <= upper)
    return points[mask]


def get_ventricle_points(vent_img):
    """獲取單側腦室的點座標(物理空間)"""
    data_raw = get_image_data(vent_img)
    data = get_largest_connected_component(data_raw)
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        return np.array([])
    points = convert_voxel_to_physical(coords, vent_img.affine)
    return points


def get_posterior_point(vent_img):
    """
    取得三腦室最後方的點（RAS 座標中 Y 最小），作為 Posterior Commissure 的近似位置。
    先做 IQR 過濾去除標記錯誤的孤立點，再取第 5 百分位數（Y 最小）的點。

    Args:
        vent_img: 三腦室影像

    Returns:
        np.array: 物理座標 (x, y, z)
    """
    data = get_image_data(vent_img)
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        raise ValueError("三腦室 mask 為空，無法取得 posterior 錨點")
    points = convert_voxel_to_physical(coords, vent_img.affine)

    # 先用 IQR 過濾去標記錯誤的極端值
    points_clean = remove_outliers_iqr(points)
    if len(points_clean) == 0:
        points_clean = points  # 如果過濾後為空，從後用原始點

    # RAS 座標中 Y 軸朝 Anterior，Y 最小即最 Posterior
    # 用第 5 百分位數（而非單點最小）提高穩定性
    y_threshold = np.percentile(points_clean[:, 1], 5)
    posterior_candidates = points_clean[points_clean[:, 1] <= y_threshold]

    # 在候選點中取質心（避免單點奮際）
    return np.mean(posterior_candidates, axis=0)


def find_line_intersection_3d(p1, d1, p2, d2):
    """
    求兩條 3D 直線的最近交點（最小二乘意義上的交點）。

    Line1: P(t) = p1 + t * d1
    Line2: Q(s) = p2 + s * d2

    Returns:
        np.array: 兩條線最近點的中點
    """
    w = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # 平行線，回傳兩點中點
        return (p1 + p2) / 2

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    closest1 = p1 + t * d1
    closest2 = p2 + s * d2
    return (closest1 + closest2) / 2


def fit_medial_wall_line(section, side, medial_fraction=0.25):
    """
    取腦室截面中朝中線那側的邊緣點群，擬合 medial wall 方向。

    臨床意義：
        - 左腦室 medial wall = X 最大的那側（面對中線，在 RAS 中 X 朝右）
        - 右腦室 medial wall = X 最小的那側（面對中線）
    這樣擬合出的線只在自己這側，不會串到另一側腦室。

    Args:
        section: (N, 3) 截面點群
        side: 'left' 或 'right'
        medial_fraction: 取 X 最端的多少比例（預設 25%）

    Returns:
        (center, direction): 直線中心點與方向向量（已正規化）
    """
    # **過濾雜訊**: 預防單一離群點（如上方雜訊）成為最高點
    # 將截面點群經過 IQR 過濾後，再取最高區域
    filtered_section = remove_outliers_iqr(section)
    if len(filtered_section) < 10:  # 如果過濾後點太少，退回使用原始截面
        filtered_section = section

    # 找出過濾後，Z 最高的區域（例如前 10%）
    z_top_threshold = np.percentile(filtered_section[:, 2], 90)
    top_points = filtered_section[filtered_section[:, 2] >= z_top_threshold]

    if side == 'left':
        # 左腦室：出發點為「最上面且最右側（X 最大）」
        # 從 top_points 中找 X 最大的點當作頂端錨點
        anchor_point = top_points[np.argmax(top_points[:, 0])]
        # 內側壁點群：取整個截面中 X 較大（靠右）的那一半
        x_mid = np.median(section[:, 0])
        medial = section[section[:, 0] >= x_mid]
    else:
        # 右腦室：出發點為「最上面且最左側（X 最小）」
        # 從 top_points 中找 X 最小的點當作頂端錨點
        anchor_point = top_points[np.argmin(top_points[:, 0])]
        # 內側壁點群：取整個截面中 X 較小（靠左）的那一半
        x_mid = np.median(section[:, 0])
        medial = section[section[:, 0] <= x_mid]

    if len(medial) < 2:
        medial = section  # 點太少就退回全截面

    # 用 SVD 擬合內側壁的整體走向
    center = np.mean(medial, axis=0)
    centered = medial - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # 最大方差方向（沿 medial wall 走向）
    direction = direction / np.linalg.norm(direction)

    # 確保方向是「由上往下」指
    if direction[2] > 0:
        direction = -direction

    # 把錨點 (最高最內側) 當作這條線的起點/代表點回傳
    return anchor_point, direction


def compute_angle_vertex(left_section, right_section):
    """
    對左右腦室截面各自擬合 medial wall 方向，
    取兩條擬合線的 3D 交點為角度頂點。

    醫學意義：沿著兩側腦室內側壁（medial wall）各自延伸，
    兩條線的交點為頂點，夾角即 Callosal Angle。

    Args:
        left_section: 左腦室截面點群 (N, 3)
        right_section: 右腦室截面點群 (N, 3)

    Returns:
        tuple (vertex, left_dir, right_dir) 或 None
    """
    if len(left_section) == 0 or len(right_section) == 0:
        return None

    left_center,  left_dir  = fit_medial_wall_line(left_section,  'left')
    right_center, right_dir = fit_medial_wall_line(right_section, 'right')

    vertex = find_line_intersection_3d(left_center, left_dir, right_center, right_dir)
    return vertex, left_dir, right_dir


def extract_ventricle_cross_section(points, coronal_plane, thickness=2.0):
    """
    在冠狀面處切取腦室截面
    
    Args:
        points: 腦室的 3D 點雲
        coronal_plane: 冠狀面參數
        thickness: 切片厚度(mm)
        
    Returns:
        np.array: 切片範圍內的點雲，並投影在平面上
    """
    if len(points) == 0:
        return np.array([])
        
    A, B, C, D = coronal_plane['A'], coronal_plane['B'], coronal_plane['C'], coronal_plane['D']
    normal = coronal_plane['normal']
    norm_sq = A**2 + B**2 + C**2
    
    # 計算每個點到平面的距離
    distances = (A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / np.sqrt(norm_sq)
    
    # 篩選在厚度範圍內的點
    mask = np.abs(distances) <= thickness
    section_points = points[mask]
    
    # 投影到平面上以便後續計算
    if len(section_points) > 0:
        return project_points_to_plane(section_points, coronal_plane)
    return np.array([])

def calculate_callosal_angle(left_vent, right_vent, third_vent, falx_img, verbose=True):
    """
    計算 Callosal Angle
    
    Args:
        left_vent: 左側腦室影像
        right_vent: 右側腦室影像
        third_vent: 三腦室影像
        falx_img: Falx 影像
        verbose: 是否印出資訊
        
    Returns:
        dict: 角度計算結果
    """
    if verbose:
        print("\n" + "=" * 70)
        print("開始計算 Callosal Angle (胼胝體角)")
        print("=" * 70)
        
    if falx_img is None or third_vent is None:
        raise ValueError("必須提供 Falx 與三腦室影像以計算 Callosal Angle!")
        
    # Step 1: 取得切面錨點
    # 用三腦室最後方的點（Y 最小）近似 Posterior Commissure 位置
    # 比質心更符合臨床標準：「在 PC 水平的冠狀切面量角」
    pc_anchor = get_posterior_point(third_vent)

    # 三腦室質心也先 IQR 過濾再取平均，與 get_posterior_point 一致
    _tv_data = get_image_data(third_vent)
    _tv_coords = np.argwhere(_tv_data > 0)
    if len(_tv_coords) == 0:
        raise ValueError("三腦室 mask 為空！")
    from model.image_processing import convert_voxel_to_physical as _cvp
    _tv_pts = _cvp(_tv_coords, third_vent.affine)
    _tv_pts_clean = remove_outliers_iqr(_tv_pts)
    if len(_tv_pts_clean) == 0:
        _tv_pts_clean = _tv_pts
    third_centroid = np.mean(_tv_pts_clean, axis=0)

    if verbose:
        print(f"  PC 近似錨點（三腦室最後方）: ({pc_anchor[0]:.2f}, {pc_anchor[1]:.2f}, {pc_anchor[2]:.2f}) mm")
        print(f"  三腦室質心（僅供參考）:       ({third_centroid[0]:.2f}, {third_centroid[1]:.2f}, {third_centroid[2]:.2f}) mm")

    # Step 2: 取得 Falx 平面與腦室的 APVI 前後徑方向
    falx_plane = fit_falx_plane(falx_img, verbose=False)
    vent_result = calculate_ventricle_ap_diameter(left_vent, right_vent, falx_img=falx_img, verbose=False)

    # 取得最長徑的端點，並計算方向向量
    ant_pt = np.array(vent_result['anterior_point'])
    post_pt = np.array(vent_result['posterior_point'])

    if np.all(ant_pt == 0) and np.all(post_pt == 0):
        ap_vector = np.array([0, 1, 0])
        if verbose: print("  ⚠️ 無法取得 APVI 前後徑方向，降級使用標準 Y 軸向量")
    else:
        ap_vector = ant_pt - post_pt

    # Step 3: 結合 Falx 和 AP 向量建立測量平面（冠狀面），以 PC 錨點為通過點
    coronal_plane = build_coronal_plane(falx_plane, ap_vector, pc_anchor, verbose=verbose)
    
    # Step 4: 橫切腦室
    left_points = get_ventricle_points(left_vent)
    right_points = get_ventricle_points(right_vent)
    
    # 將點進一步過濾以確保分層正確 (左在左側，右在右側) 
    left_points = filter_points_by_falx_side(left_points, falx_plane, 'left', verbose=False)
    right_points = filter_points_by_falx_side(right_points, falx_plane, 'right', verbose=False)
    
    # 切面取樣
    left_section = extract_ventricle_cross_section(left_points, coronal_plane, thickness=2.0)
    right_section = extract_ventricle_cross_section(right_points, coronal_plane, thickness=2.0)
    
    if verbose:
        print(f"  左腦室截面點數: {len(left_section)}")
        print(f"  右腦室截面點數: {len(right_section)}")
        
    # Step 5: 用 SVD 擬合頂壁方向，計算夾角
    # 醫學意義：沿著兩側腦室頂壁（靠胼胝體那面）各自擬合一條線，量這兩條線的夾角
    angle = 0.0
    vertex = pc_anchor
    left_wall_center = None
    right_wall_center = None
    left_dir = None
    right_dir = None

    if len(left_section) > 0 and len(right_section) > 0:
        result = compute_angle_vertex(left_section, right_section)
        if result is None:
            vertex = pc_anchor
            if verbose:
                print("  ⚠️ 無法擬合頂壁，降級使用 PC 錨點作為頂點")
        else:
            vertex, left_dir, right_dir = result
            left_wall_center = fit_medial_wall_line(left_section, 'left')[0]
            right_wall_center = fit_medial_wall_line(right_section, 'right')[0]

            if verbose:
                print(f"  角度頂點（頂壁擬合線交點）: ({vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f}) mm")

            # 角度 = 兩條 medial wall 擬合線的夾角
            dot_product = np.clip(np.dot(left_dir, right_dir), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))
    else:
        if verbose: print("  ⚠️ 無法在截面處找到腦室。可能冠狀面位置無腦室結構")

    if verbose:
        print("\n" + "-" * 70)
        print("Callosal Angle 計算結果:")
        print(f"  Callosal Angle: {angle:.1f}°")
        if angle > 0:
            if angle <= 80:
                print(f"  ⚠️  Callosal Angle ≤ 80° (實際為 {angle:.1f}°), 符合 iNPH 典型範圍 (50–80°)")
            elif angle <= 100:
                print(f"  ⚠️  Callosal Angle 80–100° (實際為 {angle:.1f}°), 邊界區域，建議進一步評估")
            else:
                print(f"  ✓  Callosal Angle > 100° (實際為 {angle:.1f}°), 正常範圍 (100–120°)")
        print("=" * 70 + "\n")

    # 視覺化用：把頂壁擬合中心當作「代表性最高點」
    l_vis = tuple(left_wall_center)  if left_wall_center  is not None else (0, 0, 0)
    r_vis = tuple(right_wall_center) if right_wall_center is not None else (0, 0, 0)

    return {
        'angle': angle,
        'center_point': pc_anchor,          # 切面通過點（PC 近似）
        'vertex': vertex,                   # 角度頂點（頂壁擬合線交點）
        'third_centroid': third_centroid,
        'left_highest_point':  l_vis,       # 左側頂壁擬合中心（視覺化用）
        'right_highest_point': r_vis,       # 右側頂壁擬合中心（視覺化用）
        'left_wall_dir':  tuple(left_dir)  if left_dir  is not None else (0, 0, 0),
        'right_wall_dir': tuple(right_dir) if right_dir is not None else (0, 0, 0),
        'coronal_plane': coronal_plane
    }
