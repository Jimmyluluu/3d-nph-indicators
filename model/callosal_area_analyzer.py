#!/usr/bin/env python3
"""
Callosal 平面三角形面積計算模組
結合三腦室錨點、Falx 平面和 APVI 前後徑來建立測量平面並計算面積
"""

import numpy as np
from scipy.spatial import ConvexHull
from model.image_processing import get_image_data, convert_voxel_to_physical
from model.calculation import (
    fit_falx_plane, filter_points_by_falx_side, 
    project_points_to_plane
)
from model.alvi_analyzer import calculate_ventricle_ap_diameter, get_largest_connected_component
from model.callosal_geometry import compute_callosal_geometry

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
        print("  建立 Callosal 面積計算用的鉛垂平面...")
        
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


def build_plane_basis(normal):
    """根據平面法向量建立平面內 2D 基底 (u, v)。"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v


def to_plane_uv(points, origin, u, v):
    """將 3D 點投影到平面 2D 座標 (u, v)。"""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[None, :]
    rel = pts - origin
    return np.column_stack((np.dot(rel, u), np.dot(rel, v)))


def triangle_area_3d(p1, p2, p3):
    """3D 三角形面積。"""
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


def points_in_triangle_2d(points_uv, tri_uv):
    """判斷 2D 點是否在三角形內（含邊界）。"""
    a, b, c = tri_uv
    v0 = c - a
    v1 = b - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) < 1e-10:
        return np.zeros(len(points_uv), dtype=bool)

    v2 = points_uv - a
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denom
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denom
    eps = 1e-8
    return (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)


def estimate_overlap_area_in_triangle(section_points, triangle_vertices, coronal_plane):
    """
    估算「某結構截面落在三角形內」的面積（mm²）。
    用結構內部點在三角形內的 2D ConvexHull 面積近似。
    """
    if len(section_points) < 3:
        return 0.0

    p1, p2, p3 = [np.asarray(p, dtype=float) for p in triangle_vertices]
    tri_area = triangle_area_3d(p1, p2, p3)
    if tri_area < 1e-8:
        return 0.0

    normal = np.asarray(coronal_plane['normal'], dtype=float)
    u, v = build_plane_basis(normal)
    origin = p3

    tri_uv = to_plane_uv(np.array([p1, p2, p3]), origin, u, v)
    section_uv = to_plane_uv(section_points, origin, u, v)

    inside_mask = points_in_triangle_2d(section_uv, tri_uv)
    inside_points = section_uv[inside_mask]
    if len(inside_points) < 3:
        return 0.0

    try:
        overlap_area = float(ConvexHull(inside_points).volume)
    except Exception:
        return 0.0

    return min(max(overlap_area, 0.0), tri_area)


def estimate_inclusion_exclusion_areas(left_section, right_section, third_section,
                                       triangle_vertices, coronal_plane, resolution_mm=0.25):
    """
    在三角形平面內用 raster mask 計算包含-排除面積。

    計算式：
        A_net = A(L) + A(R) - A(L∩R) - A(L∩T) - A(R∩T) + A(L∩R∩T)

    其中 L/R/T 皆已限制在三角形內。
    """
    from skimage.draw import polygon2mask

    p1, p2, p3 = [np.asarray(p, dtype=float) for p in triangle_vertices]
    triangle_area = triangle_area_3d(p1, p2, p3)
    if triangle_area < 1e-8:
        return {
            'left_in_triangle_area_mm2': 0.0,
            'right_in_triangle_area_mm2': 0.0,
            'third_in_triangle_area_mm2': 0.0,
            'left_right_overlap_area_mm2': 0.0,
            'left_third_overlap_area_mm2': 0.0,
            'right_third_overlap_area_mm2': 0.0,
            'triple_overlap_area_mm2': 0.0,
            'lateral_union_area_mm2': 0.0,
            'net_triangle_area_mm2': 0.0,
        }

    normal = np.asarray(coronal_plane['normal'], dtype=float)
    u, v = build_plane_basis(normal)
    origin = p3

    tri_uv = to_plane_uv(np.array([p1, p2, p3]), origin, u, v)

    # 只在三角形 bounding box 內 raster，可加速且避免不必要誤差
    u_min, v_min = np.min(tri_uv, axis=0)
    u_max, v_max = np.max(tri_uv, axis=0)

    pad = resolution_mm * 2.0
    u_min -= pad
    v_min -= pad
    u_max += pad
    v_max += pad

    width = max(int(np.ceil((u_max - u_min) / resolution_mm)) + 1, 4)
    height = max(int(np.ceil((v_max - v_min) / resolution_mm)) + 1, 4)
    shape = (height, width)

    def uv_to_rc(poly_uv):
        poly_uv = np.asarray(poly_uv, dtype=float)
        cols = (poly_uv[:, 0] - u_min) / resolution_mm
        rows = (poly_uv[:, 1] - v_min) / resolution_mm
        return np.column_stack((rows, cols))

    tri_mask = polygon2mask(shape, uv_to_rc(tri_uv))

    def section_mask(section_points_uv):
        if len(section_points_uv) < 3:
            return np.zeros(shape, dtype=bool)
        try:
            hull = ConvexHull(section_points_uv)
            poly_uv = section_points_uv[hull.vertices]
        except Exception:
            return np.zeros(shape, dtype=bool)
        return polygon2mask(shape, uv_to_rc(poly_uv))

    left_uv = to_plane_uv(left_section, origin, u, v) if len(left_section) else np.empty((0, 2))
    right_uv = to_plane_uv(right_section, origin, u, v) if len(right_section) else np.empty((0, 2))
    third_uv = to_plane_uv(third_section, origin, u, v) if len(third_section) else np.empty((0, 2))

    left_mask = section_mask(left_uv) & tri_mask
    right_mask = section_mask(right_uv) & tri_mask
    third_mask = section_mask(third_uv) & tri_mask

    px_area = resolution_mm * resolution_mm

    a_l = float(np.count_nonzero(left_mask) * px_area)
    a_r = float(np.count_nonzero(right_mask) * px_area)
    a_t = float(np.count_nonzero(third_mask) * px_area)

    a_lr = float(np.count_nonzero(left_mask & right_mask) * px_area)
    a_lt = float(np.count_nonzero(left_mask & third_mask) * px_area)
    a_rt = float(np.count_nonzero(right_mask & third_mask) * px_area)
    a_lrt = float(np.count_nonzero(left_mask & right_mask & third_mask) * px_area)

    a_lateral_union = float(np.count_nonzero((left_mask | right_mask)) * px_area)

    a_net = a_l + a_r - a_lr - a_lt - a_rt + a_lrt
    a_net = max(a_net, 0.0)
    a_net = min(a_net, float(triangle_area))

    return {
        'left_in_triangle_area_mm2': a_l,
        'right_in_triangle_area_mm2': a_r,
        'third_in_triangle_area_mm2': a_t,
        'left_right_overlap_area_mm2': a_lr,
        'left_third_overlap_area_mm2': a_lt,
        'right_third_overlap_area_mm2': a_rt,
        'triple_overlap_area_mm2': a_lrt,
        'lateral_union_area_mm2': a_lateral_union,
        'net_triangle_area_mm2': a_net,
    }


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

def calculate_callosal_area(left_vent, right_vent, third_vent, falx_img, verbose=True):
    """計算 Callosal 平面三角形淨面積（共用 callosal 幾何流程）。"""
    if verbose:
        print("\n" + "=" * 70)
        print("開始計算 Callosal 平面三角形面積")
        print("=" * 70)

    geometry = compute_callosal_geometry(
        left_vent,
        right_vent,
        third_vent,
        falx_img,
        verbose=verbose,
        thickness=2.0,
    )

    pc_anchor = geometry['pc_anchor']
    third_centroid = geometry['third_centroid']
    third_vertex = np.asarray(geometry['vertex'], dtype=float)
    coronal_plane = geometry['coronal_plane']
    left_section = geometry['left_section']
    right_section = geometry['right_section']
    third_section = geometry['third_section']
    left_wall_center = geometry['left_highest_point']
    right_wall_center = geometry['right_highest_point']

    if verbose:
        print(
            f"  PC 近似錨點（三腦室最後方）: "
            f"({pc_anchor[0]:.2f}, {pc_anchor[1]:.2f}, {pc_anchor[2]:.2f}) mm"
        )
        print(
            f"  三腦室質心（僅供參考）:       "
            f"({third_centroid[0]:.2f}, {third_centroid[1]:.2f}, {third_centroid[2]:.2f}) mm"
        )
        print(
            f"  三角形第三頂點（Angle Vertex）: "
            f"({third_vertex[0]:.2f}, {third_vertex[1]:.2f}, {third_vertex[2]:.2f}) mm"
        )
        print(f"  左腦室截面點數: {len(left_section)}")
        print(f"  右腦室截面點數: {len(right_section)}")
        print(f"  三腦室截面點數: {len(third_section)}")

    triangle_area = 0.0
    lateral_overlap_area = 0.0
    third_overlap_area = 0.0
    net_triangle_area = 0.0
    net_triangle_ratio = 0.0
    net_triangle_ratio_percent = 0.0
    left_in_triangle_area = 0.0
    right_in_triangle_area = 0.0
    left_right_overlap_area = 0.0
    left_third_overlap_area = 0.0
    right_third_overlap_area = 0.0
    triple_overlap_area = 0.0

    if len(left_section) > 0 and len(right_section) > 0 and left_wall_center != (0, 0, 0) and right_wall_center != (0, 0, 0):
        p_left = np.asarray(left_wall_center, dtype=float)
        p_right = np.asarray(right_wall_center, dtype=float)
        p_third = third_vertex

        triangle_area = float(triangle_area_3d(p_left, p_right, p_third))
        ie_areas = estimate_inclusion_exclusion_areas(
            left_section,
            right_section,
            third_section,
            (p_left, p_right, p_third),
            coronal_plane,
            resolution_mm=0.25,
        )

        left_in_triangle_area = ie_areas['left_in_triangle_area_mm2']
        right_in_triangle_area = ie_areas['right_in_triangle_area_mm2']
        left_right_overlap_area = ie_areas['left_right_overlap_area_mm2']
        left_third_overlap_area = ie_areas['left_third_overlap_area_mm2']
        right_third_overlap_area = ie_areas['right_third_overlap_area_mm2']
        triple_overlap_area = ie_areas['triple_overlap_area_mm2']
        lateral_overlap_area = ie_areas['lateral_union_area_mm2']
        third_overlap_area = ie_areas['third_in_triangle_area_mm2']
        net_triangle_area = ie_areas['net_triangle_area_mm2']

        if triangle_area > 1e-8:
            net_triangle_ratio = net_triangle_area / triangle_area
            net_triangle_ratio_percent = net_triangle_ratio * 100.0
    elif verbose:
        print("  ⚠️ 無法在截面處找到左右腦室，面積計算失敗")

    if verbose:
        print("\n" + "-" * 70)
        print("Callosal 平面三角形面積計算結果:")
        print(f"  三角形總面積: {triangle_area:.2f} mm²")
        print(f"  左腦室在三角形內面積: {left_in_triangle_area:.2f} mm²")
        print(f"  右腦室在三角形內面積: {right_in_triangle_area:.2f} mm²")
        print(f"  左右交疊面積: {left_right_overlap_area:.2f} mm²")
        print(f"  左三交疊面積: {left_third_overlap_area:.2f} mm²")
        print(f"  右三交疊面積: {right_third_overlap_area:.2f} mm²")
        print(f"  三者交疊面積: {triple_overlap_area:.2f} mm²")
        print(f"  左右聯集面積(三角內): {lateral_overlap_area:.2f} mm²")
        print(f"  三腦室在三角形內面積: {third_overlap_area:.2f} mm²")
        print(f"  淨面積(包含-排除): {net_triangle_area:.2f} mm²")
        print(f"  淨面積占比: {net_triangle_ratio_percent:.2f}%")
        print("=" * 70 + "\n")

    return {
        'triangle_area_mm2': triangle_area,
        'left_in_triangle_area_mm2': left_in_triangle_area,
        'right_in_triangle_area_mm2': right_in_triangle_area,
        'left_right_overlap_area_mm2': left_right_overlap_area,
        'left_third_overlap_area_mm2': left_third_overlap_area,
        'right_third_overlap_area_mm2': right_third_overlap_area,
        'triple_overlap_area_mm2': triple_overlap_area,
        'lateral_overlap_area_mm2': lateral_overlap_area,
        'third_overlap_area_mm2': third_overlap_area,
        'net_triangle_area_mm2': net_triangle_area,
        'net_triangle_ratio': net_triangle_ratio,
        'net_triangle_ratio_percent': net_triangle_ratio_percent,
        'center_point': pc_anchor,
        'vertex': tuple(third_vertex),
        'third_centroid': third_centroid,
        'left_highest_point': tuple(left_wall_center),
        'right_highest_point': tuple(right_wall_center),
        'left_wall_dir': (0, 0, 0),
        'right_wall_dir': (0, 0, 0),
        'coronal_plane': coronal_plane,
    }
