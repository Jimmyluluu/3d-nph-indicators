#!/usr/bin/env python3
"""
Callosal 共用幾何流程。

此模組專注在「切面與幾何取樣」，不直接計算最終指標，
供 callosal_angle 與 callosal_area 共用。
"""

import numpy as np

from model.image_processing import get_image_data, convert_voxel_to_physical
from model.calculation import fit_falx_plane, filter_points_by_falx_side, project_points_to_plane
from model.alvi_analyzer import calculate_ventricle_ap_diameter, get_largest_connected_component


def build_coronal_plane(falx_plane, ap_vector, centroid, verbose=True):
    """結合 Falx 與 AP 方向建立冠狀面。"""
    if verbose:
        print("  建立 Callosal 計算用的鉛垂平面...")

    falx_normal = falx_plane['normal']
    ap_vector = np.asarray(ap_vector, dtype=float)
    ap_vector = ap_vector / np.linalg.norm(ap_vector)

    ap_vector_proj = ap_vector - np.dot(ap_vector, falx_normal) * falx_normal
    if np.linalg.norm(ap_vector_proj) > 1e-6:
        ap_vector = ap_vector_proj / np.linalg.norm(ap_vector_proj)

    coronal_normal = ap_vector
    if coronal_normal[1] < 0:
        coronal_normal = -coronal_normal

    A, B, C = coronal_normal
    D = -np.dot(coronal_normal, centroid)

    if verbose:
        print(f"    三腦室中點: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) mm")
        print(f"    AP 方向向量: ({ap_vector[0]:.2f}, {ap_vector[1]:.2f}, {ap_vector[2]:.2f})")
        print(f"    冠狀面法向量: ({A:.4f}, {B:.4f}, {C:.4f})")

    return {
        'normal': coronal_normal,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'center': centroid,
    }


def remove_outliers_iqr(points, k=1.5):
    """用 IQR 方法過濾離群點。"""
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
    """取得單側腦室點雲（物理座標）。"""
    data_raw = get_image_data(vent_img)
    data = get_largest_connected_component(data_raw)
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        return np.array([])
    return convert_voxel_to_physical(coords, vent_img.affine)


def get_posterior_point(vent_img):
    """取得第三腦室最 posterior 的穩健錨點。"""
    data = get_image_data(vent_img)
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        raise ValueError("三腦室 mask 為空，無法取得 posterior 錨點")

    points = convert_voxel_to_physical(coords, vent_img.affine)
    points_clean = remove_outliers_iqr(points)
    if len(points_clean) == 0:
        points_clean = points

    y_threshold = np.percentile(points_clean[:, 1], 5)
    posterior_candidates = points_clean[points_clean[:, 1] <= y_threshold]
    return np.mean(posterior_candidates, axis=0)


def find_line_intersection_3d(p1, d1, p2, d2):
    """求兩條 3D 直線最近點中點。"""
    w = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return (p1 + p2) / 2

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    closest1 = p1 + t * d1
    closest2 = p2 + s * d2
    return (closest1 + closest2) / 2


def find_ray_intersection_3d(p1, d1, p2, d2):
    """求兩條 3D 射線最近點中點（t, s >= 0）。"""
    w = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return (p1 + p2) / 2

    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom

    # 射線限制：只允許往正向延伸
    t = max(0.0, t)
    s = max(0.0, s)

    closest1 = p1 + t * d1
    closest2 = p2 + s * d2
    return (closest1 + closest2) / 2


def fit_medial_wall_line(section, side):
    """對截面內側壁擬合方向線（medial wall）。"""
    filtered_section = remove_outliers_iqr(section)
    if len(filtered_section) < 10:
        filtered_section = section

    z_top_threshold = np.percentile(filtered_section[:, 2], 90)
    top_points = filtered_section[filtered_section[:, 2] >= z_top_threshold]
    if len(top_points) == 0:
        top_points = filtered_section

    if side == 'left':
        anchor_point = top_points[np.argmax(top_points[:, 0])]
        x_mid = np.median(section[:, 0])
        medial = section[section[:, 0] >= x_mid]
    else:
        anchor_point = top_points[np.argmin(top_points[:, 0])]
        x_mid = np.median(section[:, 0])
        medial = section[section[:, 0] <= x_mid]

    if len(medial) < 2:
        medial = section

    center = np.mean(medial, axis=0)
    centered = medial - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)

    if direction[2] > 0:
        direction = -direction

    return anchor_point, direction


def compute_angle_vertex(left_section, right_section):
    """對左右截面內側壁擬合後，回傳 3D 幾何頂點與方向。"""
    if len(left_section) == 0 or len(right_section) == 0:
        return None

    left_center, left_dir = fit_medial_wall_line(left_section, 'left')
    right_center, right_dir = fit_medial_wall_line(right_section, 'right')

    vertex = find_line_intersection_3d(left_center, left_dir, right_center, right_dir)
    return vertex, left_dir, right_dir


def extract_ventricle_cross_section(points, coronal_plane, thickness=2.0):
    """在冠狀面切取截面並投影到平面。"""
    if len(points) == 0:
        return np.array([])

    A, B, C, D = coronal_plane['A'], coronal_plane['B'], coronal_plane['C'], coronal_plane['D']
    norm_sq = A**2 + B**2 + C**2
    distances = (A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / np.sqrt(norm_sq)

    mask = np.abs(distances) <= thickness
    section_points = points[mask]
    if len(section_points) > 0:
        return project_points_to_plane(section_points, coronal_plane)
    return np.array([])


def compute_callosal_geometry(left_vent, right_vent, third_vent, falx_img, verbose=True, thickness=2.0):
    """共用 Callosal 幾何流程。"""
    if falx_img is None or third_vent is None:
        raise ValueError("必須提供 Falx 與三腦室影像")

    pc_anchor = get_posterior_point(third_vent)

    tv_data = get_image_data(third_vent)
    tv_coords = np.argwhere(tv_data > 0)
    if len(tv_coords) == 0:
        raise ValueError("三腦室 mask 為空！")
    tv_pts = convert_voxel_to_physical(tv_coords, third_vent.affine)
    tv_pts_clean = remove_outliers_iqr(tv_pts)
    if len(tv_pts_clean) == 0:
        tv_pts_clean = tv_pts
    third_centroid = np.mean(tv_pts_clean, axis=0)

    falx_plane = fit_falx_plane(falx_img, verbose=False)
    vent_result = calculate_ventricle_ap_diameter(
        left_vent,
        right_vent,
        falx_img=falx_img,
        verbose=False,
    )

    ant_pt = np.array(vent_result['anterior_point'])
    post_pt = np.array(vent_result['posterior_point'])
    if np.all(ant_pt == 0) and np.all(post_pt == 0):
        ap_vector = np.array([0, 1, 0])
        if verbose:
            print("  ⚠️ 無法取得 APVI 前後徑方向，降級使用標準 Y 軸向量")
    else:
        ap_vector = ant_pt - post_pt

    coronal_plane = build_coronal_plane(falx_plane, ap_vector, pc_anchor, verbose=verbose)

    left_points = get_ventricle_points(left_vent)
    right_points = get_ventricle_points(right_vent)
    third_points = get_ventricle_points(third_vent)

    left_points = filter_points_by_falx_side(left_points, falx_plane, 'left', verbose=False)
    right_points = filter_points_by_falx_side(right_points, falx_plane, 'right', verbose=False)

    left_section = extract_ventricle_cross_section(left_points, coronal_plane, thickness=thickness)
    right_section = extract_ventricle_cross_section(right_points, coronal_plane, thickness=thickness)
    third_section = extract_ventricle_cross_section(third_points, coronal_plane, thickness=thickness)

    left_wall_center = None
    right_wall_center = None
    left_dir = None
    right_dir = None
    # 依最新需求：第三頂點固定使用三腦室質心，不使用左右壁延伸交點
    vertex = np.asarray(third_centroid, dtype=float)

    if len(left_section) > 0 and len(right_section) > 0:
        left_wall_center, left_dir = fit_medial_wall_line(left_section, 'left')
        right_wall_center, right_dir = fit_medial_wall_line(right_section, 'right')

    return {
        'pc_anchor': pc_anchor,
        'third_centroid': third_centroid,
        'coronal_plane': coronal_plane,
        'left_section': left_section,
        'right_section': right_section,
        'third_section': third_section,
        'left_highest_point': tuple(left_wall_center) if left_wall_center is not None else (0, 0, 0),
        'right_highest_point': tuple(right_wall_center) if right_wall_center is not None else (0, 0, 0),
        'left_wall_dir': tuple(left_dir) if left_dir is not None else (0, 0, 0),
        'right_wall_dir': tuple(right_dir) if right_dir is not None else (0, 0, 0),
        'vertex': vertex,
    }
