#!/usr/bin/env python3
"""
Callosal Angle (胼胝體角) 計算模組。

幾何流程與 callosal_area 共用，最後只保留舊角度算法：
兩條擬合線方向向量點積求角。
"""

import numpy as np

from model.callosal_geometry import compute_callosal_geometry


def calculate_callosal_angle(left_vent, right_vent, third_vent, falx_img, verbose=True):
    """計算 Callosal Angle。"""
    if verbose:
        print("\n" + "=" * 70)
        print("開始計算 Callosal Angle (胼胝體角)")
        print("=" * 70)

    geometry = compute_callosal_geometry(
        left_vent,
        right_vent,
        third_vent,
        falx_img,
        verbose=verbose,
        thickness=2.0,
    )

    left_section = geometry['left_section']
    right_section = geometry['right_section']
    left_dir = np.asarray(geometry['left_wall_dir'], dtype=float)
    right_dir = np.asarray(geometry['right_wall_dir'], dtype=float)
    left_anchor = np.asarray(geometry['left_highest_point'], dtype=float)
    right_anchor = np.asarray(geometry['right_highest_point'], dtype=float)

    if verbose:
        pc_anchor = geometry['pc_anchor']
        third_centroid = geometry['third_centroid']
        print(
            f"  PC 近似錨點（三腦室最後方）: "
            f"({pc_anchor[0]:.2f}, {pc_anchor[1]:.2f}, {pc_anchor[2]:.2f}) mm"
        )
        print(
            f"  三腦室質心（僅供參考）:       "
            f"({third_centroid[0]:.2f}, {third_centroid[1]:.2f}, {third_centroid[2]:.2f}) mm"
        )
        print(f"  左腦室截面點數: {len(left_section)}")
        print(f"  右腦室截面點數: {len(right_section)}")

    angle = 0.0
    angle_method = "none"
    vertex = geometry['pc_anchor']
    if len(left_section) > 0 and len(right_section) > 0:
        vertex = geometry['vertex']

        # 優先使用「視覺化同一組幾何」：left_anchor -> vertex -> right_anchor
        if not np.allclose(left_anchor, [0.0, 0.0, 0.0]) and not np.allclose(right_anchor, [0.0, 0.0, 0.0]):
            left_vec = left_anchor - np.asarray(vertex, dtype=float)
            right_vec = right_anchor - np.asarray(vertex, dtype=float)
            if np.linalg.norm(left_vec) > 1e-8 and np.linalg.norm(right_vec) > 1e-8:
                left_unit = left_vec / np.linalg.norm(left_vec)
                right_unit = right_vec / np.linalg.norm(right_vec)
                dot_product = np.clip(np.dot(left_unit, right_unit), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                angle_method = "vertex_anchor"

        # 後備：錨點不可用時才退回方向向量法
        if angle_method == "none" and np.linalg.norm(left_dir) > 0 and np.linalg.norm(right_dir) > 0:
            dot_product = np.clip(np.dot(left_dir, right_dir), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot_product))
            angle_method = "wall_dir"

        if verbose:
            print(
                f"  角度頂點（三腦室質心）: "
                f"({vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f}) mm"
            )
            if angle_method == "vertex_anchor":
                print("  角度算法: vertex->左右錨點向量夾角（與視覺化一致）")
            elif angle_method == "wall_dir":
                print("  角度算法: 左右壁方向向量夾角（錨點不可用時後備）")
            else:
                print("  ⚠️ 無有效向量可計算角度，降級為 0.0")
    elif verbose:
        print("  ⚠️ 無法在截面處找到腦室。可能冠狀面位置無腦室結構")

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

    return {
        'angle': angle,
        'angle_method': angle_method,
        'center_point': geometry['pc_anchor'],
        'vertex': vertex,
        'third_centroid': geometry['third_centroid'],
        'left_highest_point': geometry['left_highest_point'],
        'right_highest_point': geometry['right_highest_point'],
        'left_wall_dir': tuple(left_dir) if np.linalg.norm(left_dir) > 0 else geometry['left_wall_dir'],
        'right_wall_dir': tuple(right_dir) if np.linalg.norm(right_dir) > 0 else geometry['right_wall_dir'],
        'coronal_plane': geometry['coronal_plane'],
    }
