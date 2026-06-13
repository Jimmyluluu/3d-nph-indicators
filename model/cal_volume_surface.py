#!/usr/bin/env python3
"""
腦室體積與表面積計算工具
包含純計算函數，不包含其他複合指標計算
"""

import numpy as np
from model.image_processing import extract_surface_mesh, get_image_data, get_voxel_size


def calculate_surface_area(left_ventricle, right_ventricle, verbose=True):
    """
    計算左右腦室的表面積（純計算，不包含視覺化資料）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        verbose (bool): 是否顯示計算過程資訊

    Returns:
        dict: 包含表面積計算結果的字典
    """
    from skimage.measure import mesh_surface_area

    def _get_surface_area(image_obj, name):
        # 使用統一的表面提取函數 (Marching Cubes 已經提供平滑表面)
        mesh_result = extract_surface_mesh(image_obj, level=0.5, verbose=verbose)

        # 取得物理座標的頂點和面
        vertices_physical = mesh_result['vertices_physical']
        faces = mesh_result['faces']

        # 使用 Marching Cubes 結果計算表面積 (單位: mm²)
        surface_area = mesh_surface_area(vertices_physical, faces)

        # 輸出表面積結果（使用 processors.printers）
        if verbose:
            from processors.printers import print_surface_area_calculation
            print_surface_area_calculation(name, surface_area, verbose)

        return surface_area

    left_area = _get_surface_area(left_ventricle, "左腦室")
    right_area = _get_surface_area(right_ventricle, "右腦室")

    total_surface_area = left_area + right_area

    # 輸出表面積計算總結（使用 processors.printers）
    if verbose:
        from processors.printers import print_surface_area_summary
        print_surface_area_summary(left_area, right_area, total_surface_area, verbose)

    return {
        'left_surface_area': left_area,
        'right_surface_area': right_area,
        'total_surface_area': total_surface_area
    }


def calculate_volume_smooth(image_obj, verbose=True):
    """
    使用 Marching Cubes 演算法計算平滑體積（基於三角網格）

    Args:
        image_obj: nibabel 影像物件
        verbose (bool): 是否顯示計算過程資訊

    Returns:
        float: 平滑體積 (mm³)
    """
    # 輸出體積計算開始資訊（使用 processors.printers）
    if verbose:
        from processors.printers import print_volume_calculation
        print_volume_calculation(None, verbose=False)  # 先輸出開始資訊

    # 使用統一的表面提取函數 (與表面積計算相同)
    mesh_result = extract_surface_mesh(image_obj, level=0.5, verbose=False)

    # 取得物理座標的頂點和面
    vertices_physical = mesh_result['vertices_physical']
    faces = mesh_result['faces']

    # 基於三角網格計算體積
    # 使用公式：V = (1/6) * Σ((v1 × v2) · v3) 對於每個三角形
    volume = 0.0
    for face in faces:
        v1, v2, v3 = vertices_physical[face[0]], vertices_physical[face[1]], vertices_physical[face[2]]
        # 計算三角形面積並投影到原點形成四面體體積
        cross_product = np.cross(v2 - v1, v3 - v1)
        triangle_volume = np.abs(np.dot(cross_product, v1)) / 6.0
        volume += triangle_volume

    # 輸出體積計算結果（使用 processors.printers）
    if verbose:
        from processors.printers import print_volume_calculation
        print_volume_calculation(volume, verbose)

    return volume


def calculate_volume_surface_ratio(left_ventricle, right_ventricle, verbose=True):
    """
    計算左右腦室的體積與表面積比例（左右相加後計算總體比例）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        verbose (bool): 是否顯示計算過程資訊

    Returns:
        dict: 包含體積、表面積和比例計算結果的字典
    """
    from skimage.measure import mesh_surface_area
    from model.image_processing import get_image_data

    # 輸出計算開始資訊（使用 processors.printers）
    if verbose:
        from processors.printers import print_volume_surface_ratio_start
        print_volume_surface_ratio_start(verbose)

    def _calculate_volume_and_surface(image_obj, name):
        """計算單個腦室的體積和表面積"""
        # 驗證影像資料
        image_data = get_image_data(image_obj)
        if image_data is None or np.sum(image_data > 0) == 0:
            raise ValueError(f"{name} 影像資料為空或沒有有效的體素")

        # 輸出單一腦室計算開始資訊（使用 processors.printers）
        if verbose:
            from processors.printers import print_volume_surface_calculation_start
            print_volume_surface_calculation_start(name, verbose)

        # 使用統一的表面提取函數
        mesh_result = extract_surface_mesh(image_obj, level=0.5, verbose=False)

        # 取得網格資料
        vertices_physical = mesh_result['vertices_physical']
        faces = mesh_result['faces']

        # 驗證網格資料
        if len(vertices_physical) == 0 or len(faces) == 0:
            raise ValueError(f"{name} 無法提取有效的表面網格")

        # 計算平滑體積（基於物理座標網格，單位: mm³）
        volume = 0.0
        for face in faces:
            v1, v2, v3 = vertices_physical[face[0]], vertices_physical[face[1]], vertices_physical[face[2]]
            cross_product = np.cross(v2 - v1, v3 - v1)
            triangle_volume = np.abs(np.dot(cross_product, v1)) / 6.0
            volume += triangle_volume

        # 計算表面積（基於物理座標網格，單位: mm²）
        surface_area = mesh_surface_area(vertices_physical, faces)

        # 輸出單一腦室計算結果（使用 processors.printers）
        if verbose:
            from processors.printers import print_volume_surface_single_result
            print_volume_surface_single_result(name, volume, surface_area, verbose)

        return volume, surface_area

    # 分別計算左右腦室的體積和表面積
    left_volume, left_surface_area = _calculate_volume_and_surface(left_ventricle, "左腦室")
    right_volume, right_surface_area = _calculate_volume_and_surface(right_ventricle, "右腦室")

    # 計算整體數據（左右相加後計算比例）
    total_volume = left_volume + right_volume
    total_surface_area = left_surface_area + right_surface_area
    total_ratio = total_volume / total_surface_area if total_surface_area > 0 else 0.0

    # 輸出體積表面積比例計算總結（使用 processors.printers）
    if verbose:
        from processors.printers import print_volume_surface_ratio_summary
        print_volume_surface_ratio_summary(
            left_volume, left_surface_area,
            right_volume, right_surface_area,
            total_volume, total_surface_area, total_ratio, verbose
        )

    return {
        'left_volume': left_volume,
        'right_volume': right_volume,
        'total_volume': total_volume,
        'left_surface_area': left_surface_area,
        'right_surface_area': right_surface_area,
        'total_surface_area': total_surface_area,
        'total_ratio': total_ratio
    }


def calculate_csf_minus_ventricle(csf_img, left_ventricle, right_ventricle,
                                  third_ventricle, fourth_ventricle, verbose=True):
    """
    計算腦室外 CSF 體積。

    使用 mask 聯集扣除，避免左右腦室、三腦室、四腦室之間重疊時重複扣除。

    Args:
        csf_img: CSF mask 影像物件
        left_ventricle: 左腦室 mask 影像物件
        right_ventricle: 右腦室 mask 影像物件
        third_ventricle: 三腦室 mask 影像物件
        fourth_ventricle: 四腦室 mask 影像物件
        verbose (bool): 是否顯示計算過程資訊

    Returns:
        dict: 包含 CSF、各腦室、腦室聯集與扣除後體積
    """
    csf_voxel_size = get_voxel_size(csf_img)
    csf_voxel_volume = float(np.prod(csf_voxel_size))

    def _mask_volume(mask, image_obj):
        return float(np.count_nonzero(mask) * np.prod(get_voxel_size(image_obj)))

    def _project_mask_to_csf_grid(mask, image_obj, name):
        """將任意 shape 的 mask 依 affine 放回 CSF grid。"""
        coords = np.argwhere(mask)
        projected = np.zeros(csf_img.shape, dtype=bool)

        if len(coords) == 0:
            return projected, 0

        homogeneous = np.column_stack([coords, np.ones(len(coords))])
        physical_coords = (image_obj.affine @ homogeneous.T).T[:, :3]

        inverse_csf_affine = np.linalg.inv(csf_img.affine)
        csf_homogeneous = np.column_stack([physical_coords, np.ones(len(physical_coords))])
        csf_voxels = (inverse_csf_affine @ csf_homogeneous.T).T[:, :3]
        csf_indices = np.rint(csf_voxels).astype(int)

        in_bounds = (
            (csf_indices[:, 0] >= 0) & (csf_indices[:, 0] < csf_img.shape[0]) &
            (csf_indices[:, 1] >= 0) & (csf_indices[:, 1] < csf_img.shape[1]) &
            (csf_indices[:, 2] >= 0) & (csf_indices[:, 2] < csf_img.shape[2])
        )

        valid_indices = csf_indices[in_bounds]
        if len(valid_indices) > 0:
            projected[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True

        dropped_count = int(len(coords) - len(valid_indices))
        if verbose and dropped_count > 0:
            print(f"  注意: {name} 有 {dropped_count} 個體素投影後超出 CSF 範圍，已忽略")

        return projected, dropped_count

    csf_mask = get_image_data(csf_img) > 0
    left_mask = get_image_data(left_ventricle) > 0
    right_mask = get_image_data(right_ventricle) > 0
    third_mask = get_image_data(third_ventricle) > 0
    fourth_mask = get_image_data(fourth_ventricle) > 0

    if not np.any(csf_mask):
        raise ValueError("CSF mask 沒有有效體素")

    left_on_csf, left_dropped = _project_mask_to_csf_grid(left_mask, left_ventricle, "左腦室")
    right_on_csf, right_dropped = _project_mask_to_csf_grid(right_mask, right_ventricle, "右腦室")
    third_on_csf, third_dropped = _project_mask_to_csf_grid(third_mask, third_ventricle, "三腦室")
    fourth_on_csf, fourth_dropped = _project_mask_to_csf_grid(fourth_mask, fourth_ventricle, "四腦室")

    ventricle_union_mask = left_on_csf | right_on_csf | third_on_csf | fourth_on_csf
    csf_minus_ventricle_mask = csf_mask & ~ventricle_union_mask

    csf_volume = float(np.count_nonzero(csf_mask) * csf_voxel_volume)
    left_volume = _mask_volume(left_mask, left_ventricle)
    right_volume = _mask_volume(right_mask, right_ventricle)
    third_volume = _mask_volume(third_mask, third_ventricle)
    fourth_volume = _mask_volume(fourth_mask, fourth_ventricle)
    ventricle_union_volume = float(np.count_nonzero(ventricle_union_mask) * csf_voxel_volume)
    csf_minus_ventricle_volume = float(np.count_nonzero(csf_minus_ventricle_mask) * csf_voxel_volume)

    if verbose:
        from processors.printers import print_csf_minus_ventricle_summary
        print_csf_minus_ventricle_summary({
            'csf_volume': csf_volume,
            'left_ventricle_volume': left_volume,
            'right_ventricle_volume': right_volume,
            'third_ventricle_volume': third_volume,
            'fourth_ventricle_volume': fourth_volume,
            'ventricle_union_volume': ventricle_union_volume,
            'csf_minus_ventricle_volume': csf_minus_ventricle_volume,
        }, verbose=verbose)

    return {
        'csf_volume': csf_volume,
        'left_ventricle_volume': left_volume,
        'right_ventricle_volume': right_volume,
        'third_ventricle_volume': third_volume,
        'fourth_ventricle_volume': fourth_volume,
        'ventricle_union_volume': ventricle_union_volume,
        'csf_minus_ventricle_volume': csf_minus_ventricle_volume,
        'voxel_size': tuple(csf_voxel_size),
        'dropped_voxels': {
            'left_ventricle': left_dropped,
            'right_ventricle': right_dropped,
            'third_ventricle': third_dropped,
            'fourth_ventricle': fourth_dropped,
        }
    }
