#!/usr/bin/env python3
"""
計算結果輸出處理器
負責所有與 print 相關的功能，保持 model/ 模組的純計算性質
"""

def print_ventricle_loading_info(left_path, left_img, left_orig_ornt, left_new_ornt,
                                right_path, right_img, right_orig_ornt, right_new_ornt,
                                shape_same, verbose=True):
    """
    輸出腦室載入資訊

    Args:
        left_path: 左腦室檔案路徑
        left_img: 左腦室影像物件
        left_orig_ornt: 左腦室原始方向
        left_new_ornt: 左腦室新方向
        right_path: 右腦室檔案路徑
        right_img: 右腦室影像物件
        right_orig_ornt: 右腦室原始方向
        right_new_ornt: 右腦室新方向
        shape_same: 影像形狀是否相同
        verbose: 是否顯示資訊
    """
    if not verbose:
        return

    print(f"\n載入左腦室: {left_path}")
    print(f"  影像形狀: {left_img.shape}")
    print(f"  原始方向: {left_orig_ornt} → 標準方向: {left_new_ornt}")
    print(f"  體素間距: {left_img.header.get_zooms()[:3]}")

    print(f"\n載入右腦室: {right_path}")
    print(f"  影像形狀: {right_img.shape}")
    print(f"  原始方向: {right_orig_ornt} → 標準方向: {right_new_ornt}")
    print(f"  體素間距: {right_img.header.get_zooms()[:3]}")

    if shape_same:
        print("\n✓ 影像形狀相同")
    else:
        print("\n⚠ 注意：影像形狀不同（可能經過裁剪）")
        print("  將使用 affine 矩陣轉換到物理空間進行計算")

    print("✓ 座標系統驗證通過！將在物理空間中計算。")


def print_original_image_loading_info(original_path, original_img, orig_ornt, new_ornt, verbose=True):
    """
    輸出原始影像載入資訊

    Args:
        original_path: 原始影像檔案路徑
        original_img: 原始影像物件
        orig_ornt: 原始方向
        new_ornt: 新方向
        verbose: 是否顯示資訊
    """
    if not verbose:
        return

    print(f"\n載入原始影像: {original_path}")
    print(f"  影像形狀: {original_img.shape}")
    print(f"  原始方向: {orig_ornt} → 標準方向: {new_ornt}")
    print(f"  體素間距: {original_img.header.get_zooms()[:3]}")


def print_cranial_width_calculation_info(verbose=True):
    """
    輸出顱內寬度計算開始資訊

    Args:
        verbose: 是否顯示資訊
    """
    if verbose:
        print(f"\n計算顱內橫向最大寬度...")


def print_anterior_horn_distance_info(z_range, y_percentile, left_count, right_count,
                                    max_distance, left_max_point, right_max_point,
                                    verbose=True):
    """
    輸出前腳距離計算資訊

    Args:
        z_range: Z 軸範圍
        y_percentile: Y 軸百分位數
        left_count: 左側前腳點數
        right_count: 右側前腳點數
        max_distance: 最大距離
        left_max_point: 左側端點座標
        right_max_point: 右側端點座標
        verbose: 是否顯示資訊
    """
    if not verbose:
        return

    print(f"\n計算腦室前腳最大距離...")
    print(f"  前腳定義：Z 軸範圍 {z_range[0]*100}%-{z_range[1]*100}%，Y 軸前 {y_percentile}%")
    print(f"  左側前腳點數：{left_count}")
    print(f"  右側前腳點數：{right_count}")
    print(f"  正在計算最大距離...")
    print(f"  ✓ 前腳最大距離：{max_distance:.2f} mm")
    print(f"  左側端點：({left_max_point[0]:.2f}, {left_max_point[1]:.2f}, {left_max_point[2]:.2f})")
    print(f"  右側端點：({right_max_point[0]:.2f}, {right_max_point[1]:.2f}, {right_max_point[2]:.2f})")


def print_evan_index_results(anterior_distance, cranial_width, evan_index, evan_index_percent, verbose=True):
    """
    輸出 3D Evan Index 計算結果

    Args:
        anterior_distance: 前腳距離
        cranial_width: 顱內寬度
        evan_index: Evan Index
        evan_index_percent: Evan Index 百分比
        verbose: 是否顯示資訊
    """
    if not verbose:
        return

    print(f"\n3D Evan Index 計算結果：")
    print(f"  前腳最大距離：{anterior_distance:.2f} mm")
    print(f"  顱內寬度：{cranial_width:.2f} mm")
    print(f"  Evan Index：{evan_index:.4f} ({evan_index_percent:.2f}%)")


def print_surface_area_calculation(name, surface_area, verbose=True):
    """
    輸出單個腦室表面積計算資訊

    Args:
        name: 腦室名稱 ("左腦室" 或 "右腦室")
        surface_area: 表面積
        verbose: 是否顯示資訊
    """
    if verbose:
        print(f"\n計算 {name} 表面積...")
        print(f"  - 計算表面積: {surface_area:.2f} mm^2")


def print_surface_area_summary(left_area, right_area, total_area, verbose=True):
    """
    輸出表面積計算總結

    Args:
        left_area: 左腦室表面積
        right_area: 右腦室表面積
        total_area: 總表面積
        verbose: 是否顯示資訊
    """
    if not verbose:
        return

    print("\n表面積計算總結:")
    print(f"  左腦室表面積: {left_area:.2f} mm^2")
    print(f"  右腦室表面積: {right_area:.2f} mm^2")
    print(f"  總表面積: {total_area:.2f} mm^2")


def print_volume_calculation(volume, verbose=True):
    """
    輸出體積計算資訊

    Args:
        volume: 體積
        verbose: 是否顯示資訊
    """
    if verbose:
        print(f"\n計算平滑體積...")
        print(f"  - 平滑體積: {volume:.2f} mm³")


def print_volume_surface_calculation_start(name, verbose=True):
    """
    輸出體積表面積計算開始資訊

    Args:
        name: 腦室名稱
        verbose: 是否顯示資訊
    """
    if verbose:
        print(f"\n計算 {name} 體積和表面積...")


def print_volume_surface_single_result(name, volume, surface_area, verbose=True):
    """
    輸出單個腦室的體積表面積計算結果

    Args:
        name: 腦室名稱
        volume: 體積
        surface_area: 表面積
        verbose: 是否顯示資訊
    """
    if verbose:
        print(f"  {name}:")
        print(f"    體積: {volume:.2f} mm³")
        print(f"    表面積: {surface_area:.2f} mm²")


def print_volume_surface_ratio_summary(left_volume, left_surface_area,
                                      right_volume, right_surface_area,
                                      total_volume, total_surface_area, total_ratio, verbose=True):
    """
    輸出體積表面積比例計算總結

    Args:
        left_volume: 左腦室體積
        left_surface_area: 左腦室表面積
        right_volume: 右腦室體積
        right_surface_area: 右腦室表面積
        total_volume: 總體積
        total_surface_area: 總表面積
        total_ratio: 整體比例（左右相加後計算）
        verbose: 是否顯示資訊
    """
    if not verbose:
        return

    print(f"\n體積與表面積比例計算總結:")
    print(f"  左腦室:")
    print(f"    體積: {left_volume:.2f} mm³")
    print(f"    表面積: {left_surface_area:.2f} mm²")
    print(f"  右腦室:")
    print(f"    體積: {right_volume:.2f} mm³")
    print(f"    表面積: {right_surface_area:.2f} mm²")
    print(f"  整體 (左+右):")
    print(f"    總體積: {total_volume:.2f} mm³")
    print(f"    總表面積: {total_surface_area:.2f} mm²")
    print(f"    V/SA 比例: {total_ratio:.4f} mm")


def print_volume_surface_ratio_start(verbose=True):
    """
    輸出體積表面積比例計算開始資訊

    Args:
        verbose: 是否顯示資訊
    """
    if verbose:
        print(f"\n計算體積與表面積比例...")


def print_sampling_info(side, original_count, sampled_count, verbose=True):
    """
    輸出點雲降採樣資訊

    Args:
        side: "左側" 或 "右側"
        original_count: 原始點數
        sampled_count: 採樣後點數
        verbose: 是否顯示資訊
    """
    if verbose and original_count != sampled_count:
        print(f"  {side}點雲降採樣至 {sampled_count} 點")


def print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size,
                              cranial_width_mm=None, ratio=None):
    """
    格式化輸出測量結果

    Args:
        distance_mm: 距離(mm)
        left_centroid: 左質心物理座標(mm)
        right_centroid: 右質心物理座標(mm)
        voxel_size: 體素間距
        cranial_width_mm: 顱內橫向最大寬度(mm)(可選)
        ratio: 腦室距離/顱內寬度比值(可選)
    """
    print("\n" + "=" * 70)
    print("腦室質心距離測量結果")
    print("=" * 70)
    print(f"\n左腦室質心座標 (mm): ({left_centroid[0]:.2f}, {left_centroid[1]:.2f}, {left_centroid[2]:.2f})")
    print(f"右腦室質心座標 (mm): ({right_centroid[0]:.2f}, {right_centroid[1]:.2f}, {right_centroid[2]:.2f})")
    print(f"\n體素間距 (mm): {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.2f}")
    print(f"\n左右腦室質心距離: {distance_mm:.2f} mm")

    if cranial_width_mm is not None:
        print(f"顱內橫向最大寬度: {cranial_width_mm:.2f} mm")

    if ratio is not None:
        print(f"\n腦室距離/顱內寬度比值: {ratio:.4f} ({ratio*100:.2f}%)")

    print("=" * 70)

def print_surface_area_summary(surface_data):
    """
    格式化輸出表面積測量結果

    Args:
        surface_data (dict): 從 calculate_surface_area 函數回傳的字典
    """
    print("\n" + "=" * 70)
    print("腦室表面積測量結果")
    print("=" * 70)

    left_area = surface_data['left_surface_area']
    right_area = surface_data['right_surface_area']
    total_area = surface_data['total_surface_area']

    print(f"\n測量結果：")
    print(f"  左腦室表面積: {left_area:.2f} mm^2")
    print(f"  右腦室表面積: {right_area:.2f} mm^2")
    print(f"  總表面積: {total_area:.2f} mm^2")


    print("=" * 70)

def print_evan_index_summary(evan_data):
    """
    格式化輸出 3D Evan Index 測量結果

    Args:
        evan_data: 3D Evan Index 計算結果字典
    """
    print("\n" + "=" * 70)
    print("3D Evan Index 測量結果")
    print("=" * 70)

    anterior_distance = evan_data['anterior_horn_distance_mm']
    cranial_width = evan_data['cranial_width_mm']
    evan_index = evan_data['evan_index']
    evan_index_percent = evan_data['evan_index_percent']

    left_endpoint = evan_data['anterior_horn_endpoints']['left']
    right_endpoint = evan_data['anterior_horn_endpoints']['right']

    left_count = evan_data['anterior_horn_points_count']['left']
    right_count = evan_data['anterior_horn_points_count']['right']

    voxel_size = evan_data['voxel_size']

    print(f"\n前腳最大距離端點：")
    print(f"  左側端點 (mm): ({left_endpoint[0]:.2f}, {left_endpoint[1]:.2f}, {left_endpoint[2]:.2f})")
    print(f"  右側端點 (mm): ({right_endpoint[0]:.2f}, {right_endpoint[1]:.2f}, {right_endpoint[2]:.2f})")

    print(f"\n前腳點數統計：")
    print(f"  左側前腳點數: {left_count}")
    print(f"  右側前腳點數: {right_count}")

    print(f"\n體素間距 (mm): {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.2f}")

    print(f"\n測量結果：")
    print(f"  前腳最大距離: {anterior_distance:.2f} mm")
    print(f"  顱內橫向寬度: {cranial_width:.2f} mm")
    print(f"  3D Evan Index: {evan_index:.4f} ({evan_index_percent:.2f}%)")

    print("=" * 70)