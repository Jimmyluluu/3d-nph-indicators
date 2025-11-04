#!/usr/bin/env python3
"""
腦室分析工具
計算腦室質心距離並進行3D視覺化
"""

import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure
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


def visualize_ventricle_distance(left_ventricle, right_ventricle,
                                  left_centroid, right_centroid,
                                  distance_mm, output_path="ventricle_distance.png",
                                  show_plot=True, original_path=None,
                                  cranial_width_data=None):
    """
    視覺化左右腦室和質心距離（在物理空間中）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        left_centroid: 左腦室質心物理座標（mm）
        right_centroid: 右腦室質心物理座標（mm）
        distance_mm: 距離（mm）
        output_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        original_path: 原始腦部影像路徑（可選）
        cranial_width_data: 顱內橫向寬度資料 (寬度, 左端點, 右端點, 切面)（可選）

    Returns:
        plotly figure物件
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    print(f"\n準備視覺化...")

    # 載入原始腦部影像（如果提供）
    original_img = None
    original_data = None
    if original_path:
        print(f"載入原始腦部影像: {original_path}")
        original_img = nib.load(original_path)
        original_data = get_image_data(original_img)

    # 建立圖表
    fig = go.Figure()

    # 如果有原始影像，先繪製（作為背景）- 使用表面網格
    if original_data is not None:
        try:
            # 使用較低的閾值以顯示腦組織
            threshold = np.percentile(original_data[original_data > 0], 30)

            # 使用 marching cubes 提取表面（在體素空間）
            verts, faces, normals, values = measure.marching_cubes(original_data, level=threshold)

            # 將頂點從體素座標轉換到物理座標
            verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
            verts_physical = (original_img.affine @ verts_homogeneous.T).T[:, :3]

            # 使用 Mesh3d 繪製表面（降低透明度讓黃線更明顯）
            fig.add_trace(go.Mesh3d(
                x=verts_physical[:, 0],
                y=verts_physical[:, 1],
                z=verts_physical[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgray',
                opacity=0.15,  # 降低透明度
                name='Brain Surface',
                showlegend=True,
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    specular=0.2
                ),
                flatshading=False
            ))
            print(f"✓ 腦部表面已加入")
        except Exception as e:
            print(f"警告：無法提取腦部表面 - {str(e)}")

    # 左腦室 - 轉換到物理空間（完整點雲，不下採樣）
    left_coords_voxel = np.argwhere(left_data > 0)
    left_coords_homogeneous = np.column_stack([left_coords_voxel, np.ones(len(left_coords_voxel))])
    left_coords_physical = (left_ventricle.affine @ left_coords_homogeneous.T).T[:, :3]

    fig.add_trace(go.Scatter3d(
        x=left_coords_physical[:, 0],
        y=left_coords_physical[:, 1],
        z=left_coords_physical[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color='blue',
            opacity=0.3  # 降低透明度
        ),
        name='Left Ventricle',
        showlegend=True
    ))

    # 右腦室 - 轉換到物理空間（完整點雲，不下採樣）
    right_coords_voxel = np.argwhere(right_data > 0)
    right_coords_homogeneous = np.column_stack([right_coords_voxel, np.ones(len(right_coords_voxel))])
    right_coords_physical = (right_ventricle.affine @ right_coords_homogeneous.T).T[:, :3]

    fig.add_trace(go.Scatter3d(
        x=right_coords_physical[:, 0],
        y=right_coords_physical[:, 1],
        z=right_coords_physical[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color='red',
            opacity=0.3  # 降低透明度
        ),
        name='Right Ventricle',
        showlegend=True
    ))

    # 左腦室質心（大藍點）
    fig.add_trace(go.Scatter3d(
        x=[left_centroid[0]],
        y=[left_centroid[1]],
        z=[left_centroid[2]],
        mode='markers+text',
        marker=dict(
            size=10,
            color='darkblue',
            symbol='diamond'
        ),
        text=['Left Centroid'],
        textposition='top center',
        name='Left Centroid',
        showlegend=True
    ))

    # 右腦室質心（大紅點）
    fig.add_trace(go.Scatter3d(
        x=[right_centroid[0]],
        y=[right_centroid[1]],
        z=[right_centroid[2]],
        mode='markers+text',
        marker=dict(
            size=10,
            color='darkred',
            symbol='diamond'
        ),
        text=['Right Centroid'],
        textposition='top center',
        name='Right Centroid',
        showlegend=True
    ))

    # 連接線（更明顯的金黃色）- 腦室質心距離
    fig.add_trace(go.Scatter3d(
        x=[left_centroid[0], right_centroid[0]],
        y=[left_centroid[1], right_centroid[1]],
        z=[left_centroid[2], right_centroid[2]],
        mode='lines+text',
        line=dict(
            color='gold',  # 改用更明顯的金色
            width=10  # 加粗線條
        ),
        text=['', f'{distance_mm:.2f} mm'],
        textposition='middle center',
        textfont=dict(size=16, color='gold'),  # 加大文字
        name=f'Ventricle Distance: {distance_mm:.2f} mm',
        showlegend=True
    ))

    # 顱內橫向最大寬度線（如果有提供）
    if cranial_width_data is not None:
        cranial_width, left_point, right_point, slice_idx = cranial_width_data

        fig.add_trace(go.Scatter3d(
            x=[left_point[0], right_point[0]],
            y=[left_point[1], right_point[1]],
            z=[left_point[2], right_point[2]],
            mode='lines+text',
            line=dict(
                color='cyan',  # 青綠色
                width=10
            ),
            text=['', f'{cranial_width:.2f} mm'],
            textposition='middle center',
            textfont=dict(size=16, color='cyan'),
            name=f'Cranial Width: {cranial_width:.2f} mm',
            showlegend=True
        ))

    # 更新版面配置
    fig.update_layout(
        title=f'Ventricle Centroid Distance: {distance_mm:.2f} mm',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=900,
        showlegend=True
    )

    # 儲存圖片
    print(f"\n儲存圖片到: {output_path}")
    fig.write_image(output_path)
    print(f"圖片已儲存！")

    # 同時儲存 HTML（可互動）
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path)
    print(f"互動式HTML已儲存到: {html_path}")

    # 顯示圖表
    if show_plot:
        fig.show()

    return fig


def print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size):
    """
    格式化輸出測量結果

    Args:
        distance_mm: 距離（mm）
        left_centroid: 左質心物理座標（mm）
        right_centroid: 右質心物理座標（mm）
        voxel_size: 體素間距
    """
    print("\n" + "=" * 70)
    print("腦室質心距離測量結果")
    print("=" * 70)
    print(f"\n左腦室質心座標 (mm): ({left_centroid[0]:.2f}, {left_centroid[1]:.2f}, {left_centroid[2]:.2f})")
    print(f"右腦室質心座標 (mm): ({right_centroid[0]:.2f}, {right_centroid[1]:.2f}, {right_centroid[2]:.2f})")
    print(f"\n體素間距 (mm): {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.2f}")
    print(f"\n左右腦室質心距離: {distance_mm:.2f} mm")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("腦室質心距離分析")
    print("=" * 70)

    # 指定檔案路徑
    left_path = "000016209E/Ventricle_L.nii.gz"
    right_path = "000016209E/Ventricle_R.nii.gz"
    original_path = "000016209E/original.nii.gz"

    # 載入左右腦室（使用原始座標，不重新定向）
    print("\n步驟 1: 載入左右腦室影像")
    print("-" * 70)
    left_vent, right_vent = load_ventricle_pair(left_path, right_path, verbose=True)

    # 計算質心距離
    print("\n步驟 2: 計算質心距離")
    print("-" * 70)
    distance_mm, left_centroid, right_centroid, voxel_size = calculate_centroid_distance(
        left_vent, right_vent
    )

    # 顯示結果
    print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size)

    # 視覺化並儲存
    print("\n步驟 3: 產生3D視覺化")
    print("-" * 70)
    visualize_ventricle_distance(
        left_vent, right_vent,
        left_centroid, right_centroid,
        distance_mm,
        output_path="ventricle_distance.png",
        show_plot=False,  # 設為 True 會開啟瀏覽器顯示互動圖
        original_path=original_path  # 加入原始腦部影像
    )
