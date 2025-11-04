#!/usr/bin/env python3
"""
視覺化工具
負責分析結果的3D視覺化
"""

import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure
from model.reorient import get_image_data


def visualize_ventricle_distance(left_ventricle, right_ventricle,
                                  left_centroid, right_centroid,
                                  distance_mm, output_path="ventricle_distance.png",
                                  show_plot=True, original_path=None,
                                  cranial_width_data=None, ratio=None):
    """
    視覺化左右腦室和質心距離(在物理空間中)

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        left_centroid: 左腦室質心物理座標(mm)
        right_centroid: 右腦室質心物理座標(mm)
        distance_mm: 距離(mm)
        output_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        original_path: 原始腦部影像路徑(可選)
        cranial_width_data: 顱內橫向寬度資料 (寬度, 左端點, 右端點, 切面)(可選)
        ratio: 腦室距離/顱內寬度比值(可選)

    Returns:
        plotly figure物件
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    print(f"\n準備視覺化...")

    # 載入原始腦部影像(如果提供)
    original_img = None
    original_data = None
    if original_path:
        print(f"載入原始腦部影像: {original_path}")
        original_img = nib.load(original_path)
        original_data = get_image_data(original_img)

    # 建立圖表
    fig = go.Figure()

    # 如果有原始影像,先繪製(作為背景) - 使用表面網格
    if original_data is not None:
        try:
            # 使用較低的閾值以顯示腦組織
            threshold = np.percentile(original_data[original_data > 0], 30)

            # 使用 marching cubes 提取表面(在體素空間)
            verts, faces, normals, values = measure.marching_cubes(original_data, level=threshold)

            # 將頂點從體素座標轉換到物理座標
            verts_homogeneous = np.column_stack([verts, np.ones(len(verts))])
            verts_physical = (original_img.affine @ verts_homogeneous.T).T[:, :3]

            # 使用 Mesh3d 繪製表面(降低透明度讓黃線更明顯)
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
            print(f"警告:無法提取腦部表面 - {str(e)}")

    # 左腦室 - 轉換到物理空間(完整點雲,不下採樣)
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

    # 右腦室 - 轉換到物理空間(完整點雲,不下採樣)
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

    # 左腦室質心(大藍點)
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

    # 右腦室質心(大紅點)
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

    # 連接線(更明顯的金黃色) - 腦室質心距離
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

    # 顱內橫向最大寬度線(如果有提供)
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
    title_text = f'Ventricle Centroid Distance: {distance_mm:.2f} mm'
    if ratio is not None and cranial_width_data is not None:
        cranial_width = cranial_width_data[0]
        title_text += f'<br>Cranial Width: {cranial_width:.2f} mm | Ratio: {ratio:.4f} ({ratio*100:.2f}%)'

    fig.update_layout(
        title=title_text,
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

    # 同時儲存 HTML(可互動)
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path)
    print(f"互動式HTML已儲存到: {html_path}")

    # 顯示圖表
    if show_plot:
        fig.show()

    return fig


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
