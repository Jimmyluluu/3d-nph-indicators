#!/usr/bin/env python3
"""
視覺化工具
負責分析結果的3D視覺化
"""

import numpy as np
import plotly.graph_objects as go
from model.image_processing import get_image_data, extract_surface_mesh


def visualize_ventricle_distance(left_ventricle, right_ventricle,
                                  left_centroid, right_centroid,
                                  distance_mm, output_path="ventricle_distance.png",
                                  show_plot=True, original_img=None,
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
        original_img: 原始腦部影像物件(可選,已拉正到 RAS+ 方向)
        cranial_width_data: 顱內橫向寬度資料 (寬度, 左端點, 右端點, 切面)(可選)
        ratio: 腦室距離/顱內寬度比值(可選)

    Returns:
        plotly figure物件
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    print(f"\n準備視覺化...")

    # 取得原始腦部影像資料(如果提供)
    original_data = None
    if original_img is not None:
        print(f"使用已載入的原始腦部影像")
        original_data = get_image_data(original_img)

    # 建立圖表
    fig = go.Figure()

    # 如果有原始影像,先繪製(作為背景) - 使用表面網格
    if original_data is not None:
        try:
            # 使用較低的閾值以顯示腦組織
            threshold = np.percentile(original_data[original_data > 0], 30)

            # 使用統一的表面提取函數
            brain_mesh = extract_surface_mesh(original_img, level=threshold, verbose=False)
            verts_physical = brain_mesh['vertices_physical']
            faces = brain_mesh['faces']

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

    # 左腦室 - 使用統一表面提取函數
    try:
        # 使用統一的表面提取函數
        left_mesh = extract_surface_mesh(left_ventricle, level=0.5, verbose=False)
        left_verts_physical = left_mesh['vertices_physical']
        left_faces = left_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=left_verts_physical[:, 0],
            y=left_verts_physical[:, 1],
            z=left_verts_physical[:, 2],
            i=left_faces[:, 0],
            j=left_faces[:, 1],
            k=left_faces[:, 2],
            color='blue',
            opacity=0.4,
            name='Left Ventricle',
            showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            flatshading=False
        ))
    except Exception as e:
        print(f"警告: 無法提取左腦室表面，使用點雲顯示 - {str(e)}")
        # 備用方案：使用點雲
        left_coords_voxel = np.argwhere(left_data > 0)
        left_coords_homogeneous = np.column_stack([left_coords_voxel, np.ones(len(left_coords_voxel))])
        left_coords_physical = (left_ventricle.affine @ left_coords_homogeneous.T).T[:, :3]
        fig.add_trace(go.Scatter3d(
            x=left_coords_physical[:, 0],
            y=left_coords_physical[:, 1],
            z=left_coords_physical[:, 2],
            mode='markers',
            marker=dict(size=1, color='blue', opacity=0.3),
            name='Left Ventricle',
            showlegend=True
        ))

    # 右腦室 - 使用統一表面提取函數
    try:
        # 使用統一的表面提取函數
        right_mesh = extract_surface_mesh(right_ventricle, level=0.5, verbose=False)
        right_verts_physical = right_mesh['vertices_physical']
        right_faces = right_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=right_verts_physical[:, 0],
            y=right_verts_physical[:, 1],
            z=right_verts_physical[:, 2],
            i=right_faces[:, 0],
            j=right_faces[:, 1],
            k=right_faces[:, 2],
            color='red',
            opacity=0.4,
            name='Right Ventricle',
            showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            flatshading=False
        ))
    except Exception as e:
        print(f"警告: 無法提取右腦室表面，使用點雲顯示 - {str(e)}")
        # 備用方案：使用點雲
        right_coords_voxel = np.argwhere(right_data > 0)
        right_coords_homogeneous = np.column_stack([right_coords_voxel, np.ones(len(right_coords_voxel))])
        right_coords_physical = (right_ventricle.affine @ right_coords_homogeneous.T).T[:, :3]
        fig.add_trace(go.Scatter3d(
            x=right_coords_physical[:, 0],
            y=right_coords_physical[:, 1],
            z=right_coords_physical[:, 2],
            mode='markers',
            marker=dict(size=1, color='red', opacity=0.3),
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


def visualize_3d_evan_index(left_ventricle, right_ventricle, original_img,
                              evan_data, output_path="evan_index.png",
                              show_plot=True, z_range=(0.4, 0.6), y_percentile=40):
    """
    視覺化 3D Evan Index（腦室前腳最大距離與顱內寬度）

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        original_img: 原始腦部影像物件(已拉正到 RAS+ 方向)
        evan_data: 3D Evan Index 計算結果字典
        output_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        z_range: Z 軸切面範圍（用於篩選前腳點雲顯示）
        y_percentile: Y 軸前方百分位數（用於篩選前腳點雲顯示）

    Returns:
        plotly figure物件
    """
    # 取得資料
    left_data = get_image_data(left_ventricle)
    right_data = get_image_data(right_ventricle)

    print(f"\n準備 3D Evan Index 視覺化...")

    # 取得原始腦部影像資料
    print(f"使用已載入的原始腦部影像")
    original_data = get_image_data(original_img)

    # 建立圖表
    fig = go.Figure()

    # 繪製腦部表面網格
    try:
        threshold = np.percentile(original_data[original_data > 0], 30)

        # 使用統一的表面提取函數
        brain_mesh = extract_surface_mesh(original_img, level=threshold, verbose=False)
        verts_physical = brain_mesh['vertices_physical']
        faces = brain_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=verts_physical[:, 0],
            y=verts_physical[:, 1],
            z=verts_physical[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=0.15,
            name='Brain Surface',
            showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            flatshading=False
        ))
        print(f"✓ 腦部表面已加入")
    except Exception as e:
        print(f"警告:無法提取腦部表面 - {str(e)}")

    # 顯示完整腦室 - 使用 Marching Cubes 提取平滑表面
    # 左腦室
    try:
        # 使用統一的表面提取函數
        left_mesh = extract_surface_mesh(left_ventricle, level=0.5, verbose=False)
        left_verts_physical = left_mesh['vertices_physical']
        left_faces = left_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=left_verts_physical[:, 0],
            y=left_verts_physical[:, 1],
            z=left_verts_physical[:, 2],
            i=left_faces[:, 0],
            j=left_faces[:, 1],
            k=left_faces[:, 2],
            color='blue',
            opacity=0.4,
            name='Left Ventricle',
            showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            flatshading=False
        ))
    except Exception as e:
        print(f"警告: 無法提取左腦室表面 - {str(e)}")

    # 右腦室
    try:
        # 使用統一的表面提取函數
        right_mesh = extract_surface_mesh(right_ventricle, level=0.5, verbose=False)
        right_verts_physical = right_mesh['vertices_physical']
        right_faces = right_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=right_verts_physical[:, 0],
            y=right_verts_physical[:, 1],
            z=right_verts_physical[:, 2],
            i=right_faces[:, 0],
            j=right_faces[:, 1],
            k=right_faces[:, 2],
            color='red',
            opacity=0.4,
            name='Right Ventricle',
            showlegend=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            flatshading=False
        ))
    except Exception as e:
        print(f"警告: 無法提取右腦室表面 - {str(e)}")

    # 前腳最大距離的端點標記
    left_endpoint = evan_data['anterior_horn_endpoints']['left']
    right_endpoint = evan_data['anterior_horn_endpoints']['right']

    fig.add_trace(go.Scatter3d(
        x=[left_endpoint[0]],
        y=[left_endpoint[1]],
        z=[left_endpoint[2]],
        mode='markers',
        marker=dict(size=10, color='darkblue', symbol='diamond'),
        name='Left Max Point',
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=[right_endpoint[0]],
        y=[right_endpoint[1]],
        z=[right_endpoint[2]],
        mode='markers',
        marker=dict(size=10, color='darkred', symbol='diamond'),
        name='Right Max Point',
        showlegend=True
    ))

    # 前腳最大距離連線（紫色）
    anterior_distance = evan_data['anterior_horn_distance_mm']
    fig.add_trace(go.Scatter3d(
        x=[left_endpoint[0], right_endpoint[0]],
        y=[left_endpoint[1], right_endpoint[1]],
        z=[left_endpoint[2], right_endpoint[2]],
        mode='lines+text',
        line=dict(color='purple', width=10),
        text=['', f'{anterior_distance:.2f} mm'],
        textposition='middle center',
        textfont=dict(size=16, color='purple'),
        name=f'Anterior Horn Distance: {anterior_distance:.2f} mm',
        showlegend=True
    ))

    # 顱內寬度連線（青色）
    cranial_width = evan_data['cranial_width_mm']
    cranial_left = evan_data['cranial_width_endpoints']['left']
    cranial_right = evan_data['cranial_width_endpoints']['right']

    fig.add_trace(go.Scatter3d(
        x=[cranial_left[0], cranial_right[0]],
        y=[cranial_left[1], cranial_right[1]],
        z=[cranial_left[2], cranial_right[2]],
        mode='lines+text',
        line=dict(color='cyan', width=10),
        text=['', f'{cranial_width:.2f} mm'],
        textposition='middle center',
        textfont=dict(size=16, color='cyan'),
        name=f'Cranial Width: {cranial_width:.2f} mm',
        showlegend=True
    ))

    # 更新版面配置
    evan_index = evan_data['evan_index']
    evan_index_percent = evan_data['evan_index_percent']

    title_text = f'3D Evan Index: {evan_index:.4f} ({evan_index_percent:.2f}%)'
    title_text += f'<br>Anterior Horn Distance: {anterior_distance:.2f} mm | Cranial Width: {cranial_width:.2f} mm'

    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
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


def visualize_surface_area(surface_data, output_path="surface_area.png", show_plot=True):
    """
    使用 Plotly 視覺化平滑後的腦室表面積

    Args:
        surface_data (dict): 從 calculate_surface_area 函數回傳的字典
        output_path (str): 輸出圖片路徑
        show_plot (bool): 是否顯示互動式圖表

    Returns:
        plotly figure物件
    """
    print(f"\n準備表面積視覺化...")

    left_area = surface_data['left_surface_area']
    right_area = surface_data['right_surface_area']
    total_area = surface_data['total_surface_area']
    
    left_verts = surface_data['left_vertices']
    left_faces = surface_data['left_faces']
    right_verts = surface_data['right_vertices']
    right_faces = surface_data['right_faces']

    fig = go.Figure()

    # 左腦室
    fig.add_trace(go.Mesh3d(
        x=left_verts[:, 0], y=left_verts[:, 1], z=left_verts[:, 2],
        i=left_faces[:, 0], j=left_faces[:, 1], k=left_faces[:, 2],
        color='blue',
        opacity=0.8,
        name=f'Left Ventricle ({left_area:.2f} mm^2)',
        showlegend=True,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        flatshading=False
    ))
    print(f"✓ 左腦室網格已加入")

    # 右腦室
    fig.add_trace(go.Mesh3d(
        x=right_verts[:, 0], y=right_verts[:, 1], z=right_verts[:, 2],
        i=right_faces[:, 0], j=right_faces[:, 1], k=right_faces[:, 2],
        color='red',
        opacity=0.8,
        name=f'Right Ventricle ({right_area:.2f} mm^2)',
        showlegend=True,
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
        flatshading=False
    ))
    print(f"✓ 右腦室網格已加入")

    # 更新版面配置
    title_text = f"Ventricle Surface Area<br>Total: {total_area:.2f} mm^2"
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


def visualize_volume_surface_ratio(left_ventricle, right_ventricle, ratio_data,
                                   output_path="volume_surface_ratio.png", show_plot=True):
    """
    視覺化體積與表面積比例分析結果

    Args:
        left_ventricle: 左腦室影像物件
        right_ventricle: 右腦室影像物件
        ratio_data: 體積表面積比例計算結果
        output_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表

    Returns:
        plotly figure物件
    """
    print(f"\n準備視覺化體積與表面積比例...")

    # 建立圖表
    fig = go.Figure()

    # 左腦室 - 使用統一表面提取函數
    try:
        left_mesh = extract_surface_mesh(left_ventricle, level=0.5, verbose=False)
        left_verts_physical = left_mesh['vertices_physical']
        left_faces = left_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=left_verts_physical[:, 0],
            y=left_verts_physical[:, 1],
            z=left_verts_physical[:, 2],
            i=left_faces[:, 0],
            j=left_faces[:, 1],
            k=left_faces[:, 2],
            color='blue',
            opacity=0.4,
            name=f'Left Ventricle<br>V: {ratio_data["left_volume"]:.1f}mm³<br>S: {ratio_data["left_surface_area"]:.1f}mm²<br>Ratio: {ratio_data["left_ratio"]:.3f}mm',
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.2
            ),
            flatshading=False
        ))
        print(f"✓ 左腦室表面已加入")
    except Exception as e:
        print(f"警告:無法提取左腦室表面 - {str(e)}")

    # 右腦室 - 使用統一表面提取函數
    try:
        right_mesh = extract_surface_mesh(right_ventricle, level=0.5, verbose=False)
        right_verts_physical = right_mesh['vertices_physical']
        right_faces = right_mesh['faces']

        fig.add_trace(go.Mesh3d(
            x=right_verts_physical[:, 0],
            y=right_verts_physical[:, 1],
            z=right_verts_physical[:, 2],
            i=right_faces[:, 0],
            j=right_faces[:, 1],
            k=right_faces[:, 2],
            color='red',
            opacity=0.4,
            name=f'Right Ventricle<br>V: {ratio_data["right_volume"]:.1f}mm³<br>S: {ratio_data["right_surface_area"]:.1f}mm²<br>Ratio: {ratio_data["right_ratio"]:.3f}mm',
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.2
            ),
            flatshading=False
        ))
        print(f"✓ 右腦室表面已加入")
    except Exception as e:
        print(f"警告:無法提取右腦室表面 - {str(e)}")

    # 設定圖表佈局
    fig.update_layout(
        title=dict(
            text=f'腦室體積與表面積比例分析<br>' +
                 f'總體積: {ratio_data["total_volume"]:.1f} mm³, 總表面積: {ratio_data["total_surface_area"]:.1f} mm²<br>' +
                 f'整體比例: {ratio_data["total_ratio"]:.3f} mm, 差異: {ratio_data["ratio_difference"]:.3f} mm ({ratio_data["ratio_difference_percent"]:.1f}%)',
            x=0.5,
            font=dict(size=14)
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=80),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    # 顯示或儲存圖表
    if show_plot:
        fig.show()

    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
        print(f"✓ 3D 互動圖表已儲存: {output_path.replace('.png', '.html')}")

    return fig


def print_volume_surface_ratio_summary(ratio_data):
    """
    輸出體積與表面積比例的計算摘要

    Args:
        ratio_data: 體積表面積比例計算結果字典
    """
    print("\n" + "=" * 70)
    print("📊 體積與表面積比例分析摘要")
    print("=" * 70)

    print(f"\n🔵 左腦室:")
    print(f"   體積: {ratio_data['left_volume']:.2f} mm³")
    print(f"   表面積: {ratio_data['left_surface_area']:.2f} mm²")
    print(f"   體積/表面積比例: {ratio_data['left_ratio']:.4f} mm")

    print(f"\n🔴 右腦室:")
    print(f"   體積: {ratio_data['right_volume']:.2f} mm³")
    print(f"   表面積: {ratio_data['right_surface_area']:.2f} mm²")
    print(f"   體積/表面積比例: {ratio_data['right_ratio']:.4f} mm")

    print(f"\n📈 整體分析:")
    print(f"   總體積: {ratio_data['total_volume']:.2f} mm³")
    print(f"   總表面積: {ratio_data['total_surface_area']:.2f} mm²")
    print(f"   整體比例: {ratio_data['total_ratio']:.4f} mm")

    print(f"\n⚖️ 差異分析:")
    print(f"   比例差異: {ratio_data['ratio_difference']:.4f} mm")
    print(f"   差異百分比: {ratio_data['ratio_difference_percent']:.2f}%")

    # 解釋比例的意義
    avg_ratio = (ratio_data['left_ratio'] + ratio_data['right_ratio']) / 2
    print(f"\n💡 比例解釋:")
    print(f"   體積/表面積比例反映形狀的球形度")
    print(f"   比例越大，形狀越接近球形")
    print(f"   平均比例: {avg_ratio:.4f} mm")

    print("=" * 70)
