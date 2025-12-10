
#!/usr/bin/env python3
"""
影像重新定向工具
處理 NIfTI 影像的方向標準化
"""

import nibabel as nib
from pathlib import Path
import numpy as np
from skimage import measure


def reorient_image(image_path: str, verbose: bool = True):
    """
    將影像重新定向到標準RAS+方向（不儲存檔案）

    Args:
        image_path: 影像檔案路徑
        verbose: 是否顯示處理資訊

    Returns:
        tuple: (重新定向後的影像物件, 原始方向, 新方向)
    """
    # 讀取影像
    img = nib.load(image_path)

    # 獲取原始方向
    orig_ornt = nib.aff2axcodes(img.affine)

    # 轉換到canonical方向（最接近RAS+）
    canonical_img = nib.as_closest_canonical(img)

    # 獲取新方向
    new_ornt = nib.aff2axcodes(canonical_img.affine)

    if verbose:
        print(f"檔案: {Path(image_path).name}")
        print(f"影像形狀: {img.shape}")
        print(f"資料範圍: {img.get_fdata().min():.2f} 到 {img.get_fdata().max():.2f}")
        print(f"原始方向: {orig_ornt}")
        print(f"新方向: {new_ornt}")
        if orig_ornt == new_ornt:
            print("已經是標準方向")
        else:
            print("成功重新定向到標準方向")

    return canonical_img, orig_ornt, new_ornt


def get_image_data(img):
    """
    從影像物件取得numpy陣列資料

    Args:
        img: nibabel影像物件

    Returns:
        numpy陣列
    """
    if hasattr(img, 'get_fdata'):
        return img.get_fdata()
    else:
        return img


def get_voxel_size(img):
    """
    取得影像的體素間距（voxel spacing）

    Args:
        img: nibabel影像物件

    Returns:
        tuple: (x_spacing, y_spacing, z_spacing) 單位為 mm
    """
    # 從 affine 矩陣取得體素大小
    pixdim = img.header.get_zooms()
    return pixdim[:3]


def save_image(img, output_path: str):
    """
    儲存影像到檔案

    Args:
        img: nibabel影像物件
        output_path: 輸出檔案路徑
    """
    nib.save(img, output_path)
    print(f"已儲存: {output_path}")


def process_all_images(input_dir: str, output_dir: str = None, save_files: bool = False):
    """
    批次處理目錄中所有的 .nii.gz 檔案

    Args:
        input_dir: 輸入目錄路徑
        output_dir: 輸出目錄路徑（如果為None且save_files=True，則自動建立）
        save_files: 是否儲存重新定向後的檔案

    Returns:
        dict: {檔案名稱: (重新定向影像, 原始方向, 新方向)}
    """
    input_path = Path(input_dir)

    # 如果需要儲存且沒有指定輸出目錄，則建立一個新的
    if save_files and output_dir is None:
        output_dir = str(input_path) + "_reoriented"

    if save_files:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"輸出目錄: {output_dir}\n")

    # 找出所有 .nii.gz 檔案
    nii_files = sorted(input_path.glob("*.nii.gz"))

    if not nii_files:
        print(f"在 {input_dir} 中沒有找到 .nii.gz 檔案")
        return {}

    print(f"找到 {len(nii_files)} 個影像檔案")
    print("=" * 70)

    results = {}

    # 處理每個檔案
    for i, nii_file in enumerate(nii_files, 1):
        print(f"\n[{i}/{len(nii_files)}] 處理: {nii_file.name}")
        print("-" * 70)

        # 重新定向
        reoriented_img, orig_ornt, new_ornt = reorient_image(str(nii_file), verbose=True)

        # 儲存結果
        results[nii_file.name] = (reoriented_img, orig_ornt, new_ornt)

        # 如果需要儲存檔案
        if save_files:
            output_file = output_path / nii_file.name
            save_image(reoriented_img, str(output_file))

    print("\n" + "=" * 70)
    print(f"批次處理完成！共處理 {len(results)} 個檔案")

    return results


def convert_voxel_to_physical(verts_voxel, affine):
    """
    將體素座標轉換為物理座標

    Args:
        verts_voxel: 體素座標陣列，形狀為 (N, 3)
        affine: 4x4 仿射矩陣

    Returns:
        numpy陣列: 物理座標，形狀為 (N, 3)
    """
    # 轉換為齊次座標
    verts_homogeneous = np.column_stack([verts_voxel, np.ones(len(verts_voxel))])
    # 應用仿射變換
    verts_physical = (affine @ verts_homogeneous.T).T[:, :3]
    return verts_physical


def extract_surface_mesh(image_obj, level=0.5, verbose=False):
    """
    使用 Marching Cubes 演算法提取 3D 表面網格

    Args:
        image_obj: nibabel 影像物件
        level: Marching Cubes 等級值
        verbose: 是否顯示處理資訊

    Returns:
        dict: 包含以下鍵值的字典:
            - 'vertices_physical': 物理座標頂點 (N, 3)
            - 'faces': 三角面索引 (M, 3)
            - 'vertices_voxel': 體素座標頂點 (N, 3)
    """
    if verbose:
        print(f"提取表面網格 - level: {level}")

    # 取得影像資料和體素間距
    image_data = get_image_data(image_obj)
    voxel_size = get_voxel_size(image_obj)

    try:
        # 執行 Marching Cubes
        # 注意: 不傳入 spacing，讓它回傳體素索引座標
        # 之後再透過 affine 矩陣一次轉換為物理座標
        verts, faces, _, _ = measure.marching_cubes(
            image_data,
            level=level
        )

        # 轉換為物理座標
        verts_physical = convert_voxel_to_physical(verts, image_obj.affine)

        return {
            'vertices_physical': verts_physical,
            'faces': faces,
            'vertices_voxel': verts
        }

    except Exception as e:
        raise RuntimeError(f"表面提取失敗: {str(e)}")
