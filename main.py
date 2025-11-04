#!/usr/bin/env python3
"""
3D NPH Indicators - 主程式
正常壓力型水腦症（NPH）3D影像指標計算工具
"""

from pathlib import Path
import nibabel as nib
from model.calculation import (
    load_ventricle_pair,
    calculate_centroid_distance,
    calculate_cranial_width,
    calculate_ventricle_to_cranial_ratio
)
from model.visualization import (
    visualize_ventricle_distance,
    print_measurement_summary
)


def process_case(data_dir, output_image_path, show_plot=False, verbose=True):
    """
    處理單一案例

    Args:
        data_dir: 資料目錄路徑
        output_image_path: 輸出圖片路徑
        show_plot: 是否顯示互動式圖表
        verbose: 是否顯示詳細資訊

    Returns:
        dict: 包含所有測量結果的字典，如果失敗則返回包含錯誤訊息的字典
    """
    try:
        if verbose:
            print("=" * 70)
            print(f"處理案例: {data_dir}")
            print("=" * 70)

        # 偵測檔案命名模式
        data_path = Path(data_dir)
        case_name = data_path.name

        # 模式 1: 標準命名
        left_ventricle_path = data_path / "Ventricle_L.nii.gz"
        right_ventricle_path = data_path / "Ventricle_R.nii.gz"
        original_brain_path = data_path / "original.nii.gz"

        # 模式 2: data_ 開頭的命名
        if case_name.startswith('data_'):
            data_num = case_name.replace('data_', '')
            left_ventricle_path_alt = data_path / f"mask_Ventricle_L_{data_num}.nii.gz"
            right_ventricle_path_alt = data_path / f"mask_Ventricle_R_{data_num}.nii.gz"
            original_brain_path_alt = data_path / f"original_{data_num}.nii.gz"

            # 使用 data_ 模式的路徑
            if left_ventricle_path_alt.exists():
                left_ventricle_path = left_ventricle_path_alt
                right_ventricle_path = right_ventricle_path_alt
                original_brain_path = original_brain_path_alt

        # 轉換為字串
        left_ventricle_path = str(left_ventricle_path)
        right_ventricle_path = str(right_ventricle_path)
        original_brain_path = str(original_brain_path)

        # 步驟 1: 載入左右腦室（使用原始座標系統）
        if verbose:
            print("\n步驟 1: 載入左右腦室影像")
            print("-" * 70)
        left_vent, right_vent = load_ventricle_pair(
            left_ventricle_path, right_ventricle_path, verbose=verbose
        )

        # 步驟 2: 計算質心距離
        if verbose:
            print("\n步驟 2: 計算左右腦室質心距離")
            print("-" * 70)
        distance_mm, left_centroid, right_centroid, voxel_size = calculate_centroid_distance(
            left_vent, right_vent
        )

        # 步驟 3: 計算顱內橫向最大寬度
        if verbose:
            print("\n步驟 3: 計算顱內橫向最大寬度")
            print("-" * 70)

        original_brain = nib.load(original_brain_path)
        cranial_width, left_point, right_point, slice_idx = calculate_cranial_width(original_brain)

        if verbose:
            print(f"\n顱內橫向最大寬度: {cranial_width:.2f} mm")
            print(f"位置: Z 切面 #{slice_idx}")
            print(f"左端點座標 (mm): ({left_point[0]:.2f}, {left_point[1]:.2f}, {left_point[2]:.2f})")
            print(f"右端點座標 (mm): ({right_point[0]:.2f}, {right_point[1]:.2f}, {right_point[2]:.2f})")

        # 步驟 4: 計算腦室距離與顱內寬度的比值
        if verbose:
            print("\n步驟 4: 計算腦室距離與顱內寬度的比值")
            print("-" * 70)
        ratio = calculate_ventricle_to_cranial_ratio(distance_mm, cranial_width)
        if verbose:
            print(f"腦室距離/顱內寬度比值: {ratio:.4f} ({ratio*100:.2f}%)")

        # 顯示完整測量摘要
        if verbose:
            print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size,
                                     cranial_width_mm=cranial_width, ratio=ratio)

        # 步驟 5: 產生視覺化圖片
        if verbose:
            print("\n步驟 5: 產生3D視覺化圖片")
            print("-" * 70)

        # 準備顱內寬度資料
        cranial_width_data = (cranial_width, left_point, right_point, slice_idx)

        visualize_ventricle_distance(
            left_vent, right_vent,
            left_centroid, right_centroid,
            distance_mm,
            output_path=str(output_image_path),
            show_plot=show_plot,
            original_path=original_brain_path,
            cranial_width_data=cranial_width_data,
            ratio=ratio
        )

        # 返回結果字典
        return {
            'status': 'success',
            'ventricle_distance_mm': distance_mm,
            'cranial_width_mm': cranial_width,
            'ratio': ratio,
            'ratio_percent': ratio * 100,
            'left_centroid': list(left_centroid),
            'right_centroid': list(right_centroid),
            'voxel_size': list(voxel_size),
            'cranial_width_endpoints': {
                'left': list(left_point),
                'right': list(right_point),
                'slice_index': slice_idx
            }
        }

    except Exception as e:
        if verbose:
            print(f"\n錯誤: {str(e)}")
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }


def main():
    """主程式：計算左右腦室質心距離"""
    print("=" * 70)
    print("3D NPH 指標計算 - 腦室質心距離分析")
    print("=" * 70)

    # 設定資料目錄
    data_dir = "000016209E"

    # 建立 result 資料夾
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    # 設定輸出檔案路徑
    output_image = result_dir / f"{data_dir}_ventricle_distance.png"

    # 處理案例
    result = process_case(data_dir, output_image, show_plot=False, verbose=True)

    if result['status'] == 'success':
        output_html = output_image.with_suffix('.html')
        print("\n" + "=" * 70)
        print("分析完成！")
        print(f"結果圖片: {output_image}")
        print(f"互動式HTML: {output_html}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("分析失敗！")
        print(f"錯誤: {result.get('error_message', '未知錯誤')}")
        print("=" * 70)


if __name__ == "__main__":
    main()
