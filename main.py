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


def main():
    """主程式：計算左右腦室質心距離"""
    print("=" * 70)
    print("3D NPH 指標計算 - 腦室質心距離分析")
    print("=" * 70)

    # 設定資料目錄
    data_dir = "000016209E"

    # 檔案路徑
    left_ventricle_path = f"{data_dir}/Ventricle_L.nii.gz"
    right_ventricle_path = f"{data_dir}/Ventricle_R.nii.gz"
    original_brain_path = f"{data_dir}/original.nii.gz"

    # 步驟 1: 載入左右腦室（使用原始座標系統）
    print("\n步驟 1: 載入左右腦室影像")
    print("-" * 70)
    left_vent, right_vent = load_ventricle_pair(
        left_ventricle_path, right_ventricle_path, verbose=True
    )

    # 步驟 2: 計算質心距離
    print("\n步驟 2: 計算左右腦室質心距離")
    print("-" * 70)
    distance_mm, left_centroid, right_centroid, voxel_size = calculate_centroid_distance(
        left_vent, right_vent
    )

    # 步驟 3: 計算顱內橫向最大寬度
    print("\n步驟 3: 計算顱內橫向最大寬度")
    print("-" * 70)

    original_brain = nib.load(original_brain_path)
    cranial_width, left_point, right_point, slice_idx = calculate_cranial_width(original_brain)

    print(f"\n顱內橫向最大寬度: {cranial_width:.2f} mm")
    print(f"位置: Z 切面 #{slice_idx}")
    print(f"左端點座標 (mm): ({left_point[0]:.2f}, {left_point[1]:.2f}, {left_point[2]:.2f})")
    print(f"右端點座標 (mm): ({right_point[0]:.2f}, {right_point[1]:.2f}, {right_point[2]:.2f})")

    # 步驟 4: 計算腦室距離與顱內寬度的比值
    print("\n步驟 4: 計算腦室距離與顱內寬度的比值")
    print("-" * 70)
    ratio = calculate_ventricle_to_cranial_ratio(distance_mm, cranial_width)
    print(f"腦室距離/顱內寬度比值: {ratio:.4f} ({ratio*100:.2f}%)")

    # 顯示完整測量摘要
    print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size,
                             cranial_width_mm=cranial_width, ratio=ratio)

    # 步驟 5: 產生視覺化圖片
    print("\n步驟 5: 產生3D視覺化圖片")
    print("-" * 70)

    # 建立 result 資料夾
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    # 設定輸出檔案路徑
    output_image = result_dir / f"{data_dir}_ventricle_distance.png"

    # 準備顱內寬度資料
    cranial_width_data = (cranial_width, left_point, right_point, slice_idx)

    visualize_ventricle_distance(
        left_vent, right_vent,
        left_centroid, right_centroid,
        distance_mm,
        output_path=str(output_image),
        show_plot=False,  # 設為 True 會在瀏覽器開啟互動圖表
        original_path=original_brain_path,  # 加入原始腦部影像
        cranial_width_data=cranial_width_data,  # 加入顱內寬度資料
        ratio=ratio  # 加入腦室距離/顱內寬度比值
    )

    output_html = output_image.with_suffix('.html')

    print("\n" + "=" * 70)
    print("分析完成！")
    print(f"結果圖片: {output_image}")
    print(f"互動式HTML: {output_html}")
    print("=" * 70)


if __name__ == "__main__":
    main()
