#!/usr/bin/env python3
"""
3D NPH Indicators - 主程式
正常壓力型水腦症（NPH）3D影像指標計算工具
"""

from pathlib import Path
from model.ventricle_analysis import (
    load_ventricle_pair,
    calculate_centroid_distance,
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

    # 左右腦室檔案路徑
    left_ventricle_path = f"{data_dir}/Ventricle_L.nii.gz"
    right_ventricle_path = f"{data_dir}/Ventricle_R.nii.gz"

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

    # 步驟 3: 顯示結果
    print_measurement_summary(distance_mm, left_centroid, right_centroid, voxel_size)

    # 步驟 4: 產生視覺化圖片
    print("\n步驟 3: 產生3D視覺化圖片")
    print("-" * 70)

    # 建立 result 資料夾
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    # 設定輸出檔案路徑
    output_image = result_dir / f"{data_dir}_ventricle_distance.png"

    visualize_ventricle_distance(
        left_vent, right_vent,
        left_centroid, right_centroid,
        distance_mm,
        output_path=str(output_image),
        show_plot=False  # 設為 True 會在瀏覽器開啟互動圖表
    )

    output_html = output_image.with_suffix('.html')

    print("\n" + "=" * 70)
    print("分析完成！")
    print(f"結果圖片: {output_image}")
    print(f"互動式HTML: {output_html}")
    print("=" * 70)


if __name__ == "__main__":
    main()
