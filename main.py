#!/usr/bin/env python3
"""
3D NPH Indicators - 主程式入口
正常壓力型水腦症（NPH）3D影像指標計算工具

統一的 CLI 入口,支援:
- 單案例處理
- 批次處理
- 三種指標類型 (centroid_ratio, evan_index, surface_area)
"""

import argparse
from pathlib import Path
from processors.batch_processor import batch_process
from processors.case_processor import process_case_indicator_ratio, process_case_evan_index, process_case_surface_area, process_case_volume_surface_ratio


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description='3D NPH 指標計算工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
指標類型說明:
  centroid_ratio      - 腦室質心距離/顱內寬度比值（預設）
  evan_index          - 腦室前腳最大距離/顱內寬度比值（3D Evan Index）
  surface_area        - 腦室表面積（mm^2）
  volume_surface_ratio - 體積與表面積比例（mm，球形度指標）

使用範例:
  # 批次處理
  python main.py batch --type centroid_ratio
  python main.py batch --type evan_index --data-dir /path/to/data
  python main.py batch --type surface_area --smoothing-iterations 150
  python main.py batch --type volume_surface_ratio

  # 單案例處理
  python main.py single --case-dir 000016209E --type centroid_ratio
  python main.py single --case-dir data_5 --type surface_area
  python main.py single --case-dir data_5 --type volume_surface_ratio
        """
    )

    # 建立子命令
    subparsers = parser.add_subparsers(dest='command', help='處理模式', required=True)

    # ===== 批次處理子命令 =====
    batch_parser = subparsers.add_parser(
        'batch',
        help='批次處理多個案例',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    batch_parser.add_argument(
        '--type', '-t',
        choices=['centroid_ratio', 'evan_index', 'surface_area', 'volume_surface_ratio'],
        default='centroid_ratio',
        help='指標類型（預設: centroid_ratio）'
    )

    batch_parser.add_argument(
        '--data-dir', '-d',
        default='/Volumes/Kuro醬の1TSSD/標記好的資料',
        help='資料目錄路徑'
    )

    batch_parser.add_argument(
        '--skip-not-ok',
        action='store_true',
        default=True,
        help='跳過標記為 _not_ok 的資料夾（預設: True）'
    )

    batch_parser.add_argument(
        '--z-range',
        nargs=2,
        type=float,
        default=[0.3, 0.9],
        metavar=('MIN', 'MAX'),
        help='Z 軸切面範圍（僅用於 evan_index，預設: 0.3 0.9）'
    )

    batch_parser.add_argument(
        '--y-percentile',
        type=int,
        default=4,
        help='Y 軸前方百分位數（僅用於 evan_index，預設: 4）'
    )

    # Surface Area 相關參數
    batch_parser.add_argument(
        '--smoothing-iterations', type=int, default=100,
        help='平滑迭代次數（僅用於 surface_area，預設: 100）'
    )
    batch_parser.add_argument(
        '--smoothing-factor', type=float, default=0.1,
        help='平滑因子（僅用於 surface_area，預設: 0.1）'
    )

    # ===== 單案例處理子命令 =====
    single_parser = subparsers.add_parser(
        'single',
        help='處理單一案例',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    single_parser.add_argument(
        '--case-dir', '-c',
        required=True,
        help='案例資料夾路徑或名稱'
    )

    single_parser.add_argument(
        '--type', '-t',
        choices=['centroid_ratio', 'evan_index', 'surface_area', 'volume_surface_ratio'],
        default='centroid_ratio',
        help='指標類型（預設: centroid_ratio）'
    )

    single_parser.add_argument(
        '--output', '-o',
        help='輸出圖片路徑（預設: result/<case_id>_<type>.png）'
    )

    single_parser.add_argument(
        '--show-plot',
        action='store_true',
        help='顯示互動式圖表'
    )

    single_parser.add_argument(
        '--z-range',
        nargs=2,
        type=float,
        default=[0.3, 0.9],
        metavar=('MIN', 'MAX'),
        help='Z 軸切面範圍（僅用於 evan_index，預設: 0.3 0.9）'
    )

    single_parser.add_argument(
        '--y-percentile',
        type=int,
        default=4,
        help='Y 軸前方百分位數（僅用於 evan_index，預設: 4）'
    )

    # Surface Area 相關參數
    single_parser.add_argument(
        '--smoothing-iterations', type=int, default=100,
        help='平滑迭代次數（僅用於 surface_area，預設: 100）'
    )
    single_parser.add_argument(
        '--smoothing-factor', type=float, default=0.1,
        help='平滑因子（僅用於 surface_area，預設: 0.1）'
    )

    # 解析參數
    args = parser.parse_args()

    print("=" * 70)
    print("3D NPH 指標計算工具")
    print("=" * 70)

    # 根據子命令執行不同的處理
    if args.command == 'batch':
        # 批次處理
        print(f"模式: 批次處理")
        print(f"指標類型: {args.type}")
        print(f"資料目錄: {args.data_dir}")
        print(f"輸出目錄: result/{args.type}")
        print(f"跳過 _not_ok: {'是' if args.skip_not_ok else '否'}")

        if args.type == 'evan_index':
            print(f"前腳定義 - Z 範圍: {args.z_range[0]*100}%-{args.z_range[1]*100}%, Y 百分位: {args.y_percentile}%")

        print("=" * 70)
        print()

        # 執行批次處理
        batch_process(
            data_dir=args.data_dir,
            indicator_type=args.type,
            skip_not_ok=args.skip_not_ok,
            z_range=tuple(args.z_range),
            y_percentile=args.y_percentile
        )

    elif args.command == 'single':
        # 單案例處理
        case_dir = Path(args.case_dir)
        case_id = case_dir.name

        # 設定輸出路徑
        if args.output:
            output_path = Path(args.output)
        else:
            result_dir = Path("result")
            result_dir.mkdir(exist_ok=True)
            output_path = result_dir / f"{case_id}_{args.type}.png"

        print(f"模式: 單案例處理")
        print(f"案例目錄: {args.case_dir}")
        print(f"指標類型: {args.type}")
        print(f"輸出路徑: {output_path}")

        if args.type == 'evan_index':
            print(f"前腳定義 - Z 範圍: {args.z_range[0]*100}%-{args.z_range[1]*100}%, Y 百分位: {args.y_percentile}%")

        print("=" * 70)
        print()

        # 選擇處理函數
        if args.type == 'centroid_ratio':
            result = process_case_indicator_ratio(
                data_dir=str(case_dir),
                output_image_path=str(output_path),
                show_plot=args.show_plot,
                verbose=True
            )
        elif args.type == 'evan_index':
            result = process_case_evan_index(
                data_dir=str(case_dir),
                output_image_path=str(output_path),
                show_plot=args.show_plot,
                verbose=True,
                z_range=tuple(args.z_range),
                y_percentile=args.y_percentile
            )
        elif args.type == 'surface_area':
            result = process_case_surface_area(
                data_dir=str(case_dir),
                output_image_path=str(output_path),
                show_plot=args.show_plot,
                verbose=True
            )
        elif args.type == 'volume_surface_ratio':
            result = process_case_volume_surface_ratio(
                data_dir=str(case_dir),
                output_image_path=str(output_path),
                show_plot=args.show_plot,
                verbose=True
            )

        # 顯示結果
        print("\n" + "=" * 70)
        if result['status'] == 'success':
            print("分析完成！")
            print(f"結果圖片: {output_path}")
            print(f"互動式HTML: {output_path.with_suffix('.html')}")
        else:
            print("分析失敗！")
            print(f"錯誤: {result.get('error_message', '未知錯誤')}")
        print("=" * 70)


if __name__ == "__main__":
    main()
