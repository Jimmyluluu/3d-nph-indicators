#!/usr/bin/env python3
"""
批次處理腳本
自動處理外接硬碟中的所有 NPH 案例資料
"""

import time
from pathlib import Path
from datetime import datetime
from main import process_case
from model.data_export import DataExporter, ProcessLogger


def scan_data_directory(base_dir, skip_not_ok=True):
    """
    掃描資料目錄，找出所有有效的案例資料夾

    Args:
        base_dir: 基礎資料目錄路徑
        skip_not_ok: 是否跳過標記為 _not_ok 的資料夾

    Returns:
        list: 案例資料夾路徑列表
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"資料目錄不存在: {base_dir}")

    # 取得所有子目錄
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    # 過濾掉隱藏檔案（._ 開頭）和 _not_ok 標記的資料夾
    valid_dirs = []
    for d in all_dirs:
        # 跳過隱藏目錄
        if d.name.startswith('.'):
            continue

        # 如果需要跳過 _not_ok
        if skip_not_ok and '_not_ok' in d.name:
            continue

        # 檢查是否包含必要的檔案（兩種命名模式）
        # 模式 1: 標準命名 (000016209E 等)
        required_files_pattern1 = [
            d / "Ventricle_L.nii.gz",
            d / "Ventricle_R.nii.gz",
            d / "original.nii.gz"
        ]

        # 模式 2: data_ 開頭的命名
        # 嘗試找出編號
        if d.name.startswith('data_'):
            # 提取編號，例如 data_1 -> 1
            data_num = d.name.replace('data_', '')
            required_files_pattern2 = [
                d / f"mask_Ventricle_L_{data_num}.nii.gz",
                d / f"mask_Ventricle_R_{data_num}.nii.gz",
                d / f"original_{data_num}.nii.gz"
            ]
        else:
            required_files_pattern2 = []

        # 檢查任一模式是否存在
        if all(f.exists() for f in required_files_pattern1):
            valid_dirs.append(d)
        elif required_files_pattern2 and all(f.exists() for f in required_files_pattern2):
            valid_dirs.append(d)

    return sorted(valid_dirs)


def format_time(seconds):
    """格式化時間顯示"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} 分鐘"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} 小時"


def generate_markdown_report(results, output_path, total_time, success_count, error_count):
    """
    產生 Markdown 格式的報表

    Args:
        results: 所有案例的結果列表
        output_path: 輸出檔案路徑
        total_time: 總處理時間
        success_count: 成功數量
        error_count: 失敗數量
    """
    # 水腦症案例列表
    nph_cases = [
        "000235496D",
        "000206288G",
        "000152785B",
        "000137208D",
        "000096384I",
        "000087554H"
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        # 標題
        f.write("# 3D NPH 指標批次處理報表\n\n")
        f.write(f"**處理時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 摘要統計
        f.write("## 處理摘要\n\n")
        f.write(f"- **總案例數**: {len(results)}\n")
        f.write(f"- **成功**: {success_count} 個\n")
        f.write(f"- **失敗**: {error_count} 個\n")
        f.write(f"- **成功率**: {success_count/len(results)*100:.1f}%\n")
        f.write(f"- **總耗時**: {format_time(total_time)}\n")
        f.write(f"- **平均每案例**: {format_time(total_time/len(results))}\n\n")

        # 成功案例表格
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            f.write("## 測量結果\n\n")
            f.write("| 案例 ID | 腦室距離 (mm) | 顱內寬度 (mm) | 比值 | 百分比 | 處理時間 |\n")
            f.write("|---------|---------------|---------------|------|--------|----------|\n")

            for result in successful_results:
                case_id = result.get('case_id', 'N/A')
                distance = result.get('ventricle_distance_mm', 0)
                width = result.get('cranial_width_mm', 0)
                ratio = result.get('ratio', 0)
                percent = result.get('ratio_percent', 0)
                time_str = result.get('processing_time', 'N/A')

                # 檢查是否為水腦症案例
                if case_id in nph_cases:
                    case_id_display = f"{case_id} ⚠️ NPH"
                else:
                    case_id_display = case_id

                f.write(f"| {case_id_display} | {distance:.2f} | {width:.2f} | {ratio:.4f} | {percent:.2f}% | {time_str} |\n")

            # 統計資訊
            distances = [r['ventricle_distance_mm'] for r in successful_results]
            widths = [r['cranial_width_mm'] for r in successful_results]
            ratios = [r['ratio'] for r in successful_results]

            f.write("\n### 統計數據（全部案例）\n\n")
            f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
            f.write("|------|--------|--------|--------|--------|\n")
            f.write(f"| 腦室距離 (mm) | {min(distances):.2f} | {max(distances):.2f} | {sum(distances)/len(distances):.2f} | {sorted(distances)[len(distances)//2]:.2f} |\n")
            f.write(f"| 顱內寬度 (mm) | {min(widths):.2f} | {max(widths):.2f} | {sum(widths)/len(widths):.2f} | {sorted(widths)[len(widths)//2]:.2f} |\n")
            f.write(f"| 比值 | {min(ratios):.4f} | {max(ratios):.4f} | {sum(ratios)/len(ratios):.4f} | {sorted(ratios)[len(ratios)//2]:.4f} |\n")

            # 水腦症案例統計
            nph_results = [r for r in successful_results if r.get('case_id') in nph_cases]
            if nph_results:
                nph_distances = [r['ventricle_distance_mm'] for r in nph_results]
                nph_widths = [r['cranial_width_mm'] for r in nph_results]
                nph_ratios = [r['ratio'] for r in nph_results]

                f.write("\n### 水腦症案例統計 (NPH)\n\n")
                f.write(f"**共 {len(nph_results)} 個水腦症案例**\n\n")
                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| 腦室距離 (mm) | {min(nph_distances):.2f} | {max(nph_distances):.2f} | {sum(nph_distances)/len(nph_distances):.2f} | {sorted(nph_distances)[len(nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(nph_widths):.2f} | {max(nph_widths):.2f} | {sum(nph_widths)/len(nph_widths):.2f} | {sorted(nph_widths)[len(nph_widths)//2]:.2f} |\n")
                f.write(f"| 比值 | {min(nph_ratios):.4f} | {max(nph_ratios):.4f} | {sum(nph_ratios)/len(nph_ratios):.4f} | {sorted(nph_ratios)[len(nph_ratios)//2]:.4f} |\n")

            # 非水腦症案例統計
            non_nph_results = [r for r in successful_results if r.get('case_id') not in nph_cases]
            if non_nph_results:
                non_nph_distances = [r['ventricle_distance_mm'] for r in non_nph_results]
                non_nph_widths = [r['cranial_width_mm'] for r in non_nph_results]
                non_nph_ratios = [r['ratio'] for r in non_nph_results]

                f.write("\n### 非水腦症案例統計 (非 NPH)\n\n")
                f.write(f"**共 {len(non_nph_results)} 個非水腦症案例**\n\n")
                f.write("| 指標 | 最小值 | 最大值 | 平均值 | 中位數 |\n")
                f.write("|------|--------|--------|--------|--------|\n")
                f.write(f"| 腦室距離 (mm) | {min(non_nph_distances):.2f} | {max(non_nph_distances):.2f} | {sum(non_nph_distances)/len(non_nph_distances):.2f} | {sorted(non_nph_distances)[len(non_nph_distances)//2]:.2f} |\n")
                f.write(f"| 顱內寬度 (mm) | {min(non_nph_widths):.2f} | {max(non_nph_widths):.2f} | {sum(non_nph_widths)/len(non_nph_widths):.2f} | {sorted(non_nph_widths)[len(non_nph_widths)//2]:.2f} |\n")
                f.write(f"| 比值 | {min(non_nph_ratios):.4f} | {max(non_nph_ratios):.4f} | {sum(non_nph_ratios)/len(non_nph_ratios):.4f} | {sorted(non_nph_ratios)[len(non_nph_ratios)//2]:.4f} |\n")

        # 失敗案例
        failed_results = [r for r in results if r.get('status') == 'error']
        if failed_results:
            f.write("\n## 失敗案例\n\n")
            f.write("| 案例 ID | 錯誤類型 | 錯誤訊息 |\n")
            f.write("|---------|----------|----------|\n")

            for result in failed_results:
                case_id = result.get('case_id', 'N/A')
                error_type = result.get('error_type', 'Unknown')
                error_msg = result.get('error_message', 'N/A')

                # 截斷過長的錯誤訊息
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."

                f.write(f"| {case_id} | {error_type} | {error_msg} |\n")

        f.write("\n---\n\n")
        f.write("*由 3D NPH Indicators 自動產生*\n")


def batch_process(data_dir, output_dir="result", skip_not_ok=True):
    """
    批次處理所有案例

    Args:
        data_dir: 資料目錄路徑
        output_dir: 輸出目錄路徑
        skip_not_ok: 是否跳過標記為 _not_ok 的資料夾
    """
    # 建立輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 初始化資料匯出器和日誌記錄器
    exporter = DataExporter(output_path)
    log_file = output_path / "processing.log"

    with ProcessLogger(log_file) as logger:
        logger.info(f"開始批次處理")
        logger.info(f"資料目錄: {data_dir}")
        logger.info(f"輸出目錄: {output_dir}")
        logger.info(f"跳過 _not_ok: {skip_not_ok}")

        # 掃描資料目錄
        try:
            case_dirs = scan_data_directory(data_dir, skip_not_ok=skip_not_ok)
            total_cases = len(case_dirs)

            if total_cases == 0:
                logger.warning("沒有找到有效的案例資料夾！")
                return

            logger.info(f"找到 {total_cases} 個有效案例")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"掃描資料目錄失敗: {str(e)}", e)
            return

        # 處理統計
        results = []
        success_count = 0
        error_count = 0
        start_time = time.time()

        # 逐一處理每個案例
        for i, case_dir in enumerate(case_dirs, 1):
            case_id = case_dir.name
            case_start_time = time.time()

            logger.info(f"\n[{i}/{total_cases}] 處理案例: {case_id}")

            try:
                # 設定輸出路徑
                output_image_path = exporter.get_image_path(case_id)

                # 處理案例（關閉詳細輸出）
                result = process_case(
                    str(case_dir),
                    str(output_image_path),
                    show_plot=False,
                    verbose=False
                )

                # 計算處理時間
                processing_time = time.time() - case_start_time
                result['processing_time'] = f"{processing_time:.2f}s"
                result['case_id'] = case_id

                # 記錄結果
                if result['status'] == 'success':
                    success_count += 1
                    logger.success(
                        f"  ✓ 成功"
                    )
                    logger.info(f"     腦室距離: {result['ventricle_distance_mm']:.2f} mm")
                    logger.info(f"     顱內寬度: {result['cranial_width_mm']:.2f} mm")
                    logger.info(f"     比值: {result['ratio']:.4f} ({result['ratio_percent']:.2f}%)")
                    logger.info(f"     處理時間: {processing_time:.1f}s")

                else:
                    error_count += 1
                    logger.error(
                        f"  ✗ 失敗 - {result.get('error_message', '未知錯誤')}",
                        None
                    )

                results.append(result)

                # 顯示進度和預估時間
                elapsed_time = time.time() - start_time
                avg_time_per_case = elapsed_time / i
                remaining_cases = total_cases - i
                estimated_remaining = avg_time_per_case * remaining_cases

                logger.info(
                    f"  進度: {i}/{total_cases} ({i/total_cases*100:.1f}%) | "
                    f"成功: {success_count} | 失敗: {error_count} | "
                    f"預估剩餘: {format_time(estimated_remaining)}"
                )

            except Exception as e:
                error_count += 1
                processing_time = time.time() - case_start_time
                logger.error(f"  處理案例時發生例外: {str(e)}", e)

                # 記錄錯誤結果
                results.append({
                    'case_id': case_id,
                    'status': 'error',
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'processing_time': f"{processing_time:.2f}s"
                })

        # 所有案例處理完成
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("批次處理完成！")
        logger.info(f"總共處理: {total_cases} 個案例")
        logger.info(f"成功: {success_count} 個")
        logger.info(f"失敗: {error_count} 個")
        logger.info(f"總耗時: {format_time(total_time)}")
        logger.info(f"平均每案例: {format_time(total_time/total_cases)}")

        # 產生 Markdown 報表
        logger.info("\n產生 Markdown 報表...")
        md_path = output_path / "results_summary.md"
        generate_markdown_report(results, md_path, total_time, success_count, error_count)
        logger.success(f"Markdown 報表已儲存: {md_path}")

        logger.info("\n結果檔案位置:")
        logger.info(f"  圖片: {exporter.images_dir}")
        logger.info(f"  HTML: {exporter.html_dir}")
        logger.info(f"  報表: {md_path}")
        logger.info(f"  日誌: {log_file}")
        logger.info("=" * 70)


def main():
    """主程式"""
    print("=" * 70)
    print("3D NPH 指標批次處理工具")
    print("=" * 70)
    print()

    # 設定參數
    data_directory = "/Volumes/Kuro醬の1TSSD/標記好的資料"
    output_directory = "result"
    skip_not_ok = True

    print(f"資料目錄: {data_directory}")
    print(f"輸出目錄: {output_directory}")
    print(f"跳過 _not_ok 資料夾: {'是' if skip_not_ok else '否'}")
    print()

    # 執行批次處理
    batch_process(
        data_dir=data_directory,
        output_dir=output_directory,
        skip_not_ok=skip_not_ok
    )


if __name__ == "__main__":
    main()
