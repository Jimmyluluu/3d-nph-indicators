#!/usr/bin/env python3
"""
批次處理模組
負責批次處理多個案例的指標計算
"""

import time
from pathlib import Path
from processors.logger import ProcessLogger
from processors.case_processor import process_case_indicator_ratio, process_case_evan_index, process_case_surface_area, process_case_volume_surface_ratio
from model.report_generator import generate_markdown_report, INDICATOR_CONFIGS, format_time


def scan_data_directory(base_dir, indicator_type, skip_not_ok=True):
    """
    掃描資料目錄，找出所有有效的案例資料夾

    Args:
        base_dir: 基礎資料目錄路徑
        indicator_type: 指標類型, 用於決定需要哪些檔案
        skip_not_ok: 是否跳過標記為 _not_ok 的資料夾

    Returns:
        list: 案例資料夾路徑列表
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"資料目錄不存在: {base_dir}")

    # 取得所有子目錄
    all_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    requires_original = indicator_type not in ['surface_area', 'volume_surface_ratio']

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
        # 模式 1: 標準命名
        required_files_p1 = [d / "Ventricle_L.nii.gz", d / "Ventricle_R.nii.gz"]
        if requires_original:
            required_files_p1.append(d / "original.nii.gz")

        # 模式 2: data_ 開頭的命名
        if d.name.startswith('data_'):
            data_num = d.name.replace('data_', '')
            required_files_p2 = [
                d / f"mask_Ventricle_L_{data_num}.nii.gz",
                d / f"mask_Ventricle_R_{data_num}.nii.gz"
            ]
            if requires_original:
                required_files_p2.append(d / f"original_{data_num}.nii.gz")
        else:
            required_files_p2 = []

        # 檢查任一模式是否存在
        if all(f.exists() for f in required_files_p1):
            valid_dirs.append(d)
        elif required_files_p2 and all(f.exists() for f in required_files_p2):
            valid_dirs.append(d)

    return sorted(valid_dirs)


def batch_process(data_dir, indicator_type="centroid_ratio", skip_not_ok=True,
                  z_range=(0.3, 0.9), y_percentile=4):
    """
    批次處理所有案例

    Args:
        data_dir: 資料目錄路徑
        indicator_type: 指標類型
        skip_not_ok: 是否跳過標記為 _not_ok 的資料夾
        z_range: Z 軸切面範圍（僅用於 evan_index）
        y_percentile: Y 軸前方百分位數（僅用於 evan_index）
    """
    # 設定輸出目錄
    output_dir = f"result/{indicator_type}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "processing.log"

    # 驗證指標類型並取得配置
    if indicator_type not in INDICATOR_CONFIGS:
        raise ValueError(f"不支援的指標類型: {indicator_type}。可用的類型: {list(INDICATOR_CONFIGS.keys())}")

    # 選擇處理函數
    if indicator_type == "centroid_ratio":
        process_func = process_case_indicator_ratio
        indicator_name = "腦室質心距離比值"
    elif indicator_type == "evan_index":
        process_func = lambda data_dir, output_path, show_plot=False, verbose=True: process_case_evan_index(data_dir, output_path, show_plot=show_plot, verbose=verbose, z_range=z_range, y_percentile=y_percentile)
        indicator_name = "3D Evan Index"
    elif indicator_type == "surface_area":
        process_func = lambda data_dir, output_path, show_plot=False, verbose=True: process_case_surface_area(data_dir, output_path, show_plot=show_plot, verbose=verbose)
        indicator_name = "腦室表面積"
    elif indicator_type == "volume_surface_ratio":
        process_func = lambda data_dir, output_path, show_plot=False, verbose=True: process_case_volume_surface_ratio(data_dir, output_path, show_plot=show_plot, verbose=verbose)
        indicator_name = "體積與表面積比例"
    else:
        raise ValueError(f"不支援的指標類型: {indicator_type}")

    with ProcessLogger(log_file) as logger:
        logger.info(f"開始批次處理 - {indicator_name}")
        logger.info(f"資料目錄: {data_dir}")
        logger.info(f"輸出目錄: {output_dir}")
        logger.info(f"跳過 _not_ok: {skip_not_ok}")

        if indicator_type == "evan_index":
            logger.info(f"前腳定義 - Z 範圍: {z_range}, Y 百分位: {y_percentile}")

        # 掃描資料目錄
        try:
            case_dirs = scan_data_directory(data_dir, indicator_type, skip_not_ok=skip_not_ok)
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
                # 為每個案例建立獨立資料夾
                case_output_dir = output_path / case_id
                case_output_dir.mkdir(exist_ok=True)

                output_image_path = case_output_dir / f"{case_id}.png"

                # 處理案例
                result = process_func(
                    str(case_dir),
                    str(output_image_path),
                    show_plot=False,
                    verbose=False
                )

                processing_time = time.time() - case_start_time
                result['processing_time'] = f"{processing_time:.2f}s"
                result['case_id'] = case_id

                # 記錄結果
                if result['status'] == 'success':
                    success_count += 1
                    logger.success(f"  ✓ 成功")

                    if indicator_type == "centroid_ratio":
                        logger.info(f"     腦室距離: {result['ventricle_distance_mm']:.2f} mm")
                        logger.info(f"     顱內寬度: {result['cranial_width_mm']:.2f} mm")
                        logger.info(f"     比值: {result['ratio']:.4f} ({result['ratio_percent']:.2f}%)")
                    elif indicator_type == "evan_index":
                        logger.info(f"     前腳距離: {result['anterior_horn_distance_mm']:.2f} mm")
                        logger.info(f"     顱內寬度: {result['cranial_width_mm']:.2f} mm")
                        logger.info(f"     Evan Index: {result['evan_index']:.4f} ({result['evan_index_percent']:.2f}%)")
                    elif indicator_type == "surface_area":
                        logger.info(f"     左腦室表面積: {result['left_surface_area']:.2f} mm^2")
                        logger.info(f"     右腦室表面積: {result['right_surface_area']:.2f} mm^2")
                        logger.info(f"     總表面積: {result['total_surface_area']:.2f} mm^2")
                    elif indicator_type == "volume_surface_ratio":
                        logger.info(f"     左腦室體積: {result['left_volume']:.2f} mm³")
                        logger.info(f"     右腦室體積: {result['right_volume']:.2f} mm³")
                        logger.info(f"     總體積: {result['total_volume']:.2f} mm³")
                        logger.info(f"     左腦室表面積: {result['left_surface_area']:.2f} mm²")
                        logger.info(f"     右腦室表面積: {result['right_surface_area']:.2f} mm²")
                        logger.info(f"     總表面積: {result['total_surface_area']:.2f} mm²")
                        logger.info(f"     左腦室比例: {result['left_ratio']:.4f} mm")
                        logger.info(f"     右腦室比例: {result['right_ratio']:.4f} mm")
                        logger.info(f"     整體比例: {result['total_ratio']:.4f} mm")
                        logger.info(f"     比例差異: {result['ratio_difference']:.4f} mm ({result['ratio_difference_percent']:.2f}%)")

                    logger.info(f"     處理時間: {processing_time:.1f}s")
                else:
                    error_count += 1
                    logger.error(f"  ✗ 失敗 - {result.get('error_message', '未知錯誤')}", None)

                results.append(result)

                # 顯示進度
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

                results.append({
                    'case_id': case_id,
                    'status': 'error',
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'processing_time': f"{processing_time:.2f}s"
                })

        # 處理完成
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
        generate_markdown_report(results, md_path, total_time, success_count, error_count, indicator_type)
        logger.success(f"Markdown 報表已儲存: {md_path}")

        logger.info("\n結果檔案位置:")
        logger.info(f"  案例資料夾: {output_path}/<case_id>/")
        logger.info(f"    - <case_id>.png (圖片)")
        logger.info(f"    - <case_id>.html (互動式圖表)")
        logger.info(f"  報表: {md_path}")
        logger.info(f"  日誌: {log_file}")
        logger.info("=" * 70)
