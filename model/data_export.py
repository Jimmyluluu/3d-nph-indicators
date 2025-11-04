#!/usr/bin/env python3
"""
資料匯出模組
負責將分析結果匯出為 CSV、JSON 等格式
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class DataExporter:
    """資料匯出器"""

    def __init__(self, output_dir: Path):
        """
        初始化資料匯出器

        Args:
            output_dir: 輸出目錄路徑
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 建立子目錄
        self.images_dir = self.output_dir / "images"
        self.html_dir = self.output_dir / "html"
        self.json_dir = self.output_dir / "json"

        self.images_dir.mkdir(exist_ok=True)
        self.html_dir.mkdir(exist_ok=True)
        self.json_dir.mkdir(exist_ok=True)

    def export_json(self, case_id: str, data: Dict[str, Any]) -> Path:
        """
        匯出單一案例的 JSON 數據

        Args:
            case_id: 案例 ID
            data: 要匯出的數據字典

        Returns:
            JSON 檔案路徑
        """
        json_path = self.json_dir / f"{case_id}.json"

        # 添加時間戳記
        data['export_time'] = datetime.now().isoformat()
        data['case_id'] = case_id

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return json_path

    def export_summary_csv(self, results: List[Dict[str, Any]], csv_path: Path = None) -> Path:
        """
        匯出彙總 CSV 報表

        Args:
            results: 所有案例的結果列表
            csv_path: CSV 檔案路徑（可選，預設為 summary.csv）

        Returns:
            CSV 檔案路徑
        """
        if csv_path is None:
            csv_path = self.output_dir / "summary.csv"

        if not results:
            return csv_path

        # 定義 CSV 欄位
        fieldnames = [
            'case_id',
            'ventricle_distance_mm',
            'cranial_width_mm',
            'ratio',
            'ratio_percent',
            'left_centroid_x',
            'left_centroid_y',
            'left_centroid_z',
            'right_centroid_x',
            'right_centroid_y',
            'right_centroid_z',
            'voxel_spacing_x',
            'voxel_spacing_y',
            'voxel_spacing_z',
            'processing_time',
            'status',
            'error_message'
        ]

        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # 準備每一行的數據
                row = {
                    'case_id': result.get('case_id', ''),
                    'ventricle_distance_mm': result.get('ventricle_distance_mm', ''),
                    'cranial_width_mm': result.get('cranial_width_mm', ''),
                    'ratio': result.get('ratio', ''),
                    'ratio_percent': result.get('ratio_percent', ''),
                    'processing_time': result.get('processing_time', ''),
                    'status': result.get('status', ''),
                    'error_message': result.get('error_message', '')
                }

                # 添加質心座標
                left_centroid = result.get('left_centroid', [None, None, None])
                row['left_centroid_x'] = left_centroid[0] if len(left_centroid) > 0 else ''
                row['left_centroid_y'] = left_centroid[1] if len(left_centroid) > 1 else ''
                row['left_centroid_z'] = left_centroid[2] if len(left_centroid) > 2 else ''

                right_centroid = result.get('right_centroid', [None, None, None])
                row['right_centroid_x'] = right_centroid[0] if len(right_centroid) > 0 else ''
                row['right_centroid_y'] = right_centroid[1] if len(right_centroid) > 1 else ''
                row['right_centroid_z'] = right_centroid[2] if len(right_centroid) > 2 else ''

                # 添加體素間距
                voxel_size = result.get('voxel_size', [None, None, None])
                row['voxel_spacing_x'] = voxel_size[0] if len(voxel_size) > 0 else ''
                row['voxel_spacing_y'] = voxel_size[1] if len(voxel_size) > 1 else ''
                row['voxel_spacing_z'] = voxel_size[2] if len(voxel_size) > 2 else ''

                writer.writerow(row)

        return csv_path

    def get_image_path(self, case_id: str) -> Path:
        """取得圖片輸出路徑"""
        return self.images_dir / f"{case_id}.png"

    def get_html_path(self, case_id: str) -> Path:
        """取得 HTML 輸出路徑"""
        return self.html_dir / f"{case_id}.html"


class ProcessLogger:
    """處理日誌記錄器"""

    def __init__(self, log_file: Path):
        """
        初始化日誌記錄器

        Args:
            log_file: 日誌檔案路徑
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True, parents=True)

        # 開啟日誌檔案
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')

        # 寫入標頭
        self._log(f"處理日誌 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 70)

    def _log(self, message: str):
        """內部日誌寫入方法"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.file_handle.write(log_message + '\n')
        self.file_handle.flush()

    def info(self, message: str):
        """記錄一般訊息"""
        self._log(f"INFO: {message}")

    def success(self, message: str):
        """記錄成功訊息"""
        self._log(f"SUCCESS: {message}")

    def warning(self, message: str):
        """記錄警告訊息"""
        self._log(f"WARNING: {message}")

    def error(self, message: str, exception: Exception = None):
        """記錄錯誤訊息"""
        self._log(f"ERROR: {message}")
        if exception:
            self._log(f"  Exception: {type(exception).__name__}: {str(exception)}")

    def close(self):
        """關閉日誌檔案"""
        self._log("=" * 70)
        self._log("處理完成")
        self.file_handle.close()

    def __enter__(self):
        """支援 with 語句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支援 with 語句"""
        self.close()
        return False
