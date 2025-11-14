#!/usr/bin/env python3
"""
處理日誌記錄模組
負責批次處理過程的日誌記錄
"""

from pathlib import Path
from datetime import datetime


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
