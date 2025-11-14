# 3D NPH Indicators

正常壓力型水腦症（Normal Pressure Hydrocephalus, NPH）3D 影像指標計算工具

## 主要功能

### 支援的 NPH 指標

本工具支援兩種 NPH 診斷指標:

#### 1. 質心距離比值 (Centroid Ratio)

**測量內容**：
- 左右腦室質心之間的 3D 歐式距離
- 顱內橫向最大寬度
- 腦室質心距離 / 顱內寬度比值

**特點**：
- 在物理空間（世界座標系統）中計算質心位置
- 採用加權質心演算法（使用體素強度值作為權重）
- 自動拉正影像到 RAS+ 方向,確保測量一致性
- 每個 Z 切面測量顱內寬度,取最大值

#### 2. 3D Evan Index

**測量內容**：
- 左右腦室前腳（anterior horn）之間的最大距離
- 顱內橫向最大寬度
- 前腳最大距離 / 顱內寬度比值

**前腳定義**：
- Z 軸範圍：預設取 30%-90% 的上方區域（可調整）
- Y 軸過濾：預設取前方 4% 的點（可調整）
- 計算左右前腳點雲之間的最大距離

**特點**：
- 針對前腳區域的專門測量
- 參數可調整以適應不同影像特徵
- 提供更精確的前腳擴張評估

### 3D 視覺化

- 產生互動式 3D 視覺化圖表（HTML 格式）
- 靜態圖片輸出（PNG 格式）
- 視覺化內容包括：
  - 腦部表面網格（半透明灰色，Marching Cubes 提取）
  - 左右腦室平滑表面（藍色與紅色，Marching Cubes 提取）
  - 測量端點標記（鑽石形狀）
  - 測量線條（金色/紫色，標註距離）
  - 顱內橫向寬度連線（青色，標註寬度）
  - 完整的測量數據顯示於標題

## 專案結構

```plaintext
3d-nph-indicators/
├── main.py                      # 統一 CLI 入口（支援單案例/批次處理）
│
├── processors/                  # 處理流程模組
│   ├── case_processor.py        # 單案例處理邏輯
│   ├── batch_processor.py       # 批次處理邏輯
│   └── logger.py                # 日誌記錄
│
├── model/                       # 核心計算模組
│   ├── calculation.py           # 計算邏輯（質心、Evan Index、影像載入）
│   ├── visualization.py         # 3D 視覺化（Plotly + Marching Cubes）
│   ├── report_generator.py      # Markdown 報表生成
│   └── reorient.py             # 影像座標轉換（RAS+ 拉正）
│
├── result/                      # 輸出結果目錄
│   ├── centroid_ratio/          # 質心距離比值結果
│   │   ├── {case_id}/           # 各案例獨立資料夾
│   │   │   ├── {case_id}.png    # PNG 圖片
│   │   │   └── {case_id}.html   # 互動式 HTML
│   │   ├── results_summary.md   # 批次處理報表
│   │   └── processing.log       # 處理日誌
│   └── evan_index/              # 3D Evan Index 結果
│       └── ... (同上)
│
├── requirements.txt             # Python 依賴套件
├── README.md                    # 專案說明文件
└── CLAUDE.md                    # 開發指南（給 AI 助手）
```

### 模組職責說明

- **`model/`** - 純計算和視覺化邏輯,可被其他專案重用
- **`processors/`** - 處理流程協調,整合 model/ 中的功能
- **`main.py`** - 統一 CLI 入口,支援單案例和批次處理

## 安裝與環境設定

### 系統需求

- Python 3.8 或以上版本
- 建議使用虛擬環境

### 安裝步驟

1. 複製專案到本地端

```bash
git clone <repository-url>
cd 3d-nph-indicators
```

2. 建立並啟動虛擬環境（建議）

```bash
python3 -m venv env
source env/bin/activate  # macOS/Linux
# 或
env\Scripts\activate  # Windows
```

3. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 依賴套件說明

- `nibabel >= 5.0.0`: 讀取和處理 NIfTI 格式的醫學影像
- `numpy >= 1.24.0`: 數值計算和陣列操作
- `plotly >= 5.17.0`: 產生互動式 3D 視覺化圖表
- `kaleido >= 0.2.1`: 將 Plotly 圖表匯出為靜態圖片
- `scikit-image >= 0.21.0`: 影像處理（Marching Cubes 演算法）

## 使用方法

### 準備資料

確保你有以下 NIfTI 格式的影像檔案：

- `Ventricle_L.nii.gz` - 左腦室分割影像
- `Ventricle_R.nii.gz` - 右腦室分割影像
- `original.nii.gz` - 原始腦部影像（用於視覺化背景）

**支援兩種命名模式**：
1. 標準模式：`Ventricle_L.nii.gz`, `Ventricle_R.nii.gz`, `original.nii.gz`
2. Data 模式：`mask_Ventricle_L_N.nii.gz`, `mask_Ventricle_R_N.nii.gz`, `original_N.nii.gz`

將這些檔案放在資料目錄中（例如：`000016209E/` 或 `data_1/`）

### 批次處理

```bash
# 質心距離比值
python main.py batch --type centroid_ratio

# 3D Evan Index
python main.py batch --type evan_index

# 指定資料目錄
python main.py batch --type centroid_ratio --data-dir /path/to/data

# Evan Index 進階參數
python main.py batch --type evan_index --z-range 0.3 0.9 --y-percentile 4
```

### 單案例處理

```bash
# 處理單一案例
python main.py single --case-dir 000016209E --type centroid_ratio

# 顯示互動式圖表
python main.py single --case-dir data_5 --type evan_index --show-plot

# 指定輸出路徑
python main.py single --case-dir 000016209E --output my_result.png
```

### 查看幫助

```bash
python main.py --help
python main.py batch --help
python main.py single --help
```

### 參數說明

#### 指標類型參數

- `--type`, `-t`: 選擇指標類型
  - `centroid_ratio`: 腦室質心距離比值（預設）
  - `evan_index`: 3D Evan Index

#### 批次處理參數

- `--data-dir`, `-d`: 資料目錄路徑（預設: `/Volumes/Kuro醬の1TSSD/標記好的資料`）
- `--skip-not-ok`: 跳過標記為 `_not_ok` 的資料夾

#### 單案例處理參數

- `--case-dir`, `-c`: 案例資料夾路徑（必填）
- `--output`, `-o`: 輸出圖片路徑（可選）
- `--show-plot`: 顯示互動式圖表視窗

#### Evan Index 進階參數

- `--z-range MIN MAX`: Z 軸範圍（預設: `0.3 0.9`）
  - 調整前腳的垂直範圍
  - 如果前腳位置較高,可調整為 `0.5 0.9`
  - 如果前腳位置較低,可調整為 `0.3 0.7`

- `--y-percentile N`: Y 軸前方百分位（預設: `4`）
  - 調整前腳的前後範圍
  - 增加數值會包含更多後方的點
  - 減少數值會更集中在前方區域

### 輸出結果

#### 批次處理輸出

在 `result/{指標類型}/` 目錄中產生：

1. **各案例資料夾** `{case_id}/`
   - `{case_id}.png` - 高解析度 3D 視覺化（1200 x 900 像素）
   - `{case_id}.html` - 互動式圖表（可在瀏覽器中開啟）

2. **results_summary.md** - 批次處理報表
   - 處理摘要（總數、成功率、耗時）
   - 所有案例測量結果表格
   - 統計數據（最小值、最大值、平均值、中位數）
   - NPH vs 非 NPH 分組統計
   - 失敗案例列表

3. **processing.log** - 詳細處理日誌
   - 時間戳記
   - 每個案例的處理狀態
   - 錯誤訊息（如有）

#### 單案例輸出

在 `result/` 目錄或指定路徑產生：
- `{case_id}_{type}.png` - 視覺化圖片
- `{case_id}_{type}.html` - 互動式圖表

## 核心特性

### 1. 自動影像拉正

所有影像在載入時會自動拉正到 RAS+ 方向（Right-Anterior-Superior）:
- R (Right): X 軸正方向指向右側
- A (Anterior): Y 軸正方向指向前方
- S (Superior): Z 軸正方向指向頭頂

這確保所有測量都在統一的座標系統中進行,提高結果的準確性和可重複性。

### 2. NPH 案例識別

批次處理會自動識別 NPH 案例並進行分組統計:
- 從 `nph-list.txt` 讀取 NPH 案例列表
- 在報表中標記 NPH 案例（⚠️ NPH）
- 生成 NPH vs 非 NPH 的比較統計

### 3. 錯誤處理

- 完整的錯誤捕獲和記錄
- 失敗案例不會中斷批次處理
- 詳細的錯誤訊息和類型記錄
- 在報表中列出所有失敗案例

### 4. 進度追蹤

批次處理時提供即時進度資訊:
- 當前處理案例編號
- 成功/失敗統計
- 預估剩餘時間
- 平均處理速度

## 技術細節

### 座標系統

- **影像載入**: 自動轉換到 RAS+ 方向
- **質心計算**: 在物理空間（世界座標）中進行
- **距離測量**: 使用實際的毫米單位

### 視覺化技術

- **表面提取**: Marching Cubes 演算法
- **3D 渲染**: Plotly 互動式圖表
- **靜態輸出**: Kaleido 引擎

### 計算方法

- **質心**: 加權質心演算法（體素強度作為權重）
- **距離**: 3D 歐式距離
- **顱內寬度**: 每個 Z 切面的最大橫向距離

## 開發者資訊

### 專案架構設計

本專案採用清晰的模組化架構:

- **分層設計**: CLI → Processors → Model
- **職責分離**: 計算邏輯與處理流程分離
- **可擴展性**: 易於新增新的指標類型
- **向後相容**: 保留舊介面,不影響現有使用者

### 新增指標

詳細的開發指南請參考 `CLAUDE.md` 文件,其中包含:
- 核心設計原則
- 開發新指標的標準流程
- 常見錯誤與解決方案
- 完整的程式碼範例

## 授權

[請根據實際情況填寫授權資訊]

## 聯絡資訊

[請根據實際情況填寫聯絡資訊]
