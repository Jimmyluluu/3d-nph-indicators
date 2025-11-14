# 3D NPH Indicators

正常壓力型水腦症（Normal Pressure Hydrocephalus, NPH）3D 影像指標計算工具

## 主要功能

### 支援的 NPH 指標

本工具支援兩種 NPH 診斷指標，可透過批次處理程式選擇：

#### 1. 質心距離比值 (Centroid Ratio)

**測量內容**：
- 左右腦室質心之間的 3D 歐式距離
- 顱內橫向最大寬度
- 腦室質心距離 / 顱內寬度比值

**特點**：
- 在物理空間（世界座標系統）中計算質心位置
- 採用加權質心演算法（使用體素強度值作為權重）
- 每個 Z 切面測量顱內寬度，取最大值

#### 2. 3D Evan Index

**測量內容**：
- 左右腦室前腳（anterior horn）之間的最大距離
- 顱內橫向最大寬度
- 前腳最大距離 / 顱內寬度比值

**前腳定義**：
- Z 軸範圍：預設取 40%-60% 的上方區域（可調整）
- Y 軸過濾：預設取前方 40% 的點（可調整）
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
├── main.py                    # 單案例處理程式
├── batch_process.py           # 批次處理程式（支援兩種指標）
├── model/
│   ├── calculation.py         # 計算模組（質心、Evan Index、顱內寬度）
│   ├── visualization.py       # 視覺化模組（3D 圖表、Marching Cubes）
│   ├── data_export.py         # 批次處理日誌記錄
│   └── reorient.py            # 影像處理工具（座標轉換等）
├── result/                    # 輸出結果目錄
│   ├── centroid_ratio/        # 質心距離比值結果
│   │   ├── {case_id}/         # 各案例獨立資料夾
│   │   │   ├── {case_id}.png  # PNG 圖片
│   │   │   └── {case_id}.html # 互動式 HTML
│   │   ├── results_summary.md # 批次處理報表
│   │   └── processing.log     # 處理日誌
│   └── evan_index/            # 3D Evan Index 結果
│       ├── {case_id}/         # 各案例獨立資料夾
│       │   ├── {case_id}.png  # PNG 圖片
│       │   └── {case_id}.html # 互動式 HTML
│       ├── results_summary.md # 批次處理報表
│       └── processing.log     # 處理日誌
├── requirements.txt           # Python 依賴套件
└── README.md                  # 專案說明文件
```

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

### 批次處理（推薦）

批次處理可以一次處理多個案例，並自動產生統計報表。

#### 基本使用

**處理質心距離比值**（預設）：
```bash
python3 batch_process.py --type centroid_ratio
```

**處理 3D Evan Index**：
```bash
python3 batch_process.py --type evan_index
```

#### 完整參數說明

```bash
python3 batch_process.py \
  --type centroid_ratio \              # 指標類型：centroid_ratio 或 evan_index
  --data-dir /path/to/data \           # 資料目錄路徑
  --skip-not-ok                        # 跳過標記為 _not_ok 的資料夾
```

#### Evan Index 進階參數

```bash
python batch_process.py \
  --type evan_index \
  --z-range 0.4 0.6 \                  # Z 軸範圍（預設：0.4 0.6）
  --y-percentile 40                    # Y 軸前方百分位（預設：40）
```

**參數調整建議**：
- `--z-range`：調整前腳的垂直範圍
  - 預設 `0.4 0.6` 表示取 40%-60% 的上方切面
  - 如果前腳位置較高，可調整為 `0.5 0.7`
  - 如果前腳位置較低，可調整為 `0.3 0.5`
- `--y-percentile`：調整前腳的前後範圍
  - 預設 `40` 表示取前方 40% 的點
  - 增加數值會包含更多後方的點
  - 減少數值會更集中在前方區域

#### 查看說明

```bash
python batch_process.py --help
```

### 批次處理輸出

批次處理完成後，會在 `result/{指標類型}/` 目錄中產生：

1. **{case_id}/** - 各案例獨立資料夾
   - `{case_id}.png` - 高解析度 3D 視覺化（1200 x 900 像素）
   - `{case_id}.html` - 互動式圖表（可在瀏覽器中開啟，支援 360° 旋轉、縮放、平移）

2. **results_summary.md** - 批次處理報表
   - 處理摘要（總數、成功率、耗時）
   - 所有案例測量結果表格
   - 統計數據（最小值、最大值、平均值、中位數）
   - NPH 案例與非 NPH 案例分組統計
   - 失敗案例列表

3. **processing.log** - 詳細處理日誌
   - 每個案例的處理狀態
   - 測量數值
   - 錯誤訊息（如有）
   - 處理時間

### 單案例處理

如果只需要處理單一案例：

1. 修改 `main.py` 中的資料目錄設定：

```python
data_dir = "000016209E"  # 你的案例資料夾名稱
```

2. 執行主程式：

```bash
python main.py
```

3. 輸出結果會在 `result/` 目錄中

## 輸出範例

```plaintext
======================================================================
3D NPH 指標計算 - 腦室質心距離分析
======================================================================

步驟 1: 載入左右腦室影像
----------------------------------------------------------------------

載入左腦室: 000016209E/Ventricle_L.nii.gz
  影像形狀: (512, 512, 29)
  方向: ('R', 'A', 'S')
  體素間距: (0.4297, 0.4297, 5.0)

載入右腦室: 000016209E/Ventricle_R.nii.gz
  影像形狀: (512, 512, 29)
  方向: ('R', 'A', 'S')
  體素間距: (0.4297, 0.4297, 5.0)

✓ 影像形狀相同
✓ 座標系統驗證通過！將在物理空間中計算。

步驟 2: 計算左右腦室質心距離
----------------------------------------------------------------------

步驟 3: 計算顱內橫向最大寬度
----------------------------------------------------------------------

計算顱內橫向最大寬度...

顱內橫向最大寬度: 142.35 mm
位置: Z 切面 #15
左端點座標 (mm): (-71.18, 25.43, 75.00)
右端點座標 (mm): (71.17, 28.76, 75.00)

步驟 4: 計算腦室距離與顱內寬度的比值
----------------------------------------------------------------------
腦室距離/顱內寬度比值: 0.2543 (25.43%)

======================================================================
腦室質心距離測量結果
======================================================================

左腦室質心座標 (mm): (25.67, 18.32, 85.23)
右腦室質心座標 (mm): (-10.45, 20.18, 82.67)

體素間距 (mm): 0.4297 x 0.4297 x 5.00

左右腦室質心距離: 36.21 mm
顱內橫向最大寬度: 142.35 mm

腦室距離/顱內寬度比值: 0.2543 (25.43%)
======================================================================
```

## 技術細節

### 座標系統處理

- 所有計算均在物理空間（世界座標系統）中進行
- 使用 NIfTI affine 矩陣進行體素空間到物理空間的轉換
- 支援不同方向（RAS, LAS 等）的影像
- 確保左右腦室影像座標系統一致性

### 質心計算方法

- 採用加權質心演算法
- 使用體素強度值作為權重
- 提高計算精確度

### 視覺化技術

- Marching Cubes 演算法提取腦部和腦室平滑表面
- Smooth shading 產生專業的 3D 視覺效果
- 使用 Plotly 產生互動式 WebGL 圖表
- Kaleido 引擎將圖表轉換為高品質 PNG

### 資料處理流程

**計算階段（使用原始資料）**：
- 所有測量計算基於**原始體素資料**（二值化 mask: 0 和 1）
- 質心計算、距離測量、前腳識別都使用原始體素座標
- 保證計算結果的精確性和可重現性

**視覺化階段（平滑表面）**：
- Marching Cubes 算法從原始資料提取平滑表面（閾值 = 0.5）
- 僅用於 3D 圖表顯示，不影響任何計算結果
- 提供專業美觀的視覺呈現

這種設計確保了**測量準確性**與**視覺美觀性**的平衡。

## 使用範例

### 範例 1：批次處理質心距離比值

```bash
# 使用預設設定處理所有案例
python batch_process.py --type centroid_ratio

# 指定資料目錄
python batch_process.py --type centroid_ratio --data-dir /Volumes/MyData/NPH_Cases
```

**輸出**：
- `result/centroid_ratio/{case_id}/` - 各案例資料夾
- `result/centroid_ratio/results_summary.md` - 統計報表

### 範例 2：批次處理 3D Evan Index

```bash
# 使用預設參數
python3 batch_process.py --type evan_index

# 調整前腳範圍參數
python3 batch_process.py --type evan_index --z-range 0.3 0.7 --y-percentile 50
```

**輸出**：
- `result/evan_index/{case_id}/` - 各案例資料夾
- `result/evan_index/results_summary.md` - 統計報表

### 範例 3：處理單一案例

```bash
# 編輯 main.py，設定 data_dir = "000016209E"
python main.py
```

## 常見問題

### Q1: 如何調整前腳識別的範圍？

使用 `--z-range` 和 `--y-percentile` 參數：
```bash
python batch_process.py --type evan_index --z-range 0.3 0.7 --y-percentile 50
```

### Q2: 批次處理如何跳過某些案例？

將需要跳過的案例資料夾重新命名，加上 `_not_ok` 後綴：
```bash
mv 000123456A 000123456A_not_ok
```

### Q3: 視覺化圖表太大，如何調整？

編輯 `model/visualization.py`，修改圖表尺寸：
```python
width=1200,  # 改為你想要的寬度
height=900,  # 改為你想要的高度
```

### Q4: 如何查看處理失敗的原因？

檢查 `result/{指標類型}/processing.log` 檔案，其中包含詳細的錯誤訊息。

## 授權與引用

本工具為研究用途開發，如使用本工具發表研究成果，請適當引用。
