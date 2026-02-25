# 3D NPH Indicators 開發指南

## 基本規則

- 不用測試
- 回覆使用繁體中文

---

## 🏗️ 專案架構

```
3d-nph-indicators/
├── main.py                      # 統一 CLI 入口（5 種指標類型）
│
├── processors/                  # 處理流程邏輯
│   ├── logger.py               # 日誌記錄
│   ├── printers.py             # 計算結果輸出（print 相關）
│   ├── case_processor.py       # 單案例處理（5 種指標）
│   └── batch_processor.py      # 批次處理
│
└── model/                       # 純計算和視覺化模組
    ├── calculation.py          # 基礎計算（含統一載入函數、Falx 工具函數）
    ├── cal_volume_surface.py   # 體積與表面積計算
    ├── alvi_analyzer.py        # ALVI 計算模組
    ├── evan_analyzer.py        # 3D Evan Index 計算模組
    ├── result_analyzer.py      # 批次結果分析（統計、ROC 曲線）
    ├── visualization.py        # 3D 視覺化
    ├── image_processing.py     # 影像處理工具
    └── report_generator.py    # 報表產生
```

**職責劃分:**
- `model/` - 純計算、視覺化、報表邏輯 (不含處理流程)
- `processors/` - 處理流程、日誌、print 輸出、協調邏輯
- `main.py` - CLI 入口

---

## 🎯 核心原則

### 1. 影像載入統一規則

**所有影像必須透過統一函數載入,自動拉正到 RAS+ 方向**

```python
from model.calculation import load_ventricle_pair, load_original_image

# ✅ 正確:使用統一載入函數
left_vent, right_vent = load_ventricle_pair(left_path, right_path)
original_img = load_original_image(original_path)

# ❌ 錯誤:直接使用 nibabel.load() 會沒拉正
import nibabel as nib
original_img = nib.load(original_path)  # 禁止!
```

### 2. 視覺化函數參數規則

**傳影像物件,不傳路徑**

```python
# ✅ 正確:接受影像物件
def visualize_something(left_vent, right_vent, original_img, ...):
    original_data = get_image_data(original_img)

# ❌ 錯誤:接受路徑會導致重複載入
def visualize_something(left_vent, right_vent, original_path, ...):
    original_img = nib.load(original_path)  # 重複載入!
```

### 3. 報表產生統一規則

**使用 `generate_markdown_report()` 和 `INDICATOR_CONFIGS`**

```python
from model.report_generator import generate_markdown_report, INDICATOR_CONFIGS

# 1. 在 model/report_generator.py 的 INDICATOR_CONFIGS 新增配置
INDICATOR_CONFIGS['new_metric'] = {
    'title': '新指標批次處理報表',
    'distance_field': 'metric_distance_mm',
    'distance_label': '指標距離 (mm)',
    'ratio_field': 'metric_ratio',
    'ratio_label': '新指標比值',
    'ratio_percent_field': 'metric_ratio_percent',
    'footer': 'New Metric Calculator'
}

# 2. 呼叫統一報表函數
generate_markdown_report(results, output_path, total_time,
                        success_count, error_count,
                        indicator_type='new_metric')
```

---

## 📝 開發新指標標準流程

### ⚠️ 表面積計算的特殊處理

對於表面積計算，請注意：
- **無視覺化需求** - 表面積計算為純計算模式，不需要產生 3D 圖表
- **返回值格式** - 只需返回數值結果，不需要網格資料
- **處理函數** - 不需呼叫視覺化函數，直接返回計算結果

參考現有的 `calculate_surface_area()` 和 `process_case_surface_area()` 實作。

### 步驟 1: 在 `model/calculation.py` 新增計算函數

```python
def calculate_new_metric(left_vent, right_vent, original_img):
    """
    計算新指標

    Args:
        left_vent: 左腦室 (已拉正到 RAS+)
        right_vent: 右腦室 (已拉正到 RAS+)
        original_img: 原始影像 (已拉正到 RAS+)

    Returns:
        dict: 計算結果
    """
    # 你的計算邏輯
    return {'metric_value': 0.123, ...}
```

### 步驟 2: 在 `processors/case_processor.py` 新增處理函數

```python
def process_case_new_metric(data_dir, output_image_path,
                            show_plot=False, verbose=True):
    """處理單一案例 - 新指標"""
    try:
        # 1. 找檔案
        data_path = Path(data_dir)
        left_path = data_path / "Ventricle_L.nii.gz"
        right_path = data_path / "Ventricle_R.nii.gz"
        original_path = data_path / "original.nii.gz"

        # 2. 載入影像 (自動拉正到 RAS+)
        from model.calculation import load_ventricle_pair, load_original_image

        left_vent, right_vent = load_ventricle_pair(
            str(left_path), str(right_path), verbose=verbose
        )
        original_img = load_original_image(
            str(original_path), verbose=verbose
        )

        # 3. 計算指標
        metric_data = calculate_new_metric(left_vent, right_vent, original_img)

        # 4. 視覺化 (傳物件不傳路徑!)
        visualize_new_metric(
            left_vent, right_vent, original_img,  # ✅ 傳物件
            metric_data,
            output_path=str(output_image_path),
            show_plot=show_plot
        )

        # 5. 返回結果
        return {
            'status': 'success',
            'metric_distance_mm': metric_data['value'],
            # 欄位名稱需對應 INDICATOR_CONFIGS
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'error_type': type(e).__name__
        }
```

### 步驟 3: 在 `processors/batch_processor.py` 更新

在 `batch_process()` 函數中新增支援:

```python
# 選擇處理函數
if indicator_type == "centroid_ratio":
    process_func = process_case_indicator_ratio
elif indicator_type == "evan_index":
    process_func = process_case_evan_index
elif indicator_type == "surface_area":
    process_func = process_case_surface_area
elif indicator_type == "volume_surface_ratio":
    process_func = process_case_volume_surface_ratio
elif indicator_type == "alvi":
    process_func = process_case_alvi
elif indicator_type == "new_metric":  # ✅ 新增
    process_func = process_case_new_metric
```

### 步驟 4: 更新 CLI 入口

在 `main.py` 的 argparse choices 中新增:

```python
parser.add_argument(
    '--type', '-t',
    choices=['centroid_ratio', 'evan_index', 'surface_area', 'volume_surface_ratio', 'alvi', 'new_metric'],  # ✅ 新增
    default='centroid_ratio',
    help='指標類型'
)
```

---

## ⚠️ 常見錯誤

### 錯誤 1: 重複載入影像

```python
# ❌ 錯誤
original_img = load_original_image(path)
visualize_something(original_path, ...)  # visualization 內又載入一次

# ✅ 正確
original_img = load_original_image(path)
visualize_something(original_img, ...)  # 直接傳物件
```

### 錯誤 2: 使用 nib.load() 沒拉正

```python
# ❌ 錯誤
import nibabel as nib
img = nib.load(path)

# ✅ 正確
from model.calculation import load_original_image
img = load_original_image(path)
```

### 錯誤 3: 在 model/ 中放處理邏輯

```python
# ❌ 錯誤:在 model/ 中放日誌、處理流程
model/batch_handler.py  # 應該放在 processors/

# ✅ 正確:model/ 只放純計算和視覺化
model/calculation.py     # 純計算函數
model/visualization.py   # 純視覺化函數
```

---

## 📁 重要函數位置

```
model/calculation.py
  ├── load_ventricle_pair()       ✅ 載入腦室 (會自動拉正)
  ├── load_original_image()       ✅ 載入原始影像 (會自動拉正)
  ├── load_falx_image()           ✅ 載入 Falx mask (會自動拉正)
  ├── fit_falx_plane()            ✅ 擬合 Falx 平面 (SVD)
  ├── filter_points_by_falx_side() 依 Falx 平面過濾點雲
  ├── project_points_to_plane()   投影點雲到平面
  ├── find_max_diameter_convex_hull() Convex Hull 最長徑
  ├── calculate_centroid_3d()     計算 3D 質心
  ├── calculate_centroid_distance() 計算質心距離
  └── calculate_ventricle_to_cranial_ratio() 計算比值

model/cal_volume_surface.py
  ├── calculate_surface_area()    計算腦室表面積
  ├── calculate_volume_smooth()   Marching Cubes 平滑體積
  └── calculate_volume_surface_ratio() 體積/表面積比例

model/alvi_analyzer.py
  ├── calculate_ventricle_ap_diameter() 腦室前後徑 (Falx 投影)
  ├── calculate_skull_ap_diameter()    顱骨前後徑
  └── calculate_alvi()                 ALVI 計算

model/evan_analyzer.py
  ├── calculate_anterior_horn_distance_with_falx() 前腳距離 (Falx 方法)
  ├── calculate_anterior_horn_max_distance()       前腳距離 (質心方法)
  ├── calculate_cranial_width()                    顱內橫向寬度
  └── calculate_3d_evan_index()                    3D Evan Index

model/result_analyzer.py
  ├── INDICATOR_CONFIGS           ✅ 指標配置字典（含 evan_index/alvi/volume_surface_ratio/ventricle_volume）
  ├── BaseResultAnalyzer          通用結果分析器基礎類別
  │   ├── load_data()             載入結果摘要
  │   ├── get_statistics()        統計數據
  │   ├── evaluate_threshold()    評估診斷閾值
  │   ├── generate_roc_curve()    生成 ROC 曲線
  │   └── generate_report()       生成分析報告
  └── create_analyzer()           工廠函數

model/visualization.py
  └── visualize_*()               視覺化函數 (接受影像物件)

model/report_generator.py
  ├── INDICATOR_CONFIGS           ✅ 報表配置字典
  └── generate_markdown_report()  ✅ 統一報表產生

model/image_processing.py
  ├── reorient_image()            ⚠️ 不要直接用！透過 load_* 函數呼叫
  ├── get_image_data()            ✅ 取得影像資料
  ├── get_voxel_size()            ✅ 取得體素大小
  ├── convert_voxel_to_physical() 體素座標轉物理座標
  └── extract_surface_mesh()      ✅ Marching Cubes 表面提取

processors/case_processor.py
  ├── find_case_files()                    尋找案例檔案
  ├── process_case_indicator_ratio()       質心距離比值
  ├── process_case_evan_index()            3D Evan Index
  ├── process_case_surface_area()          腦室表面積
  ├── process_case_volume_surface_ratio()  體積/表面積比例
  └── process_case_alvi()                  ALVI

processors/batch_processor.py
  ├── scan_data_directory()
  └── batch_process()

processors/printers.py
  └── print_*()                   所有 print 輸出函數（保持 model/ 純計算）

processors/logger.py
  └── ProcessLogger               日誌記錄器
```

---

## ✅ 開發新指標檢查清單

- [ ] 在 `model/` 新增計算模組（如 `model/xxx_analyzer.py`）或在 `model/calculation.py` 新增計算函數
- [ ] 在 `model/visualization.py` 新增視覺化函數（接受物件不接受路徑）
- [ ] 在 `processors/case_processor.py` 新增 `process_case_xxx()` 處理函數
- [ ] 使用 `load_ventricle_pair()`、`load_original_image()`、`load_falx_image()` 載入影像
- [ ] print 輸出邏輯放在 `processors/printers.py`，不放在 model/
- [ ] 在 `processors/batch_processor.py` 新增支援
- [ ] 在 `model/report_generator.py` 的 `INDICATOR_CONFIGS` 新增配置
- [ ] 更新 `main.py` 的 CLI 參數 choices

---

## 🔍 模組職責說明

### model/ - 純計算模組
- **只包含**: 計算函數、視覺化函數、報表生成
- **不包含**: 檔案掃描、日誌記錄、處理流程
- **原則**: 可以被其他專案重用的純邏輯

### processors/ - 處理協調模組
- **包含**: 單案例處理、批次處理、日誌記錄
- **職責**: 協調 model/ 中的函數,處理檔案 I/O
- **原則**: 專案特定的處理流程

### main.py - CLI 入口
- **職責**: 解析命令列參數,呼叫 processors/
- **原則**: 薄層,只做介面不做邏輯

---

**記住三個核心原則:**
1. **統一載入** - 使用 `load_ventricle_pair()` 和 `load_original_image()`
2. **傳物件不傳路徑** - 視覺化函數接受已載入的影像物件
3. **職責分離** - model/ 純邏輯, processors/ 處理流程, CLI 只做介面
