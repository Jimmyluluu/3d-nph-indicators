# 3D NPH Indicators 開發指南

## 基本規則

- 不用測試
- 回覆使用繁體中文

---

## 🏗️ 專案架構

```
3d-nph-indicators/
├── main.py                      # 統一 CLI 入口
│
├── processors/                  # 處理流程邏輯
│   ├── logger.py               # 日誌記錄
│   ├── case_processor.py       # 單案例處理
│   └── batch_processor.py      # 批次處理
│
└── model/                       # 純計算和視覺化模組
    ├── calculation.py          # 計算邏輯(含統一載入函數)
    ├── visualization.py        # 3D 視覺化
    ├── reorient.py            # 影像拉正工具
    └── report_generator.py    # 報表產生
```

**職責劃分:**
- `model/` - 純計算、視覺化、報表邏輯 (不含處理流程)
- `processors/` - 處理流程、日誌、協調邏輯
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
    process_func = ...
elif indicator_type == "new_metric":  # ✅ 新增
    process_func = process_case_new_metric
```

### 步驟 4: 更新 CLI 入口

在 `main.py` 的 argparse choices 中新增:

```python
parser.add_argument(
    '--type', '-t',
    choices=['centroid_ratio', 'evan_index', 'new_metric'],  # ✅ 新增
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
  ├── load_ventricle_pair()    ✅ 載入腦室 (會自動拉正)
  ├── load_original_image()    ✅ 載入原始影像 (會自動拉正)
  └── calculate_*()            計算函數

model/visualization.py
  └── visualize_*()            視覺化函數 (接受影像物件)

model/report_generator.py
  ├── INDICATOR_CONFIGS        ✅ 指標配置字典
  └── generate_markdown_report()  ✅ 統一報表產生

processors/case_processor.py
  ├── process_case_indicator_ratio()
  └── process_case_evan_index()

processors/batch_processor.py
  ├── scan_data_directory()
  └── batch_process()

processors/logger.py
  └── ProcessLogger            日誌記錄器

model/reorient.py
  ├── reorient_image()         ⚠️ 不要直接用!透過 load_* 函數呼叫
  ├── get_image_data()         ✅ 取得影像資料
  └── get_voxel_size()         ✅ 取得體素大小
```

---

## ✅ 開發新指標檢查清單

- [ ] 在 `model/calculation.py` 新增計算函數
- [ ] 在 `model/visualization.py` 新增視覺化函數(接受物件不接受路徑)
- [ ] 在 `processors/case_processor.py` 新增處理函數
- [ ] 使用 `load_ventricle_pair()` 和 `load_original_image()` 載入影像
- [ ] 在 `processors/batch_processor.py` 新增支援
- [ ] 在 `model/report_generator.py` 的 `INDICATOR_CONFIGS` 新增配置
- [ ] 更新 `main.py` 的 CLI 參數

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
