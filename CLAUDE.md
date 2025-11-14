# 3D NPH Indicators 開發指南

## 基本規則

- 不用測試
- 回覆使用繁體中文

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

## 📝 開發新指標標準模板

```python
def process_case_new_metric(data_dir, output_image_path, show_plot=False, verbose=True):
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

### 錯誤 3: 座標系統不一致

```python
# ❌ 錯誤
left_vent = load_ventricle_pair(...)  # 已拉正
original_img = nib.load(path)  # 沒拉正,座標不一致!

# ✅ 正確
left_vent = load_ventricle_pair(...)  # 已拉正
original_img = load_original_image(path)  # 已拉正,座標一致
```

---

## 📁 重要函數位置

```text
model/calculation.py
  ├── load_ventricle_pair()    ✅ 載入腦室 (會自動拉正)
  ├── load_original_image()    ✅ 載入原始影像 (會自動拉正)
  └── calculate_*()            計算函數

model/visualization.py
  └── visualize_*()            視覺化函數 (接受影像物件不接受路徑)

model/report_generator.py
  ├── INDICATOR_CONFIGS        ✅ 指標配置字典
  └── generate_markdown_report()  ✅ 統一報表產生

model/reorient.py
  ├── reorient_image()         ⚠️ 不要直接用!透過 load_* 函數呼叫
  ├── get_image_data()         ✅ 取得影像資料
  └── get_voxel_size()         ✅ 取得體素大小
```

---

## ✅ 開發新指標檢查清單

- [ ] 使用 `load_ventricle_pair()` 載入腦室
- [ ] 使用 `load_original_image()` 載入原始影像
- [ ] 不直接使用 `nibabel.load()`
- [ ] 視覺化函數接受影像物件,不接受路徑
- [ ] 在 `INDICATOR_CONFIGS` 新增配置
- [ ] 使用 `generate_markdown_report()` 產生報表

---

**記住: 所有影像載入都走統一函數,視覺化傳物件不傳路徑,報表用統一配置!**
