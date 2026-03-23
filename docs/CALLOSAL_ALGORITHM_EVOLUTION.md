# Callosal 演算法調整歷程與前後對比

## 1) 這次一路調整的歷程（重點版）

### A. 先處理文件與實作不一致
- 一開始先比對 `docs/ALGORITHMS.md` 與實作，確認流程敘述有落差（例如量角線定義、角度計算描述）。
- 文件後續已改成接近實作版本，但在多次迭代後，建議再做一次最終同步。

### B. 幾何流程重構成共用模組
- 新增 `model/callosal_geometry.py`，把共同幾何流程集中：
  - PC 近似錨點（第三腦室後方）
  - Falx + AP 建冠狀面
  - 左右/第三腦室截面切取與投影
  - 左右內側壁擬合
  - `vertex` 求解
- `callosal_angle` 與 `callosal_area` 都改為呼叫 `compute_callosal_geometry(...)`。

### C. 內側/外側壁與方向策略多輪嘗試
- 中途曾切到「外側壁」版本，後來已改回「內側壁」。
- 曾嘗試過「朝上/朝中線」與「射線交會」等抑制穿底策略，後續依實際表現回退部分策略。

### D. 視覺化與角度公式對齊
- 中途有發現「畫的角線」與「角度公式」來源不同。
- 現在 `callosal_angle` 主算法改為：
  - 優先用 `vertex->left_anchor` 與 `vertex->right_anchor` 夾角（與圖一致）
  - 錨點不可用時才 fallback 到 `left_dir/right_dir`。

### E. 面積第三頂點改為共用 `vertex`
- `callosal_area` 原先第三點是 `third_centroid`。
- 現在改為 `vertex`，讓 angle 與 area 的幾何第三點一致。

### F. 舊版面積流程移除
- 已刪除 `_legacy_calculate_callosal_area`，只保留新版 `calculate_callosal_area(...)`。

### G. `+0.5` 高度限制最後移除
- 曾加入「`vertex` 不低於三腦室質心 + 0.5mm」限制。
- 你最後要求移除，現況已拿掉。

## 2) 最初（對話一開始純版）vs 最後（對照表）

| 項目 | 最初 | 最後（目前） |
| --- | --- | --- |
| 幾何流程來源 | `angle` / `area` 各自實作，重複邏輯多 | 共用 `compute_callosal_geometry(...)` |
| 平面定義 | 冠狀面通過 PC 近似點（已有） | 同上，且 `vertex` 最後會再投影回該平面 |
| 側腦室壁擬合 | 內側壁 SVD | 內側壁 SVD |
| `vertex` 求法 | 兩條擬合線 3D 最近交點 | 同樣為線-線最近交點，並投影回平面 |
| `vertex` 高度限制 | 無 | 目前無 `+0.5` 強制抬高 |
| 角度數值 | 主要用 `dot(left_dir, right_dir)` | 優先用 `vertex->左右錨點` 夾角；無錨點時 fallback `left_dir/right_dir` |
| 角度視覺化 | 折線 `left_anchor -> center_point -> right_anchor`（示意畫法） | 折線 `left_anchor -> vertex -> right_anchor`，與主算法一致 |
| 面積第三頂點 | `third_centroid` | `vertex`（與 angle 共用） |
| 面積視覺化第三點 | 質心 | `vertex`（並保留質心作參考點） |
| 舊版流程 | `_legacy_calculate_callosal_area` 存在 | 已移除 |

## 3) 目前實際算法（簡版）

1. 載入並清理第三腦室點雲，取 PC 近似點。  
2. 用 Falx + AP 向量建立冠狀面（通過 PC）。  
3. 切取左右/第三腦室截面（`±2mm`）並投影。  
4. 左右內側壁擬合線，求 `vertex`。  
5. `vertex` 再投影回冠狀面（保證在同一平面）。  
6. Angle：優先 `vertex->左右錨點` 夾角。  
7. Area：三角形第三點用 `vertex`，再做 inclusion/exclusion 面積。  

## 4) 目前你最在意的關鍵狀態

- `angle` 和 `area` 已共用同一個 `vertex`。  
- `+0.5mm` 的人工抬高限制已移除。  
- 幾何點位仍會被投影回你定義的冠狀平面。  
