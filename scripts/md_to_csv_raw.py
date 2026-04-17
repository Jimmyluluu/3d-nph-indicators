"""
解析 ALVI、Evan Index、Volume/Surface Ratio 的 results_summary.md
合併成 CSV 供 MLP / KAN 模型訓練使用

此版本不擷取 alvi 和 evan_index 比值，只保留原始測量值
輸出: result/nph_indicators_raw.csv
"""

import re
import csv
from pathlib import Path


def parse_markdown_table(filepath: str) -> list[dict]:
    """解析 markdown 報表中的測量結果表格"""
    text = Path(filepath).read_text(encoding='utf-8')

    section_match = re.search(r'## 測量結果\s*\n(.*?)(?=\n### |\n## )', text, re.DOTALL)
    if not section_match:
        raise ValueError(f"找不到測量結果區塊: {filepath}")

    section = section_match.group(1)
    lines = [l.strip() for l in section.strip().split('\n') if l.strip().startswith('|')]
    data_lines = lines[2:]

    results = []
    for line in data_lines:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        case_id_raw = cells[0].strip()

        is_nph = '⚠️ NPH' in case_id_raw
        case_id = case_id_raw.replace('⚠️ NPH', '').strip()

        results.append({
            'case_id': case_id,
            'is_nph': is_nph,
            'cells': cells,
        })

    return results


def parse_angle_deg(raw_value: str) -> float:
    """將角度字串（例如 '29.3°'）轉為 float。"""
    cleaned = raw_value.replace('°', '').strip()
    return float(cleaned)


def main():
    base = Path(__file__).parent.parent / 'result'

    # === 解析 ALVI (只取原始測量值) ===
    alvi_data = {}
    for row in parse_markdown_table(base / 'alvi' / 'results_summary.md'):
        alvi_data[row['case_id']] = {
            'is_nph': row['is_nph'],
            'ventricle_ap_diameter_mm': float(row['cells'][1]),
            'skull_ap_diameter_mm': float(row['cells'][2]),
        }

    # === 解析 Evan Index (只取原始測量值) ===
    evan_data = {}
    for row in parse_markdown_table(base / 'evan_index' / 'results_summary.md'):
        evan_data[row['case_id']] = {
            'frontal_horn_distance_mm': float(row['cells'][1]),
            'cranial_width_mm': float(row['cells'][2]),
        }

    # === 解析 Volume/Surface Ratio ===
    vsr_data = {}
    for row in parse_markdown_table(base / 'volume_surface_ratio' / 'results_summary.md'):
        total_vol = float(row['cells'][3])
        vsa_ratio = float(row['cells'][4])
        vsr_data[row['case_id']] = {
            'left_ventricle_volume_mm3': float(row['cells'][1]),
            'right_ventricle_volume_mm3': float(row['cells'][2]),
            'total_surface_area_mm2': round(total_vol / vsa_ratio, 1) if vsa_ratio > 0 else 0,
        }

    # === 解析 Callosal Angle (非必要欄位) ===
    callosal_data = {}
    callosal_path = base / 'callosal_angle' / 'results_summary.md'
    if callosal_path.exists():
        for row in parse_markdown_table(callosal_path):
            try:
                angle = parse_angle_deg(row['cells'][1])
                if angle == 0.0:
                    # 視為缺值，後續回填 NaN
                    continue
                callosal_data[row['case_id']] = {
                    'callosal_angle_deg': angle,
                }
            except (ValueError, IndexError):
                # 角度解析失敗視為缺值，後續回填 NaN
                continue
    else:
        print(f"⚠️ 找不到 Callosal Angle 報表: {callosal_path}")
        print("   將以 NaN 填入 callosal_angle_deg")

    # === 合併 ===
    all_case_ids = sorted(alvi_data.keys())

    output_path = base.parent / 'result' / 'nph_indicators_raw.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'case_id',
        'label',  # 1=NPH, 0=nonNPH
        'ventricle_ap_diameter_mm',
        'skull_ap_diameter_mm',
        'frontal_horn_distance_mm',
        'cranial_width_mm',
        'left_ventricle_volume_mm3',
        'right_ventricle_volume_mm3',
        'total_surface_area_mm2',
        'callosal_angle_deg',
    ]

    rows_written = 0
    skipped = []

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cid in all_case_ids:
            alvi = alvi_data[cid]

            if cid not in evan_data or cid not in vsr_data:
                skipped.append(cid)
                continue

            evan = evan_data[cid]
            vsr = vsr_data[cid]
            callosal = callosal_data.get(cid)

            if callosal is None:
                callosal = {
                    'callosal_angle_deg': float('nan'),
                }

            row = {
                'case_id': cid,
                'label': 1 if alvi['is_nph'] else 0,
                **{k: v for k, v in alvi.items() if k != 'is_nph'},
                **evan,
                **vsr,
                **callosal,
            }
            writer.writerow(row)
            rows_written += 1

    print(f"✅ CSV 已產生: {output_path}")
    print(f"   總筆數: {rows_written}")
    if skipped:
        print(f"   跳過 {len(skipped)} 筆 (缺少其他指標): {skipped}")

    nph_count = sum(1 for cid in all_case_ids
                    if cid not in skipped and alvi_data[cid]['is_nph'])
    non_nph_count = rows_written - nph_count
    print(f"   NPH: {nph_count}, 非 NPH: {non_nph_count}")


if __name__ == '__main__':
    main()
