# EDA: scada_pipeline

**Rows:** 1,000  |  **Cols:** 18  |  **Verdict:** `usable_small`

## Columns
`timestamp`, `segment_id`, `pressure`, `flow_rate`, `temperature`, `valve_status`, `pump_state`, `pump_speed`, `compressor_state`, `energy_consumption`, `alarm_triggered`, `event_type`, `target`, `rpm`, `power`, `status_label`, `leak_label`, `dataset_name`

## Null Summary
No nulls detected.

## Numeric Stats
- **pressure**: mean=76.2025, std=9.4817, min=43.11, max=115.3, nulls=0
- **flow_rate**: mean=4.359, std=0.8947, min=1.01, max=7.53, nulls=0
- **temperature**: mean=32.0977, std=1.9767, min=25.92, max=38.52, nulls=0
- **rpm**: mean=937.1917, std=640.7524, min=0.0, max=1678.8, nulls=0
- **power**: mean=26.8535, std=9.2702, min=5.42, max=58.18, nulls=0

## Labels
Classes: 2  |  Majority: 69.4%
- `0`: 694
- `1`: 306

**Labels usable:** True

## Timestamp
Range: 2024-01-01 00:00:00 → 2024-01-01 00:16:00
Unique timestamps: 17
Usable: True