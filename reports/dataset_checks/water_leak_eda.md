# EDA: water_leak

**Rows:** 1,000  |  **Cols:** 13  |  **Verdict:** `usable_small`

## Columns
`Timestamp`, `Sensor_ID`, `Pressure (bar)`, `Flow Rate (L/s)`, `Temperature (°C)`, `Leak Status`, `Burst Status`, `timestamp`, `pressure`, `flow_rate`, `sensor_id`, `leak_label`, `dataset_name`

## Null Summary
No nulls detected.

## Numeric Stats
- **pressure**: mean=3.2207, std=0.489, min=0.911, max=3.9954, nulls=0
- **flow_rate**: mean=125.0381, std=44.1214, min=50.6545, max=331.7541, nulls=0

## Labels
Classes: 2  |  Majority: 98.1%
- `0`: 981
- `1`: 19

**Labels usable:** True

## Timestamp
Range: 2024-01-01 00:00:00 → 2024-01-04 11:15:00
Unique timestamps: 1000
Usable: True