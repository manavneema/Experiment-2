"""
Conservative HUMAN_PRIOR_WHITELIST for the causal baseline.

Structure:
HUMAN_PRIOR_WHITELIST = [
    ("from_metric", "to_metric"),
    ...
]

Only include high-confidence upstream -> downstream mappings. Edit this file
if you want to add or remove entries before running `run_baseline_pipeline`.
"""

HUMAN_PRIOR_WHITELIST = [
    # Core count/row propagation
    ("raw_input_record_count", "bronze_input_rows"),
    ("bronze_output_rows", "silver_output_rows"),

    # Aggregates (distance / speed / ingestion time)
    ("raw_distance_mean", "bronze_distance_km_mean"),
    ("raw_avg_speed_mean", "silver_avg_speed_imputed"),
    ("raw_ingestion_duration_sec", "bronze_ingestion_duration_sec"),

    # Survival / pipeline-level quality
    ("bronze_survival_rate", "silver_survival_rate"),

    # Unit / dedup / high-level mappings
    ("raw_unique_units", "bronze_output_rows"),

    # Temporal / run-coverage -> downstream runtime/rows
    ("raw_temporal_coverage_hours", "bronze_ingestion_duration_sec"),
    ("raw_min_trip_start_ts", "bronze_input_rows"),
    ("raw_max_trip_end_ts", "bronze_output_rows"),
]

# Tip: programmatically extend or modify at runtime after pivoting columns:
# from HUMAN_PRIOR_WHITELIST import HUMAN_PRIOR_WHITELIST
# HUMAN_PRIOR_WHITELIST.extend([(...), ...])
