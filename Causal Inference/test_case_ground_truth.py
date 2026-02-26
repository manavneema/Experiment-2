# Ground Truth Configuration for All 15 RCA Test Cases
# Use this in your RCA evaluation notebook

GROUND_TRUTH = {
    # =================================================================
    # CATEGORY 1: Raw Data Quality - NULL Injections (Cases 1, 5, 6)
    # =================================================================
    
    "case1_unit_id_nulls": {
        "category": "Raw Data Quality - NULL Injections",
        "description": "40% of unit_id values set to NULL (primary key failure)",
        "fault_date": "2026-02-06",
        "fault_type": "foreign_key_nulls",
        "fault_severity": "high",
        "true_root_causes": {
            "raw_null_count_unit_id"  # Direct root cause
        },
        "expected_downstream": {
            "bronze_null_primary_key_rows",  # Validation detects nulls
            "bronze_rows_dropped_by_rules"   # Rows dropped during validation
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "raw_null_count_unit_id → bronze_null_primary_key_rows → bronze_rows_dropped_by_rules"
    },
    
    "case5_sensor_reads_nulls": {
        "category": "Raw Data Quality - NULL Injections",
        "description": "35% of fuel_consumption readings set to NULL (sensor calibration failure)",
        "fault_date": "2026-02-12",
        "fault_type": "sensor_nulls",
        "fault_severity": "medium",
        "true_root_causes": {
            "raw_mean_fuel_consumption_ecol"  # sensor-level fuel mean becomes NaN (mapped)
        },
        "expected_downstream": {
            "raw_mean_fuel_consumption_ecol",  # Statistical property affected (raw-layer mean)
            "bronze_rows_dropped_by_rules"  # Validation drops incomplete rows
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "raw_mean_fuel_consumption_ecol → bronze_mean_fuel_consumption_ecol → bronze_rows_dropped_by_rules"
    },
    
    "case6_vehicle_id_nulls": {
        "category": "Raw Data Quality - NULL Injections",
        "description": "25% of vehicle_id values set to NULL (master data sync failure)",
        "fault_date": "2026-02-13",
        "fault_type": "foreign_key_nulls",
        "fault_severity": "medium",
        "true_root_causes": {
            "raw_null_count_unit_id"  # Direct root cause (graph-native)
        },
        "expected_downstream": {
            "bronze_null_primary_key_rows",  # Validation detects nulls
            "silver_vehicle_info_join_miss_rate"  # Join fails on null vehicle_id
        },
        "affected_tiers": ["raw", "bronze", "silver"],
        "causal_path": "raw_null_count_vehicle_id → bronze_null_primary_key_rows → silver_vehicle_info_join_miss_rate"
    },
    
    # =================================================================
    # CATEGORY 2: Raw Data Quality - Validation Issues (Cases 2, 7, 8)
    # =================================================================
    
    "case2_distance_gps_nulls": {
        "category": "Raw Data Quality - Validation Issues",
        "description": "35% of rows have null distance, start_latitude, start_longitude (GPS coverage loss)",
        "fault_date": "2026-02-09",
        "fault_type": "gps_sensor_failure",
        "fault_severity": "high",
        "true_root_causes": {
            "raw_null_count_gps_coverage",
            "raw_null_count_start",
            "raw_distance_std"
        },
        "expected_downstream": {
            "bronze_rows_dropped_by_rules",  # Distance-dependent validation drops rows
            "bronze_invalid_avg_speed_rows"  # Can't compute speed without distance
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "raw_null_count_gps_coverage → bronze_invalid_avg_speed_rows → bronze_rows_dropped_by_rules"
    },
    
    "case7_extreme_values": {
        "category": "Raw Data Quality - Validation Issues",
        "description": "20% of rows have extreme values (10x normal) in speed and fuel (sensor noise)",
        "fault_date": "2026-02-14",
        "fault_type": "extreme_values_outliers",
        "fault_severity": "medium",
        "true_root_causes": {
            "raw_avg_speed_mean",  # Mean increases 10x
            "raw_std_fuel_consumption_ecol"  # Fuel variance (mapped)
        },
        "expected_downstream": {
            "bronze_invalid_avg_speed_rows",  # Validation detects outliers
            "bronze_rows_dropped_by_rules"  # Extreme values fail validation
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "raw_avg_speed_mean → raw_fuel_consumption_std → bronze_invalid_avg_speed_rows"
    },
    
    "case8_invalid_ranges": {
        "category": "Raw Data Quality - Validation Issues",
        "description": "30% of rows have invalid ranges (start AFTER end, negative durations)",
        "fault_date": "2026-02-15",
        "fault_type": "temporal_constraint_violation",
        "fault_severity": "high",
        "true_root_causes": {
            "bronze_duration_mean",  # Duration aggregated in bronze
            "bronze_rows_dropped_by_rules"  # validation count
        },
        "expected_downstream": {
            "bronze_rows_dropped_by_rules",  # Validation fails on negative duration
            "bronze_survival_rate"  # Percentage of rows surviving validation decreases
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "bronze_duration_mean → bronze_rows_dropped_by_rules"
    },
    
    # =================================================================
    # CATEGORY 3: ML Layer - Model Quality (Cases 3, 9, 10)
    # =================================================================
    
    "case3_fuel_sensor_drift": {
        "category": "ML Layer - Model Quality",
        "description": "15% of fuel readings show 2-3.5x drift (sensor calibration failure)",
        "fault_date": "2026-02-10",
        "fault_type": "sensor_drift",
        "fault_severity": "high",
        "true_root_causes": {
            "silver_ml_large_error_count",  # ML detects large prediction errors
            "p95_fuel_per_100km"  # KPI affected by drift
        },
        "expected_downstream": {
            "silver_ml_residual_mean",  # Residuals become non-zero
            "silver_ml_residual_mean"  # imputation proxy -> residuals in current graph
        },
        "affected_tiers": ["bronze", "silver", "ml"],
        "causal_path": "silver_ml_large_error_count → p95_fuel_per_100km → silver_ml_residual_mean"
    },
    
    "case9_speed_prediction_drift": {
        "category": "ML Layer - Model Quality",
        "description": "20% of trips have speed predictions 1.5-2.5x actual (model drift)",
        "fault_date": "2026-02-16",
        "fault_type": "ml_model_drift",
        "fault_severity": "medium",
        "true_root_causes": {
            "silver_ml_large_error_count",  # ML detects large prediction errors
            "silver_ml_prediction_std"  # Prediction variance increases
        },
        "expected_downstream": {
            "silver_ml_residual_mean",  # Residuals non-zero
            "p95_speed"  # Speed KPI increases
        },
        "affected_tiers": ["bronze", "silver", "ml"],
        "causal_path": "silver_ml_large_error_count → silver_ml_prediction_std → silver_ml_residual_mean"
    },
    
    "case10_prediction_outliers": {
        "category": "ML Layer - Model Quality",
        "description": "15% of rows have extreme predictions (20x normal) from ML model (edge case failure)",
        "fault_date": "2026-02-17",
        "fault_type": "ml_outliers",
        "fault_severity": "medium",
        "true_root_causes": {
            "silver_ml_large_error_count"  # ML error detection triggers
        },
        "expected_downstream": {
            "silver_ml_residual_mean",  # imputation proxy in graph
            "bronze_invalid_avg_speed_rows"  # Validation may fail on extreme values
        },
        "affected_tiers": ["bronze", "silver", "ml"],
        "causal_path": "silver_ml_large_error_count → silver_ml_residual_mean → bronze_invalid_avg_speed_rows"
    },
    
    # =================================================================
    # CATEGORY 4: Temporal/Duration Issues (Cases 4, 11, 12)
    # =================================================================
    
    "case4_clock_skew": {
        "category": "Temporal/Duration Issues",
        "description": "Device clock sync failure: 10% future dates, 10% negative durations (temporal inconsistency)",
        "fault_date": "2026-02-11",
        "fault_type": "clock_synchronization_failure",
        "fault_severity": "high",
        "true_root_causes": {
            "raw_max_trip_end_ts",  # Shows future timestamps
            "bronze_rows_dropped_by_rules"  # Validation catches bad durations
        },
        "expected_downstream": {
            "raw_duration_mean",  # Duration becomes erratic
            "bronze_survival_rate"  # Fewer rows survive validation
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "raw_max_trip_end_ts → bronze_rows_dropped_by_rules → bronze_survival_rate"
    },
    
    "case11_duration_anomalies": {
        "category": "Temporal/Duration Issues",
        "description": "20% of rows have anomalous durations (6+ hours added, GPS tracking loss)",
        "fault_date": "2026-02-18",
        "fault_type": "temporal_anomaly",
        "fault_severity": "medium",
        "true_root_causes": {
            "bronze_duration_mean",  # Duration increases dramatically
            "bronze_duration_std"  # Variance increases
        },
        "expected_downstream": {
            "bronze_rows_dropped_by_rules",  # Anomalous durations fail validation
            "bronze_survival_rate"  # Fewer rows survive
        },
        "affected_tiers": ["raw", "bronze"],
        "causal_path": "bronze_duration_mean → bronze_duration_std → bronze_rows_dropped_by_rules"
    },
    
    "case12_timestamp_inconsistencies": {
        "category": "Temporal/Duration Issues",
        "description": "25% of rows have timestamp inconsistencies (start/end offsets, multi-device sync failure)",
        "fault_date": "2026-02-19",
        "fault_type": "timestamp_inconsistency",
        "fault_severity": "medium",
        "true_root_causes": {
            "raw_max_trip_end_ts",  # End timestamps become inconsistent (graph uses end ts)
            "raw_max_trip_end_ts"
        },
        "expected_downstream": {
            "bronze_duplicate_rows_removed",  # Deduplication logic triggered
            "silver_vehicle_info_join_miss_rate"  # Time-based join fails
        },
        "affected_tiers": ["raw", "bronze", "silver"],
        "causal_path": "raw_max_trip_start_ts → raw_max_trip_end_ts → bronze_duplicate_rows_removed"
    },
    
    # =================================================================
    # CATEGORY 5: Bronze Layer - Transformation Issues (Cases 13, 14, 15)
    # =================================================================
    
    "case13_aggregation_errors": {
        "category": "Bronze Layer - Transformation Issues",
        "description": "15% of rows duplicated due to aggregation window misconfiguration",
        "fault_date": "2026-02-20",
        "fault_type": "aggregation_error",
        "fault_severity": "medium",
        "true_root_causes": {
            "bronze_duplicate_rows_removed"  # Deduplication logic detects duplicates
        },
        "expected_downstream": {
            "bronze_rows_dropped_by_rules",  # Duplicates dropped by validation
            "bronze_duplicate_rows_removed"  # represented in graph
        },
        "affected_tiers": ["bronze"],
        "causal_path": "bronze_duplicate_rows_removed → bronze_rows_dropped_by_rules → raw_unique_units"
    },
    
    "case14_join_failures": {
        "category": "Bronze Layer - Transformation Issues",
        "description": "20% join failures in bronze→silver (vehicle master data incomplete/mismatched)",
        "fault_date": "2026-02-21",
        "fault_type": "join_failure",
        "fault_severity": "high",
        "true_root_causes": {
            "silver_vehicle_info_join_miss_rate"  # Join fails on corrupted vehicle_id
        },
        "expected_downstream": {
            "silver_vehicle_type_nulls",  # Join returns nulls
            "bronze_null_primary_key_rows"  # Invalid join keys detected
        },
        "affected_tiers": ["bronze", "silver"],
        "causal_path": "silver_vehicle_info_join_miss_rate → silver_vehicle_type_nulls → bronze_null_primary_key_rows"
    },
    
    "case15_duplicate_handling": {
        "category": "Bronze Layer - Transformation Issues",
        "description": "30% of rows duplicated, deduplication fails (both versions reach KPIs)",
        "fault_date": "2026-02-22",
        "fault_type": "duplicate_handling",
        "fault_severity": "high",
        "true_root_causes": {
            "bronze_duplicate_rows_removed"  # Duplicate detection triggers
        },
        "expected_downstream": {
            "bronze_duplicate_rows_removed",  # graph-native indicator
            "silver_avg_speed_imputed"  # imputed speed metric present in graph
        },
        "affected_tiers": ["bronze", "silver"],
        "causal_path": "bronze_duplicate_rows_removed → raw_unique_units → silver_avg_speed"
    }
    ,"case16_vehicle_id_nulls_low": {
        "category": "Raw Data Quality - NULL Injections",
        "description": "5% of vehicle_id values set to NULL (low-severity)",
        "fault_date": "2025-10-12",
        "fault_type": "foreign_key_nulls",
        "fault_severity": "low",
        "true_root_causes": {"raw_null_count_unit_id"},
        "expected_downstream": {"bronze_null_primary_key_rows", "silver_vehicle_info_join_miss_rate"},
        "affected_tiers": ["raw","bronze","silver"],
        "causal_path": "raw_null_count_unit_id → bronze_null_primary_key_rows → silver_vehicle_info_join_miss_rate"
    },

    "case17_extreme_values_low": {
        "category": "Raw Data Quality - Validation Issues",
        "description": "Low-severity extreme values (5%)",
        "fault_date": "2025-10-13",
        "fault_type": "extreme_values_outliers",
        "fault_severity": "low",
        "true_root_causes": {"raw_avg_speed_mean","raw_std_fuel_consumption_ecol"},
        "expected_downstream": {"bronze_invalid_avg_speed_rows","bronze_rows_dropped_by_rules"},
        "affected_tiers": ["raw","bronze"],
        "causal_path": "raw_avg_speed_mean → raw_std_fuel_consumption_ecol → bronze_invalid_avg_speed_rows"
    },

    "case18_invalid_ranges_low": {
        "category": "Raw Data Quality - Validation Issues",
        "description": "Low-severity invalid ranges (10%)",
        "fault_date": "2025-10-14",
        "fault_type": "temporal_constraint_violation",
        "fault_severity": "low",
        "true_root_causes": {"bronze_duration_mean","bronze_rows_dropped_by_rules"},
        "expected_downstream": {"bronze_rows_dropped_by_rules","bronze_survival_rate"},
        "affected_tiers": ["bronze"],
        "causal_path": "bronze_duration_mean → bronze_rows_dropped_by_rules"
    },

    "case19_speed_prediction_drift_low": {
        "category": "ML Layer - Model Quality",
        "description": "Low-severity speed prediction drift (10%)",
        "fault_date": "2025-10-15",
        "fault_type": "ml_model_drift",
        "fault_severity": "low",
        "true_root_causes": {"silver_ml_large_error_count","silver_ml_prediction_std"},
        "expected_downstream": {"silver_ml_residual_mean","p95_fuel_per_100km"},
        "affected_tiers": ["silver","ml"],
        "causal_path": "silver_ml_large_error_count → silver_ml_prediction_std → silver_ml_residual_mean"
    },

    "case20_prediction_outliers_low": {
        "category": "ML Layer - Model Quality",
        "description": "Low-severity prediction outliers (5%)",
        "fault_date": "2025-10-16",
        "fault_type": "ml_outliers",
        "fault_severity": "low",
        "true_root_causes": {"silver_ml_large_error_count"},
        "expected_downstream": {"silver_ml_residual_mean","bronze_invalid_avg_speed_rows"},
        "affected_tiers": ["bronze","silver","ml"],
        "causal_path": "silver_ml_large_error_count → silver_ml_residual_mean → bronze_invalid_avg_speed_rows"
    },

    "case21_duration_anomalies_low": {
        "category": "Temporal/Duration Issues",
        "description": "Low-severity duration anomalies (10%)",
        "fault_date": "2025-10-17",
        "fault_type": "temporal_anomaly",
        "fault_severity": "low",
        "true_root_causes": {"bronze_duration_mean","bronze_duration_std"},
        "expected_downstream": {"bronze_rows_dropped_by_rules","bronze_survival_rate"},
        "affected_tiers": ["bronze"],
        "causal_path": "bronze_duration_mean → bronze_duration_std → bronze_rows_dropped_by_rules"
    },

    "case22_timestamp_inconsistencies_low": {
        "category": "Temporal/Duration Issues",
        "description": "Low-severity timestamp inconsistencies (15%)",
        "fault_date": "2025-10-18",
        "fault_type": "timestamp_inconsistency",
        "fault_severity": "low",
        "true_root_causes": {"raw_max_trip_end_ts"},
        "expected_downstream": {"bronze_duplicate_rows_removed","silver_vehicle_info_join_miss_rate"},
        "affected_tiers": ["raw","bronze","silver"],
        "causal_path": "raw_max_trip_end_ts → bronze_duplicate_rows_removed"
    }

}

# Summary of all test cases
TEST_CASE_SUMMARY = {
    "total_cases": 22,
    "categories": 5,
    "cases_per_category": null,
    "date_range": ("2025-10-01", "2026-02-22"),
    "category_breakdown": {
        "Raw Data Quality - NULL Injections": 4,  # Cases 1,5,6,16
        "Raw Data Quality - Validation Issues": 5,  # Cases 2,7,8,17,18
        "ML Layer - Model Quality": 5,  # Cases 3,9,10,19,20
        "Temporal/Duration Issues": 5,  # Cases 4,11,12,21,22
        "Bronze Layer - Transformation Issues": 3  # Cases 13, 14, 15
    }
}

# Helper function for RCA evaluation
def get_ground_truth_for_case(case_id):
    """Get ground truth configuration for a specific test case."""
    case_key = f"case{case_id}"
    # Find matching case in GROUND_TRUTH
    for key, config in GROUND_TRUTH.items():
        if key.startswith(case_key):
            return config
    return None

def get_true_roots(case_id):
    """Get true root causes for a specific test case."""
    config = get_ground_truth_for_case(case_id)
    if config:
        return config['true_root_causes']
    return set()

def get_expected_downstream(case_id):
    """Get expected downstream metrics for a specific test case."""
    config = get_ground_truth_for_case(case_id)
    if config:
        return config['expected_downstream']
    return set()
