Case 1: Testing with 40% Null values in unit_id column

======================================================================
VALIDATING GRAPH FILES
======================================================================
  ✓ hybrid_filtered: All files present

======================================================================
TEST DATE: 2026-02-06
======================================================================
Ground truth case: case1_unit_id_nulls
True root causes: {'raw_unique_units', 'raw_null_count_unit_id'}

Loading test run for date: 2026-02-06
  Loaded 91 metrics

------------------------------------------------------------
GRAPH: Hybrid PC-NOTEARS-Bootstrap (Filtered)
------------------------------------------------------------
  Edges loaded: 22
  Upstream nodes: 16
  Downstream nodes: 14
  Anomalies detected: 35

  Top-10 Candidates (downstream traversal):
     1. raw_null_count_unit_id                     (score=2.909) *✓ ROOT
     2. raw_distance_mean                          (score=2.669) *
     3. bronze_distance_km_mean                    (score=2.371) *
     4. raw_avg_speed_mean                         (score=2.084) *
     5. raw_null_count_distance                    (score=2.072) *
     6. bronze_invalid_avg_speed_rows              (score=2.046) *
     7. raw_null_count_start_longitude             (score=2.041) *
     8. bronze_rows_dropped_by_rules               (score=2.034) *
     9. raw_max_trip_end_ts                        (score=2.028) *
    10. raw_null_count_avg_speed                   (score=2.023) *

  Evaluation Summary:
    Top-3 Accuracy: 1.000
    MRR: 1.000
    Recall@5: 0.500
    ✗ raw_unique_units: Rank 38
    ✓ raw_null_count_unit_id: Rank 1


