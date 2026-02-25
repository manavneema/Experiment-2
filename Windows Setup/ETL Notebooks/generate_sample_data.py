"""
Generate sample data for local ETL notebook testing.
Creates synthetic trips and vehicle data in parquet format.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(
    output_dir="./artifacts",
    num_trips=1000,
    num_vehicles=50,
    run_date="2025-12-15"
):
    """Generate sample trips and vehicle info data."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate run date
    run_date_obj = pd.to_datetime(run_date)
    
    # Generate vehicles
    print(f"Generating {num_vehicles} vehicles...")
    vehicle_data = {
        'unit_id': [f'UNIT_{i:04d}' for i in range(1, num_vehicles + 1)],
        'vehicle_class': np.random.choice(['sedan', 'suv', 'truck', 'van'], num_vehicles),
        'engine_cc': np.random.choice([1200, 1500, 1800, 2000, 2500, 3000], num_vehicles),
        'fuel_type': np.random.choice(['petrol', 'diesel', 'hybrid'], num_vehicles),
        'transmission': np.random.choice(['manual', 'automatic'], num_vehicles),
        'model_year': np.random.randint(2015, 2023, num_vehicles)
    }
    vehicle_df = pd.DataFrame(vehicle_data)
    vehicle_file = os.path.join(output_dir, 'vehicle_info.parquet')
    vehicle_df.to_parquet(vehicle_file, index=False)
    print(f"✓ Saved {len(vehicle_df)} vehicles to {vehicle_file}")
    
    # Generate trips
    print(f"Generating {num_trips} trips...")
    trips_data = []
    
    for i in range(num_trips):
        unit_id = np.random.choice(vehicle_df['unit_id'].values)
        
        # Random trip on run_date
        hour = np.random.randint(6, 22)  # 6 AM to 10 PM
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        start_ts = run_date_obj + timedelta(hours=hour, minutes=minute, seconds=second)
        
        # Duration: 300 seconds to 7200 seconds (5 min to 2 hours)
        duration_sec = np.random.randint(300, 7200)
        end_ts = start_ts + timedelta(seconds=duration_sec)
        
        # Distance: 2 km to 200 km
        distance_m = np.random.uniform(2000, 200000)
        
        # Average speed based on distance and duration
        speed_kmh = (distance_m / 1000) / (duration_sec / 3600)
        speed_kmh = np.clip(speed_kmh, 5, 150)  # Realistic speeds
        
        # Idle time: 0% to 30% of trip duration
        idle_time_sec = int(np.random.uniform(0, duration_sec * 0.3))
        
        # Fuel consumption (L) - varies by vehicle and driving
        fuel_per_100km = np.random.uniform(5, 12)
        fuel_consumed = (fuel_per_100km * distance_m / 1000) / 100.0
        
        # GPS coverage (0.0 to 1.0)
        gps_coverage = np.random.uniform(0.6, 1.0)
        
        # Trip type (1=normal, 4=correction)
        trip_type = 1 if np.random.random() > 0.05 else 4
        
        trips_data.append({
            'unit_id': unit_id,
            'date': run_date_obj.date(),
            'start': start_ts,
            'end': end_ts,
            'distance': int(distance_m),
            'distance_km': distance_m / 1000.0,
            'avg_speed': speed_kmh,
            'idle_time': idle_time_sec,
            'fuel_consumption': fuel_consumed,
            'gps_coverage': gps_coverage,
            'trip_type': trip_type
        })
    
    trips_df = pd.DataFrame(trips_data)
    trips_file = os.path.join(output_dir, 'trips_raw.parquet')
    trips_df.to_parquet(trips_file, index=False)
    print(f"✓ Saved {len(trips_df)} trips to {trips_file}")
    
    # Print summary stats
    print("\n" + "="*60)
    print("SAMPLE DATA GENERATION SUMMARY")
    print("="*60)
    print(f"Vehicles: {len(vehicle_df)}")
    print(f"  - Vehicle classes: {vehicle_df['vehicle_class'].unique()}")
    print(f"  - Fuel types: {vehicle_df['fuel_type'].unique()}")
    print(f"\nTrips: {len(trips_df)}")
    print(f"  - Date range: {trips_df['date'].min()} to {trips_df['date'].max()}")
    print(f"  - Distance: {trips_df['distance_km'].min():.1f} to {trips_df['distance_km'].max():.1f} km")
    print(f"  - Duration: {trips_df['start'].min()} to {trips_df['end'].max()}")
    print(f"  - Avg speed: {trips_df['avg_speed'].min():.1f} to {trips_df['avg_speed'].max():.1f} km/h")
    print(f"  - Fuel consumption: {trips_df['fuel_consumption'].min():.2f} to {trips_df['fuel_consumption'].max():.2f} L")
    print("="*60)
    
    return vehicle_df, trips_df


if __name__ == "__main__":
    vehicle_df, trips_df = generate_sample_data(
        output_dir="./artifacts",
        num_trips=1000,
        num_vehicles=50,
        run_date="2025-12-15"
    )
    print("\n✓ Sample data ready for ETL notebooks!")
