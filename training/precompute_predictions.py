# precompute_predictions.py
"""
Pre-compute all predictions for deployment.
Run this locally once, then deploy the cache files.
"""

import pandas as pd
import pickle
from tqdm import tqdm
from wellapp import utils
from statsmodels.tsa.seasonal import STL
import os

def precompute_all_predictions():
    """Pre-compute XGBoost, Gaussian Process, and STL for all stations."""
    
    # Load data and model
    print("Loading data...")
    df_daily, df_monthly, quality_codes, stations_df = utils.load_data()
    model = utils.load_ml_model()
    
    # Storage for all predictions
    predictions_cache = {}
    
    # Get list of stations with monthly data
    stations_with_data = df_monthly['STATION'].unique()
    print(f"Pre-computing predictions for {len(stations_with_data)} stations...")
    
    for station_id in tqdm(stations_with_data):
        try:
            # Get station data
            station_df = df_monthly.loc[df_monthly.STATION == station_id]
            
            # ---- XGBoost Predictions ----
            predictions_xgb, dates_xgb = utils.generate_ml_predictions(
                station_id, station_df, stations_df, df_monthly, model
            )
            
            # ---- Gaussian Process Predictions ----
            try:
                gsp_mean, gsp_std = utils.generate_gsp(station_df, dates_xgb)
            except Exception as e:
                print(f"  GP failed for {station_id}: {e}")
                gsp_mean = None
                gsp_std = None
            
            # ---- STL Decomposition (only up to last data point) ----
            try:
                last_data_date = station_df['MSMT_DATE'].max()
                date_range_stl = pd.date_range(
                    start=dates_xgb[0],
                    end=last_data_date,
                    freq='ME'
                )
                
                predictions_stl = predictions_xgb[:len(date_range_stl)]
                
                # STL on XGBoost predictions
                stl_xgb = STL(predictions_stl, period=12).fit()
                
                # STL on Gaussian Process (if available)
                if gsp_mean is not None:
                    gsp_stl_data = gsp_mean[:len(date_range_stl)]
                    stl_gsp = STL(gsp_stl_data, period=12).fit()
                else:
                    stl_gsp = None
                
            except Exception as e:
                print(f"  STL failed for {station_id}: {e}")
                stl_xgb = None
                stl_gsp = None
                date_range_stl = None
            
            # Store everything
            predictions_cache[station_id] = {
                # XGBoost predictions (full range)
                'xgb_predictions': predictions_xgb,
                'xgb_dates': dates_xgb,
                
                # Gaussian Process (full range)
                'gsp_mean': gsp_mean,
                'gsp_std': gsp_std,
                'gsp_dates': dates_xgb,
                
                # STL decomposition (up to last data point)
                'stl_dates': date_range_stl,
                'xgb_predictions_trend': predictions_stl if stl_xgb else None,
                'xgb_trend': stl_xgb.trend if stl_xgb else None,
                'xgb_seasonal': stl_xgb.seasonal if stl_xgb else None,
                'xgb_resid': stl_xgb.resid if stl_xgb else None,
                'gsp_predictions_trend': gsp_stl_data if stl_gsp else None,
                'gsp_trend': stl_gsp.trend if stl_gsp else None,
                'gsp_seasonal': stl_gsp.seasonal if stl_gsp else None,
                'gsp_resid': stl_gsp.resid if stl_gsp else None,
            }
            
        except Exception as e:
            print(f"  Failed for {station_id}: {e}")
            continue
    
    # Save to disk
    print(f"\nSaving cache with {len(predictions_cache)} stations...")
    with open('models/predictions_cache.pkl', 'wb') as f:
        pickle.dump(predictions_cache, f)
    
    # Calculate size
    size_mb = os.path.getsize('models/predictions_cache.pkl') / (1024 * 1024)
    print(f"Cache saved: {size_mb:.1f} MB")
    print("Done!")

if __name__ == '__main__':
    precompute_all_predictions()