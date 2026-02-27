import numpy as np
import asyncio
import websockets
import json
import os
import pandas as pd
from scipy.interpolate import interp1d

"""
Class to convert raw ADC values to Pressure (Pa) for a specific region.

Example usage:
converter = PressureConverter()
pressure = converter.get_pressure(adc_values, 'i3') # region of hand as described in handregions.png
"""
class PressureConverter:
    def __init__(self, calibration_folder='calibration_data'):
        self.folder = calibration_folder
        self.converters = {} # Cache for loaded regions

    def get_pressure(self, adc_values, roi_name):
        """
        Maps raw ADC values to Pressure (Pa) for a specific region.
        """
        # 1. Load converter if not already in memory
        if roi_name not in self.converters:
            self._load_converter(roi_name)
        
        converter_func = self.converters[roi_name]
        
        # 2. Apply conversion
        return converter_func(adc_values)

    def _load_converter(self, roi_name):
        csv_path = os.path.join(self.folder, f"calibration_{roi_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No calibration found for {roi_name}")
            
        df = pd.read_csv(csv_path)
        
        # X axis = ADC Reading (Input)
        # Y axis = Pressure Pa (Output)
        x_adc = df['ADC_Reading'].values
        y_pressure = df['Pressure_Pa'].values
        
        # Sort by ADC to ensure interpolation works
        sorted_indices = np.argsort(x_adc)
        x_adc = x_adc[sorted_indices]
        y_pressure = y_pressure[sorted_indices]
        
        # --- DEFINE CAPS ---
        # Low cap: The pressure at the lowest valid ADC reading
        min_pressure = y_pressure[0]
        # High cap: The pressure at the highest valid ADC reading
        max_pressure = y_pressure[-1]
        
        # --- CREATE INTERPOLATION ---
        # bounds_error=False: Don't crash on out-of-bounds values
        # fill_value=(low_val, high_val): Tuple defines caps for (below_min, above_max)
        f = interp1d(x_adc, y_pressure, kind='linear', 
                     bounds_error=False, 
                     fill_value=(min_pressure, max_pressure))
        
        self.converters[roi_name] = f


