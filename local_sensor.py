"""
Local Offline Sensor Data Module
Reads and manages sensor data from local JSON file
Supports manual updates and optional simulation mode
"""

import json
import os
from datetime import datetime
import random

class SensorData:
    """Local sensor data management"""
    
    def __init__(self, sensor_file="sensor_data.json", simulate=False):
        """
        Initialize with local sensor file
        
        Args:
            sensor_file: Path to sensor JSON file
            simulate: Enable random simulation for testing
        """
        self.sensor_file = sensor_file
        self.simulate = simulate
        self.data = self._load_data()
        print(f"✅ Local sensor data initialized (simulate={simulate})")
    
    def _load_data(self):
        """Load sensor data from JSON file"""
        if os.path.exists(self.sensor_file):
            try:
                with open(self.sensor_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading sensor data: {e}")
                return self._get_default_data()
        return self._get_default_data()
    
    def _get_default_data(self):
        """Return default sensor data"""
        return {
            'temperature': 24.5,
            'humidity': 55.0,
            'soil_moisture': 45.0,
            'gas_level': 280,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_data(self):
        """
        Get current sensor data
        
        Returns:
            dict: Sensor data with optional simulation
        """
        if self.simulate:
            self.data = self._simulate()
        
        # Ensure timestamp is current
        self.data['timestamp'] = datetime.now().isoformat()
        return self.data
    
    def _simulate(self):
        """Simulate sensor values for testing"""
        return {
            'temperature': round(20 + random.gauss(5, 3), 1),
            'humidity': round(50 + random.gauss(0, 10), 1),
            'soil_moisture': round(40 + random.gauss(0, 15), 1),
            'gas_level': round(250 + random.gauss(0, 50), 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def update_data(self, temperature=None, humidity=None, soil_moisture=None, gas_level=None):
        """
        Manually update sensor data
        
        Args:
            temperature: New temperature value
            humidity: New humidity value
            soil_moisture: New soil moisture value
            gas_level: New gas level value
        """
        if temperature is not None:
            self.data['temperature'] = float(temperature)
        if humidity is not None:
            self.data['humidity'] = float(humidity)
        if soil_moisture is not None:
            self.data['soil_moisture'] = float(soil_moisture)
        if gas_level is not None:
            self.data['gas_level'] = int(gas_level)
        
        self.data['timestamp'] = datetime.now().isoformat()
        self._save_data()
    
    def _save_data(self):
        """Save sensor data to JSON file"""
        try:
            with open(self.sensor_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving sensor data: {e}")

# Global instance
_sensor_data = None

def get_sensor_data(simulate=False):
    """Get or create sensor data instance"""
    global _sensor_data
    if _sensor_data is None:
        _sensor_data = SensorData(simulate=simulate)
    return _sensor_data
