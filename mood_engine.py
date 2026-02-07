"""
Mood Engine for Digital Pet Plant
Analyzes sensor data and generates emotional responses
"""

class MoodEngine:
    def __init__(self):
        self.moods = {
            "happy": {"emoji": "🌱", "color": "#4CAF50"},
            "thirsty": {"emoji": "🥀", "color": "#FF9800"},
            "stressed": {"emoji": "😰", "color": "#F44336"},
            "sick": {"emoji": "🤒", "color": "#9C27B0"},
            "emergency": {"emoji": "🚨", "color": "#FF0000"}
        }
    
    def analyze(self, sensor_data):
        """
        Analyze sensor data and return mood + situation report
        
        Args:
            sensor_data (dict): {
                'temperature': float,
                'humidity': float,
                'soil_moisture': float,
                'gas_level': str ('normal', 'warning', 'critical')
            }
        
        Returns:
            dict: {
                'mood': str,
                'situation_report': str,
                'emoji': str,
                'color': str,
                'severity': int (0-5)
            }
        """
        
        temperature = float(sensor_data.get('temperature', 25))
        humidity = float(sensor_data.get('humidity', 50))
        soil_moisture = float(sensor_data.get('soil_moisture', 50))
        gas_level = sensor_data.get('gas_level', 'normal')
        
        # Calculate severity scores
        severity = 0
        issues = []
        
        # Soil Moisture Analysis
        soil_status = self._analyze_soil(soil_moisture)
        issues.append(f"🌍 Soil Moisture: {soil_moisture:.1f}% - {soil_status['status']}")
        severity += soil_status['severity']
        
        # Temperature Analysis
        temp_status = self._analyze_temperature(temperature)
        issues.append(f"🌡️ Temperature: {temperature:.1f}°C - {temp_status['status']}")
        severity += temp_status['severity']
        
        # Humidity Analysis
        humidity_status = self._analyze_humidity(humidity)
        issues.append(f"💧 Humidity: {humidity:.1f}% - {humidity_status['status']}")
        severity += humidity_status['severity']
        
        # Gas Level Analysis
        gas_status = self._analyze_gas(gas_level)
        issues.append(f"⚠️ Air Quality: {gas_level} - {gas_status['status']}")
        severity += gas_status['severity']
        
        # Determine mood based on severity
        mood = self._determine_mood(severity, gas_level)
        
        # Create situation report
        situation_report = "\n".join(issues)
        
        return {
            'mood': mood,
            'situation_report': situation_report,
            'emoji': self.moods[mood]['emoji'],
            'color': self.moods[mood]['color'],
            'severity': min(severity, 5)
        }
    
    def _analyze_soil(self, moisture):
        """Analyze soil moisture level"""
        if moisture < 10:
            return {'status': 'CRITICAL - DROUGHT', 'severity': 3}
        elif moisture < 20:
            return {'status': 'Very Dry - Thirsty!', 'severity': 2}
        elif moisture < 35:
            return {'status': 'Dry - Needs water soon', 'severity': 1}
        elif moisture < 50:
            return {'status': 'Slightly Dry', 'severity': 0}
        elif moisture <= 70:
            return {'status': 'Perfect 🎯', 'severity': 0}
        elif moisture <= 80:
            return {'status': 'Slightly Moist', 'severity': 0}
        else:
            return {'status': 'Soggy - Risk of root rot!', 'severity': 1}
    
    def _analyze_temperature(self, temp):
        """Analyze temperature level"""
        if temp < 5:
            return {'status': 'FREEZING - Danger!', 'severity': 3}
        elif temp < 10:
            return {'status': 'Very Cold - Stressed', 'severity': 2}
        elif temp < 15:
            return {'status': 'Cold', 'severity': 1}
        elif temp <= 25:
            return {'status': 'Perfect 🎯', 'severity': 0}
        elif temp <= 30:
            return {'status': 'Warm', 'severity': 0}
        elif temp < 35:
            return {'status': 'Hot - Stressed', 'severity': 1}
        else:
            return {'status': 'SCORCHING - Danger!', 'severity': 2}
    
    def _analyze_humidity(self, humidity):
        """Analyze humidity level"""
        if humidity < 20:
            return {'status': 'Very Dry - Uncomfortable', 'severity': 1}
        elif humidity < 40:
            return {'status': 'Dry - Could use moisture', 'severity': 0}
        elif humidity <= 60:
            return {'status': 'Perfect 🎯', 'severity': 0}
        elif humidity <= 75:
            return {'status': 'Humid', 'severity': 0}
        else:
            return {'status': 'Too Humid - Fungal risk', 'severity': 1}
    
    def _analyze_gas(self, gas_level):
        """Analyze air quality/gas levels"""
        if gas_level == 'critical':
            return {'status': 'TOXIC AIR - Emergency!', 'severity': 3}
        elif gas_level == 'warning':
            return {'status': 'Polluted - Uncomfortable', 'severity': 2}
        else:
            return {'status': 'Fresh Air 🌬️', 'severity': 0}
    
    def _determine_mood(self, severity, gas_level):
        """Determine overall mood based on severity"""
        if gas_level == 'critical':
            return 'emergency'
        elif severity >= 4:
            return 'sick'
        elif severity >= 2:
            return 'stressed'
        elif severity >= 1:
            return 'thirsty'
        else:
            return 'happy'
