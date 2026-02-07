"""
Interactive Pet Plant - Flask Application
An emotionally-aware digital plant that reacts to real-time sensor data
100% Offline - No Firebase dependency
"""

import os
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
from functools import wraps
from datetime import datetime, timedelta, timezone

# Import custom modules - LOCAL ONLY (no Firebase) - these already exist in your project
from mqtt_test import start_mqtt_logger
from mood_engine import MoodEngine
from rag_pipeline import RAGPipeline
from local_auth import get_auth
from local_sensor import get_sensor_data

# Initialize Flask
app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production"  # Change this!

# Initialize managers - ALL LOCAL
auth = get_auth()  # Local authentication
sensor_data_mgr = get_sensor_data(simulate=False)  # Set True for random simulation
mood_engine = MoodEngine()

try:
    rag = RAGPipeline(care_guide_path="plant_care_guide.txt")
    print("✅ RAG Pipeline loaded")
except Exception as e:
    print(f"⚠️ RAG initialization failed: {e}")
    rag = None

# Session timeout
SESSION_TIMEOUT = timedelta(hours=1)

# ==================== DECORATORS ====================

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))

        # Check session timeout
        if 'last_activity' in session:
            if datetime.now(timezone.utc) - session['last_activity'] > SESSION_TIMEOUT:
                session.clear()
                return redirect(url_for('login'))

        session['last_activity'] = datetime.now(timezone.utc)
        return f(*args, **kwargs)

    return decorated_function

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page - redirect to dashboard or login"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page - LOCAL AUTHENTICATION"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Use local authentication
        user = auth.authenticate(username, password)

        if user:
            session['user'] = user
            session['last_activity'] = datetime.now(timezone.utc)
            return redirect(url_for('dashboard'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with pet plant"""
    user = session['user']
    plant_name = user.get('plant_name', 'My Plant')

    return render_template_string(DASHBOARD_TEMPLATE, plant_name=plant_name)

@app.route('/api/sensors')
@login_required
def get_sensors():
    """API endpoint to get current sensor data - LOCAL"""
    try:
        # Get sensor data from local file (sensor_data.json)
        sensor_data = sensor_data_mgr.get_data()
        mood_info = mood_engine.analyze(sensor_data)

        return jsonify({
            'success': True,
            'sensors': sensor_data,
            'mood': mood_info
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """API endpoint for plant chat - LOCAL"""
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400

        # Get current sensor data and mood from local file
        sensor_data = sensor_data_mgr.get_data()
        mood_info = mood_engine.analyze(sensor_data)
        situation_report = mood_info['situation_report']

        # Get plant response using RAG
        if rag:
            plant_response = rag.get_plant_response(
                user_message,
                situation_report,
                personality="sassy"  # Can be customized per plant
            )
        else:
            # Fallback if RAG not available
            plant_response = f"*tries to respond but is too focused on my situation*\n\n{situation_report}\n\nCan you help? 🥺"

        return jsonify({
            'success': True,
            'response': plant_response,
            'mood': mood_info['mood'],
            'emoji': mood_info['emoji']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency-status')
@login_required
def emergency_status():
    """Check if there's an emergency (high gas levels) - LOCAL"""
    try:
        # Get current sensor data from local file
        sensor_data = sensor_data_mgr.get_data()

        # Emergency if gas level is critical (>1000)
        is_emergency = sensor_data.get('gas_level', 0) > 1000

        return jsonify({
            'is_emergency': is_emergency,
            'gas_level': sensor_data.get('gas_level', 0)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== TEMPLATES ====================

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pet Plant - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .login-container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 400px;
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
        }

        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }

        input:focus {
            outline: none;
            border-color: #667eea;
        }

        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 10px;
        }

        button:hover {
            transform: translateY(-2px);
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #c62828;
        }

        .demo-credentials {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 0.9rem;
            color: #2e7d32;
        }

        .plant-emoji {
            text-align: center;
            font-size: 4rem;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="plant-emoji">🌱</div>
        <h1>Pet Plant</h1>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required>
            </div>

            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>

            <button type="submit">🌿 Login</button>
        </form>

        <div class="demo-credentials">
            <strong>Demo Credentials:</strong><br>
            Username: dhrruv<br>
            Password: demo123
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ plant_name }} - Pet Plant Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2rem;
        }

        .logout-btn {
            padding: 10px 20px;
            background: #f44336;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
        }

        .logout-btn:hover {
            background: #d32f2f;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        .plant-container {
            background: white;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .plant-avatar {
            font-size: 8rem;
            margin-bottom: 20px;
            transition: transform 0.5s;
        }

        .plant-avatar.happy { animation: happyBounce 0.5s; }
        .plant-avatar.thirsty { animation: thirstyWilt 0.5s; }
        .plant-avatar.stressed { animation: stressedShake 0.5s; }
        .plant-avatar.sick { animation: sickSway 0.5s; }
        .plant-avatar.emergency { animation: emergencyPulse 0.5s infinite; }

        @keyframes happyBounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        @keyframes thirstyWilt {
            0%, 100% { transform: rotate(0deg); }
            50% { transform: rotate(-5deg); }
        }

        @keyframes stressedShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }

        @keyframes sickSway {
            0%, 100% { transform: rotate(0deg); }
            50% { transform: rotate(3deg); }
        }

        @keyframes emergencyPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .plant-name {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #667eea;
            font-weight: bold;
        }

        .mood-indicator {
            display: inline-block;
            padding: 15px 30px;
            border-radius: 50px;
            color: white;
            font-weight: bold;
            font-size: 1.1rem;
            margin-top: 20px;
        }

        .sensors-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 30px;
        }

        .sensor-card {
            background: rgba(0,0,0,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }

        .sensor-label {
            font-size: 0.9rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }

        .sensor-value {
            font-size: 1.5rem;
            font-weight: bold;
        }

        .situation-report {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .situation-report h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }

        .situation-text {
            white-space: pre-wrap;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #555;
            background: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .chat-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            min-height: 500px;
        }

        .chat-container h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
            max-height: 300px;
        }

        .message {
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 8px;
            animation: messageSlide 0.3s;
        }

        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            background: #667eea;
            color: white;
            margin-left: 20px;
            border-radius: 0 8px 8px 8px;
        }

        .plant-message {
            background: #e8f5e9;
            color: #2e7d32;
            margin-right: 20px;
            border-radius: 8px 0 8px 8px;
        }

        .chat-input-group {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            outline: none;
            border-color: #667eea;
        }

        .chat-send-btn {
            padding: 12px 25px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.3s;
        }

        .chat-send-btn:hover {
            background: #5568d3;
        }

        .emergency-banner {
            display: none;
            background: #ff0000;
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            font-size: 1.2rem;
            font-weight: bold;
            text-align: center;
            animation: emergencyFlash 0.5s infinite;
        }

        .emergency-banner.show {
            display: block;
        }

        @keyframes emergencyFlash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌱 Pet Plant</h1>
            <a href="/logout" class="logout-btn">Logout</a>
        </div>

        <div class="emergency-banner" id="emergencyBanner">
            🚨 EMERGENCY! HIGH GAS LEVELS DETECTED! 🚨
        </div>

        <div class="situation-report" id="situationReport">
            <h3>📊 Plant Status Report</h3>
            <div class="situation-text" id="situationText">Loading...</div>
        </div>

        <div class="main-grid">
            <div class="plant-container">
                <div class="plant-name">{{ plant_name }}</div>
                <div class="plant-avatar happy" id="plantAvatar">🌱</div>
                <div class="mood-indicator" id="moodIndicator">
                    Loading...
                </div>

                <div class="sensors-grid">
                    <div class="sensor-card">
                        <div class="sensor-label">🌡️ Temperature</div>
                        <div class="sensor-value" id="tempValue">--°C</div>
                    </div>
                    <div class="sensor-card">
                        <div class="sensor-label">💧 Humidity</div>
                        <div class="sensor-value" id="humValue">--%</div>
                    </div>
                    <div class="sensor-card">
                        <div class="sensor-label">🌍 Soil Moisture</div>
                        <div class="sensor-value" id="soilValue">--%</div>
                    </div>
                    <div class="sensor-card">
                        <div class="sensor-label">⚠️ Gas Level</div>
                        <div class="sensor-value" id="gasValue">--</div>
                    </div>
                </div>
            </div>

            <div class="chat-container">
                <h3>💬 Chat with {{ plant_name }}</h3>
                <div class="chat-messages" id="chatMessages"></div>
                <div class="chat-input-group">
                    <input
                        type="text"
                        class="chat-input"
                        id="chatInput"
                        placeholder="Ask me something... 🌿"
                    >
                    <button class="chat-send-btn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update sensors and mood every 2 seconds
        function updateSensors() {
            fetch('/api/sensors')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const sensors = data.sensors;
                        const mood = data.mood;

                        // Update sensor values
                        document.getElementById('tempValue').textContent = sensors.temperature.toFixed(1) + '°C';
                        document.getElementById('humValue').textContent = sensors.humidity.toFixed(1) + '%';
                        document.getElementById('soilValue').textContent = sensors.soil_moisture.toFixed(1) + '%';
                        document.getElementById('gasValue').textContent = sensors.gas_level;

                        // Update situation report
                        document.getElementById('situationText').textContent = mood.situation_report;

                        // Update mood indicator
                        const indicator = document.getElementById('moodIndicator');
                        indicator.textContent = mood.emoji + ' ' + mood.mood.toUpperCase();
                        indicator.style.background = mood.color;

                        // Update plant avatar
                        const avatar = document.getElementById('plantAvatar');
                        avatar.className = 'plant-avatar ' + mood.mood;
                        avatar.textContent = mood.emoji;

                        // Check emergency
                        checkEmergency();
                    }
                })
                .catch(e => console.error('Sensor error:', e));
        }

        function checkEmergency() {
            fetch('/api/emergency-status')
                .then(r => r.json())
                .then(data => {
                    const banner = document.getElementById('emergencyBanner');
                    if (data.is_emergency) {
                        banner.classList.add('show');
                    } else {
                        banner.classList.remove('show');
                    }
                });
        }

        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();

            if (!message) return;

            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';

            // Send to API
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        addMessage(data.response, 'plant');
                    }
                })
                .catch(e => console.error('Chat error:', e));
        }

        function addMessage(text, sender) {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = 'message ' + sender + '-message';
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        // Keyboard support for chat
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // Initial load and auto-refresh
        updateSensors();
        setInterval(updateSensors, 2000);
    </script>
</body>
</html>
"""

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌱 INTERACTIVE PET PLANT - FLASK APP")
    print("="*60)
    print("✅ Features:")
    print("   - Login/Authentication")
    print("   - Real-time Sensor Data")
    print("   - Mood Engine")
    print("   - RAG + Local LLM Chat")
    print("   - Emergency Detection")
    print("   - MQTT -> sensor_data.json + CSV logging (NEW)")
    print("\n🚀 Starting server on http://localhost:5000")
    print("="*60 + "\n")

    # ✅ Start MQTT logger ONCE (safe)
    start_mqtt_logger()

    # ✅ IMPORTANT: use_reloader=False prevents Flask from running the app twice in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)