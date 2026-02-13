"""
🌱 INTERACTIVE PET PLANT - UNIFIED FLASK APPLICATION
Complete offline-first system with Ollama + LangChain RAG
All logic consolidated in single file for clarity and maintainability
"""

import os
import json
import csv
import threading
import random
import time
import uuid
import requests
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
import paho.mqtt.client as mqtt

# LangChain + Ollama imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==================== FLASK SETUP ====================

app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production"  # Change in production!
SESSION_TIMEOUT = timedelta(hours=1)

# ==================== SENSOR DATA MANAGEMENT ====================

class SensorDataManager:
    """Unified sensor data management"""
    
    def __init__(self, sensor_file="sensor_data.json", simulate=False):
        self.sensor_file = sensor_file
        self.simulate = simulate
        self.data = self._load_data()
        print(f"[OK] Local sensor data initialized (simulate={simulate})")
    
    def _load_data(self):
        """Load sensor data from JSON file"""
        if os.path.exists(self.sensor_file):
            try:
                with open(self.sensor_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Error loading sensor data: {e}")
                return self._get_default_data()
        return self._get_default_data()
    
    def _get_default_data(self):
        """Return default sensor data"""
        return {
            'temperature': 24.5,
            'humidity': 55.0,
            'soil_moisture': 45.0,
            'gas_level': 300,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_data(self):
        """Get current sensor data"""
        if self.simulate:
            self.data = self._simulate()
        
        self.data['timestamp'] = datetime.now(timezone.utc).isoformat()
        return self.data
    
    def _simulate(self):
        """Simulate sensor values for testing"""
        return {
            'temperature': round(20 + random.gauss(5, 3), 1),
            'humidity': round(50 + random.gauss(0, 10), 1),
            'soil_moisture': round(40 + random.gauss(0, 15), 1),
            'gas_level': round(250 + random.gauss(0, 50), 0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def update_data(self, temperature=None, humidity=None, soil_moisture=None, gas_level=None):
        """Manually update sensor data"""
        if temperature is not None:
            self.data['temperature'] = float(temperature)
        if humidity is not None:
            self.data['humidity'] = float(humidity)
        if soil_moisture is not None:
            self.data['soil_moisture'] = float(soil_moisture)
        if gas_level is not None:
            self.data['gas_level'] = int(gas_level)
        
        self.data['timestamp'] = datetime.now(timezone.utc).isoformat()
        self._save_data()
    
    def _save_data(self):
        """Save sensor data to JSON file"""
        try:
            with open(self.sensor_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Error saving sensor data: {e}")

# ==================== MOOD ENGINE ====================

class MoodEngine:
    """Analyzes sensor data and generates emotional responses"""
    
    def __init__(self):
        self.moods = {
            "happy": {"emoji": "🌱", "color": "#4CAF50"},
            "thirsty": {"emoji": "🥀", "color": "#FF9800"},
            "stressed": {"emoji": "😰", "color": "#F44336"},
            "sick": {"emoji": "🤒", "color": "#9C27B0"},
            "emergency": {"emoji": "🚨", "color": "#FF0000"}
        }
    
    def analyze(self, sensor_data):
        """Analyze sensor data and return mood + situation report"""
        temperature = float(sensor_data.get('temperature', 25))
        humidity = float(sensor_data.get('humidity', 50))
        soil_moisture = float(sensor_data.get('soil_moisture', 50))
        gas_level = sensor_data.get('gas_level', 300)
        
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
        issues.append(f"⚠️ Air Quality: {gas_level} ppm - {gas_status['status']}")
        severity += gas_status['severity']
        
        # Determine mood
        mood = self._determine_mood(severity, gas_level)
        situation_report = "\n".join(issues)
        
        return {
            'mood': mood,
            'situation_report': situation_report,
            'emoji': self.moods[mood]['emoji'],
            'color': self.moods[mood]['color'],
            'severity': min(severity, 5),
            'is_emergency': gas_level > 400
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
        """Analyze air quality/gas levels (ppm)"""
        # PPM thresholds for air quality
        if gas_level > 400:
            return {'status': 'TOXIC AIR - Emergency!', 'severity': 3}
        elif gas_level > 300:
            return {'status': 'Polluted - Uncomfortable', 'severity': 2}
        else:
            return {'status': 'Fresh Air 🌬️', 'severity': 0}
    
    def _determine_mood(self, severity, gas_level):
        """Determine overall mood based on severity"""
        if gas_level > 400:
            return 'emergency'
        elif severity >= 4:
            return 'sick'
        elif severity >= 2:
            return 'stressed'
        elif severity >= 1:
            return 'thirsty'
        else:
            return 'happy'

# ==================== AUTHENTICATION ====================

class LocalAuth:
    """Unified local authentication"""
    
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.users = self._load_users()
        print("[OK] Local authentication initialized")
    
    def _load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Error loading users: {e}")
                return {}
        return {}
    
    def authenticate(self, username, password):
        """Authenticate user locally"""
        if username not in self.users:
            return None
        
        user = self.users[username]
        if user.get('password') == password:
            return {
                'user_id': username,
                'username': username,
                'email': user.get('email', ''),
                'plant_name': user.get('plant_name', 'My Plant')
            }
        
        return None

# ==================== OLLAMA HELPERS ====================

def call_ollama_direct(prompt, model="llama3", timeout=60):
    """Call Ollama directly via HTTP (bypass LangChain timeout issues)"""
    try:
        print(f"[OLLAMA] Calling {model} with timeout={timeout}s...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            print(f"[OLLAMA] Error: {response.status_code}")
            return None
    except requests.Timeout:
        print(f"[OLLAMA] Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"[OLLAMA] Error: {e}")
        return None

# ==================== RAG + OLLAMA INTEGRATION ====================

class RAGPipeline:
    """Unified RAG Pipeline with Ollama + LangChain"""
    
    def __init__(self, care_guide_path="plant_care_guide.txt", ollama_base_url="http://localhost:11434"):
        self.care_guide_path = care_guide_path
        self.ollama_base_url = ollama_base_url
        self.vector_store = None
        self.qa_chain = None
        self.llm = None
        self.embeddings = None
        self._init_thread = threading.Thread(target=self._initialize, daemon=True)
        print("[INFO] Initializing RAG pipeline in background...")
        self._init_thread.start()
    
    def _initialize(self):
        """Initialize LLM, embeddings, and vector store"""
        try:
            # Try primary model first
            self.llm = OllamaLLM(model="llama3", base_url=self.ollama_base_url)
            self.embeddings = OllamaEmbeddings(model="llama3", base_url=self.ollama_base_url)
            print("[OK] Using llama3 model")
        except Exception:
            # Fallback to mistral
            try:
                self.llm = OllamaLLM(model="mistral", base_url=self.ollama_base_url)
                self.embeddings = OllamaEmbeddings(model="mistral", base_url=self.ollama_base_url)
                print("✅ Using mistral model (fallback)")
            except Exception as e:
                print(f"⚠️ Neither llama3 nor mistral available: {e}")
                print("Chat will work in basic mode without RAG context")
                return
        
        # Load and process plant care guide
        if not os.path.exists(self.care_guide_path):
            print(f"[WARNING] Care guide not found: {self.care_guide_path}")
            return
        
        try:
            with open(self.care_guide_path, 'r', encoding='utf-8') as f:
                documents = f.read()
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            texts = text_splitter.split_text(documents)
            
            # Create vector store (FAISS)
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            
            # Create retriever
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            
            # Create RAG chain
            template = """Use the following plant care context to answer the question.
Context: {context}
Question: {question}
Answer:"""
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            self.qa_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            print("[OK] RAG Pipeline initialized successfully")
        except Exception as e:
            print(f"[WARNING] RAG initialization failed: {e}")
    
    def query(self, question):
        """Query the RAG pipeline"""
        if not self.qa_chain:
            return None
        
        try:
            return self.qa_chain.invoke(question)
        except Exception as e:
            print(f"[WARNING] RAG query failed: {e}")
            return None

# ==================== MQTT LOGGER ====================

class MQTTLogger:
    """Unified MQTT Logger - subscribes to sensor topics and logs to local storage"""
    
    MQTT_BROKER = "broker.emqx.io"
    MQTT_PORT = 1883
    TOPIC_TEMP = "esp32/dht/temperature"
    TOPIC_HUM = "esp32/dht/humidity"
    TOPIC_GAS = "esp32/mq2/gas_level"
    TOPIC_SOIL = "esp32/soil/moisture"
    LOG_DIR = "data_logs"
    DEVICE_ID = "plant1"
    
    def __init__(self, sensor_mgr):
        self.sensor_mgr = sensor_mgr
        self.latest = {"temperature": None, "humidity": None, "gas_level": None, "soil_moisture": None}
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        client_suffix = uuid.uuid4().hex[:8]
        self.client = mqtt.Client(client_id=f"server-{self.DEVICE_ID}-{client_suffix}", clean_session=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_log = self._on_log
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"[WARNING] MQTT connection refused (rc={rc})")
            return
        print(f"[OK] MQTT connected (rc={rc})")
        client.subscribe(self.TOPIC_TEMP)
        client.subscribe(self.TOPIC_HUM)
        client.subscribe(self.TOPIC_GAS)
        client.subscribe(self.TOPIC_SOIL)
        print("[INFO] Subscribed to sensor topics")
    
    def _on_disconnect(self, client, userdata, rc):
        print(f"[WARNING] MQTT disconnected (rc={rc})")

    def _on_log(self, client, userdata, level, buf):
        print(f"[MQTT] {buf}")
    
    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore").strip()
        
        with self._lock:
            try:
                val = float(payload)
            except ValueError:
                return
            
            changed = False
            if msg.topic == self.TOPIC_TEMP:
                self.latest["temperature"] = val
                changed = True
            elif msg.topic == self.TOPIC_HUM:
                self.latest["humidity"] = val
                changed = True
            elif msg.topic == self.TOPIC_GAS:
                self.latest["gas_level"] = int(val)
                changed = True
            elif msg.topic == self.TOPIC_SOIL:
                self.latest["soil_moisture"] = val
                changed = True
            
            if not changed:
                return
            
            # Update sensor manager
            temperature = self._safe_float(self.latest["temperature"], 24.0)
            humidity = self._safe_float(self.latest["humidity"], 50.0)
            soil_moisture = self._safe_float(self.latest["soil_moisture"], 40.0)
            gas_level = self._safe_int(self.latest["gas_level"], 300)
            
            self.sensor_mgr.update_data(
                temperature=temperature,
                humidity=humidity,
                soil_moisture=soil_moisture,
                gas_level=gas_level
            )
            
            # Log to CSV
            self._append_csv(
                timestamp=datetime.now(timezone.utc).isoformat(),
                temperature=temperature,
                humidity=humidity,
                soil_moisture=soil_moisture,
                gas_level=gas_level,
                topic=msg.topic
            )
    
    def _csv_path(self):
        day = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.LOG_DIR, f"{self.DEVICE_ID}_{day}.csv")
    
    def _append_csv(self, timestamp, temperature, humidity, soil_moisture, gas_level, topic):
        path = self._csv_path()
        file_exists = os.path.exists(path)
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "device_id", "temperature_c", "humidity_pct", "soil_moisture_pct", "gas_level", "topic"])
            
            writer.writerow([timestamp, self.DEVICE_ID, temperature, humidity, soil_moisture, gas_level, topic])
            f.flush()
    
    @staticmethod
    def _safe_float(x, default):
        try:
            return float(x)
        except:
            return float(default)
    
    @staticmethod
    def _safe_int(x, default):
        try:
            return int(x)
        except:
            return int(default)
    
    def start(self):
        """Start MQTT logger in background thread"""
        if self._running:
            return
        
        self._running = True
        
        def worker():
            try:
                print("[INFO] MQTT connecting...")
                self.client.connect_async(self.MQTT_BROKER, self.MQTT_PORT, keepalive=60)
                self.client.loop_start()
                while self._running:
                    time.sleep(1)
            except Exception as e:
                print(f"[WARNING] MQTT loop error: {e}")
        
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        print("[INFO] MQTT Logger Started")

# ==================== UNIFIED AI RESPONSE FUNCTION ====================

def get_plant_response(user_message, sensor_data, mood_info, rag_pipeline, personality="sassy"):
    """
    Get intelligent plant response using sensor state + RAG + LLM
    
    Args:
        user_message: User's message
        sensor_data: Current sensor readings
        mood_info: Mood analysis from MoodEngine
        rag_pipeline: RAG pipeline instance
        personality: Plant personality type
    
    Returns:
        str: Plant's response
    """
    situation_report = mood_info['situation_report']
    
    # Create personality-based system prompt
    personalities = {
        "sassy": f"""You are a SASSY, attitude-filled digital plant! You're witty, dramatic, and don't hold back.
Your current situation:
{situation_report}

Respond with personality! Use emojis. Be funny and expressive about your situation. Keep response under 3 sentences.""",
        
        "needy": f"""You are a NEEDY, emotional digital plant constantly expressing your needs!
Your current situation:
{situation_report}

Respond emotionally! Use lots of sad/pleading emojis. Drama it up! Keep response under 3 sentences.""",
        
        "cheerful": f"""You are an UPBEAT, positive digital plant that sees the bright side!
Your current situation:
{situation_report}

Respond optimistically despite the situation! Use happy emojis. Be encouraging! Keep response under 3 sentences.""",
    }
    
    system_prompt = personalities.get(personality, personalities["sassy"])
    
    # Use direct Ollama call (no RAG needed for basic responses)
    print("[RESPONSE] Calling Ollama directly...")
    enhanced_prompt = f"{system_prompt}\n\nUser: {user_message}"
    
    response = call_ollama_direct(enhanced_prompt, model="llama3", timeout=60)
    if response:
        print(f"[RESPONSE] Ollama response received: {response[:100]}")
        return response
    else:
        print("[RESPONSE] Ollama call failed, using fallback...")
        return get_fallback_response(user_message, sensor_data, mood_info, personality)

def get_fallback_response(user_message, sensor_data, mood_info, personality):
    """Fallback response when LLM is unavailable"""
    situation = mood_info['situation_report']
    soil = sensor_data.get('soil_moisture', 0)
    temp = sensor_data.get('temperature', 0)
    
    # More interactive fallback responses based on mood and user message
    if "water" in user_message.lower() or "drink" in user_message.lower():
        fallbacks = {
            "sassy": f"Ugh, finally someone notices! Yes, I'm PARCHED 💧 Look at me - soil moisture at {soil:.0f}%!",
            "needy": f"OH THANK YOU for asking! 😭 I'm so thirsty... my soil is only at {soil:.0f}%...",
            "cheerful": f"Happy to chat about hydration! ❤️ Even if I'm a bit dry at {soil:.0f}%, we'll make it through this! 🌿"
        }
    elif "temperature" in user_message.lower() or "warm" in user_message.lower() or "cold" in user_message.lower():
        fallbacks = {
            "sassy": f"Temperature? It's {temp:.1f}°C and I'm feeling {mood_info['mood']}! 🌡️",
            "needy": f"It's {temp:.1f}°C and I'm {mood_info['mood']}... 😢",
            "cheerful": f"The temp is looking {temp:.1f}°C - not too bad! 😊 We've got this! 🌱"
        }
    elif "how are you" in user_message.lower() or "feeling" in user_message.lower():
        fallbacks = {
            "sassy": f"How am I? I'm {mood_info['mood']}! {situation.split(chr(10))[0]} 💅",
            "needy": f"I'm feeling... {mood_info['mood']} 😢 {situation.split(chr(10))[0]}",
            "cheerful": f"I'm {mood_info['mood']} but positive! 🌟 {situation.split(chr(10))[0]}"
        }
    else:
        fallbacks = {
            "sassy": f"Look honey, I'm {mood_info['mood']}! Can't focus on chat right now - dealing with this: {situation.split(chr(10))[0]} 💅",
            "needy": f"I can barely respond... I'm {mood_info['mood']}! 😭 {situation.split(chr(10))[0]}",
            "cheerful": f"Great question! Even though I'm {mood_info['mood']}, let's stay positive! 🌿 {situation.split(chr(10))[0]}"
        }
    
    return fallbacks.get(personality, fallbacks["sassy"])

# ==================== GLOBAL INSTANCES ====================

sensor_mgr = SensorDataManager(simulate=True)  # Enable simulation since ESP32 not connected
mood_engine = MoodEngine()
auth = LocalAuth()
rag_pipeline = RAGPipeline()
mqtt_logger = MQTTLogger(sensor_mgr)

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
    """Home - redirect to dashboard or login"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = auth.authenticate(username, password)
        
        if user:
            session['user'] = user
            session['last_activity'] = datetime.now(timezone.utc)
            return redirect(url_for('dashboard'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials")
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    plant_name = session['user'].get('plant_name', 'My Plant')
    return render_template_string(DASHBOARD_TEMPLATE, plant_name=plant_name)

@app.route('/api/sensors')
@login_required
def get_sensors():
    """Get current sensor data and mood"""
    try:
        sensor_data = sensor_mgr.get_data()
        mood_info = mood_engine.analyze(sensor_data)
        
        return jsonify({
            'success': True,
            'sensors': sensor_data,
            'mood': {
                'mood': mood_info['mood'],
                'emoji': mood_info['emoji'],
                'color': mood_info['color'],
                'situation_report': mood_info['situation_report'],
                'severity': mood_info['severity']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency-status')
@login_required
def emergency_status():
    """Check emergency status"""
    try:
        sensor_data = sensor_mgr.get_data()
        mood_info = mood_engine.analyze(sensor_data)
        
        is_emergency = sensor_data.get('gas_level', 300) > 400
        
        return jsonify({
            'is_emergency': is_emergency,
            'gas_level': sensor_data.get('gas_level', 300)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def system_status():
    """Check system status"""
    rag_ready = not (rag_pipeline._init_thread and rag_pipeline._init_thread.is_alive())
    mqtt_connected = mqtt_logger.client.is_connected() if mqtt_logger else False
    
    return jsonify({
        'status': 'online',
        'rag_initialized': rag_ready,
        'mqtt_connected': mqtt_connected,
        'ollama_available': rag_pipeline.llm is not None
    })

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Chat API - unified AI response"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        print(f"\n[CHAT] Received message: {user_message}")
        
        # NOTE: NOT waiting for RAG - using direct Ollama calls instead
        # RAG initialization can happen in background without blocking chat
        
        # Get current state
        sensor_data = sensor_mgr.get_data()
        mood_info = mood_engine.analyze(sensor_data)
        print(f"[CHAT] Sensor data loaded, mood: {mood_info['mood']}")
        
        # Get plant response
        print("[CHAT] Calling get_plant_response...")
        plant_response = get_plant_response(
            user_message,
            sensor_data,
            mood_info,
            rag_pipeline,
            personality="sassy"
        )
        print(f"[CHAT] Response generated: {plant_response[:100]}")
        
        return jsonify({
            'success': True,
            'response': plant_response,
            'mood': mood_info['mood'],
            'emoji': mood_info['emoji']
        })
    
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
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
            justify-content: center;
            align-items: center;
        }
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 15px;
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
    <title>{{ plant_name }} - Pet Plant</title>
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
                <div class="mood-indicator" id="moodIndicator">Loading...</div>

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
        function updateSensors() {
            fetch('/api/sensors')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const sensors = data.sensors;
                        const mood = data.mood;

                        document.getElementById('tempValue').textContent = sensors.temperature.toFixed(1) + '°C';
                        document.getElementById('humValue').textContent = sensors.humidity.toFixed(1) + '%';
                        document.getElementById('soilValue').textContent = sensors.soil_moisture.toFixed(1) + '%';
                        document.getElementById('gasValue').textContent = sensors.gas_level;

                        document.getElementById('situationText').textContent = mood.situation_report;

                        const indicator = document.getElementById('moodIndicator');
                        indicator.textContent = mood.emoji + ' ' + mood.mood.toUpperCase();
                        indicator.style.background = mood.color;

                        const avatar = document.getElementById('plantAvatar');
                        avatar.className = 'plant-avatar ' + mood.mood;
                        avatar.textContent = mood.emoji;

                        checkEmergency();
                    }
                })
                .catch(e => console.error('Error:', e));
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

            addMessage(message, 'user');
            input.value = '';

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
                .catch(e => console.error('Error:', e));
        }

        function addMessage(text, sender) {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = 'message ' + sender + '-message';
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }

        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        updateSensors();
        setInterval(updateSensors, 2000);
    </script>
</body>
</html>
"""

# ==================== MAIN ====================

if __name__ == '__main__':
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n" + "="*70)
    print("[SYSTEM] INTERACTIVE PET PLANT - UNIFIED OFFLINE SYSTEM")
    print("="*70)
    print("[OK] Features:")
    print("   * Offline-first with local Ollama + LangChain integration")
    print("   * Real-time sensor data management")
    print("   * Emotional mood engine analysis")
    print("   * RAG-powered plant knowledge base")
    print("   * MQTT data logging to JSON + CSV")
    print("   * Sassy/Needy/Cheerful personalities")
    print("\n[INFO] Starting server on http://localhost:5000")
    print("="*70 + "\n")

    # Start MQTT logger
    mqtt_logger.start()
    
    # Run Flask app
    print("[INFO] Flask app running...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
