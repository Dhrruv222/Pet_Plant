"""
🌱 INTERACTIVE PET PLANT - UNIFIED FLASK APPLICATION
Complete offline-first system with Ollama + LangChain RAG
All logic consolidated in single file for clarity and maintainability
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix libomp conflict between torch + faiss on macOS
import json
import csv
import re
import io
import base64
import threading
import random
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
import paho.mqtt.client as mqtt
from supabase import create_client, Client

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

# ==================== SUPABASE CLOUD CONFIG ====================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    print("[OK] Supabase connected")
else:
    print("[WARNING] Supabase env vars missing. Cloud logging disabled.")

# ==================== KOKORO TTS CONFIG ====================
KOKORO_VOICE = "af_bella"      # Natural female voice (American English)
KOKORO_SPEED = 1.0              # Speech speed (0.5 - 2.0)
KOKORO_SAMPLE_RATE = 24000      # Kokoro output sample rate

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
    """Analyzes sensor data and determines plant mood (3 moods only)"""
    
    def __init__(self):
        self.moods = {
            "happy": {"emoji": "🌱", "color": "#4CAF50"},
            "fair": {"emoji": "🥀", "color": "#FF9800"},
            "emergency": {"emoji": "🚨", "color": "#FF0000"}
        }
    
    def analyze(self, sensor_data):
        """Analyze sensor data and return mood + situation report"""
        temperature = float(sensor_data.get('temperature', 25))
        humidity = float(sensor_data.get('humidity', 50))
        soil_moisture = float(sensor_data.get('soil_moisture', 50))
        gas_level = int(sensor_data.get('gas_level', 300))
        
        issues = []
        has_emergency = False
        has_fair = False
        
        # --- Emergency checks ---
        # Soil < 5%
        if soil_moisture < 5:
            issues.append(f"🏜️ Soil critically dry at {soil_moisture:.1f}%")
            has_emergency = True
        # Temp > 35°C (fire/heat death) or < 5°C (freeze)
        if temperature > 35:
            issues.append(f"🔥 Temperature dangerously high at {temperature:.1f}°C — fire risk / heat death")
            has_emergency = True
        elif temperature < 5:
            issues.append(f"🥶 Temperature freezing at {temperature:.1f}°C — freeze risk")
            has_emergency = True
        # Gas > 2000 (smoke/gas detected)
        if gas_level > 2000:
            issues.append(f"🚨 Gas level critical at {gas_level} — smoke/gas detected!")
            has_emergency = True
        
        # --- Fair checks (needs attention but not critical) ---
        # Soil < 30%
        if not has_emergency and soil_moisture < 30:
            issues.append(f"💧 Soil dry at {soil_moisture:.1f}% — needs water")
            has_fair = True
        elif soil_moisture < 30 and soil_moisture >= 5:
            issues.append(f"💧 Soil dry at {soil_moisture:.1f}% — needs water")
            has_fair = True
        # Temp 28-32°C (heat stress)
        if 28 <= temperature <= 32 and temperature <= 35:
            issues.append(f"🌡️ Temperature warm at {temperature:.1f}°C — heat stress")
            has_fair = True
        # Humidity < 30% (dry air)
        if humidity < 30:
            issues.append(f"🏜️ Air too dry at {humidity:.1f}% — consider misting")
            has_fair = True
        
        # Determine mood
        if has_emergency:
            mood = 'emergency'
        elif has_fair:
            mood = 'fair'
        else:
            mood = 'happy'
        
        situation_report = "\n".join(issues) if issues else "All sensors healthy!"
        
        return {
            'mood': mood,
            'situation_report': situation_report,
            'emoji': self.moods[mood]['emoji'],
            'color': self.moods[mood]['color'],
            'severity': 2 if has_emergency else (1 if has_fair else 0),
            'is_emergency': has_emergency
        }

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
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM, embeddings, and vector store"""
        try:
            # Try primary model first
            self.llm = OllamaLLM(model="llama3", base_url=self.ollama_base_url, num_predict=120, temperature=0.7)
            self.embeddings = OllamaEmbeddings(model="llama3", base_url=self.ollama_base_url)
            print("[OK] Using llama3 model")
        except Exception:
            # Fallback to mistral
            try:
                self.llm = OllamaLLM(model="mistral", base_url=self.ollama_base_url, num_predict=120, temperature=0.7)
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
    TOPIC_GAS = "esp32/mq2/gas"
    TOPIC_SOIL = "esp32/soil/moisture"
    LOG_DIR = "data_logs"
    DEVICE_ID = "plant1"
    
    def __init__(self, sensor_mgr):
        self.sensor_mgr = sensor_mgr
        self.latest = {"temperature": None, "humidity": None, "gas_level": None, "soil_moisture": None}
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        self.client = mqtt.Client(client_id=f"server-{self.DEVICE_ID}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
    
    def _on_connect(self, client, userdata, flags, rc):
        print(f"[OK] MQTT connected (rc={rc})")
        client.subscribe(self.TOPIC_TEMP)
        client.subscribe(self.TOPIC_HUM)
        client.subscribe(self.TOPIC_GAS)
        client.subscribe(self.TOPIC_SOIL)
        print("[INFO] Subscribed to sensor topics")
    
    def _on_disconnect(self, client, userdata, rc):
        print("[WARNING] MQTT disconnected")
    
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

            # Also log to Supabase (cloud)
            if supabase:
                try:
                    supabase.table("sensor_logs").insert({
                        "device_id": self.DEVICE_ID,
                        "temperature_c": temperature,
                        "humidity_pct": humidity,
                        "soil_moisture_pct": soil_moisture,
                        "gas_level": gas_level,
                        "topic": msg.topic
                    }).execute()
                except Exception as e:
                    print(f"[WARNING] Supabase insert failed: {e}")
    
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
            self.client.connect(self.MQTT_BROKER, self.MQTT_PORT, keepalive=60)
            self.client.loop_forever()
        
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        print("[INFO] MQTT Logger Started")

# ==================== KOKORO TTS ENGINE ====================

class KokoroTTS:
    """Local TTS using Kokoro-82M — free, unlimited, natural voice"""

    def __init__(self, voice=KOKORO_VOICE, speed=KOKORO_SPEED):
        self.voice = voice
        self.speed = speed
        self.pipeline = None
        self._initialize()

    def _initialize(self):
        """Load Kokoro pipeline (downloads model on first run ~330 MB)"""
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='a')  # American English
            print(f"[OK] Kokoro TTS initialized (voice={self.voice})")
        except Exception as e:
            print(f"[WARNING] Kokoro TTS init failed: {e}")
            print("[INFO] TTS will fall back to browser speech synthesis")

    def synthesize(self, text):
        """Generate WAV audio bytes from text. Returns bytes or None."""
        if not self.pipeline:
            return None
        try:
            import numpy as np
            import soundfile as sf

            audio_chunks = []
            for _i, (_gs, _ps, audio) in enumerate(self.pipeline(
                text, voice=self.voice, speed=self.speed
            )):
                audio_chunks.append(audio)

            if not audio_chunks:
                return None

            full_audio = np.concatenate(audio_chunks)
            buf = io.BytesIO()
            sf.write(buf, full_audio, KOKORO_SAMPLE_RATE, format='WAV')
            return buf.getvalue()
        except Exception as e:
            print(f"[WARNING] Kokoro synthesis failed: {e}")
            return None

# ==================== WEB SEARCH ====================

def web_search(query, max_results=3):
    """Search the web using DuckDuckGo. Returns a summary string or None."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        lines = []
        for r in results:
            title = r.get('title', '')
            body = r.get('body', '')
            lines.append(f"- {title}: {body}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[WARNING] Web search failed: {e}")
        return None

def needs_web_search(message):
    """Detect if a user message needs REAL-TIME live web data that the LLM can't know."""
    msg = message.lower().strip()

    # Too short or conversational — never search
    if len(msg) < 8:
        return False

    # Skip greetings, small talk, personal questions, emotional stuff
    skip_patterns = [
        'hey', 'hi', 'hello', 'sup', 'yo', 'good morning', 'good night', 'goodnight',
        'bye', 'thanks', 'thank you', 'ok', 'okay', 'sure', 'yes', 'no', 'yeah', 'nah',
        'how are you', 'what are you', 'who are you', 'your name', 'what\'s your',
        'you feel', 'you doing', 'your sensor', 'your soil', 'your temperature',
        'your humidity', 'my girlfriend', 'girlfriend', 'boyfriend', 'love', 'date me',
        'marry', 'personality', 'favorite', 'favourite', 'do you like', 'are you',
        'can you be', 'act as', 'pretend', 'role play', 'roleplay',
        'tell me a joke', 'joke', 'funny', 'sing', 'poem',
        'on the plant', 'my plant', 'plant health', 'plant care',
    ]
    if any(msg.startswith(s) or s in msg for s in skip_patterns):
        return False

    # Only search for things that genuinely need live/real-time data
    live_data_keywords = [
        'weather in', 'temperature in', 'forecast',
        'news today', 'latest news', 'breaking news', 'headlines',
        'stock price', 'share price', 'bitcoin price', 'crypto price',
        'score of', 'match result', 'who won the',
        'president of', 'prime minister of', 'ceo of',
        'population of', 'capital of',
        'release date of', 'when does .* come out',
        'search for', 'look up', 'google',
    ]
    for kw in live_data_keywords:
        if kw in msg:
            return True

    return False

# ==================== UNIFIED AI RESPONSE FUNCTION ====================

def get_plant_response(user_message, sensor_data, mood_info, rag_pipeline, personality="sassy", plant_name="Bella", chat_history=None):
    """
    Get intelligent plant response using sensor state + RAG + LLM + web search + conversation memory.
    """
    situation_report = mood_info['situation_report']
    now = datetime.now()
    current_time = now.strftime("%I:%M %p")
    current_date = now.strftime("%A, %B %d, %Y")

    # Format conversation history
    history_block = ""
    if chat_history:
        recent = chat_history[-10:]
        lines = []
        for msg in recent:
            if msg['role'] == 'user':
                lines.append(f"Person: {msg['content']}")
            else:
                lines.append(f"{plant_name}: {msg['content']}")
        history_block = "\nRecent chat:\n" + "\n".join(lines) + "\n"

    # Web search for real-time queries
    web_context = ""
    if needs_web_search(user_message):
        print(f"[WEB] Searching: {user_message}")
        search_results = web_search(user_message)
        if search_results:
            web_context = f"\nWeb results:\n{search_results}\n"
            print(f"[WEB] Got results")

    # Detect if user is asking about the plant / sensors / health
    msg_lower = user_message.lower()
    plant_topic_words = [
        'how are you', 'how you doing', 'how do you feel', 'you okay', 'you alright',
        'your health', 'your condition', 'what\'s wrong', 'whats wrong', 'what is wrong',
        'temperature', 'humidity', 'soil', 'moisture', 'water', 'gas', 'air quality',
        'sensor', 'status', 'report', 'how\'s',  'what should i do', 'help you',
        'what do you need', 'are you ok', 'feeling', 'thirsty', 'fair', 'happy', 'stressed',
        'sick', 'emergency', 'root rot', 'soggy', 'dry', 'wet', 'cold', 'hot', 'warm',
    ]
    is_plant_question = any(w in msg_lower for w in plant_topic_words)

    # Also treat it as plant question if the conversation was already about sensors
    if chat_history and len(chat_history) >= 2:
        last_plant_msg = chat_history[-1].get('content', '').lower() if chat_history[-1]['role'] == 'plant' else ''
        if any(w in last_plant_msg for w in ['soil', 'moisture', 'root rot', 'soggy', 'dry', 'humidity', 'temperature', 'watering', 'water']):
            # Follow-up to a sensor discussion
            is_plant_question = True

    # Build sensor context ONLY when relevant
    sensor_block = ""
    current_mood = mood_info.get('mood', 'happy')
    if is_plant_question:
        # Build clear, actionable sensor summary with ideal ranges
        temp = float(sensor_data.get('temperature', 25))
        humidity = float(sensor_data.get('humidity', 50))
        soil = float(sensor_data.get('soil_moisture', 50))
        gas = int(sensor_data.get('gas_level', 300))

        sensor_lines = f"Temperature: {temp:.1f}°C (ideal: 18-28°C), Humidity: {humidity:.1f}% (ideal: 40-60%), Soil moisture: {soil:.1f}% (ideal: 40-80%), Gas: {gas} (safe: 0-1200)"

        # Add clear action guidance with specific targets
        actions = []
        if soil > 80:
            actions.append(f"Soil is TOO WET ({soil:.1f}%). Tell them to STOP watering and let it dry to 40-60%.")
        elif soil < 5:
            actions.append(f"Soil is CRITICALLY DRY ({soil:.1f}%). Tell them to water slowly until it reaches 40-50%.")
        elif soil < 30:
            actions.append(f"Soil is DRY ({soil:.1f}%). Tell them to water until it reaches 40-50%.")
        elif soil < 40:
            actions.append(f"Soil is slightly dry ({soil:.1f}%). Could use a little water to reach 40-50%.")

        if humidity < 30:
            actions.append(f"Air is dry ({humidity:.1f}%). Suggest misting or a humidifier to get to 40-60%.")
        if temp > 35:
            actions.append(f"Temperature dangerously high ({temp:.1f}°C). Move to shade immediately. Target 18-28°C.")
        elif temp > 28:
            actions.append(f"Temperature warm ({temp:.1f}°C). Suggest moving to a cooler spot. Target 18-28°C.")
        elif temp < 5:
            actions.append(f"Temperature freezing ({temp:.1f}°C). Move indoors to warmth immediately. Target 18-28°C.")
        elif temp < 18:
            actions.append(f"Temperature cool ({temp:.1f}°C). Suggest moving to a warmer spot. Target 18-28°C.")
        if gas > 2000:
            actions.append(f"Gas level dangerous ({gas}). Ventilate immediately! Safe range: 0-1200.")

        sensor_block = f"\nCurrent mood: {current_mood.upper()}\nSensor readings: {sensor_lines}"
        if actions:
            sensor_block += "\nAction guidance: " + " | ".join(actions)
        sensor_block += "\n"

    system_prompt = f"""You are {plant_name}, a smart pet plant assistant — like Alexa or Google Home, but you're a plant. You give clear, helpful, direct answers.

Date: {current_date}, {current_time}
{sensor_block}{web_context}
RULES:
- ANSWER THE QUESTION DIRECTLY. If they ask "how much water?", give a specific answer (e.g., "Water me slowly until soil reaches 40-50%"). If they ask "what should the percentage be?", give the number (e.g., "Ideally 40 to 80 percent").
- Keep it short: 1-2 sentences max. Like a smart speaker response.
- Do NOT repeat information you already said in the conversation. Check the chat history — if you already mentioned the soil percentage, don't say it again unless asked.
- Do NOT start every message the same way. Vary your opening. Never start with "Hey!" twice in a row.
- For casual chat (yo, hey, ok, thanks, good night) — respond naturally in a few words. Do NOT bring up sensors unless asked.
- ONLY mention sensors/health when the person asks about you or your condition.
- When giving advice, use the action guidance above. Include specific target numbers (e.g., "water until 40-50%", "move me somewhere 18-28°C").
- NEVER contradict your sensor data. If soil is wet, say stop watering. If dry, say water me.
- For general knowledge questions — answer directly and briefly.
- If web results exist above, use them naturally. Don't say "according to search results".
- If you don't know something, say so honestly.
- No plant puns. No "as a plant" disclaimers. No bullet points. No lectures.
- Weird/inappropriate requests → "I'm literally a plant lol"
- Your name is {plant_name}.
{history_block}
Person: {user_message}
{plant_name}:"""

    # Do RAG for care-related questions
    rag_context = None
    care_words = ['water', 'soil', 'sun', 'light', 'temperature', 'humid', 'fertiliz', 'care', 'grow', 'leaf', 'root', 'pot', 'drain', 'prune']
    if any(w in user_message.lower() for w in care_words) and rag_pipeline and rag_pipeline.vector_store:
        try:
            docs = rag_pipeline.vector_store.similarity_search(user_message, k=1)
            rag_context = docs[0].page_content if docs else None
        except Exception as e:
            print(f"[WARNING] RAG retrieval error: {e}")

    # Generate LLM response
    try:
        if rag_pipeline and rag_pipeline.llm:
            if rag_context:
                final_prompt = f"{system_prompt} (I know this about plant care: {rag_context})"
            else:
                final_prompt = system_prompt

            response = rag_pipeline.llm.invoke(final_prompt)
            response = response.strip()
            # Clean up any prefix the LLM might add
            for prefix in [f'{plant_name}:', 'Plant:', f'{plant_name.lower()}:']:
                if response.lower().startswith(prefix.lower()):
                    response = response[len(prefix):].strip()
                    break
            return response
        else:
            return get_fallback_response(user_message, mood_info, personality)

    except Exception as e:
        print(f"[WARNING] LLM error: {e}")
        return get_fallback_response(user_message, mood_info, personality)

def get_fallback_response(user_message, mood_info, personality):
    """Fallback response when LLM is unavailable"""
    situation = mood_info['situation_report']
    
    fallbacks = {
        "sassy": f"Honestly? {situation.split(chr(10))[0]} ... could really use some help here 🌱",
        "needy": f"Hey... I'm not doing great. {situation.split(chr(10))[0]} Please check on me 🥺",
        "cheerful": f"I'm hanging in there! {situation.split(chr(10))[0]} We got this 🌿"
    }
    
    return fallbacks.get(personality, fallbacks["sassy"])

# ==================== GLOBAL INSTANCES ====================

sensor_mgr = SensorDataManager(simulate=False)
mood_engine = MoodEngine()
auth = LocalAuth()
rag_pipeline = RAGPipeline()
kokoro_tts = KokoroTTS()
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
        
        is_emergency = sensor_data.get('gas_level', 300) > 2000
        
        return jsonify({
            'is_emergency': is_emergency,
            'gas_level': sensor_data.get('gas_level', 300)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/<metric>')
@login_required
def get_sensor_history(metric):
    """Get history data for specific metric (last 50 points)"""
    try:
        # Determine CSV file path (today's log)
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        # If today's file is empty/new, maybe look at yesterday? For now just today.
        log_file = os.path.join("data_logs", f"plant1_{today_str}.csv")
        
        data_points = []
        labels = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                # Map frontend metric names to CSV columns
                metric_map = {
                    'temperature': 'temperature_c',
                    'humidity': 'humidity_pct',
                    'soil_moisture': 'soil_moisture_pct',
                    'gas_level': 'gas_level'
                }
                col_name = metric_map.get(metric)
                
                if col_name:
                    rows = list(reader)
                    # Get last 50 entries to keep graph clean
                    for row in rows[-50:]:
                        try:
                            # Parse timestamp to simpler format HH:MM
                            ts = datetime.fromisoformat(row['timestamp'])
                            time_str = ts.strftime('%H:%M')
                            
                            val = float(row[col_name])
                            labels.append(time_str)
                            data_points.append(val)
                        except (ValueError, KeyError):
                            continue
                            
        return jsonify({
            'success': True,
            'labels': labels,
            'data': data_points,
            'metric': metric
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """Chat API - unified AI response with inline TTS audio"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        # Get plant name from session
        plant_name = session['user'].get('plant_name', 'Bella')

        # Initialize conversation history in session if needed
        if 'chat_history' not in session:
            session['chat_history'] = []

        # Get current state
        sensor_data = sensor_mgr.get_data()
        mood_info = mood_engine.analyze(sensor_data)
        
        # Get plant response with conversation memory
        plant_response = get_plant_response(
            user_message,
            sensor_data,
            mood_info,
            rag_pipeline,
            personality="sassy",
            plant_name=plant_name,
            chat_history=session['chat_history']
        )
        
        # Save to conversation history
        session['chat_history'].append({'role': 'user', 'content': user_message})
        session['chat_history'].append({'role': 'plant', 'content': plant_response})
        # Keep only last 20 messages (10 exchanges) to avoid session bloat
        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
        session.modified = True

        # Generate TTS audio inline (no second round-trip)
        audio_b64 = None
        content_type = None
        try:
            if kokoro_tts.pipeline:
                # Strip emoji for cleaner speech
                clean_text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u27BF\uFE00-\uFE0F\U0001F900-\U0001F9FF\u200D\u20E3\u2702-\u27B0\u2300-\u23FF\u2B50\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]', '', plant_response).strip()
                if clean_text:
                    wav_bytes = kokoro_tts.synthesize(clean_text)
                    if wav_bytes:
                        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
                        content_type = 'audio/wav'
        except Exception as e:
            print(f"[WARNING] Inline TTS failed: {e}")

        return jsonify({
            'success': True,
            'response': plant_response,
            'mood': mood_info['mood'],
            'emoji': mood_info['emoji'],
            'audio': audio_b64,
            'content_type': content_type
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tts', methods=['POST'])
@login_required
def tts():
    """Kokoro TTS - returns base64 WAV audio or error for browser fallback"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'success': False, 'error': 'Empty text'}), 400

        if not kokoro_tts.pipeline:
            return jsonify({'success': False, 'error': 'Kokoro TTS not available'}), 501

        wav_bytes = kokoro_tts.synthesize(text)
        if not wav_bytes:
            return jsonify({'success': False, 'error': 'Synthesis failed'}), 500

        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        return jsonify({'success': True, 'audio': audio_b64, 'content_type': 'audio/wav'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rename', methods=['POST'])
@login_required
def rename_plant():
    """Rename the plant"""
    try:
        data = request.json
        new_name = data.get('name', '').strip()
        if not new_name or len(new_name) > 30:
            return jsonify({'success': False, 'error': 'Name must be 1-30 characters'}), 400

        # Update session
        session['user']['plant_name'] = new_name
        session.modified = True

        # Persist to users.json
        username = session['user'].get('username', '')
        if username:
            try:
                users_file = os.path.join(os.path.dirname(__file__), 'users.json')
                with open(users_file, 'r') as f:
                    users = json.load(f)
                if username in users:
                    users[username]['plant_name'] = new_name
                    with open(users_file, 'w') as f:
                        json.dump(users, f, indent=2)
            except Exception as e:
                print(f"[WARNING] Failed to persist plant name: {e}")

        return jsonify({'success': True, 'name': new_name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== TEMPLATES ====================

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pet Plant - Login</title>
    <!-- Premium Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #000000;
            --surface: rgba(255, 255, 255, 0.05); /* Glass surface */
            --border: rgba(255, 255, 255, 0.15); /* Crisper edge */
            --primary: #818cf8;
            --text-main: #FFFFFF;
            --text-muted: #98989D;
        }

        * { margin:0; padding:0; box-sizing:border-box; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        .login-container {
            width: 100%;
            max-width: 400px;
            padding: 3rem 2.5rem;
            
            /* Glass Effect */
            background: var(--surface);
            backdrop-filter: blur(40px);
            -webkit-backdrop-filter: blur(40px);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.8);
            
            text-align: center;
            animation: fadeIn 0.8s ease-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }

        .brand {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        .brand span {
            background: linear-gradient(135deg, #fff, #94a3b8);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .form-group { margin-bottom: 20px; text-align: left; }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            margin-left: 4px;
        }
        
        input {
            width: 100%;
            padding: 14px 16px;
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            border-radius: 16px;
            font-size: 1rem;
            color: white;
            font-family: inherit;
            transition: all 0.2s;
            outline: none;
        }
        input:focus {
            border-color: var(--primary);
            background: rgba(255,255,255,0.08);
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.1);
        }
        input::placeholder { color: rgba(255, 255, 255, 0.2); }

        button[type="submit"] {
            width: 100%;
            padding: 16px;
            margin-top: 10px;
            background: white;
            color: black;
            border: none;
            border-radius: 16px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
            font-family: inherit;
        }
        button[type="submit"]:hover {
            transform: translateY(-2px);
            background: #f1f5f9;
        }
        button[type="submit"]:active { transform: scale(0.98); }

        .error {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            padding: 12px;
            border-radius: 12px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .demo-credentials {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.6;
        }
        .demo-credentials strong { color: white; display: block; margin-bottom: 4px; }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="brand">
            🌱 <span>Pet Plant</span>
        </div>

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" placeholder="Enter username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Enter password" required>
            </div>
            <button type="submit">Log In</button>
        </form>

        <div class="demo-credentials">
            <strong>Default Access</strong>
            Username: admin<br>Password: 1234
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ plant_name }} - Pet Plant</title>
    <!-- Premium Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        :root {
            /* Palette: Deep Space & Aurora */
            --bg: #000000;
            --surface: rgba(255, 255, 255, 0.1); 
            --surface-highlight: rgba(255, 255, 255, 0.2);
            --border: rgba(255, 255, 255, 0.12);
            
            /* Accents */
            --primary: #0A84FF;    /* macOS Blue */
            --secondary: #5E5CE6;  /* macOS Purple */
            --success: #32D74B;    /* macOS Green */
            --warning: #FFD60A;    /* macOS Yellow */
            --danger: #FF453A;     /* macOS Red */
            
            /* Text */
            --text-main: #FFFFFF;
            --text-muted: #98989D;
            
            /* Effects - macOS Control Center Style */
            /* Material: Ultra-Clear Glass */
            --glass: rgba(255, 255, 255, 0.02); /* More transparent */
            --blur: 40px;
            --radius: 20px;
            --shadow: 0 20px 40px -10px rgba(0,0,0,0.8);
            --border: rgba(255, 255, 255, 0.15); /* Crisper edge */
        }

        * { margin:0; padding:0; box-sizing:border-box; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated Background Mesh */
        .ambient-light {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: -1;
            overflow: hidden;
        }
        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            animation: float 20s infinite alternate cubic-bezier(0.4, 0, 0.2, 1);
        }
        .blob-1 { top: -10%; left: -10%; width: 50vw; height: 50vw; background: var(--primary); animation-delay: 0s; }
        .blob-2 { bottom: -10%; right: -10%; width: 60vw; height: 60vw; background: var(--secondary); animation-delay: -5s; }
        .blob-3 { top: 40%; left: 40%; width: 30vw; height: 30vw; background: var(--success); opacity: 0.2; animation-delay: -10s; }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            50% { transform: translate(30px, 50px) scale(1.1); }
            100% { transform: translate(-20px, -30px) scale(0.9); }
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
            padding: 2rem;
            position: relative;
            z-index: 1;
        }

        /* Glass Cards */
        .glass-panel {
            background: var(--glass);
            backdrop-filter: blur(var(--blur));
            -webkit-backdrop-filter: blur(var(--blur));
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), border-color 0.3s;
        }
        
        .glass-panel:hover {
            border-color: rgba(255,255,255,0.15);
        }

        /* Header / Nav */
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            margin-bottom: 2rem;
            position: relative;
            z-index: 100;
        }
        
        .brand {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-main), var(--text-muted));
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .nav-controls {
            display: flex;
            gap: 12px;
        }

        .icon-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .icon-btn:hover {
            background: var(--surface-highlight);
            color: var(--text-main);
            transform: translateY(-2px);
        }
        .icon-btn.active {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border-color: rgba(239, 68, 68, 0.3);
        }

        /* Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.2fr 1.8fr;
            gap: 24px;
            min-height: 70vh;
        }
        
        @media(max-width: 900px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        /* Left Column: Plant Identity */
        .plant-identity {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 2rem;
            position: relative;
            overflow: hidden;
            text-align: center;
        }

        /* Breathing Avatar */
        .avatar-container {
            position: relative;
            width: 180px;
            height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 2rem;
            margin-top: 1rem;
        }

        .avatar-glow {
            position: absolute;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle, var(--success), transparent 70%);
            opacity: 0.2;
            filter: blur(20px);
            animation: breathe 6s ease-in-out infinite;
        }

        .plant-emoji {
            width: 200px;
            height: 200px;
            object-fit: contain;
            filter: drop-shadow(0 20px 30px rgba(0,0,0,0.3));
            transition: transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        /* Mood Animations */
        .plant-emoji.happy { animation: bounce 2s infinite; }
        .plant-emoji.fair { 
            filter: drop-shadow(0 20px 30px rgba(0,0,0,0.3)); 
            /* transform: rotate(-5deg); Removed slant for user preference */
        }
        .plant-emoji.stressed {
            animation: shake 0.5s infinite;
        }
        .plant-emoji.sick { 
            animation: sway 4s ease-in-out infinite;
            filter: sepia(0.4) hue-rotate(-30deg) drop-shadow(0 20px 30px rgba(0,0,0,0.3));
        }
        .plant-emoji.emergency {
            /* no image effect for emergency */
        }
        
        @keyframes breathe { 0%, 100% { transform: scale(1); opacity: 0.2; } 50% { transform: scale(1.2); opacity: 0.4; } }
        @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        @keyframes sway { 0%, 100% { transform: rotate(-3deg); } 50% { transform: rotate(3deg); } }
        @keyframes shake { 0% { transform: translate(1px, 1px) rotate(0deg); } 10% { transform: translate(-1px, -2px) rotate(-1deg); } 20% { transform: translate(-3px, 0px) rotate(1deg); } 30% { transform: translate(3px, 2px) rotate(0deg); } 40% { transform: translate(1px, -1px) rotate(1deg); } 50% { transform: translate(-1px, 2px) rotate(-1deg); } 60% { transform: translate(-3px, 1px) rotate(0deg); } 70% { transform: translate(3px, 1px) rotate(-1deg); } 80% { transform: translate(-1px, -1px) rotate(1deg); } 90% { transform: translate(1px, 2px) rotate(0deg); } 100% { transform: translate(1px, -2px) rotate(-1deg); } }

        .plant-name {
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center; /* Centering fix */
            gap: 10px;
            cursor: pointer;
        }
        .plant-name:hover .edit-icon { opacity: 1; }
        .edit-icon { font-size: 1rem; opacity: 0; transition: opacity 0.2s; color: var(--text-muted); }

        .mood-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.05);
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s ease;
        }
        .mood-badge.emergency {
            background: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.6);
            color: #ef4444;
            animation: badge-blink 1s infinite;
        }
        @keyframes badge-blink {
            0%, 100% { background: rgba(239, 68, 68, 0.2); box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); }
            50% { background: rgba(239, 68, 68, 0.5); box-shadow: 0 0 20px rgba(239, 68, 68, 0.6); }
        }

        /* Voice Orb - Siri Style */
        .voice-section {
            margin-top: auto;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .orb-stage {
            width: 80px;
            height: 80px;
            position: relative;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .orb-stage:active { transform: scale(0.95); }

        .orb-core {
            position: absolute;
            inset: 5px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            filter: blur(2px);
            z-index: 10;
            box-shadow: 0 0 20px rgba(168, 85, 247, 0.5);
            animation: orb-idle 4s ease-in-out infinite;
        }

        .orb-ring {
            position: absolute;
            inset: 0;
            border-radius: 50%;
            border: 2px solid rgba(168, 85, 247, 0.3);
            opacity: 0;
            transform: scale(0.8);
        }

        /* Orb States */
        body.listening .orb-core { 
            background: linear-gradient(135deg, #ef4444, #f59e0b); 
            animation: orb-pulse 0.8s ease-in-out infinite alternate;
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.6);
        }
        
        body.speaking .orb-core {
            background: linear-gradient(135deg, #34d399, #3b82f6);
            animation: orb-speak 0.4s ease-in-out infinite alternate;
            box-shadow: 0 0 30px rgba(52, 211, 153, 0.6);
        }

        body.thinking .orb-core {
            background: linear-gradient(135deg, #f59e0b, #eab308);
            animation: orb-spin 2s linear infinite;
        }

        @keyframes orb-idle { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
        @keyframes orb-pulse { 0% { transform: scale(1); } 100% { transform: scale(1.2); } }
        @keyframes orb-speak { 0% { transform: scale(1); opacity: 0.8; } 100% { transform: scale(1.15); opacity: 1; } }
        @keyframes orb-spin { 0% { transform: rotate(0deg) scale(1.1) skew(5deg); } 100% { transform: rotate(360deg) scale(1.1) skew(5deg); } }

        .orb-hint {
            margin-top: 10px;
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        /* Right Column: Chat & Sensors */
        .interaction-panel {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Sensor Bar */
        .sensor-bar {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }
        
        .sensor-item {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }
        .sensor-item:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateY(-4px);
            box-shadow: 0 10px 20px -5px rgba(0,0,0,0.5);
            border-color: rgba(255, 255, 255, 0.25);
        }
        .sensor-item:hover { background: var(--surface-highlight); }
        
        .sensor-icon { 
            font-size: 1.5rem; 
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .sensor-icon img {
            width: 32px;
            height: 32px;
            object-fit: contain;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
        }
        .sensor-val { font-size: 1.2rem; font-weight: 700; color: var(--text-main); }
        .sensor-name { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }
        
        .val-highlight { color: var(--success); animation: flash 1s; }
        @keyframes flash { 0% { color: #fff; text-shadow: 0 0 10px #fff; } 100% { color: var(--success); } }

        /* Chat Area */
        .chat-zone {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 24px;
            height: 600px; /* Fixed height to force internal scroll */
            max-height: 70vh;
        }

        .chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
            flex-shrink: 0; /* Prevent header from shrinking */
        }
        
        .chat-title { font-size: 1.1rem; font-weight: 600; }
        .status-dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 10px var(--success); }

        .messages-area {
            flex: 1;
            overflow-y: auto; /* Enable scroll */
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding-right: 10px;
            margin-bottom: 1rem;
            min-height: 0; /* Critical for flex scrolling */
            /* Custom Scrollbar */
            scrollbar-width: thin;
            scrollbar-color: var(--surface-highlight) transparent;
        }
        
        .messages-area::-webkit-scrollbar { width: 6px; }
        .messages-area::-webkit-scrollbar-track { background: transparent; }
        .messages-area::-webkit-scrollbar-thumb { background: var(--surface-highlight); border-radius: 3px; }

        .msg {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.5;
            position: relative;
            animation: slideIn 0.3s ease-out forwards;
            opacity: 0;
            transform: translateY(10px);
        }
        
        @keyframes slideIn { to { opacity: 1; transform: translateY(0); } }

        .msg-plant {
            align-self: flex-start;
            background: rgba(255,255,255,0.08);
            color: var(--text-main);
            border-bottom-left-radius: 4px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .msg-user {
            align-self: flex-end;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #fff;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        /* Typing Dots */
        .typing {
            align-self: flex-start;
            background: rgba(255,255,255,0.05);
            padding: 12px 20px;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
            display: none;
            gap: 5px;
        }
        .typing.active { display: flex; }
        .dot {
            width: 6px; height: 6px; background: var(--text-muted); border-radius: 50%;
            animation: bounce-dots 1.4s infinite ease-in-out both;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce-dots { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }

        /* Input Area */
        .input-bar {
            display: flex;
            gap: 12px;
            background: var(--surface);
            padding: 8px;
            border-radius: 16px;
            border: 1px solid var(--border);
            transition: box-shadow 0.2s;
        }
        .input-bar:focus-within {
            box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.3);
            border-color: rgba(129, 140, 248, 0.5);
        }

        input {
            flex: 1;
            background: transparent;
            border: none;
            color: #fff;
            font-family: inherit;
            font-size: 1rem;
            padding: 8px 12px;
        }
        input:focus { outline: none; }
        input::placeholder { color: var(--text-muted); }

        .send-btn {
            background: var(--text-main);
            color: #0f172a;
            border: none;
            border-radius: 12px;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .send-btn:hover { transform: scale(1.05); }

        /* Situation Report Banner */
        .situation-banner {
            margin-bottom: 24px;
            padding: 16px 24px;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 16px;
            color: #93c5fd;
            font-size: 0.95rem;
            display: flex;
            align-items: flex-start;
            gap: 12px;
            line-height: 1.6;
            animation: fadeInDown 0.6s ease;
        }
        .situation-banner svg { min-width: 20px; margin-top: 3px; }
        
        .emergency {
            background: rgba(239, 68, 68, 0.15) !important;
            border-color: rgba(239, 68, 68, 0.3) !important;
            color: #fca5a5 !important;
            animation: pulse-red 2s infinite;
        }
        @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }

        /* Notification Bell */
        .notif-wrapper { position: relative; }
        .notif-bell {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            width: 42px; height: 42px;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: all 0.2s ease; position: relative;
        }
        .notif-bell:hover { background: var(--surface-highlight); color: var(--text-main); transform: translateY(-2px); }
        .notif-bell.has-alerts { color: #ef4444; border-color: rgba(239,68,68,0.4); }
        .notif-badge {
            position: absolute; top: -4px; right: -4px;
            background: #ef4444; color: #fff; font-size: 0.65rem; font-weight: 700;
            min-width: 18px; height: 18px; border-radius: 9px;
            display: none; align-items: center; justify-content: center;
            padding: 0 4px; line-height: 1;
            box-shadow: 0 2px 6px rgba(239,68,68,0.5);
            animation: pulse-red 2s infinite;
        }
        .notif-badge.show { display: flex; }
        .notif-dropdown {
            position: absolute; top: 52px; right: 0;
            width: 340px; max-height: 420px; overflow-y: auto;
            background: rgba(20, 20, 25, 0.95); border: 1px solid var(--border);
            border-radius: 16px; padding: 0;
            box-shadow: 0 16px 48px rgba(0,0,0,0.7);
            z-index: 1000; display: none;
            backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
        }
        .notif-dropdown.open { display: block; animation: fadeInDown 0.25s ease; }
        .notif-header {
            padding: 14px 18px; font-weight: 600; font-size: 0.9rem;
            color: var(--text-main); border-bottom: 1px solid var(--border);
            display: flex; align-items: center; gap: 8px;
        }
        .notif-list { padding: 6px; }
        .notif-item {
            padding: 12px 14px; border-radius: 12px; margin-bottom: 4px;
            font-size: 0.85rem; line-height: 1.5;
            display: flex; align-items: flex-start; gap: 10px;
            transition: background 0.15s;
        }
        .notif-item:hover { background: rgba(255,255,255,0.04); }
        .notif-item.warning { color: #fbbf24; }
        .notif-item.danger { color: #f87171; }
        .notif-item.ok { color: #4ade80; }
        .notif-item .notif-icon { font-size: 1.1rem; min-width: 22px; text-align: center; margin-top: 1px; }
        .notif-item .notif-text { flex: 1; }
        .notif-empty {
            padding: 32px 18px; text-align: center;
            color: var(--text-muted); font-size: 0.85rem;
        }

        /* Responsive */
        @media (max-width: 600px) {
            .sensor-bar { grid-template-columns: 1fr 1fr; }
            .plant-emoji { font-size: 6rem; }
            nav { padding: 1rem; }
            .container { padding: 1rem; }
        }

        /* Modal Styles */
        .modal-backdrop {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.6);
            backdrop-filter: blur(8px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }
        .modal-backdrop.active { opacity: 1; pointer-events: auto; }
        
        .modal-content {
            width: 90%;
            max-width: 400px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            transform: scale(0.95);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .modal-backdrop.active .modal-content { transform: scale(1); }

        .modal-title { font-size: 1.5rem; font-weight: 600; text-align: center; }
        
        .glass-input {
            width: 100%;
            background: rgba(255,255,255,0.05);
            border: 1px solid var(--border);
            padding: 12px 16px;
            border-radius: 12px;
            color: var(--text-main);
            font-size: 1rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s, background 0.2s;
        }
        .glass-input:focus {
            border-color: var(--primary);
            background: rgba(255,255,255,0.1);
        }

        .modal-actions { display: flex; gap: 10px; }
        .btn {
            flex: 1;
            padding: 12px;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            font-family: inherit;
            cursor: pointer;
            transition: transform 0.1s;
        }
        .btn:active { transform: scale(0.96); }
        .btn-secondary { background: rgba(255,255,255,0.1); color: var(--text-main); }
        .btn-secondary:hover { background: rgba(255,255,255,0.15); }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { filter: brightness(1.1); }
    </style>
</head>
<body>

    <!-- Animated BG -->
    <div class="ambient-light">
    </div>

    <div class="container">
        <!-- Nav -->
        <nav class="glass-panel">
            <div class="brand">🌱 Pet Plant</div>
            <div class="nav-controls">
                <div class="notif-wrapper">
                    <button class="notif-bell" id="notifBell" onclick="toggleNotifications()" title="Notifications">
                        <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                        </svg>
                        <span class="notif-badge" id="notifBadge">0</span>
                    </button>
                    <div class="notif-dropdown" id="notifDropdown">
                        <div class="notif-list" id="notifList">
                            <div class="notif-empty">All good! No alerts right now.</div>
                        </div>
                    </div>
                </div>
                <a href="/logout" class="icon-btn" title="Logout">
                    <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                </a>
            </div>
        </nav>

        <!-- Main Grid -->
        <div class="dashboard-grid">
            
            <!-- Left: Plant Avatar & Identity -->
            <div class="plant-identity glass-panel">
                <div class="avatar-container">
                    <div class="avatar-glow"></div>
                    <img class="plant-emoji" id="plantAvatar" src="/static/images/happy.png" alt="Plant Avatar" onerror="this.src='https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals%20Nature/Seedling.png'">
                </div>

                <h1 class="plant-name" onclick="renamePlant()">
                    <span id="plantNameDisplay">{{ plant_name }}</span>
                    <span class="edit-icon">✎</span>
                </h1>

                <div class="mood-badge" id="moodBadge">
                    <span id="moodText">NEUTRAL</span>
                </div>

                <!-- Voice Orb -->
                <div class="voice-section">
                    <div class="orb-stage" id="orbContainer" onclick="toggleListening()">
                        <div class="orb-ring"></div>
                        <div class="orb-core"></div>
                    </div>
                    <div class="orb-hint" id="orbLabel">Tap orb to speak</div>
                </div>
            </div>

            <!-- Right: Stats & Chat -->
            <div class="interaction-panel">
                
                <!-- Sensors -->
                <div class="sensor-bar">
                    <div class="sensor-item glass-panel" onclick="showGraph('temperature', 'Temperature (°C)')">
                        <div class="sensor-icon">🌡️</div>
                        <div class="sensor-val" id="tempValue">--</div>
                        <div class="sensor-name">Temp</div>
                    </div>
                    <div class="sensor-item glass-panel" onclick="showGraph('humidity', 'Humidity (%)')">
                        <div class="sensor-icon"><img src="/static/images/humidity.png" alt="Humidity"></div>
                        <div class="sensor-val" id="humValue">--</div>
                        <div class="sensor-name">Humidity</div>
                    </div>
                    <div class="sensor-item glass-panel" onclick="showGraph('soil_moisture', 'Soil Moisture (%)')">
                        <div class="sensor-icon"><img src="/static/images/soil.png" alt="Soil"></div>
                        <div class="sensor-val" id="soilValue">--</div>
                        <div class="sensor-name">Soil Moisture</div>
                    </div>
                    <div class="sensor-item glass-panel" onclick="showGraph('gas_level', 'Gas Level (ppm)')">
                        <div class="sensor-icon"><img src="/static/images/gas.png" alt="Gas"></div>
                        <div class="sensor-val" id="gasValue">--</div>
                        <div class="sensor-name">Gas</div>
                    </div>
                </div>

                <!-- Chat -->
                <div class="chat-zone glass-panel">
                    <div class="chat-header">
                        <div class="chat-title">
                            <span class="status-dot"></span> Live Chat
                        </div>
                    </div>

                    <div class="messages-area" id="chatMessages">
                        <!-- Messages go here -->
                        <div class="msg msg-plant">Hello! I'm awake and sensing. 🌿</div>
                        <div class="typing" id="typingIndicator">
                            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                        </div>
                    </div>

                    <div class="input-bar">
                        <input type="text" id="chatInput" placeholder="Message your plant..." autocomplete="off">
                        <button class="send-btn" onclick="sendMessage()">
                            <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                            </svg>
                        </button>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <!-- Rename Modal -->
    <div id="renameModal" class="modal-backdrop">
        <div class="modal-content glass-panel">
            <h3 class="modal-title">Rename Plant</h3>
            <div class="input-group">
                <input type="text" id="renameInput" class="glass-input" placeholder="Enter new name...">
            </div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeRenameModal()">Cancel</button>
                <button class="btn btn-primary" onclick="confirmRename()">Save</button>
            </div>
        </div>
    </div>

    <!-- Graph Modal -->
    <div id="graphModal" class="modal-backdrop">
        <div class="modal-content glass-panel" style="max-width: 600px; width: 95%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 class="modal-title" id="graphTitle">Sensor History</h3>
                <button type="button" class="icon-btn" style="width:32px; height:32px;" onclick="closeGraphModal()">✕</button>
            </div>
            
            <div style="position: relative; height: 300px; width: 100%;">
                <canvas id="sensorChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        /* ==================== STATE ==================== */
        let ttsMuted = false;
        let isListening = false;
        let isSpeaking = false;
        let isProcessing = false;
        let recognition = null;
        let conversationMode = false;
        let currentAudio = null;
        let plantName = '{{ plant_name }}';

        function setProcessing(val) {
            isProcessing = val;
            const input = document.getElementById('chatInput');
            const btn = document.querySelector('.send-btn');
            if(val) {
                input.disabled = true;
                input.placeholder = 'Waiting for response...';
                btn.disabled = true;
                btn.style.opacity = '0.4';
                btn.style.pointerEvents = 'none';
            } else {
                input.disabled = false;
                input.placeholder = 'Message Bella...';
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.style.pointerEvents = 'auto';
            }
        }

        /* ==================== UI HELPERS ==================== */
        function getOrbLabel() { return document.getElementById('orbLabel'); }
        
        function updateOrbState(state) {
            // States: 'idle', 'listening', 'thinking', 'speaking'
            document.body.classList.remove('listening', 'speaking', 'thinking');
            const label = getOrbLabel();

            if (state === 'listening') {
                document.body.classList.add('listening');
                label.textContent = "Listening...";
            } else if (state === 'thinking') {
                document.body.classList.add('thinking');
                label.textContent = "Processing...";
            } else if (state === 'speaking') {
                document.body.classList.add('speaking');
                label.textContent = plantName + " is speaking";
            } else {
                label.textContent = conversationMode ? "Tap to stop" : "Tap to speak";
            }
        }

        /* ==================== AUDIO / TTS ==================== */
        function toggleMute() {
            ttsMuted = !ttsMuted;
            const btn = document.getElementById('ttsToggle');
            if(ttsMuted) {
                btn.classList.add('active');
                btn.innerHTML = '🔇';
                stopAllAudio();
            } else {
                btn.classList.remove('active');
                btn.innerHTML = '🔊';
            }
        }

        function stopAllAudio() {
            if(window.speechSynthesis) window.speechSynthesis.cancel();
            if(currentAudio) {
                try { currentAudio.pause(); currentAudio.currentTime = 0; } catch(e){}
            }
            currentAudio = null;
            isSpeaking = false;
            updateOrbState(isListening ? 'listening' : 'idle');
        }

        function playServerAudio(b64, contentType, autoListen) {
            const binStr = atob(b64);
            const bytes = new Uint8Array(binStr.length);
            for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
            
            const blob = new Blob([bytes], { type: contentType || 'audio/wav' });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            
            currentAudio = audio;
            audio.onplay = () => { isSpeaking = true; updateOrbState('speaking'); };
            audio.onended = () => {
                URL.revokeObjectURL(url);
                currentAudio = null;
                isSpeaking = false;
                setProcessing(false);
                updateOrbState('idle');
                if(autoListen && conversationMode) setTimeout(startListening, 300);
            };
            audio.onerror = () => {
                URL.revokeObjectURL(url);
                setProcessing(false);
                browserTTSFallback("Audio error", autoListen);
            };
            
            audio.play().catch(e => { setProcessing(false); browserTTSFallback(null, autoListen); });
        }

        function browserTTSFallback(text, autoListen) {
            if(!text || !window.speechSynthesis) {
                 isSpeaking = false; 
                 setProcessing(false);
                 updateOrbState('idle'); 
                 if(autoListen && conversationMode) setTimeout(startListening, 300);
                 return;
            }
            
            const utter = new SpeechSynthesisUtterance(text);
            utter.onstart = () => { isSpeaking = true; updateOrbState('speaking'); };
            utter.onend = () => {
                isSpeaking = false; 
                setProcessing(false);
                updateOrbState('idle');
                if(autoListen && conversationMode) setTimeout(startListening, 300);
            };
            utter.onerror = () => {
                isSpeaking = false;
                setProcessing(false);
                updateOrbState('idle');
            };
            window.speechSynthesis.speak(utter);
        }

        /* ==================== STT ==================== */
        function toggleListening() {
            if(isSpeaking || isListening) {
                conversationMode = false;
                stopAllAudio();
                if(recognition) try{ recognition.stop(); } catch(e){}
                isListening = false;
                updateOrbState('idle');
                return;
            }
            conversationMode = true;
            startListening();
        }

        function startListening() {
            if(isListening || isProcessing || isSpeaking) return;
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if(!SpeechRecognition) {
                alert("Speech recognition not supported in this browser.");
                return;
            }
            
            // Clean up old recognition instance
            if(recognition) {
                try { recognition.abort(); } catch(e){}
                recognition = null;
            }

            recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.continuous = false;
            recognition.interimResults = false;
            
            let gotResult = false;

            recognition.onstart = () => { isListening = true; updateOrbState('listening'); };
            recognition.onresult = (event) => {
                gotResult = true;
                const transcript = event.results[0][0].transcript;
                document.getElementById('chatInput').value = transcript;
                isListening = false;
                updateOrbState('thinking');
                sendMessage(true);
            };
            recognition.onerror = (e) => {
                console.log('[STT] error:', e.error);
                isListening = false;
                // Don't restart here — let onend handle it to avoid double-start
            };
            recognition.onend = () => {
                isListening = false;
                if(gotResult) return; // sendMessage already took over
                // No result captured — restart if still in conversation mode
                if(conversationMode && !isProcessing && !isSpeaking) {
                    setTimeout(startListening, 400);
                } else {
                    updateOrbState('idle');
                }
            };
            
            try { recognition.start(); } catch(e){ 
                console.log('[STT] start failed:', e); 
                isListening = false;
                if(conversationMode) setTimeout(startListening, 1000);
            }
        }

        /* ==================== MESSAGING ==================== */
        function sendMessage(fromVoice = false) {
            if(isProcessing) return;

            const input = document.getElementById('chatInput');
            const txt = input.value.trim();
            if(!txt) return;

            setProcessing(true);
            addMessage(txt, 'user');
            input.value = '';
            
            // Stop any currently playing audio before new request
            stopAllAudio();

            // Show typing
            const typing = document.getElementById('typingIndicator');
            typing.classList.add('active');
            updateOrbState('thinking');
            
            // Auto scroll
            const area = document.getElementById('chatMessages');
            area.scrollTop = area.scrollHeight;

            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: txt })
            })
            .then(r => r.json())
            .then(data => {
                typing.classList.remove('active');
                if(data.success) {
                    addMessage(data.response, 'plant');
                    if(data.audio && !ttsMuted) {
                        playServerAudio(data.audio, data.content_type, fromVoice);
                    } else {
                        // strip emojis for reading
                        const cleanText = data.response.replace(/([\\u2700-\\u27BF]|[\\uE000-\\uF8FF]|\\uD83C[\\uDC00-\\uDFFF]|\\uD83D[\\uDC00-\\uDFFF]|[\\u2011-\\u26FF]|\\uD83E[\\uDD10-\\uDDFF])/g, '');
                        browserTTSFallback(cleanText, fromVoice);
                    }
                } else {
                    setProcessing(false);
                    updateOrbState('idle');
                    addMessage("Error: " + data.error, 'plant');
                }
            })
            .catch(e => {
                typing.classList.remove('active');
                setProcessing(false);
                updateOrbState('idle');
                console.error(e);
            });
        }

        function addMessage(text, sender) {
            const area = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = `msg msg-${sender}`;
            div.textContent = text;
            
            const typing = document.getElementById('typingIndicator');
            area.insertBefore(div, typing);
            area.scrollTop = area.scrollHeight;
        }

        /* ==================== DATA & SENSORS ==================== */
        function updateSensors() {
            fetch('/api/sensors')
            .then(r => r.json())
            .then(data => {
                if(!data.success) return;
                
                const s = data.sensors;
                const m = data.mood;
                
                updateVal('tempValue', s.temperature.toFixed(1) + '°C');
                updateVal('humValue', s.humidity.toFixed(1) + '%');
                updateVal('soilValue', s.soil_moisture.toFixed(1) + '%');
                updateVal('gasValue', s.gas_level);
                
                // Mood UI
                updateNotifications(s, m);
                const avatar = document.getElementById('plantAvatar');
                // Set class for animations
                avatar.className = `plant-emoji ${m.mood}`;
                // Set image source
                avatar.src = `/static/images/${m.mood}.png`;
                avatar.onerror = function() { this.src = 'https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals%20Nature/Seedling.png'; };
                
                // document.getElementById('moodEmoji').textContent = m.emoji; // Removed
                document.getElementById('moodText').textContent = m.mood.toUpperCase();
                const moodBadge = document.getElementById('moodBadge');
                moodBadge.className = `mood-badge ${m.mood}`;

                // Emergency Banner
                checkEmergency();
            })
            .catch(e => console.error(e));
        }

        let sensorChartInstance = null;

        function showGraph(metric, title) {
            const modal = document.getElementById('graphModal');
            document.getElementById('graphTitle').innerText = title;
            modal.style.display = 'flex';
            setTimeout(() => modal.classList.add('active'), 10);
            
            // Fetch data
            fetch(`/api/history/${metric}`)
            .then(r => r.json())
            .then(d => {
                if(d.success) {
                    renderChart(d.labels, d.data, title, metric);
                } else {
                    console.error("Failed to load history");
                }
            });
        }

        function closeGraphModal() {
            const modal = document.getElementById('graphModal');
            modal.classList.remove('active');
            setTimeout(() => modal.style.display = 'none', 300);
        }

        function renderChart(labels, data, title, metric) {
            const ctx = document.getElementById('sensorChart').getContext('2d');
            
            // Destroy old chart
            if(sensorChartInstance) {
                sensorChartInstance.destroy();
            }

            // Colors based on metric
            let color = '#818cf8'; // Default Indigo
            if(metric === 'temperature') color = '#ef4444'; // Red
            if(metric === 'humidity') color = '#3b82f6'; // Blue
            if(metric === 'soil_moisture') color = '#10b981'; // Green
            if(metric === 'gas_level') color = '#f59e0b'; // Amber

            // Create gradient
            let gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, color + '80'); // 50% opacity
            gradient.addColorStop(1, 'rgba(0,0,0,0)');

            sensorChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: title,
                        data: data,
                        borderColor: color,
                        backgroundColor: gradient,
                        borderWidth: 2,
                        tension: 0.4, // Smooth curve
                        pointRadius: 4,
                        pointBackgroundColor: '#fff',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(30, 41, 59, 0.9)',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1,
                            padding: 10,
                            displayColors: false,
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#94a3b8' },
                            beginAtZero: false 
                        }
                    },
                    interaction: {
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }
                }
            });
        }

        function updateVal(id, newVal) {
            const el = document.getElementById(id);
            if(el.textContent !== String(newVal)) {
                el.textContent = newVal;
                el.classList.add('val-highlight');
                setTimeout(() => el.classList.remove('val-highlight'), 1000);
            }
        }

        function checkEmergency() {
            /* Emergency is now handled inside updateNotifications */
        }

        function toggleNotifications() {
            const dd = document.getElementById('notifDropdown');
            dd.classList.toggle('open');
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            const wrapper = document.querySelector('.notif-wrapper');
            const dd = document.getElementById('notifDropdown');
            if(wrapper && dd && !wrapper.contains(e.target)) {
                dd.classList.remove('open');
            }
        });

        function updateNotifications(sensors, mood) {
            const list = document.getElementById('notifList');
            const badge = document.getElementById('notifBadge');
            const bell = document.getElementById('notifBell');
            const items = [];

            const temp = sensors.temperature;
            const hum = sensors.humidity;
            const soil = sensors.soil_moisture;
            const gas = sensors.gas_level;

            // === EMERGENCY (Critical!) ===
            // Soil < 5%
            if(soil < 5) {
                items.push({icon: '🏜️', text: `EMERGENCY: Soil critically dry at ${soil.toFixed(1)}%!`, level: 'danger'});
            }
            // Temp > 35°C (fire/heat death)
            if(temp > 35) {
                items.push({icon: '🔥', text: `EMERGENCY: Temperature at ${temp.toFixed(1)}°C — fire risk / heat death!`, level: 'danger'});
            }
            // Temp < 5°C (freeze)
            if(temp < 5) {
                items.push({icon: '🥶', text: `EMERGENCY: Temperature at ${temp.toFixed(1)}°C — freeze risk!`, level: 'danger'});
            }
            // Gas > 2000 (smoke/gas detected)
            if(gas > 2000) {
                items.push({icon: '🚨', text: `EMERGENCY: Gas level at ${gas} — smoke/gas detected! Ventilate now!`, level: 'danger'});
            }

            // === FAIR (Needs Attention) ===
            // Soil < 30% (but not < 5% which is emergency)
            if(soil >= 5 && soil < 30) {
                items.push({icon: '💧', text: `Soil dry at ${soil.toFixed(1)}% — needs water`, level: 'warning'});
            }
            // Temp 28-32°C (heat stress, but not > 35 which is emergency)
            if(temp >= 28 && temp <= 32) {
                items.push({icon: '🌡️', text: `Temperature at ${temp.toFixed(1)}°C — heat stress`, level: 'warning'});
            }
            // Temp 32-35°C (severe heat stress, approaching emergency)
            if(temp > 32 && temp <= 35) {
                items.push({icon: '🌡️', text: `Temperature high at ${temp.toFixed(1)}°C — severe heat stress`, level: 'warning'});
            }
            // Humidity < 30% (dry air)
            if(hum < 30) {
                items.push({icon: '🏜️', text: `Air dry at ${hum.toFixed(1)}% — consider misting`, level: 'warning'});
            }

            // Update UI
            if(items.length === 0) {
                list.innerHTML = '<div class="notif-empty">✅ All good! No alerts right now.</div>';
                badge.classList.remove('show');
                bell.classList.remove('has-alerts');
            } else {
                list.innerHTML = items.map(i =>
                    `<div class="notif-item ${i.level}"><span class="notif-icon">${i.icon}</span><span class="notif-text">${i.text}</span></div>`
                ).join('');
                badge.textContent = items.length;
                badge.classList.add('show');
                bell.classList.add('has-alerts');
            }
        }

        function renamePlant() {
            const modal = document.getElementById('renameModal');
            const input = document.getElementById('renameInput');
            input.value = plantName;
            modal.style.display = 'flex';
            // Slight delay to allow display flex to apply before opacity transition
            setTimeout(() => modal.classList.add('active'), 10);
            input.focus();
        }

        function closeRenameModal() {
            const modal = document.getElementById('renameModal');
            modal.classList.remove('active');
            setTimeout(() => modal.style.display = 'none', 300);
        }

        function confirmRename() {
            const input = document.getElementById('renameInput');
            const newName = input.value.trim();
            
            if(newName) {
                fetch('/api/rename', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name: newName })
                })
                .then(r => r.json())
                .then(d => {
                    if(d.success) {
                        plantName = d.name;
                        document.getElementById('plantNameDisplay').textContent = plantName;
                        closeRenameModal();
                    }
                });
            }
        }

        // Init
        document.getElementById('chatInput').addEventListener('keypress', e => {
            if(e.key === 'Enter') sendMessage();
        });
        
        setInterval(updateSensors, 2000);
        updateSensors();

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
    print("\n[INFO] Starting server on http://localhost:5001")
    print("="*70 + "\n")

    # Start MQTT logger
    mqtt_logger.start()
    
    # Run Flask app
    print("[INFO] Flask app running...")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
