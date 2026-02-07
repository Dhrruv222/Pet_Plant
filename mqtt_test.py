import os
import csv
import json
import threading
from datetime import datetime
import paho.mqtt.client as mqtt

from local_sensor import get_sensor_data

# ===================== CONFIG =====================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883

# MUST match your ESP32 publish topics
TOPIC_TEMP = "esp32/dht/temperature"
TOPIC_HUM = "esp32/dht/humidity"
TOPIC_GAS = "esp32/mq2/gas"
TOPIC_SOIL = "esp32/soil/moisture"

LOG_DIR = "data_logs"
DEVICE_ID = "plant1"
# ==================================================

_lock = threading.Lock()


class MqttToLocalStorage:
    def __init__(self):
        self.sensor_mgr = get_sensor_data(simulate=False)

        self.latest = {
            "temperature": None,
            "humidity": None,
            "gas_level": None,
            "soil_moisture": None,
        }

        os.makedirs(LOG_DIR, exist_ok=True)

        self.client = mqtt.Client(client_id=f"server-{DEVICE_ID}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self._thread = None
        self._running = False

    # ================= MQTT =================

    def _on_connect(self, client, userdata, flags, rc):
        print(f"✅ MQTT connected (rc={rc})")

        client.subscribe(TOPIC_TEMP)
        client.subscribe(TOPIC_HUM)
        client.subscribe(TOPIC_GAS)
        client.subscribe(TOPIC_SOIL)

        print("📡 Subscribed to topics")

    def _on_disconnect(self, client, userdata, rc):
        print("⚠️ MQTT disconnected")

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="ignore").strip()

        with _lock:
            changed = False
            now_iso = datetime.now().astimezone().isoformat()

            try:
                val = float(payload)
            except ValueError:
                return

            if msg.topic == TOPIC_TEMP:
                self.latest["temperature"] = val
                changed = True
            elif msg.topic == TOPIC_HUM:
                self.latest["humidity"] = val
                changed = True
            elif msg.topic == TOPIC_GAS:
                self.latest["gas_level"] = int(val)
                changed = True
            elif msg.topic == TOPIC_SOIL:
                self.latest["soil_moisture"] = val
                changed = True

            if not changed:
                return

            temperature = self._safe_float(self.latest["temperature"], 24.0)
            humidity = self._safe_float(self.latest["humidity"], 50.0)
            soil_moisture = self._safe_float(self.latest["soil_moisture"], 40.0)
            gas_level = self._safe_int(self.latest["gas_level"], 300)

            # Update JSON (dashboard reads this)
            self.sensor_mgr.update_data(
                temperature=temperature,
                humidity=humidity,
                soil_moisture=soil_moisture,
                gas_level=gas_level
            )

            # Log CSV
            self._append_csv(
                timestamp=now_iso,
                temperature=temperature,
                humidity=humidity,
                soil_moisture=soil_moisture,
                gas_level=gas_level,
                topic=msg.topic
            )

    # ================= CSV =================

    def _csv_path(self):
        day = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(LOG_DIR, f"{DEVICE_ID}_{day}.csv")

    def _append_csv(self, timestamp, temperature, humidity, soil_moisture, gas_level, topic):
        path = self._csv_path()
        file_exists = os.path.exists(path)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "device_id",
                    "temperature_c",
                    "humidity_pct",
                    "soil_moisture_pct",
                    "gas_level",
                    "topic"
                ])

            writer.writerow([
                timestamp,
                DEVICE_ID,
                temperature,
                humidity,
                soil_moisture,
                gas_level,
                topic
            ])

            f.flush()

    # ================= Helpers =================

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

    # ================= Start =================

    def start_background(self):
        if self._running:
            return

        self._running = True

        def worker():
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self.client.loop_forever()

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
        print("🚀 MQTT Logger Started")


_mqtt_instance = None


def start_mqtt_logger():
    global _mqtt_instance
    if _mqtt_instance is None:
        _mqtt_instance = MqttToLocalStorage()
        _mqtt_instance.start_background()