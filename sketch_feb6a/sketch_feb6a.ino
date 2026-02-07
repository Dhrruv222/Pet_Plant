#include <WiFi.h>
#include <WebServer.h>
#include <PubSubClient.h>
#include <ESPmDNS.h>
#include <DHT.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>

// ================== CONFIG ==================
// NOTE: Move credentials to separate config file or environment variables in production
#define WIFI_SSID "Dhrruv"
#define WIFI_PASS "12345678"
#define MQTT_HOST "broker.emqx.io"

const char* ssid      = WIFI_SSID;
const char* password  = WIFI_PASS;
const char* mqtt_host = MQTT_HOST;

#define MQTT_PORT 1883
#define MQTT_PUB_INTERVAL 10000

#define DHTPIN 6
#define DHTTYPE DHT22

// OLED Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C    // Default I2C address for SSD1306 (some boards use 0x3D)
// ============================================

WebServer server(80);
WiFiClient espClient;
PubSubClient* mqttClient = nullptr;
DHT dht(DHTPIN, DHTTYPE);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

float temperature = NAN;
float humidity    = NAN;
bool sensor_update_pending = false;
unsigned long lastMqttAttempt = 0;
unsigned long lastWifiAttempt = 0;
bool mqtt_connect_pending = false;

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-S2 Fixed</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #1d2671, #c33764); margin: 0; color: white; text-align: center; }
  .container { max-width: 400px; margin: 40px auto; background: rgba(255,255,255,0.12); border-radius: 16px; padding: 25px; }
  .value { font-size: 2.5rem; font-weight: bold; }
</style>
</head>
<body>
  <div class="container">
    <h2>🌡️ ESP32-S2 Stable</h2>
    <div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px;">
      <p>Temp: <span id="temp">--</span>°C</p>
      <p>Hum: <span id="hum">--</span>%</p>
    </div>
  </div>
<script>
async function fetchData() {
  try {
    const res = await fetch('/data');
    const d = await res.json();
    document.getElementById('temp').innerText = Number(d.temperature).toFixed(1);
    document.getElementById('hum').innerText  = Number(d.humidity).toFixed(1);
  } catch(e) {}
}
setInterval(fetchData, 5000);
fetchData();
</script>
</body>
</html>
)rawliteral";

// ================== STABILITY HELPERS ==================

void connectToWiFi() {
  Serial.println("WiFi: Connecting...");
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // CRITICAL: Keeps radio stable on S2
  WiFi.begin(ssid, password);
}

void connectToMqtt() {
  if (WiFi.isConnected() && mqttClient && !mqttClient->connected()) {
    if (millis() - lastMqttAttempt > 5000) {  // Throttle reconnect attempts
      Serial.println("MQTT: Connecting...");
      if (mqttClient->connect("ESP32S2Client")) {
        Serial.println("MQTT: Online");
      } else {
        Serial.print("MQTT: Failed, rc=");
        Serial.println(mqttClient->state());
      }
      lastMqttAttempt = millis();
    }
  }
}

void WiFiEvent(WiFiEvent_t event) {
  if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    Serial.println("WiFi: Connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());

    // Web server will be started in setup()

    // Trigger MQTT connect from main loop
    mqtt_connect_pending = true;
  }
  else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    Serial.println("WiFi: Disconnected. Reconnecting...");
    // Trigger WiFi reconnect from main loop
    lastWifiAttempt = 0;
  }
}

void displayTask(void* pv) {
  (void)pv;
  for (;;) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);

    // Title
    display.println("ESP32-S2 Sensor");
    display.println("---");

    // Temperature
    display.setTextSize(2);
    display.print("T: ");
    if (!isnan(temperature)) {
      display.print(temperature, 1);
      display.println("C");
    } else {
      display.println("--");
    }

    // Humidity
    display.print("H: ");
    if (!isnan(humidity)) {
      display.print(humidity, 1);
      display.println("%");
    } else {
      display.println("--");
    }

    // Status
    display.setTextSize(1);
    display.println("---");
    if (WiFi.isConnected()) {
      display.println("WiFi: OK");
    } else {
      display.println("WiFi: Connecting...");
    }
    if (mqttClient && mqttClient->connected()) {
      display.println("MQTT: OK");
    } else {
      display.println("MQTT: Offline");
    }

    display.display();
    vTaskDelay(pdMS_TO_TICKS(2000)); // Update every 2 seconds
  }
}

void sensorTask(void* pv) {
  (void)pv;
  uint8_t retries = 0;
  for (;;) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();

    // Adafruit library returns NaN on error; retry up to 3 times
    if (!isnan(t) && !isnan(h)) {
      temperature = t;
      humidity = h;
      sensor_update_pending = true;
      retries = 0; // Reset retry counter on success
    } else {
      retries++;
      if (retries >= 3) {
        Serial.println("DHT: Failed to read after 3 retries. Check wiring/power/pullup.");
        retries = 0;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(MQTT_PUB_INTERVAL));
  }
}

// ================== SETUP ==================

void setup() {
  Serial.begin(115200);
  delay(3000); // Wait for S2 USB Serial
  Serial.println("\n--- STAGED BOOT START ---");

  // Create networking objects AFTER RTOS is fully initialized
  mqttClient = new PubSubClient(espClient);
  mqttClient->setServer(mqtt_host, MQTT_PORT);
  mqttClient->setKeepAlive(60);
  mqttClient->setSocketTimeout(30);

  // Initialize OLED display
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 allocation failed");
    while (1);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Initializing...");
  display.display();

  dht.begin();

  // WiFi events + start WiFi
  WiFi.onEvent(WiFiEvent);
  connectToWiFi();

  // Web server routes (setup handlers, but don't start yet)
  server.on("/", HTTP_GET, []() {
    server.send_P(200, "text/html", index_html);
  });

  server.on("/data", HTTP_GET, []() {
    char jsonBuf[64];
    snprintf(jsonBuf, sizeof(jsonBuf),
      "{\"temperature\":%.1f,\"humidity\":%.1f}",
      isnan(temperature) ? 0.0f : temperature,
      isnan(humidity) ? 0.0f : humidity);
    server.send(200, "application/json", jsonBuf);
  });

  server.begin();
  Serial.println("Web server started");

  // Background sensor task
  xTaskCreate(sensorTask, "Sensors", 8192, nullptr, 1, nullptr);

  // OLED display task
  xTaskCreate(displayTask, "Display", 4096, nullptr, 1, nullptr);

  Serial.println("--- BOOT COMPLETE ---");
}

void loop() {
  // WiFi reconnect handling (main loop only)
  if (WiFi.status() == WL_DISCONNECTED) {
    if (millis() - lastWifiAttempt > 5000) {
      Serial.println("WiFi: Reconnect attempt...");
      WiFi.reconnect();
      lastWifiAttempt = millis();
    }
  }

  // Handle incoming HTTP requests
  server.handleClient();

  // Handle MQTT connection and publishing
  if (mqttClient) {
    if (WiFi.isConnected()) {
      if (!mqttClient->connected()) {
        if (mqtt_connect_pending || (millis() - lastMqttAttempt > 5000)) {
          connectToMqtt();
          mqtt_connect_pending = false;
        }
      }
      mqttClient->loop();  // Process MQTT messages

      // Publish sensor data when available
      if (sensor_update_pending && mqttClient->connected()) {
        char tbuf[16], hbuf[16];
        snprintf(tbuf, sizeof(tbuf), "%.1f", temperature);
        snprintf(hbuf, sizeof(hbuf), "%.1f", humidity);

        mqttClient->publish("esp32/dht/temperature", tbuf, true);
        mqttClient->publish("esp32/dht/humidity", hbuf, true);

        sensor_update_pending = false;
      }
    }
  }

  vTaskDelay(pdMS_TO_TICKS(1)); // FreeRTOS compatible delay for TCP stability on S2
}