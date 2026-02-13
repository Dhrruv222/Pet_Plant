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

// ================== SENSOR PIN CONFIGURATION ==================
// MQ-2 Gas Sensor
#define GAS_SENSOR_PIN 5            // GPIO 5 - ADC input
#define GAS_SENSOR_THRESHOLD 1.5f   // Safety threshold: 1.5V indicates dangerous gas levels

// Soil Moisture Sensor
#define SOIL_SENSOR_PIN 2           // GPIO 2 - ADC input
#define SOIL_MOISTURE_THRESHOLD 500 // ADC reading threshold (0-4095): below = needs water

// ADC Configuration
#define ADC_RESOLUTION 4095         // 12-bit resolution
#define REF_VOLTAGE 3.3f            // Reference voltage

// OLED Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDR 0x3C              // Default I2C address for SSD1306 (some boards use 0x3D)
// ============================================

WebServer server(80);
WiFiClient espClient;
PubSubClient* mqttClient = nullptr;
DHT dht(DHTPIN, DHTTYPE);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

float temperature = NAN;
float humidity    = NAN;
float gas_level   = 0.0f;      // MQ-2 gas sensor reading (0-3.3V)
int soil_moisture = 0;         // Soil moisture analog reading (0-4095)
bool needs_water = false;      // Plant status: soil too dry
bool gas_emergency = false;    // Plant status: dangerous gas levels
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
    display.println("Plant Status");
    display.println("---");

    // Temperature
    display.setTextSize(1);
    display.print("T: ");
    if (!isnan(temperature)) {
      display.print(temperature, 1);
      display.print("C ");
    } else {
      display.print("-- ");
    }

    // Humidity
    display.print("H: ");
    if (!isnan(humidity)) {
      display.println(humidity, 0);
    } else {
      display.println("--%");
    }

    // Soil Moisture
    display.print("Soil: ");
    display.print(soil_moisture);
    display.println("/4095");

    // Gas Level
    display.print("Gas: ");
    display.print(gas_level, 2);
    display.println("V");

    // Plant Status
    display.println("---");
    if (needs_water) {
      display.println("NEEDS WATER!");
    } else {
      display.println("Water: OK");
    }

    if (gas_emergency) {
      display.println("GAS WARNING!");
    } else {
      display.println("Gas: SAFE");
    }

    // Connectivity Status
    display.println("---");
    display.print("WiFi: ");
    display.println(WiFi.isConnected() ? "OK" : "...");
    display.print("MQTT: ");
    display.println((mqttClient && mqttClient->connected()) ? "OK" : "off");

    display.display();
    vTaskDelay(pdMS_TO_TICKS(2000)); // Update every 2 seconds
  }
}

void sensorTask(void* pv) {
  (void)pv;
  uint8_t retries = 0;
  
  for (;;) {
    // ===== Read DHT22 (Temperature & Humidity) =====
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

    // ===== Read Gas Sensor (GPIO 5) =====
    int gas_adc = analogRead(GAS_SENSOR_PIN);  // Read 12-bit ADC (0-4095)
    gas_level = (gas_adc / (float)ADC_RESOLUTION) * REF_VOLTAGE;  // Convert to voltage (0-3.3V)
    
    // Update plant status: gas emergency if exceeds threshold
    gas_emergency = (gas_level > GAS_SENSOR_THRESHOLD);
    
    Serial.print("Gas - ADC: ");
    Serial.print(gas_adc);
    Serial.print(" | Voltage: ");
    Serial.print(gas_level, 3);
    Serial.print("V | Emergency: ");
    Serial.println(gas_emergency ? "YES" : "NO");

    // ===== Read Soil Moisture Sensor (GPIO 2) =====
    soil_moisture = analogRead(SOIL_SENSOR_PIN);  // Read 12-bit ADC (0-4095)
    
    // Update plant status: needs water if below threshold
    needs_water = (soil_moisture < SOIL_MOISTURE_THRESHOLD);
    
    Serial.print("Soil - ADC: ");
    Serial.print(soil_moisture);
    Serial.print(" | Threshold: ");
    Serial.print(SOIL_MOISTURE_THRESHOLD);
    Serial.print(" | Needs Water: ");
    Serial.println(needs_water ? "YES" : "NO");

    vTaskDelay(pdMS_TO_TICKS(MQTT_PUB_INTERVAL));
  }
}

// ================== SETUP ==================

void setup() {
  Serial.begin(115200);
  delay(3000); // Wait for S2 USB Serial
  Serial.println("\n--- STAGED BOOT START ---");

  // --- I2C PIN INITIALIZATION ---
  // This explicitly maps SDA to GPIO 8 and SCL to GPIO 9 for your OLED
  Wire.begin(8, 9); 

  // Create networking objects AFTER RTOS is fully initialized
  mqttClient = new PubSubClient(espClient);
  mqttClient->setServer(mqtt_host, MQTT_PORT);
  mqttClient->setKeepAlive(60);
  mqttClient->setSocketTimeout(30);

  // Initialize OLED display (Using the Wire pins defined above)
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("SSD1306 allocation failed. Check wiring on GPIO 8/9");
    while (1);
  }
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Initializing...");
  display.display();

  dht.begin();

  // Configure sensor pins
  pinMode(GAS_SENSOR_PIN, INPUT);
  pinMode(SOIL_SENSOR_PIN, INPUT);

  // WiFi events + start WiFi
  WiFi.onEvent(WiFiEvent);
  connectToWiFi();

  // Web server routes
  server.on("/", HTTP_GET, []() {
    server.send_P(200, "text/html", index_html);
  });

  server.on("/data", HTTP_GET, []() {
    char jsonBuf[256];
    snprintf(jsonBuf, sizeof(jsonBuf),
      "{\"temperature\":%.1f,\"humidity\":%.1f,\"gas_level\":%.3f,\"soil_moisture\":%d,\"needs_water\":%s,\"gas_emergency\":%s}",
      isnan(temperature) ? 0.0f : temperature,
      isnan(humidity) ? 0.0f : humidity,
      gas_level,
      soil_moisture,
      needs_water ? "true" : "false",
      gas_emergency ? "true" : "false");
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
        char tbuf[16], hbuf[16], gbuf[16], sbuf[16], nwbuf[8], gebuf[8];
        snprintf(tbuf, sizeof(tbuf), "%.1f", temperature);
        snprintf(hbuf, sizeof(hbuf), "%.1f", humidity);
        snprintf(gbuf, sizeof(gbuf), "%.3f", gas_level);
        snprintf(sbuf, sizeof(sbuf), "%d", soil_moisture);
        snprintf(nwbuf, sizeof(nwbuf), "%s", needs_water ? "true" : "false");
        snprintf(gebuf, sizeof(gebuf), "%s", gas_emergency ? "true" : "false");

        mqttClient->publish("esp32/dht/temperature", tbuf, true);
        mqttClient->publish("esp32/dht/humidity", hbuf, true);
        mqttClient->publish("esp32/mq2/gas_level", gbuf, true);
        mqttClient->publish("esp32/soil/moisture", sbuf, true);
        mqttClient->publish("esp32/plant/needs_water", nwbuf, true);
        mqttClient->publish("esp32/plant/gas_emergency", gebuf, true);

        sensor_update_pending = false;
      }
    }
  }

  vTaskDelay(pdMS_TO_TICKS(1)); // FreeRTOS compatible delay for TCP stability on S2
}