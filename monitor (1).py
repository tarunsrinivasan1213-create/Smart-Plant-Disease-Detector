import os
import time
import csv
import subprocess
from pathlib import Path
from collections import deque, Counter

from dotenv import load_dotenv
from twilio.rest import Client
from ultralytics import YOLO
import cv2
import numpy as np

# ───────────── Sensors ─────────────
import board
import adafruit_bh1750
import adafruit_sht4x
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ───────────── Load secrets from .env ─────────────
load_dotenv()
ACCOUNT_SID  = os.environ["TWILIO_ACCOUNT_SID"]
AUTH_TOKEN   = os.environ["TWILIO_AUTH_TOKEN"]
FROM_NUMBER  = os.environ["TWILIO_FROM_NUMBER"]
TO_NUMBER    = os.environ["TWILIO_TO_NUMBER"]

# ───────────── Paths ─────────────
MODEL_PATH = "models/best.pt"
DATA_DIR   = Path("data")
OUT_DIR    = Path("out")
LOG_PATH   = Path("logs/detections.csv")

# ───────────── Camera settings ─────────────
CAPTURE_NAME = "test.jpg"
CAPTURE_W    = "640"
CAPTURE_H    = "640"

# ───────────── Tiling settings ─────────────
TILE_SIZE        = 640
TILE_OVERLAP     = 0.25
MAX_TILES        = 60
SAVE_TILED_DEBUG = False

# ───────────── Detection thresholds ─────────────
RAW_CONF    = 0.15   # minimum conf to consider a detection at all
REPORT_CONF = 0.25   # minimum conf to report / log

# ───────────── Voting / alert settings ─────────────
VOTE_WINDOW  = 5    # frames to consider for stable vote
VOTE_REQUIRE = 3    # votes needed to confirm a label
INTERVAL_SEC = 3    # seconds between capture cycles
COOLDOWN_SEC = 60   # minimum seconds between SMS alerts

# ───────────── Soil settings ─────────────
# IMPORTANT: Use uppercase P0/P1/P2/P3
SOIL_CHANNEL       = "P0"
SOIL_DRY_THRESHOLD = 1.8   # volts — above this = dry
SOIL_WET_THRESHOLD = 1.0   # volts — below this = wet

# ───────────── Fallback remedies (used when LLM is unavailable) ─────────────
FALLBACK_REMEDIES = {
    "mold_related":   "Improve airflow, keep leaves dry, remove infected leaves.",
    "leaf_curling":   "Provide shade for heat stress, check pests, water consistently.",
    "spots_or_damage":"Remove spotted leaves, avoid overhead watering, sanitize tools.",
    "unknown":        "Retake a clearer picture in better light and closer to the leaf."
}


# ───────────── Label collapsing ─────────────
def collapse_label(raw_name: str) -> str:
    """Map fine-grained YOLO class names to broad category labels."""
    if raw_name in ("fuzzy_mold_growth", "powdery_white_patch"):
        return "mold_related"
    if raw_name == "leaf_curling":
        return "leaf_curling"
    if raw_name in ("leaf_spots_lesions", "patchy_spots", "holes_chew_damage"):
        return "spots_or_damage"
    return "unknown"


# ───────────── Temporal voting ─────────────
def stabilized_vote(history: deque) -> str:
    """
    Require VOTE_REQUIRE consistent detections within the last VOTE_WINDOW
    frames before reporting a stable label. Reduces false positives.
    """
    if len(history) < VOTE_WINDOW:
        return "uncertain"
    counts = Counter(history)
    label, cnt = counts.most_common(1)[0]
    return label if (label != "uncertain" and cnt >= VOTE_REQUIRE) else "uncertain"


# ───────────── Directory / log setup ─────────────
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def ensure_log():
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "image",
                "raw_label", "raw_conf",
                "reported_label", "stable_label",
                "temp_c", "humidity_pct", "lux",
                "soil_v", "soil_state",
                "sms_sid"
            ])


def log_row(img_name, raw_label, raw_conf, reported, stable, sd, sms_sid):
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            img_name,
            raw_label,
            f"{raw_conf:.3f}",
            reported,
            stable,
            sd.get("temperature_c", ""),
            sd.get("humidity_pct", ""),
            sd.get("lux", ""),
            sd.get("soil_voltage", ""),
            sd.get("soil_moisture", ""),
            sms_sid or ""
        ])


# ───────────── Camera capture ─────────────
def capture_image(path: Path):
    """
    Capture full-resolution image with rpicam-still, then
    center-crop to 640x640 using ImageMagick convert.
    """
    tmp = DATA_DIR / "tmp_full.jpg"
    subprocess.run(
        ["rpicam-still", "--zsl", "--width", "2304", "--height", "1604", "-o", str(tmp)],
        check=True
    )
    subprocess.run(
        ["convert", str(tmp), "-gravity", "center", "-crop", "640x640+0+0", "+repage", str(path)],
        check=True,
    )


# ───────────── Image tiling ─────────────
def iter_tiles_bgr(img_bgr: np.ndarray, tile_size: int, overlap: float):
    """
    Yield (x0, y0, tile) for a sliding window over img_bgr.
    Falls back to yielding the whole image if it fits in one tile.
    Overlap fraction reduces missed detections at tile boundaries.
    """
    h, w = img_bgr.shape[:2]
    if w <= tile_size and h <= tile_size:
        yield (0, 0, img_bgr)
        return

    step = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    ys = list(range(0, max(1, h - tile_size + 1), step))

    if xs[-1] != w - tile_size:
        xs.append(max(0, w - tile_size))
    if ys[-1] != h - tile_size:
        ys.append(max(0, h - tile_size))

    for y0 in ys:
        for x0 in xs:
            yield (x0, y0, img_bgr[y0:y0 + tile_size, x0:x0 + tile_size])


# ───────────── YOLO inference ─────────────
def predict_one(model: YOLO, img_path: Path):
    """
    Run inference on a single image.
    For images larger than TILE_SIZE, uses sliding window tiling
    to improve small-object detection accuracy.

    Returns: (raw_label, raw_conf, detected_bool)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return ("", 0.0, False)

    h, w = img.shape[:2]

    # Single-pass inference for small images
    if w <= TILE_SIZE and h <= TILE_SIZE:
        results = model.predict(
            source=img,
            conf=RAW_CONF,
            save=True,
            project=str(OUT_DIR),
            name="predict",
            exist_ok=True,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return ("", 0.0, False)

        confs  = results.boxes.conf.tolist()
        clss   = results.boxes.cls.tolist()
        best_i = max(range(len(confs)), key=lambda i: confs[i])
        return (model.names[int(clss[best_i])], float(confs[best_i]), True)

    # Tiled inference for large images
    best_conf = 0.0
    best_cls  = None
    tiles_seen = 0

    for (x0, y0, tile) in iter_tiles_bgr(img, TILE_SIZE, TILE_OVERLAP):
        tiles_seen += 1
        if tiles_seen > MAX_TILES:
            break

        r = model.predict(source=tile, conf=RAW_CONF, save=False, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        confs = r.boxes.conf.tolist()
        clss  = r.boxes.cls.tolist()
        bi    = max(range(len(confs)), key=lambda i: confs[i])
        c     = float(confs[bi])

        if c > best_conf:
            best_conf = c
            best_cls  = int(clss[bi])

    if best_cls is None:
        return ("", 0.0, False)

    return (model.names[best_cls], float(best_conf), True)


# ───────────── Sensor initialization ─────────────
def init_sensors():
    """
    Initialize all I2C sensors. Each sensor fails gracefully —
    the system continues running even if a sensor is unavailable.
    """
    sensors = {"bh1750": None, "sht41": None, "soil": None}

    try:
        i2c = board.I2C()
    except Exception as e:
        print(f"⚠ I2C init failed: {e}")
        return sensors

    try:
        sensors["bh1750"] = adafruit_bh1750.BH1750(i2c)
        print("✓ BH1750 light sensor ready")
    except Exception as e:
        print(f"⚠ BH1750 init failed: {e}")

    try:
        sht = adafruit_sht4x.SHT4x(i2c)
        sht.mode = adafruit_sht4x.Mode.NOHEAT_HIGHPRECISION
        sensors["sht41"] = sht
        print("✓ SHT41 temp/humidity sensor ready")
    except Exception as e:
        print(f"⚠ SHT41 init failed (non-critical): {e}")

    try:
        ads      = ADS.ADS1115(i2c)
        ch_map   = {"P0": ADS.P0, "P1": ADS.P1, "P2": ADS.P2, "P3": ADS.P3}
        soil_pin = (SOIL_CHANNEL or "P0").upper()
        sensors["soil"] = AnalogIn(ads, ch_map[soil_pin])
        print(f"✓ Soil moisture sensor ready on ADS1115 {soil_pin}")
    except Exception as e:
        print(f"⚠ Soil sensor init failed: {e}")

    return sensors


# ───────────── Sensor reading ─────────────
def read_sensors(sensors: dict) -> dict:
    """Read all available sensors. Missing sensors return None values."""
    sd = {
        "temperature_c": None,
        "humidity_pct":  None,
        "lux":           None,
        "soil_voltage":  None,
        "soil_moisture": None,
    }

    if sensors.get("bh1750"):
        try:
            sd["lux"] = float(sensors["bh1750"].lux)
        except Exception:
            pass

    if sensors.get("sht41"):
        try:
            t, h = sensors["sht41"].measurements
            sd["temperature_c"] = float(t)
            sd["humidity_pct"]  = float(h)
        except Exception:
            pass

    if sensors.get("soil"):
        try:
            v = float(sensors["soil"].voltage)
            sd["soil_voltage"]  = v
            sd["soil_moisture"] = (
                "dry"   if v > SOIL_DRY_THRESHOLD else
                "wet"   if v < SOIL_WET_THRESHOLD else
                "moist"
            )
        except Exception:
            pass

    return sd


# ───────────── LLM remedy generation (Qwen 2.5 via Ollama) ─────────────
def qwen_remedy_text(category_label: str, raw_conf: float, sd: dict) -> str:
    """
    Query local Qwen 2.5 1.5B model via Ollama for a treatment recommendation.
    Sensor data is injected into the prompt for context-aware advice.
    Falls back to static remedies on timeout or error.
    """
    if raw_conf < REPORT_CONF:
        return FALLBACK_REMEDIES.get(category_label, FALLBACK_REMEDIES["unknown"])

    env_parts = []
    if sd.get("temperature_c") is not None:
        env_parts.append(f"Temp={sd['temperature_c']:.1f}C")
    if sd.get("humidity_pct") is not None:
        env_parts.append(f"Humidity={sd['humidity_pct']:.0f}%")
    if sd.get("lux") is not None:
        env_parts.append(f"Light={sd['lux']:.0f} lux")
    if sd.get("soil_voltage") is not None:
        env_parts.append(f"SoilV={sd['soil_voltage']:.2f}V")
    if sd.get("soil_moisture") is not None:
        env_parts.append(f"Soil={sd['soil_moisture']}")

    env_line = ", ".join(env_parts) if env_parts else "No sensor data."

    prompt = (
        "you are a tomato plant assistant.\n"
        "give ONLY safe, general gardening advice.\n"
        "do not mention chemicals.\n"
        "output short 3 bullet points, each under 15 words.\n"
        f"Condition category: {category_label}.\n"
        f"Sensor context: {env_line}.\n"
        "if unsure, say: not confident, retake photo in better light.\n"
    )

    try:
        out = subprocess.check_output(
            ["ollama", "run", "qwen2.5:1.5b", prompt],
            text=True,
            timeout=20
        ).strip()

        if (not out) or len(out) > 900:
            return FALLBACK_REMEDIES.get(category_label, FALLBACK_REMEDIES["unknown"])
        return out

    except subprocess.TimeoutExpired:
        return FALLBACK_REMEDIES.get(category_label, "Qwen timeout. Try again.")
    except Exception as e:
        print(f"QWEN ERROR: {e}")
        return FALLBACK_REMEDIES.get(category_label, "Qwen error. Try again.")


# ───────────── SMS alerting ─────────────
_twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms(message: str) -> str:
    msg = _twilio_client.messages.create(
        body=message,
        from_=FROM_NUMBER,
        to=TO_NUMBER
    )
    return msg.sid


# ───────────── Main loop ─────────────
def main():
    ensure_dirs()
    ensure_log()

    model   = YOLO(MODEL_PATH)
    sensors = init_sensors()
    history = deque(maxlen=VOTE_WINDOW)
    last_alert = 0

    print("MODEL class mapping:", model.names)
    print("Starting monitor loop... (Ctrl+C to stop)\n")

    while True:
        try:
            # 1. Read environment sensors
            sd = read_sensors(sensors)

            # 2. Capture and preprocess image
            img_path = DATA_DIR / CAPTURE_NAME
            capture_image(img_path)

            # 3. Run YOLO inference (with tiling for large images)
            raw_label, raw_conf, detected = predict_one(model, img_path)

            # 4. Map to category label
            reported = (
                collapse_label(raw_label)
                if detected and raw_conf >= REPORT_CONF
                else "uncertain"
            )

            # 5. Temporal voting — require consistent detections
            history.append(reported)
            stable = stabilized_vote(history)

            print(
                f"[SENSORS] T={sd.get('temperature_c')}C  "
                f"H={sd.get('humidity_pct')}%  "
                f"Lux={sd.get('lux')}  "
                f"Soil={sd.get('soil_moisture')} (V={sd.get('soil_voltage')})"
            )
            print(
                f"[YOLO]    raw={raw_label or 'none'}  conf={raw_conf:.2f}  "
                f"-> reported={reported}  -> stable={stable}  "
                f"history={list(history)}\n"
            )

            sms_sid = None

            # 6. Alert only on stable disease + cooldown respected
            if stable and stable != "uncertain" and (time.time() - last_alert > COOLDOWN_SEC):
                remedy = qwen_remedy_text(stable, raw_conf, sd)

                if (not remedy) or ("not confident" in remedy.lower()):
                    remedy = FALLBACK_REMEDIES.get(stable, FALLBACK_REMEDIES["unknown"])

                msg = (
                    f"🌿 Plant Disease Detected\n"
                    f"Raw label : {raw_label} (conf={raw_conf:.2f})\n"
                    f"Category  : {stable}\n\n"
                    f"Environment:\n"
                    f"  Temp     : {sd.get('temperature_c')} C\n"
                    f"  Humidity : {sd.get('humidity_pct')} %\n"
                    f"  Light    : {sd.get('lux')} lux\n"
                    f"  Soil     : {sd.get('soil_moisture')} (V={sd.get('soil_voltage')})\n\n"
                    f"Recommendation:\n{remedy}"
                )

                print("Sending SMS alert...")
                sms_sid = send_sms(msg)
                print(f"SMS sent — SID: {sms_sid}\n")
                last_alert = time.time()

            # 7. Log to CSV
            log_row(img_path.name, raw_label, raw_conf, reported, stable, sd, sms_sid)

            time.sleep(INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
