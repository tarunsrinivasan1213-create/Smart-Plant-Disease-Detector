# Smart-Plant-Disease-Detector
<div align="center">

###  2nd Place — ACSEF 2025 (Alameda County Science & Engineering Fair)

*A fully autonomous, offline Edge AI system for real-time plant disease detection and diagnosis.*

*Runs entirely on a Raspberry Pi 5 — no cloud, no internet dependency.*


</div>


## Overview

Early-stage plant disease detection is critical for preventing crop loss, yet most existing solutions rely on cloud infrastructure, expensive GPU hardware, or manual inspection. These constraints make them impractical for small-scale farmers, greenhouses, and remote agricultural environments.

This project presents a **fully autonomous edge AI system** capable of real-time plant disease detection and diagnosis without internet access. Designed for **low-cost and continuous monitoring**, the system demonstrates how advanced AI can operate entirely on embedded hardware.

The system integrates three tightly coupled subsystems:

* **Computer Vision (YOLOv11n)** — Detects disease patterns on leaf surfaces
* **Environmental Sensing (I2C array)** — Captures temperature, humidity, soil moisture, and light
* **On-Device Language Model (Qwen 2.5 1.5B)** — Generates context-aware treatment recommendations

All inference runs locally on a Raspberry Pi 5, enabling:

* Zero cloud dependency
* Low latency
* Privacy-preserving operation
* Deployment in offline or rural environments

This system demonstrates a **multimodal edge AI pipeline**, combining vision + sensor data + language reasoning in a single embedded device.



##  Recognition

| Award        | Event                                                  |
| ------------ | ------------------------------------------------------ |
| 🥈 2nd Place | ACSEF 2025 — Alameda County Science & Engineering Fair |


##  Real-World Impact

* Enables **early disease detection** to reduce crop loss
* Accessible to **low-resource farmers** without internet
* Reduces reliance on **manual inspection**
* Demonstrates scalable architecture for **precision agriculture**



##  System Architecture

```
Camera → Preprocessing → YOLO Detection → Temporal Voting → LLM Reasoning → SMS Alert
                         ↑
                    Sensor Data
```



##  Hardware Components

| Component             | Specification    | Purpose             |
| --------------------- | ---------------- | ------------------- |
| Raspberry Pi 5        | 8GB RAM          | Edge compute        |
| Pi Camera Module 3    | 12MP             | Image capture       |
| SHT41                 | Temp & humidity  | Environment sensing |
| Soil Sensor + ADS1115 | Analog → digital | Soil moisture       |
| BH1750                | Lux sensor       | Light intensity     |



## Detection Classes

| Class               | Category        | Description        |
| ------------------- | --------------- | ------------------ |
| fuzzy_mold_growth   | mold_related    | Fungal growth      |
| powdery_white_patch | mold_related    | Powdery mildew     |
| leaf_curling        | leaf_curling    | Stress or pests    |
| leaf_spots_lesions  | spots_or_damage | Bacterial damage   |
| patchy_spots        | spots_or_damage | Infection patterns |
| holes_chew_damage   | spots_or_damage | Physical damage    |



## Performance Metrics

| Metric                   | Value                                     |
| ------------------------ | ----------------------------------------- |
| Model trained            | YOLOv8n (custom trained)                  |
| Model used               | YOLOv11n                                  |
| Dataset Size             | ~350 images (PlantVillage + custom) |
| Classes                  | 6 → 3 grouped                             |
| mAP@50                   | ~75–85% (estimated range)                 |
| Inference Time           | ~120–200 ms per tile                      |
| Effective FPS            | ~1–2 FPS (full pipeline)                  |
| False Positive Reduction | ~40–60% via temporal voting               |

### Key Observations

* Temporal voting significantly reduced noise-induced detections
* Tiling improved small-feature detection accuracy
* Stable performance across moderate lighting variations



## Technical Design Decisions

### 1. Temporal Voting (False Positive Reduction)

Uses a 5-frame sliding window and requires ≥3 consistent detections before triggering alerts.

### 2. Sliding Window Tiling

Splits high-resolution images into overlapping 640×640 tiles to improve detection of small features.

### 3. Sensor-Augmented LLM Reasoning

Combines detection output with environmental data for context-aware recommendations.

### 4. Graceful Degradation

System continues operating even if sensors, LLM, or SMS fail.

### 5. Edge Optimization

* Lightweight models (YOLOv11n)
* Sequential pipeline scheduling
* Efficient I2C sensor polling



##  Dataset & Training

### Dataset Sources

* PlantVillage dataset
* Custom labeled images (Label Studio)

### Training Configuration

* Model: YOLOv8n
* Image size: 640×640
* Epochs: ~50–100
* Augmentations: flipping, brightness, scaling

### Labeling Strategy

Focused on fine-grained disease patterns to improve real-world detection accuracy.



## Quickstart

```bash
git clone https://github.com/tarunsrinivasan1213-create/Smart-Plant-Disease-Detector
cd Smart-Plant-Disease-Detector

pip install -r requirements.txt

ollama pull qwen2.5:1.5b

cp .env.example .env
# Add Twilio credentials

python Plant-monitor.py
```



##  Project Structure

```
Plant-monitor.py
models/
data/
logs/
out/
```



##  Limitations

* Limited dataset diversity affects generalization
* Performance depends on lighting conditions
* LLM recommendations are heuristic-based
* Tiling increases latency



##  Future Work

* Expand dataset across plant species
* Integrate hardware accelerators (TPU/NPU)
* Build custom PCB-based system
* Optimize LLM for faster inference
* Add mobile dashboard



##  Security

* Uses `.env` for credential management
* No hardcoded secrets
* Fully local inference



##  Author

**Tarun Srinivasan Muthumari**
Edge AI · Computer Vision · Embedded Systems
Fremont, CA



##  License

MIT License



<div align="center">
<sub>Built with YOLOv8 · Raspberry Pi 5 · Ollama · Presented at ACSEF 2025</sub>
</div>
