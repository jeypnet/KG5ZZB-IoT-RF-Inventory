# Model card: modulation classifier and device fingerprinting

The app uses two pieces of machine learning:

1. a **supervised neural network** that decides how each burst is modulated and line coded, and
2. **unsupervised clustering** that groups transmissions into devices without being told how many there are.

## 1. Modulation / line-coding classifier


### What it does
Every burst the detector finds is demodulated two ways (envelope and instantaneous frequency) and
described by 17 measured features (below). A small neural network decides which of 6 classes the
burst belongs to. The decision picks the decoder (PWM, PPM, Manchester or FSK) that turns the burst
into bits. It runs on the CPU inside the app (ONNX Runtime), takes microseconds per burst and needs no internet.

### Data
- **Source:** a C# dataset generator renders randomised transmissions at 2.4 MS/s into captures
  with white noise and then runs **the app's own receive chain** on them (spectrogram, burst detector,
  channeliser, envelope and FM demodulators, feature extractor), so the network learns from exactly
  the measurements it gets in the app.
- **Training set:** seed 1, 13411 bursts (11400 train / 2011 validation for early stopping).
- **Test set:** seed 2, 4562 bursts, never used for training or model selection.
- **Classes and parameter ranges:**
  - OOK PWM: short pulse 120-700 us, long/short ratio 1.8-3.2, 12-68 bits, 1-6 repeats.
  - OOK PPM: pulse 120-800 us, short gap 250-1500 us, long/short ratio 1.8-3.2, 12-68 bits, 1-5 repeats.
  - OOK Manchester (IEEE 802.3 convention): half bit 100-600 us, 16-72 bits, 1-4 repeats.
  - 2-FSK: 2.4-50 kbit/s, deviation 0.25-1.5 x bit rate (5-60 kHz), 8-32 preamble bits + 24-152 data bits, 1-4 repeats.
  - LoRa: 125 kHz (70%), 250 or 500 kHz, SF7-SF9, 8-24 data symbols.
  - Noise: partial carriers (2-150 ms), sub-100 us clicks, one or two random tone blips, detector false alarms.
  - SNR 8-32 dB (signal power over the noise in a 25 kHz channel; LoRa in its own bandwidth), carrier offset 40-800 kHz from the tuned frequency.

### Features
1. log10 active duration (us)
2. log10 bandwidth (Hz)
3. on fraction
4. log10 pulse count
5. log10 median pulse (us)
6. log10 median gap (us)
7. pulse width CV
8. gap width CV
9. pulse groups log2 ratio
10. gap groups log2 ratio
11. period CV
12. pulse-gap correlation
13. frequency bimodality
14. log10 frequency std (Hz)
15. chirp score
16. upward step ratio
17. SNR / 20 dB

### Model
`17 -> 64 -> 32 -> 6` fully connected network, ReLU, softmax; 3430 parameters.
Feature standardisation is part of the exported ONNX graph. Cross-entropy with class weights,
Adam (lr 3e-3, weight decay 1e-4), batch 256, early stopping on validation loss.

### Results on the held-out test set
| | Accuracy | Macro F1 |
|---|---|---|
| **Neural network** | **0.976** | **0.974** |
| Rule-based baseline (`RuleModulationClassifier`) | 0.697 | 0.687 |

#### Per class (neural network)
| Class | Precision | Recall | F1 | Test bursts |
|---|---|---|---|---|
| Noise / spur | 0.995 | 0.996 | 0.995 | 1102 |
| OOK pulse-width (PWM) | 0.933 | 0.944 | 0.939 | 666 |
| OOK pulse-position (PPM) | 0.988 | 0.955 | 0.971 | 784 |
| OOK Manchester | 0.933 | 0.959 | 0.946 | 666 |
| 2-FSK | 0.993 | 0.988 | 0.990 | 678 |
| LoRa | 1.000 | 1.000 | 1.000 | 666 |

#### F1 per class: network vs rules
| Class | Neural network | Rules |
|---|---|---|
| Noise / spur | 0.995 | 0.757 |
| OOK pulse-width (PWM) | 0.939 | 0.669 |
| OOK pulse-position (PPM) | 0.971 | 0.777 |
| OOK Manchester | 0.946 | 0.615 |
| 2-FSK | 0.990 | 0.411 |
| LoRa | 1.000 | 0.892 |

#### Confusion matrix (neural network)
| true \ predicted | Noise | OOK-PWM | OOK-PPM | OOK-Manch | FSK | LoRa |
|---|---|---|---|---|---|---|
| **Noise** | 1098 | 0 | 0 | 1 | 3 | 0 |
| **OOK-PWM** | 0 | 629 | 6 | 30 | 1 | 0 |
| **OOK-PPM** | 1 | 22 | 749 | 12 | 0 | 0 |
| **OOK-Manch** | 1 | 22 | 3 | 639 | 1 | 0 |
| **FSK** | 4 | 1 | 0 | 3 | 670 | 0 |
| **LoRa** | 0 | 0 | 0 | 0 | 0 | 666 |

## 2. Device fingerprinting

Each transmission becomes a 15-number fingerprint: carrier frequency, short/long pulse and gap
durations, pulses per packet, FSK bits per packet, RSSI, bandwidth and the modulation family/coding.
Every number is divided by its typical *within-device* spread (for example 8 kHz for the carrier, 11%
for durations, 6 dB for RSSI), so distances are in "typical wobbles". The app matches a transmission
to the nearest known device within 1.8 units; otherwise it waits in a pending pool that is clustered
with **weighted DBSCAN** (radius 1.2, minimum weight 3; identical repeated packets weigh up to 4).
A dense cluster becomes a device; after the learning period that raises a NEW DEVICE alert.

<!-- clustering:start -->
## Device fingerprinting (unsupervised)

random "models" (timing families), several of which are shared by more than one device, so those
devices differ only in carrier frequency, signal strength and ID. Every transmission is rendered
into its own capture and goes through the complete receive chain (detector, demodulator, the embedded
classifier, decoder), then into the fingerprint clustering. 30 devices x 6 transmissions per population.
The DBSCAN radius (1.2), minimum weight (3) and match radius (1.8) were fixed before this evaluation; the populations (seeds 21-25) were not used to choose anything.

| Method | Needs the number of devices? | Adjusted Rand index | Purity | Devices found (true) |
|---|---|---|---|---|
| Online inventory (the app: nearest device + weighted DBSCAN) | no | 0.904 | 0.993 | 29.2 (30.0) |
| Batch weighted DBSCAN on all fingerprints | no | 0.897 | 0.987 | 29.2 (30.0) |
| k-means on the same fingerprints, told the true number of devices | yes | 0.770 | 0.822 | 30.0 (30.0) |
| Frequency only (signals within 50 kHz grouped, like a spectrum view) | no | 0.123 | 0.154 | 4.6 (30.0) |

Means over 5 populations. Transmissions the classifier called noise, and DBSCAN noise points, count as
clusters of one (they lower the score). The adjusted Rand index is 1 for a perfect grouping and about 0 for a random one.
<!-- clustering:end -->
