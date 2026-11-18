# Modellfamilien wechseln
Modellwechsel | mirei/ml-pipeline-automation @ 6f85ceb | VERIFIZIERT

## Problem → Fix

| Vorher | Fix (C#) | Nachher | Beleg |
|---|---|---|---|
| Modellliste und Ultralytics fest verdrahtet | C1 · Externe Adapter laden | Neue Familien per Konfiguration | pipeline.py:347 |
| Auswertung am Ultralytics-Modell | C2 · Gemeinsame Boxen auswerten | Testsplit, Metriken und Eingaben gemeinsam | pipeline.py:232 |
| Startbedingungen nur in Anleitung | C3 · Initialisierung ausweisen | Herkunft pro Modell im Report | pipeline.py:396 |
| Precision überschreibt Exportbezeichnung | C4 · Export separat indizieren | Variante und Precision gleichzeitig sichtbar | Notebook 03 |

Nachprüfung bestanden. Notebook 03 mit SSDLite, sechs TFLite-Exporte, Float-Teilgraph und Tabellenwerte.

## Entscheidungen

| D# | Entscheidung | Herkunft | Was die Nutzerin mitgegeben hat |
|---|---|---|---|
| D1 | Fremde Familien erweiterbar | USER_ORIGINATED | P1 Die drei vorhandenen Detektoren sollen Beispiele bleiben. Neue Modellfamilien müssen in den gemeinsamen Vergleich passen. |
| D2 | Python-Adapter für Boxdetektoren | AI_DEFAULT | |

## Bilder

[SSDLite-PR-Kurve](images/model-families/ssdlite-pr-synthetic.png) – Nachher, synthetische Testbilder.

## Dateien

- MODIFIED Notebooks/detection/pipeline.py – Registrierung, Auswertung (C1–C3)
- NEW Notebooks/detection/model_adapter.py – Schnittstelle (C1, C2)
- NEW Notebooks/detection/adapters/torchvision_ssdlite.py – SSD-Referenzadapter (C1)
- MODIFIED Notebooks/03_Automated_Object_Detection.ipynb – Modellauswahl, Berichte (C1, C3, C4)
- MODIFIED Notebooks/detection/README.md – Adapter-Vertrag (C1–C3)

## Manuelle Abnahme

**Voraussetzung.** Notebook 03 in der Python-Umgebung aus der [Detection-Anleitung](../../Notebooks/detection/README.md) öffnen. Jupyter aus dem Repository-Stamm starten. Die folgenden Werte nach der Konfigurationszelle setzen. Der vorhandene COCO128-Split dient hier als kurzer Funktionstest.

```python
models = ['yolov8n', 'ssdlite']
adapters = {'ssdlite': notebooks / 'detection/adapters/torchvision_ssdlite.py'}
dataset = runs / 'co-smoke/coco128-data'
classes = runs / 'coco128-20epochs/classes.json'
size = 320
epochs = 1
device = 'cpu'
seed = 42
```

| Schritt | Manuelle Aktion | Bestanden, wenn |
|---|---|---|
| 1 · Datensatz prüfen | In Notebook 03 die Inspektionszelle ausführen. | 96 Train-, 16 Val- und 16 Testbilder. Drei Vorschauen und Boxzahlen je Klasse erscheinen. |
| 2 · Modellfamilien wechseln | Trainings- und Ergebniszelle ausführen. In der Konfigurationszelle bei einem neuen Lauf kannst du auch `yolo11n` oder `mobilenetv3-large` ergänzen. | YOLOv8n hat Zeilen für `FP32` und `INT8`, TorchVision SSDLite eine Zeile für `FP32`. Die numerische Spalte `precision` bleibt sichtbar. Initialisierungen sind ausgewiesen. |
| 3 · Gemeinsame Auswertung | Im neuen Laufordner `run.json`, `dataset-manifest.json` und `comparison.json` öffnen. | Ein gemeinsamer Testsplit, 320 Pixel, CPU. Pro Variante erscheinen Precision, Recall, mAP und CPU-Zeit. Die Zeit umfasst den Predictor-Aufruf. |
| 4 · TFLite-Schnittstelle | Im neuen Laufordner `yolov8n/yolov8n-fp32.json` und `yolov8n/yolov8n-int8.json` öffnen. | Die JSON-Dateien zeigen je Tensor Form und Datentyp. Die INT8-Ein- und Ausgaben haben eine positive Scale sowie einen Zero-Point. Der SSDLite-Checkpoint ist im Bericht als natives Artefakt ausgewiesen. |
| 5 · Fehlerfall Adapter | In der Konfiguration für `ssdlite` einen fehlenden Dateipfad eintragen und die Inspektionszelle mit neuem `stamp` ausführen. | Die Pipeline meldet den fehlenden Adapterpfad. Danach den gültigen Pfad wiederherstellen. |
| 6 · STM32N6-Vertrag | Die Detection-Tensoren aus Schritt 4 mit dem [Firmware-Vertrag](../../README.md#4-check-the-generated-model-and-configure-the-application) vergleichen. | Der Detection-Export liefert einen Ausgabetensor. Die Segmentation-Firmware erwartet Ausgaben für Boxen/Klassen und Maskenprototypen. Ein Board-Lauf braucht einen passenden Export und einen separaten Test am Gerät. |

**Protokoll.** Je Schritt bestanden, fehlgeschlagen oder offen markieren. Den Laufordner und bei Fehlern die erste Fehlermeldung festhalten. Die Ein-Epochen-Werte dienen der Funktionsprüfung.
