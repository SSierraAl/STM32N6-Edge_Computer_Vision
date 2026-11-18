# Manuelle Projektabnahme · STM32N6 und Insektenmodelle

**Prüfstand:** Branch `mirei/ml-pipeline-automation`, Implementierung bis Commit `6f85ceb` plus lokale Korrektur der Ergebnistabelle. Stand der Laufartefakte beim Erstellen dieser Checkliste. Alle verlinkten Laufdateien liegen lokal unter `detection-runs/`.

| Workflow | Eingang | Qualitätsmessung | Nachweisstand |
|---|---|---|---|
| Detection · Notebook 03 | Bilder und YOLO-Rechtecke | Box-mAP, Precision, Recall, Host-Inferenzzeit | COCO128 abgeschlossen, IP102-Vergleich noch ohne Abschlussbericht |
| Instance Segmentation · Notebook 01 und Segmentation-CLI | Bilder und YOLO-Polygone pro Instanz | Masken-mAP, Box-mAP, Host-Inferenzzeit | Training und Export mit synthetischen Masken geprüft |
| STM32N6 · Notebook 02 | INT8-Segmentierungsmodell | Schnittstelle, Speicherbericht, Funktion und Zeit am Gerät | ST-Codegenerierung am Smoke-Modell geprüft, Board-Nachweis für eigenes Modell offen |

## 1. Dataset-Import und Datenleckage

- [ ] In [Notebook 03](../../Notebooks/03_Automated_Object_Detection.ipynb) die Inspektionszelle ausführen. Für ereignisbasierte Eingabe `group_by_event = True` setzen und korrespondierende `images/<ereignis>/` und `labels/<ereignis>/` nutzen. Mindestens drei Ereignisse mit annotierten Objekten verwenden.
- [ ] Im Inspektionsordner `event-splits.json` und `dataset-manifest.json` vergleichen. Jedes Ereignis erscheint vollständig in genau einem Split. Ein erneuter Lauf mit gleichem Seed und gleichem Bestand liefert dieselbe Zuordnung.
- [ ] Train-, Val- und Test-Vorschauen öffnen. Die Boxen sitzen auf den Objekten, die Klassennamen stimmen und eine Klasse ohne Boxen erscheint in der Zählung mit null.
- [ ] Auf einer Testkopie ein Label löschen, eine Box außerhalb des Bildes eintragen oder ein identisches Bild in zwei Splits platzieren. Jeder Fehler wird beim Import mit einer Meldung abgewiesen. Hintergrundbilder haben ein leeres Label.
- [ ] Für den IP102-Import die [Split-Zahlen](../../detection-runs/ip102-data/ip102-splits.json), die [Test-Vorschau](../../detection-runs/ip102-inspection/preview-test.jpg) und den [Inspektionsbericht](../../detection-runs/ip102-inspection/run.json) öffnen. Erwartet sind 13.662 Train-, 1.516 Val- und 3.798 Testbilder. `--min-test-images 600` akzeptiert den IP102-Split und weist den COCO128-Split mit 16 Testbildern ab. [Importanleitung](../../Notebooks/detection/README.md#prepare-the-ip102-detection-benchmark)

## 2. Reproduzierbares Training und Lernkurven

- [ ] Im [COCO128-Laufbericht](../../detection-runs/coco128-20epochs/run.json) die gemeinsamen Einstellungen kontrollieren. Erwartet sind 20 Epochen je Modell, Seed 42, 256 × 256 Pixel, CPU und 96/16/16 Bilder für Train/Val/Test.
- [ ] Pro Modell unter `training/` die Dateien `args.yaml`, `results.csv`, `results.png` und `weights/best.pt` prüfen. Erwartet sind 20 Epochenzeilen, ein Checkpoint und aufrufbare Kurven. Die Epochenmetriken beziehen sich auf Val. Die finalen Exportmetriken beziehen sich auf Test.
- [ ] [Lernkurven](../../detection-runs/coco128-20epochs/lernkurven.png) und [Messbericht](../../detection-runs/coco128-20epochs/messbericht.md) öffnen. Prüfen, dass Trainingsverlust, Validierungsmetriken und finale Testwerte jeweils zugeordnet sind.
- [ ] [IP102-Laufprotokoll](../../detection-runs/ip102-20epochs.log) prüfen. Als abgeschlossen markieren, sobald für alle drei Modelle die Epochen und beide Testvarianten vollständig vorliegen und `comparison.json` geschrieben ist.

## 3. Modellwechsel und gemeinsame Auswertung

- [ ] Die [manuelle Adapter-Abnahme](model-families-design-doc.md#manuelle-abnahme) ausführen. Sie enthält die Notebook-Konfiguration für YOLOv8n und TorchVision SSDLite, inklusive eines ungültigen Adapterpfads als Fehlerfall.
- [ ] In der Ergebnistabelle die Exportbezeichnung und die numerische Spalte `precision` prüfen. Erwartet sind separate Werte. Die Zeilen für YOLOv8n tragen `FP32` und `INT8`, die native SSDLite-Variante trägt `FP32`.
- [ ] Im Bericht die Initialisierung kontrollieren. YOLOv8n und YOLO11n verwenden vortrainierte Detektions-Checkpoints, das eingebaute MobileNetV3-Large verwendet einen ImageNet-Backbone mit neuem Detektionskopf und der SSDLite-Beispieladapter startet zufällig initialisiert.

## 4. FP32 gegen INT8, Genauigkeit und CPU-Zeit

- [ ] [Vergleichsgrafik](../../detection-runs/coco128-20epochs/messwerte-panel.png), [Rohwerte](../../detection-runs/coco128-20epochs/comparison.json) und [Tensor-Audit](../../detection-runs/coco128-20epochs/quantization-audit.json) öffnen. Erwartet sind sechs Varianten auf denselben 16 Testbildern. Pro Variante stehen Box-mAP, Precision, Recall, Bytes und ms/Bild im Bericht.
- [ ] FP32- und INT8-TFLite pro Modell öffnen oder mit den Tensor-JSONs vergleichen. Erwartet sind identische Aufgabe und passende Tensorformen, skalierte INT8-Ein- und Ausgaben sowie eine kleinere INT8-Datei. Das Audit überprüft auch die internen Teilgraphen. `calibration-images.txt` enthält ausschließlich Trainingsbilder.
- [ ] Die Werte grob mit der Referenz abgleichen. Die Zeit stammt aus der ursprünglichen Ultralytics-Messung der Modellinferenz auf dem Mac. Das neuere Adapter-Protokoll misst den gesamten Predictor-Aufruf inklusive seiner Vor- und Nachverarbeitung.

| Modell | Box-mAP50–95 FP32 → INT8 | CPU-ms/Bild FP32 → INT8 |
|---|---:|---:|
| YOLOv8n | 0,436 → 0,402 | 21,86 → 7,22 |
| YOLO11n | 0,509 → 0,452 | 17,94 → 6,74 |
| MobileNetV3-Large | 0,000 → 0,000 | 17,08 → 7,94 |

Die Tabelle dokumentiert den COCO128-Vergleich für diesen Lauf. Kleine Abweichungen bei erneuter Zeitmessung sind erwartbar. Die MobileNet-Zeile hat eine FP32-Basis von null und erlaubt keine Aussage über den Erhalt der Erkennungsqualität durch INT8. Eine Aussage für Insekten braucht die Auswertung auf dem entsprechenden Ziel-Testsplit.

## 5. Segmentation separat abnehmen

- [ ] [Maskenvorschau](../../detection-runs/segmentation-smoke/preview-test.jpg) öffnen. Erwartet sind Konturen pro Instanz. Die [Segmentation-Anleitung](../../Notebooks/segmentation/README.md) beschreibt das Format und die Befehle zur Wiederholung.
- [ ] Im [Smoke-Bericht](../../detection-runs/segmentation-smoke/comparison.json) prüfen, dass beide TFLite-Varianten auf demselben Masken-Testsplit ausgewertet wurden. Erwartet sind `mask_mAP50-95` und `box_mAP50-95`. Der synthetische Lauf prüft den Ablauf. Bei beiden Maskenvarianten lag mAP bei null.
- [ ] Die Inspektion mit IP102-Boxlabels ausführen. Erwartet ist eine Meldung über fehlende Polygonkoordinaten. Für den Qualitätsnachweis mindestens 600 passende beschriftete Insekten-Testbilder mit echten Instanzmasken prüfen. Dieser Nachweis steht noch aus.

## 6. STM32N6-Vertrag und Board-Nachweisabe

- [ ] In [Notebook 02](../../Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb) das [INT8-Manifest](../../detection-runs/segmentation-smoke/insect_model.json) prüfen. Im [generierten Header](../../detection-runs/segmentation-smoke/stai_network.h) und [ST-Bericht](../../detection-runs/segmentation-smoke/network_generate_report.txt) Ausgabereihenfolge, Tensorformen, Scale, Zero-Point und Speicherbelegung prüfen.
- [ ] Mit der [Firmware-Schnittstelle](../../README.md#4-check-the-generated-model-and-configure-the-application) vergleichen. Beim Smoke-Export kommen Maskenprototypen zuerst. Die vorhandene Firmware erwartet die Detektionsausgabe zuerst. Die generierte Eingabequantisierung des Smoke-Exports weicht ebenfalls von der Kamera-Schnittstelle ab. ST-Codegenerierung für diesen Export ist belegt.
- [ ] Sebastians Originaldemo am Gerät als Referenz prüfen. Beim späteren Zielmodell die Abweichungen am tatsächlichen Export prüfen, die Firmware entsprechend konfigurieren und am Gerät Maskenanzeige, Netzwerklatenz und Zeit von der Kamera bis zur Ausgabe dokumentieren. Eigener Modelllauf am Board und gemessene NPU-Latenz bleiben offen.

## Prüfprotokoll

| Datum | Abschnitt | Ergebnis | Beleg oder Fehlermeldung |
|---|---|---|---|
|  |  | Bestanden / fehlgeschlagen / offen |  |
