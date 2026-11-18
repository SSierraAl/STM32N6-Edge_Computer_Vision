# Insekten mit Boxen erkennen

Die Idee ist einfach. Du gibst dem Modell ein Bild. Es sagt dir, wo ein Insekt ist und zu welcher Klasse es gehört. Um jedes gefundene Insekt zeichnet es ein Rechteck. In [Notebook 03](../03_Automated_Object_Detection.ipynb) kannst du dir die Bilder mit ihren Boxen anschauen, Modelle trainieren und die Ergebnisse vergleichen.

## Kurz ausprobieren

Nimm Python 3.10 bis 3.12. Öffne das Terminal im Hauptordner vom Projekt und führe die Befehle aus. Beim ersten Lauf werden Beispieldaten und Modellgewichte geladen.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Notebooks/detection/requirements.txt
python Notebooks/detection/pipeline.py --out detection-runs/coco-smoke
```

Das Modell geht einmal durch die Trainingsbilder. So siehst du, ob der ganze Ablauf funktioniert. Für eine Aussage über die Erkennung deiner Insekten brauchst du passende Testbilder.

## Eigene Bilder verwenden

Zu jedem Bild gehört eine Textdatei mit demselben Dateinamen. In jeder Zeile steht eine Box in der Form `class_id x_center y_center width height`. Die vier Zahlen beschreiben die Mitte und Größe der Box relativ zum Bild. Bei einem Bild ohne Objekt bleibt die Datei leer. In der Klassendatei steht die Reihenfolge der Klassen, zum Beispiel `["Motte", "Fliege"]`. Die erste Klasse hat die Nummer 0.

Du kannst Bilder unter `images/<aufnahme>/` ablegen und die Labels passend unter `labels/<aufnahme>/`. Nimm mindestens drei Aufnahmen mit beschrifteten Objekten und verwende `images` nur als obersten Ordnernamen. Mit `--group-by-event` oder `group_by_event = True` im Notebook bleiben alle Bilder derselben Aufnahme zusammen. Das ist wichtig, weil zwei Bilder aus derselben Aufnahme fast gleich aussehen können.

- **Train** ist zum Lernen.
- **Val** hilft dir, während des Trainings den besten Stand auszuwählen.
- **Test** zeigt dir am Schluss, wie gut das Modell auf neuen Bildern ist.

Die Aufteilung zielt auf ungefähr 75 Prozent Training und je 12,5 Prozent für Val und Test. Ganze Aufnahmen bleiben zusammen, daher können die Anteile abweichen. In `event-splits.json` steht, wohin jede Aufnahme gekommen ist. Wenn du schon `images/{train,val,test}` und `labels/{train,val,test}` hast, lässt du `--group-by-event` weg.

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json --group-by-event \
  --models yolov8n yolo11n mobilenetv3-large \
  --epochs 50 --out detection-runs/insects-01
```

Mit `--prepare-only` kannst du erst einmal die Boxen und die Anzahl der Bilder und Klassen prüfen. Mit `--min-test-images 600` verlangt die Pipeline mindestens 600 geprüfte Testbilder. Gib jedem Lauf einen neuen Ausgabeordner.

<a id="prepare-the-ip102-detection-benchmark"></a>

## Den öffentlichen Insektendatensatz nutzen

Lege die Archive `Annotations.tar` und `JPEGImages.tar`, die Listen `test.txt` und `trainval.txt` aus `Detection/VOC2007/ImageSets/Main` und die Klassendatei `classes.txt` in `detection-runs/ip102-source/`. Das Skript bringt die Boxen ins passende Format. Es übernimmt den veröffentlichten Testsplit und wählt die Validierungsbilder aus der Trainingsliste. Exakt gleiche Dateien über Splitgrenzen hinweg werden entfernt und in `ip102-splits.json` festgehalten. [Offizieller Download](https://github.com/xpwu95/IP102)

```bash
python Notebooks/detection/prepare_ip102.py detection-runs/ip102-source detection-runs/ip102-data
python Notebooks/detection/pipeline.py \
  --dataset detection-runs/ip102-data --classes detection-runs/ip102-data/classes.json \
  --min-test-images 600 --prepare-only --out detection-runs/ip102-inspection
```

## Andere Modelle anschließen

Modelle können unterschiedlich trainiert werden und unterschiedliche Ergebnisse zurückgeben. Ein Adapter übersetzt diese Unterschiede. Für den Vergleich bekommt jedes Modell ein gleich großes Bild und gibt Boxen, Klassen und einen Wert für seine Sicherheit zurück. So kannst du dieselbe Auswertung für verschiedene Modelle nutzen.

Wähle Modelle mit `--models`. Für ein zusätzliches Modell gibst du mit `--adapter name=/pfad/zur/datei.py` die Python-Datei an. Im Notebook trägst du das Modell in `models` und seinen Dateipfad in `adapters` ein. Das Beispiel hier nutzt für beide Modelle 320 Pixel.

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json --group-by-event \
  --size 320 --models yolov8n ssdlite \
  --adapter ssdlite=Notebooks/detection/adapters/torchvision_ssdlite.py \
  --epochs 20 --out detection-runs/insects-adapter
```

Wenn du einen eigenen Adapter schreibst, orientiere dich am [Beispieladapter](adapters/torchvision_ssdlite.py) und an der [Schnittstelle](model_adapter.py). Der Adapter trainiert mit `context.train` und `context.val` und gibt aus `train(context)` ein `TrainingResult` zurück. Darin stehen der gespeicherte Modellstand, seine Initialisierung und die verfügbaren Varianten (`FP32`, `FP16`, `INT8`, `NATIVE`). Pro Variante verarbeitet die Vorhersagefunktion ein quadratisches RGB-Bild als `uint8` in HWC-Reihenfolge. Sie liefert je Objekt `[x_min, y_min, x_max, y_max, confidence, class_id]` in Pixeln des aufgefüllten Bildes. Bei keinem Treffer liefert sie ein Array der Form `(0, 6)`. Verwende für Adapter nur Python-Dateien, denen du vertraust.

## Ergebnisse lesen

In `comparison.json` siehst du, wie gut und wie schnell die Modelle waren. **Precision** zeigt, wie viele gemeldete Treffer stimmen. **Recall** zeigt, wie viele vorhandene Objekte gefunden wurden. **Box-mAP** fasst zusammen, wie gut das Modell Objekte erkennt und passende Boxen setzt.

Alle Varianten bekommen dieselben Testbilder mit derselben Bildgröße auf der CPU, jeweils ein Bild auf einmal. Die aktuelle Zeitmessung umfasst auch die Arbeit im Adapter, etwa das Aufbereiten und Auslesen der Daten. Im früheren Lauf wurde nur die Modellinferenz gemessen. In `run.json` stehen die Einstellungen. Im Bericht steht auch, wie jedes Modell angefangen hat, zum Beispiel mit Gewichten aus einem früheren Training oder mit einem neu initialisierten Detektionskopf. Das hilft beim Einordnen der Ergebnisse.

FP32 ist die Gleitkomma-Variante. INT8 nutzt kleinere Zahlen und kann das Modell kompakter machen. Ob sich dabei die Erkennungsqualität ändert, siehst du am Vergleich auf denselben Testbildern. Welche Varianten du bekommst, hängt vom Modell und seinem Adapter ab. Die Messung auf dem Rechner zeigt die CPU-Zeit. Für einen Lauf auf dem Board brauchst du einen passenden Export, eine Firmware, die seine Ausgaben versteht, und eine eigene Messung am Gerät.
