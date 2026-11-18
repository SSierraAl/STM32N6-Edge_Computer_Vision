# Insekten mit Masken erkennen

Eine Box zeigt dir, wo ein Insekt ungefähr ist. Eine Maske zeigt, welche Pixel zu diesem Insekt gehören. Wenn zwei Insekten im Bild sind, bekommt jedes seine eigene Maske. In [Notebook 01](../01_Train_and_Quantize_STM32N6.ipynb) findest du den Einstieg ins Training.

## Welche Bilder brauche ich?

Zum Trainieren und Messen brauchst du Bilder, bei denen die Konturen der einzelnen Insekten eingezeichnet sind. Die Boxen aus dem Detection-Datensatz reichen für die Maskenmessung nicht aus.

Lege die Bilder unter `images/{train,val,test}/` ab und die passenden Textdateien unter `labels/{train,val,test}/`. Jede Zeile beschreibt ein Insekt mit einer Klassennummer und den Punkten seiner Kontur, also `class x1 y1 x2 y2 x3 y3 ...`. Die Koordinaten sind relativ zum Bild. Bei einem Bild ohne Objekt bleibt die Labeldatei leer. Die Namen der Klassen stehen in einer JSON-Liste, in der die erste Klasse die Nummer 0 hat.

Bilder aus derselben Aufnahme gehören alle in denselben Split. Standardmäßig braucht der Lauf mindestens 600 geprüfte Testbilder. Für einen kurzen Funktionstest kannst du mit `--min-test-images` eine kleinere Zahl einstellen.

## Bilder prüfen und trainieren

Nutze dieselbe [Python-Umgebung wie für Detection](../detection/README.md#kurz-ausprobieren). Schau dir zuerst die Vorschauen an. So siehst du, ob die Konturen auf den Bildern richtig sitzen und ob jeder Split passt.

```bash
python Notebooks/segmentation/pipeline.py \
  --dataset /path/to/mask-dataset --classes /path/to/classes.json \
  --prepare-only --out detection-runs/masks-inspection
```

Dann kannst du mit einem vortrainierten Segmentierungsmodell starten. Der Befehl trainiert das Modell weiter und exportiert den besten Stand in zwei Varianten.

```bash
python Notebooks/segmentation/pipeline.py \
  --dataset /path/to/mask-dataset --classes /path/to/classes.json \
  --checkpoint /path/to/yolov8n-seg.pt --epochs 20 \
  --out detection-runs/masks-benchmark
```

## Was bedeutet der Vergleich?

Du prüfst FP32 und INT8 auf denselben Testbildern. Der **Masken-mAP** zeigt, wie gut die vorhergesagten Konturen zu den eingezeichneten Konturen passen. Der **Box-mAP** zeigt zusätzlich, wie gut die Rechtecke passen. In `comparison.json` findest du auch Precision, Recall, Dateigröße und die Inferenzzeit auf der CPU. Für die INT8-Kalibrierung nimmt die Pipeline nur Trainingsbilder. Die Testbilder bleiben für die abschließende Messung reserviert.

Wenn du noch ein Segmentierungsmodell ausprobieren willst, tausche den Pfad bei `--checkpoint` aus. Lass Testbilder, Bildgröße und Seed gleich und nutze einen neuen Ausgabeordner.

## Wie geht es aufs Board?

Bei 256 × 256 Pixeln schreibt die Pipeline `insect_model.json`, wenn die Formen der INT8-Tensoren zur Demo passen. Das sind die Datenblöcke, die das Modell erwartet und ausgibt. Setze in [Notebook 02](../02_Compile_and_Deploy_NUCLEO_N6.ipynb) den Wert `MANIFEST` auf diese Datei. Dort prüfst du auch die Reihenfolge der Ausgaben und wie die Kamerawerte für das Modell umgerechnet werden. Danach kannst du am Board prüfen, ob die Masken richtig angezeigt werden und wie lange ein Bild braucht.
