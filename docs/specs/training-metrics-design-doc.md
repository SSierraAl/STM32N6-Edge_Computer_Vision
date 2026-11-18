# Training und Qualitätsmetriken
Training | mirei/ml-pipeline-automation @ 7c3179d | VERIFIZIERT

## Problem → Fix

| Vorher | Fix (C#) | Nachher | Beleg |
|---|---|---|---|
| Ein-Epochen-Läufe | C1 · 20 Epochen je Modell | Lernkurven mit Messreihen | detection-runs/coco128-20epochs |
| Testmetriken im Code | C2 · Exporte und Ergebnisse prüfen | Vergleich auf demselben Testsplit | comparison.json |

Prüfung bestanden. 60 Epochen, sechs Exporte, Manifest, Kalibrierung, Checkpoints und Kurven.

## Entscheidungen

| D# | Entscheidung | Herkunft | Was die Nutzerin mitgegeben hat |
|---|---|---|---|
| D1 | Echter Lauf, vergleichbare Metriken | USER_ORIGINATED | P1 Mehr-Epochen-Training wirklich ausführen. Sebastian soll Kurven und übertragbar dokumentierte Kennzahlen unter identischen Testbedingungen sehen. |
| D2 | 20 Epochen, CPU, COCO128 | AI_DEFAULT | Lokal verfügbarer Datensatz, Seed 42, 256 px. |

## Bilder

- images/training-metrics/1-nachher-lernkurven.png – Trainingsverlust und Validierungsmetriken.

## Dateien

- RUN detection-runs/coco128-20epochs/ – Rohwerte, Kurven, Trainingsargumente, Checkpoints und Laufprotokoll.
