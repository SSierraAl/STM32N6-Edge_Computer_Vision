# INT8-Quantisierung · FP32/INT8-Vergleich
INT8 | mirei/ml-pipeline-automation @ 7c3179d | VERIFIZIERT

## Problem → Fix

| Vorher | Fix (C#) | Nachher | Beleg |
|---|---|---|---|
| Tensorprüfung nur im Hauptgraphen | C1 · FlatBuffer und Interpreter prüfen | Alle Subgraphen und Schnittstellen geprüft | pipeline.py:195, quantization-audit.json |
| Exporte und mAP-Differenz im Code | C2 · Messwerte und Dateigrößen abgleichen | Vergleich pro Modell auf identischem Testsplit | comparison.json |
| Referenzbilder für Kalibrierung im Code | C3 · Manifest und Kalibrierliste abgleichen | Kalibrierung ausschließlich mit Trainingsbildern | calibration-images.txt |

Sechs Exporte geprüft. Vier Metriken pro Export reproduziert. Float-Tensor im Hilfsgraphen abgewiesen.

## Entscheidungen

| D# | Entscheidung | Herkunft | Was die Nutzerin mitgegeben hat |
|---|---|---|---|
| D1 | FP32 und INT8 direkt vergleichen | USER_ORIGINATED | P1 Die tatsächlichen Exportdateien prüfen. Den Einfluss von INT8 auf Dateigröße und mAP auf demselben Testdatensatz messen. |

## Bilder

`detection-runs/coco128-20epochs/fp32-int8.png` – Dateigröße und Test-mAP nach Quantisierung.

## Dateien

- MODIFIED Notebooks/detection/pipeline.py – Subgraphprüfung (C1)
- NEW docs/specs/int8-quantization-design-doc.md – Prüfbeleg (C1–C3)
