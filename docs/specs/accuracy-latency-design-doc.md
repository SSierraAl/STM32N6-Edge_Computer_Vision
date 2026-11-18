# Genauigkeit und Inferenzzeit
Thesis | mirei/ml-pipeline-automation @ 7c3179d | VERIFIZIERT

## Problem → Fix

| Vorher | Fix (C#) | Nachher | Beleg |
|---|---|---|---|
| Vergleich im Code belegt | C1 · Log und Artefakte abgleichen | Gemeinsamer Testsplit und verifizierte Werte | pipeline.py:160–175 @ 7c3179d, comparison.json |
| Einstufige Latenzangabe | C2 · Messgrenze im Panel | Nur CPU-Inferenz je Bild | Laufprotokoll, run.json |
| Geänderter Adapter-Timer | C3 · Messverfahren unterscheiden | Adapterzeit umfasst Vor- und Nachverarbeitung | pipeline.py:235–237, run.json |

Bestanden: sechs Exporte, gemeinsamer Testsplit, Batch 1, 256 px, Metriken, Zeit, Dateigröße und Deltas.

## Entscheidungen

| D# | Entscheidung | Herkunft | Was die Nutzerin mitgegeben hat |
|---|---|---|---|
| D1 | Einheitliche Eingaben und CPU | USER_ORIGINATED | P1 Modell- und Quantisierungsvarianten sollen dieselben Testbilder, Bildgröße, CPU und Batchgröße nutzen. Die CPU-Inferenzzeit soll getrennt von späteren STM32N6-Messungen ausgewiesen werden. |
| D2 | COCO128-Testsplit mit 16 Bildern | AI_DEFAULT | Der vorhandene Split dient dem Machbarkeitsnachweis. |

## Bilder

[Messwerte-Panel](../../detection-runs/coco128-20epochs/messwerte-panel.png) – Nachher, mAP50–95, CPU-Inferenzzeit und Dateigröße.

## Dateien

- NEW docs/specs/accuracy-latency-design-doc.md – Messprotokoll und Nachweis (C1–C3)
- NEW detection-runs/coco128-20epochs/messwerte-panel.png – gemessene Testwerte (C1, C2)
