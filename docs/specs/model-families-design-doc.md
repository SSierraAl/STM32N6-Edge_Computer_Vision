# Modellfamilien wechseln
Modellwechsel | mirei/ml-pipeline-automation @ 7c3179d | VERIFIZIERT

## Problem → Fix

| Vorher | Fix (C#) | Nachher | Beleg |
|---|---|---|---|
| Modellliste und Ultralytics fest verdrahtet | C1 · Externe Adapter laden | Neue Familien per Konfiguration | pipeline.py:334 |
| Auswertung am Ultralytics-Modell | C2 · Gemeinsame Boxen auswerten | Testsplit, Metriken und Eingaben gemeinsam | pipeline.py:219 |
| Startbedingungen nur in Anleitung | C3 · Initialisierung ausweisen | Herkunft pro Modell im Report | pipeline.py:387 |

Tests bestanden. SSDLite-Training, sechs bestehende TFLite-Exporte, Adapterisolierung und Fehlerfälle.

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
- MODIFIED Notebooks/03_Automated_Object_Detection.ipynb – Modellauswahl, Berichte (C1, C3)
- MODIFIED Notebooks/detection/README.md – Adapter-Vertrag (C1–C3)
