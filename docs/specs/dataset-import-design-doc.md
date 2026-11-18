# 04 · Dataset-Import
04 | mirei/ml-pipeline-automation @ 7c3179d | VERIFIZIERT

## Problem → Fix

| Vorher | Fix (C#) | Nachher | Beleg |
|---|---|---|---|
| Vorgegebene Splits, Duplikatprüfung per Hash | C1 · Ereignisweise aufteilen | Aufnahme bleibt in einem Split | pipeline.py:54, CLI-Test |
| Box-Vorschauen ohne Klassenstatistik | C2 · Boxen pro Klasse zählen | Klassenabdeckung pro Split sichtbar | Notebook 03:79, CLI-Test |
| Training importiert die Quelle erneut | C3 · Geprüfte Kopie trainieren | Gleicher Datenstand im Notebook | Notebook 03:106 |

CLI bestanden: Ereignisse, Seed, Vorschauen, Klassen, Altformat, fehlerhafte Labels, Duplikate und YOLO-Ordnername.

## Entscheidungen

| D# | Entscheidung | Herkunft | Was die Nutzerin mitgegeben hat |
|---|---|---|---|
| D1 | Aufnahmeereignisse zusammenhalten | USER_ORIGINATED | P1 Aufnahmen zuerst gruppieren und erst danach aufteilen. Klassen und Boxen sollen visuell überprüfbar sein. |
| D2 | Ereignisordner, ungefähr 75/12,5/12,5 | AI_DEFAULT | |

## Bilder

[Validation-Vorschau](images/dataset-import/1-nachher-validation.jpg) – Nachher, synthetische Frames mit YOLO-Boxen.

## Dateien

- MODIFIED Notebooks/detection/pipeline.py – Gruppenimport und Klassenstatistik (C1, C2)
- MODIFIED Notebooks/03_Automated_Object_Detection.ipynb – Sichtprüfung und Datenkopie (C2, C3)
- MODIFIED Notebooks/detection/README.md – Eingabeformat und Nutzung (C1)
