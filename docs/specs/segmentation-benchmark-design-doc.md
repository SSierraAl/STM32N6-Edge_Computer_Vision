# Instance Segmentation Benchmark

**Stand:** HOST-VERIFIZIERT · **Branch:** aktueller Arbeitsstand · **Base:** Notebook 01/02 und YOLOv8n-Segmentation-Referenz.

## Nutzersignal

- **P1 · USER_ORIGINATED:** Segmentation soll die Modellauswahl bestimmen und auf dem vorhandenen STM32N6-Segmentation-Workflow aufbauen.
- **P2 · USER_ORIGINATED:** Eine Thesis-Auswertung braucht mindestens 600 Testbilder.

## Befund und Entscheidung

| ID | Befund | Umsetzung |
|---|---|---|
| C1 | Notebook 01 zeigt Segmentierung, die bisherige Vergleichspipeline bewertet Bounding Boxes. | Reproduzierbarer YOLO-Segmentation-Vergleich mit echten Masken und Maskenmetriken. |
| C2 | IP102 enthält Bounding Boxes. | Polygon-Prüfung, getrennte Splits und standardmäßig mindestens 600 Testbilder. |
| C3 | STM32N6-Referenz erwartet Box-/Klassen-/Koeffizienten-Tensor und Maskenprototypen. | INT8-Schnittstellenprüfung und Manifest für Notebook 02. Die ST-generierte Schnittstelle wird separat geprüft. |
| C4 | Ultralytics kann weitere Dateiformate laden oder Bilder überspringen. | Explizite Dateilisten, Vorprüfung der Bildgröße und Prüfung der tatsächlich ausgewerteten Testbildzahl. |

## Nachweis

- [Masken-Vorschau](images/segmentation-benchmark/1-nachher-masken-vorschau.jpg) aus synthetischen Testbildern.
- [CLI und Anleitung](../../Notebooks/segmentation/README.md).
- **Tests bestanden:** Eine YOLOv8n-seg-Epoche, FP32- und INT8-TFLite-Auswertung auf denselben drei synthetischen Testbildern, Box-Label- und 600-Bilder-Fehlerpfade. Die Masken-mAP beider Exporte ist in diesem Funktionstest 0. Die unabhängige Nachprüfung bestätigte Dateilisten und Bildzählung.
- **ST-Codegenerierung bestanden:** ST Edge AI 4.0.1 erzeugt STM32N6-Code für den synthetischen INT8-FlatBuffer. Der Header liefert Prototypen zuerst und U8-Eingabeskala 0,0036909. Notebook 02 erwartet Detektionen zuerst und 1/255. Das bestehende Firmware-Interface passt noch nicht zum Testexport.
- **Offen:** Auswertung mit mindestens 600 für die Zielarten repräsentativen, echten Testbildern mit Masken und Messung auf dem Board.
