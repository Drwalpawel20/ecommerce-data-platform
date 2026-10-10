# Raport analityczny: E-commerce Data Platform

**Autor:** Paweł Drwal  
**Data opracowania:** 4 października 2026  
**Zakres danych:** 2021-01-01 – 2025-12-31 (5 lat, dane syntetyczne)  
**Stos technologiczny:** Python · PostgreSQL · scikit-learn · statsmodels · Power BI

---

## Spis treści

1. [Streszczenie zarządcze](#1-streszczenie-zarządcze)
2. [Zakres i opis danych](#2-zakres-i-opis-danych)
3. [Metodologia](#3-metodologia)
4. [Jakość danych](#4-jakość-danych)
5. [Wyniki sprzedażowe: trend i sezonowość](#5-wyniki-sprzedażowe-trend-i-sezonowość)
6. [Wykrywanie anomalii](#6-wykrywanie-anomalii)
7. [Prognoza sprzedaży](#7-prognoza-sprzedaży)
8. [Analiza klientów](#8-analiza-klientów)
9. [Analiza produktów](#9-analiza-produktów)
10. [Analiza koszyka zakupowego](#10-analiza-koszyka-zakupowego)
11. [Rekomendacje biznesowe](#11-rekomendacje-biznesowe)
12. [Ograniczenia i ryzyka](#12-ograniczenia-i-ryzyka)
13. [Dalsze kroki](#13-dalsze-kroki)

---

## 1. Streszczenie zarządcze

Projekt buduje kompletną ścieżkę analityczną dla symulowanego sklepu internetowego: od generatora danych, przez ETL do hurtowni w modelu gwiazdy (PostgreSQL), po analizy statystyczne, modele uczenia maszynowego i dashboard Power BI.

**Najważniejsze wnioski:**

| # | Wniosek | Liczby |
|---|---|---|
| 1 | Sprzedaż rośnie stabilnie, ale tempo wzrostu hamuje | CAGR **+9,0 %**; wzrost r/r: +9,9 % → +10,3 % → +8,8 % → **+7,0 %** |
| 2 | Wzrost wynika z **liczby zamówień**, a nie z wartości koszyka | zamówienia 83,4 tys. → 116,5 tys.; AOV ≈ 585 → ≈ 591 |
| 3 | Sezonowość jest bardzo silna | grudzień ≈ **2,4×** styczeń; listopad + grudzień = **26,7 %** przychodu |
| 4 | Marża jest stabilna | ≈ **24,9 %** w każdym roku |
| 5 | Baza klientów jest silnie skoncentrowana wartościowo | 60 % klientów generuje **79,5 %** przychodu i **81,2 %** zysku |
| 6 | Katalog produktów jest silnie skoncentrowany | 24,7 % produktów = **63,6 %** przychodu; 21,9 % produktów = **3,1 %** |
| 7 | Prognoza na 2026 | ≈ **72,7 mln** (+5,6 % r/r), błąd modelu MAE **6,4 %** |
| 8 | Anulacje i zwroty to realny wyciek przychodu | ≈ **23,7 mln** (8 % wartości brutto) |

**Kluczowe wskaźniki (brutto):**

| Wskaźnik | Wartość |
|---|---:|
| Zamówienia | 500 000 |
| Pozycje zamówień | 1 147 325 |
| Sprzedane sztuki | ≈ 1,20 mln |
| Przychód | ≈ 294,6 mln |
| Zysk | ≈ 73,3 mln |
| Marża zysku | 24,9 % |
| Średnia wartość zamówienia (AOV) | ≈ 589 |
| Klienci / produkty | 50 000 / 10 000 |

---

## 2. Zakres i opis danych

| Tabela | Liczba rekordów | Opis |
|---|---:|---|
| `orders` | 500 000 | nagłówki zamówień, status, pracownik, klient |
| `order_details` | 1 147 325 | pozycje zamówień (ilość, cena, rabat, przychód) |
| `customers` | 50 000 | klienci z 243 krajów |
| `products` | 10 000 | produkty w 50 kategoriach |
| `suppliers` | 500 | dostawcy |
| `employees` | 100 | pracownicy (4 działy) |
| `categories` | 50 | kategorie produktów |

**Struktura statusów zamówień:** zrealizowane 84,0 % · wysłane 8,0 % · anulowane 5,0 % · zwrócone 3,0 %.

> Uwaga: wszystkie liczby w raporcie są **wartościami brutto** (tabela faktów zawiera także zamówienia anulowane i zwrócone, oznaczone w kolumnie `order_status`).

---

## 3. Metodologia

### 3.1 Architektura

```
Generator (Faker + NumPy, seed=42)
        ↓  CSV
PostgreSQL: schemat staging   ← COPY FROM STDIN
        ↓  upsert + ładowanie wsadowe (25 000 zamówień / batch)
PostgreSQL: schemat warehouse (model gwiazdy)
        ↓  widoki: vw_sales_daily, vw_customer_performance, vw_product_performance
Python: statystyka + ML        →   analytics/outputs, analytics/ml/outputs
        ↓
Power BI (EcommerceAnalytics_Dark.pbix)
```

### 3.2 Model danych

Tabela faktów `fact_sales` na poziomie **pozycji zamówienia** oraz wymiary: `dim_date`, `dim_customer`, `dim_product` (→ `dim_category`, `dim_supplier`), `dim_employee`.
Miary wyliczane w ETL: `revenue = ilość × cena × (1 − rabat)`, `cost = ilość × koszt produktu`, `profit = revenue − cost`.

### 3.3 Zastosowane techniki

| Obszar | Technika |
|---|---|
| ETL | `COPY`, `INSERT … ON CONFLICT DO UPDATE` (idempotentność), ładowanie wsadowe, `ANALYZE` |
| Trend i sezonowość | średnie kroczące 7/30/90 dni, regresja liniowa, dekompozycja **STL** (okres 365, wariant odporny) |
| Anomalie (statystyka) | **zespół 6 sygnałów**: Z-score globalny, IQR, Z-score kroczący (30 dni), odchylenie r/r (±30 %), reszty STL, odchylenie od modelu wartości oczekiwanej |
| Anomalie (ML) | **Isolation Forest** (300 drzew, contamination = 2 %), 15 cech |
| Segmentacja | RFM → `log1p` → `StandardScaler` → **K-Means**; porównanie z GMM i DBSCAN; PCA do wizualizacji |
| Prognoza | **HistGradientBoostingRegressor**: cechy kalendarzowe, sin/cos dnia roku, opóźnienia (1, 7, 14, 30, 365), średnie kroczące (7–90) |
| Koszyk | pary produktów, wsparcie (support), ufność (confidence), przyrost (lift) |

---

## 4. Jakość danych

Framework `sql/01_data_quality.sql` zawiera **14 testów**: liczność tabel, brakujące wartości we wszystkich kolumnach faktów, niepoprawne ilości/ceny/rabaty/przychody, zgodność formuły zysku (tolerancja 0,01), duplikaty `order_detail_id` oraz integralność referencyjna względem pięciu wymiarów.

Uzgodnienie z danymi źródłowymi: **przychód roczny z hurtowni zgadza się co do grosza z przychodem policzonym z surowych plików CSV** (np. 2022: 53 572 310; 2025: 68 832 571).

---

## 5. Wyniki sprzedażowe: trend i sezonowość

### 5.1 Wyniki roczne

| Rok | Zamówienia | Przychód | Wzrost r/r | Zysk | AOV |
|---|---:|---:|---:|---:|---:|
| 2021 | 83 392 | 48,7 mln | – | 12,1 mln | 585 |
| 2022 | 90 920 | 53,6 mln | +9,9 % | 13,3 mln | 589 |
| 2023 | 100 225 | 59,1 mln | +10,3 % | 14,7 mln | 590 |
| 2024 | 109 010 | 64,3 mln | +8,8 % | 16,0 mln | 590 |
| 2025 | 116 453 | 68,8 mln | +7,0 % | 17,1 mln | 591 |

![Trend przychodu](images/01_revenue_trend.png)

*Rys. 1. Przychód dzienny, średnie kroczące (7/30/90 dni) i trend liniowy.*

**Obserwacje:**
- Trend liniowy: nachylenie ≈ **+51 na dzień** (≈ +18,6 tys. dziennego przychodu rocznie); sam trend wyjaśnia jedynie ~22 % zmienności (R² = 0,22) – dominuje sezonowość.
- Tempo wzrostu r/r **spada** (z 10,3 % do 7,0 %) – ważny sygnał dla planowania.
- Marża ≈ 24,9 % nie wykazuje trendu.

![Marża](images/05_profit_margin_trend.png)

*Rys. 2. Dzienna marża zysku i średnia krocząca 30-dniowa.*

### 5.2 Dekompozycja STL

![STL](images/02_stl_decomposition.png)

*Rys. 3. Dekompozycja STL: obserwacje, trend, składnik sezonowy, reszty.*

Dekompozycja potwierdza prawie liniowy trend wzrostowy oraz powtarzalny, silny składnik sezonowy kulminujący w listopadzie–grudniu.

### 5.3 Profile sezonowe

| Profil miesięczny | Profil tygodniowy |
|---|---|
| ![Miesiące](images/03_monthly_seasonality.png) | ![Dni tygodnia](images/04_weekday_seasonality.png) |

| Okres | Średni przychód dzienny |
|---|---:|
| Styczeń (najsłabszy) | ≈ 114 tys. |
| Wrzesień | ≈ 180 tys. |
| Listopad | ≈ 238 tys. |
| Grudzień (najsilniejszy) | ≈ 276 tys. |
| Q1 → Q4 | 123 tys. → 223 tys. (**+81 %**) |
| Dni robocze | ≈ 168 tys. |
| Weekend | ≈ 146 tys. (**−13 %**) |

---

## 6. Wykrywanie anomalii

Zastosowano dwa niezależne podejścia, aby ograniczyć ryzyko artefaktów jednej metody.

### 6.1 Zespół sygnałów statystycznych

Wynik anomalii = liczba aktywnych sygnałów (0–6). Progi: **≥ 2 Warning**, **≥ 3 Anomaly**, **≥ 5 Extreme**.

| Poziom | Dni |
|---|---:|
| Warning | 91 |
| Anomaly | 42 |
| Extreme Anomaly | 2 |
| **Razem** | **135** |

- Dwa dni ekstremalne: **2023-11-20** (344 tys., +84 % ponad oczekiwanie) i **2025-11-24** (399 tys., +79 %).
- 130 z 135 flag (96 %) to anomalie **dodatnie** – skoki przedświąteczne.

![Anomalie – zespół](images/06_ensemble_anomalies.png)

*Rys. 4. Przychód, wartość oczekiwana i oznaczone anomalie (zespół 6 sygnałów).*

### 6.2 Isolation Forest

37 z 1 820 dni (**2,03 %**) oznaczonych jako anomalie: 27 dni nietypowo wysokich (listopad–grudzień) i 10 nietypowo niskich, głównie **1 stycznia** (2022, 2023, 2025).

![Isolation Forest](images/07_isolation_forest_anomalies.png)

*Rys. 5. Anomalie wykryte przez Isolation Forest.*

**Wniosek:** obie metody zgodnie wskazują okres przedświąteczny jako źródło największych odchyleń oraz Nowy Rok jako dzień nietypowo słaby. Anomalie te mają charakter **kalendarzowy**, a nie awaryjny – model „oczekiwanego przychodu” warto wzbogacić o kalendarz świąt/promocji.

---

## 7. Prognoza sprzedaży

Model: `HistGradientBoostingRegressor` (500 iteracji, learning rate 0,03, 20 liści, regularyzacja L2 = 1). Walidacja chronologiczna na ostatnich **30 dniach** (grudzień 2025 – sezon szczytowy).

| Metryka | Wartość |
|---|---:|
| MAE | 20 477 |
| MAE (% średniego przychodu) | **6,38 %** |
| RMSE | 26 236 |

![Prognoza](images/08_forecast_365_days.png)

*Rys. 6. Dane historyczne, predykcja testowa i prognoza 365-dniowa.*

**Prognoza na 2026:** ≈ **72,7 mln** (+5,6 % wobec 2025), co jest spójne z obserwowanym wyhamowaniem wzrostu (7,0 % → 5,6 %). Model poprawnie odtwarza kształt sezonowy – szczyt w Q4 i przedświąteczny skok.

---

## 8. Analiza klientów

### 8.1 RFM

Klienci z zamówieniami: 49 957. Mediana recency: 78 dni; średnio 10,0 zamówień na klienta; średni przychód na klienta 5 897; średnia marża 24,4 %.
Silnie prawoskośne rozkłady (np. wartość monetarna) wymagały transformacji `log1p` przed skalowaniem:

![Log monetary](images/09_rfm_log_monetary.png)

*Rys. 7. Rozkład logarytmu wartości monetarnej klienta po transformacji.*

### 8.2 Dobór modelu

| Model | Najlepszy wynik silhouette | Uwagi |
|---|---:|---|
| **K-Means (k = 2)** | **0,316** (CH 28 442, DB 1,21) | wybrany |
| GMM (k = 2) | 0,318 | gorszy indeks DB (1,33) |
| DBSCAN (25 konfiguracji) | 0,407 | rozwiązanie zdegenerowane – 96 % klientów w jednym klastrze |

| Silhouette K-Means | Porównanie modeli |
|---|---|
| ![Silhouette](images/10_kmeans_silhouette.png) | ![Porównanie](images/11_clustering_model_comparison.png) |

### 8.3 Segmenty

| Segment | Klienci | Udział | Recency | Zamówienia | Przychód / klient | Marża |
|---|---:|---:|---:|---:|---:|---:|
| **Lojalni o wysokiej wartości** | 30 061 | 60,2 % | 95 dni | 12,5 | 7 792 | 25,4 % |
| **Okazjonalni / niskiej wartości** | 19 896 | 39,8 % | 239 dni | 6,2 | 3 033 | 23,0 % |

Segment lojalny kupuje **2× częściej, ~2,5× więcej i ~2,5× świeżej**, generując **79,5 % przychodu** i **81,2 % zysku**.

| Rzut PCA | Wartość segmentów |
|---|---|
| ![PCA klientów](images/12_customer_segments_pca.png) | ![Wartość](images/13_customer_segment_value.png) |

> **Interpretacja z zastrzeżeniem:** silhouette ≈ 0,32 oznacza separację umiarkowaną. Rzut PCA pokazuje, że klienci tworzą **kontinuum** wzdłuż jednej osi (PC1 ≈ 59 % wariancji), a nie wyraźne skupiska. Dwa segmenty należy traktować jako praktyczny **podział wartościowy do targetowania**, nie jako naturalne grupy.

---

## 9. Analiza produktów

Segmentacja K-Means (k = 4) na sześciu cechach: sprzedane sztuki, zamówienia, przychód, koszt, zysk, marża (silhouette ≈ 0,25).

| Segment | Produkty | Udział | Przychód | Udział w przychodzie | Zysk | Marża |
|---|---:|---:|---:|---:|---:|---:|
| **High-Value** | 2 470 | 24,7 % | 187,2 mln | **63,6 %** | 46,4 mln | 25,2 % |
| **High-Margin** | 2 789 | 27,9 % | 70,0 mln | 23,8 % | 19,2 mln | **28,4 %** |
| **High-Selling / Low-Margin** | 2 550 | 25,5 % | 28,2 mln | 9,6 % | 5,5 mln | **21,2 %** |
| **Low-Performance** | 2 191 | 21,9 % | 9,2 mln | 3,1 % | 2,1 mln | 24,3 % |

| Rzut PCA | Przychód per segment |
|---|---|
| ![PCA produktów](images/14_product_segments_pca.png) | ![Przychód](images/15_product_segment_value.png) |

---

## 10. Analiza koszyka zakupowego

Znaleziono **21 227 unikalnych par** produktów i **42 454 reguły kierunkowe** (próg: min. 100 zamówień na produkt). Średni lift wynosi 19,2 (maks. 94,7), ale **99,4 % par współwystępuje tylko 2–3 razy**; średnie wsparcie ≈ 4,7·10⁻⁶, maksymalna ufność 3,8 %. Tylko 36 reguł opiera się na co najmniej 5 wspólnych zakupach.

![Top pary](images/16_top_product_pairs.png)

*Rys. 8. Najczęściej współwystępujące pary produktów.*

> **Wniosek:** wysokie wartości lift to efekt **małej próby** (bardzo rzadkie współwystępowania), a nie wiarygodne relacje cross-sellingowe. Dla losowo symulowanych koszyków to wynik oczekiwany. Na danych rzeczywistych należy podnieść próg minimalnego wsparcia i dodać test istotności.

---

## 11. Rekomendacje biznesowe

| Priorytet | Rekomendacja | Uzasadnienie |
|:---:|---|---|
| 1 | **Zaplanować zapasy, logistykę i obsługę klienta pod Q4** | Listopad–grudzień = 26,7 % przychodu; Q4 jest o 81 % silniejszy niż Q1 |
| 2 | **Chronić segment lojalny** (programy retencyjne, oferty premium) | 60 % klientów = 79,5 % przychodu i 81,2 % zysku |
| 3 | **Kampanie reaktywacyjne dla segmentu okazjonalnego** | średnia recency 239 dni; przychód na klienta o ~61 % niższy niż w segmencie lojalnym |
| 4 | **Przegląd cen w grupie High-Selling / Low-Margin** | najwyższy wolumen, najniższa marża (21,2 %); wzrost marży o 1 pp w tej grupie ≈ +0,3 mln zysku |
| 5 | **Racjonalizacja asortymentu Low-Performance** | 22 % katalogu, 3 % przychodu |
| 6 | **Ograniczyć anulacje i zwroty** | 8 % zamówień, ≈ 23,7 mln przychodu brutto |
| 7 | **Stymulować sprzedaż weekendową** (−13 % względem dni roboczych) i styczniowy dołek (Q1) | największy niewykorzystany potencjał popytowy |
| 8 | **Monitorować hamowanie wzrostu** (r/r: 10,3 % → 7,0 %) | prognoza 2026 zakłada +5,6 % |

---

## 12. Ograniczenia i ryzyka

**Charakter danych.** Dane są syntetyczne – wzorce (sezonowość, efekt weekendu, trend) zostały wbudowane w generator. Raport demonstruje **metodologię i potencjał platformy**, a nie rzeczywiste zachowania rynku.

**Metodyka.**
- Wskaźniki są **brutto** – nie wyłączono zamówień anulowanych i zwróconych (≈ 23,7 mln).
- Segmentacja klientów ma umiarkowaną separację (silhouette ≈ 0,32).
- Prognoza została zwalidowana na jednym 30-dniowym oknie w sezonie szczytowym i **nie została porównana z prostym modelem bazowym** (np. seasonal naïve); brak przedziałów predykcji.
- Dekompozycja STL z okresem 365 na zaledwie 5 cyklach rocznych może zaniżać reszty na krańcach szeregu (widoczne na rys. 3).
- Reguły asocjacyjne są statystycznie słabe (patrz sekcja 10).

**Spójność repozytorium i odtwarzalność.**
1. Plik `sales_forecast_12_months.csv` zawiera **stałą wartość** (≈ 4,92 mln) dla każdego miesiąca – nie jest zgodny z prognozą 365-dniową (≈ 72,7 mln rocznie). Wymaga poprawy eksportu.
2. W repozytorium **brakuje skryptów DDL** (schematy `staging`/`warehouse`, trzy widoki) oraz **skryptu porównania modeli klastrowania** (K-Means k = 2–10, GMM, DBSCAN), który wygenerował pliki porównawcze.
3. Plik `data/raw/products.csv` nie jest spójny z kosztami użytymi w hurtowni: ceny jednostkowe w `order_details.csv` nie korelują z cennikiem produktów (r ≈ 0), a nazwy produktów w wynikach (np. „Vision Smartphones Plus 09046”) różnią się od tych w CSV. Przychód zgadza się w 100 %, natomiast **zysk i marża z surowych plików CSV nie odtworzą wartości z hurtowni** (≈ 24,9 %). Zalecane jest ponowne wygenerowanie i załadowanie spójnego zestawu danych.

---

## 13. Dalsze kroki

1. Uzupełnić repozytorium o `sql/00_schema.sql` (tabele + widoki) oraz skrypt benchmarku klastrowania.
2. Naprawić eksport prognozy 12-miesięcznej; dodać model bazowy, przedziały predykcji i walidację rolling-origin.
3. Wprowadzić miarę **przychodu netto** (bez anulacji i zwrotów).
4. Dodać kalendarz świąt i promocji do modelu oczekiwanego oraz prognozy.
5. **Rozbudować dashboard Power BI** o wyniki ML: segmenty klientów i produktów, prognozę z przedziałem, anomalie i wskaźniki retencji – z zachowaniem motywu *Ecommerce Dark*.
6. Automatyzacja: orkiestracja (Airflow/Prefect), testy `pytest`, CI w GitHub Actions.
