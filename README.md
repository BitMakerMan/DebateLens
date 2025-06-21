# DebateLens - Analisi Comparativa della Comunicazione

[](https://www.python.org/)  
[](https://flask.palletsprojects.com/)  
[](https://www.google.com/search?q=https://ai.google.dev/gemini)  
[](https://www.google.com/search?q=https://rizzo.ai)

---

## 🎯 Panoramica del Progetto

**DebateLens** è un'applicazione web basata su Flask progettata per eseguire **analisi comparative della comunicazione**. Sviluppata come "Craicek's Version" per la Rizzo AI Academy, usa l'intelligenza artificiale (tramite Google Gemini AI) o un'analisi euristica di fallback per valutare i testi di diversi partecipanti su criteri specifici.

Lo scopo principale del progetto è fornire un'analisi obiettiva e dettagliata di come i diversi oratori comunicano, evidenziando **punti di forza e debolezza** su metriche come rigorosità tecnica, uso di dati oggettivi, approccio divulgativo, stile comunicativo, focalizzazione sull'argomento e orientamento pratico. I risultati vengono presentati tramite un **report comparativo** e un'accattivante **visualizzazione grafica a radar**.

## ✨ Funzionalità Principali

- **Analisi AI Avanzata:** Utilizza l'API di Google Gemini (modello `gemini-1.5-flash`) per un'analisi approfondita e contestuale dei testi, assegnando punteggi e generando spiegazioni per ogni criterio.
- **Analisi Euristica di Fallback:** Se non c'è una chiave API Gemini valida, il sistema ricade su un'analisi euristica intelligente che valuta i testi basandosi su parole chiave e struttura, garantendo comunque un'analisi funzionale.
- **Metriche di Valutazione Dettagliate:** Valuta la comunicazione su 6 criteri chiave:
  - Rigorosità Tecnica
  - Uso di Dati Oggettivi
  - Approccio Divulgativo
  - Stile Comunicativo
  - Focalizzazione Argomento
  - Orientamento Pratico
- **Visualizzazione Comparativa:** Genera un grafico a radar (stile "Iron Man") per visualizzare le performance comparative di più partecipanti.
- **Report Comparativo:** Produce un report testuale con riepiloghi, confronti dettagliati e insight automatici tra i partecipanti.
- **API RESTful:** Espone un endpoint `/api/analyze` per ricevere i dati dei partecipanti e restituire i risultati dell'analisi in formato JSON.
- **Interfaccia Web Semplice:** Il progetto include un front-end minimale per l'interazione (servito da `index.html` e altri file statici).

## 🚀 Come Avviare il Progetto

Segui questi passaggi per configurare ed eseguire DebateLens sul tuo sistema locale.

### Prerequisiti

- **Python 3.9+** installato.
- **Git** installato (se non lo hai già).

### 1. Clona il Repository

Se non l'hai ancora fatto, clona il repository sul tuo computer:

```bash
git clone https://github.com/BitMakerMan/DebateLens.git
cd DebateLens
```

### 2. Configura l'Ambiente Virtuale

È consigliabile creare e attivare un ambiente virtuale per gestire le dipendenze:

```bash
python -m venv venv
# Su Windows:
venv\Scripts\activate
# Su macOS/Linux:
source venv/bin/activate
```

### 3. Installa le Dipendenze

Installa tutte le librerie Python richieste. Assicurati di avere un file `requirements.txt` nella directory principale del progetto. Se non ce l'hai, puoi crearlo con le seguenti librerie:

```bash
pip install Flask Flask-Cors python-dotenv google-generativeai matplotlib numpy
```

Oppure, se hai un file `requirements.txt` (consigliato):

```bash
pip install -r requirements.txt
```

### 4. Configura la Chiave API di Google Gemini (Opzionale ma Consigliato)

Per usare le funzionalità complete di analisi AI basate su Google Gemini, devi ottenere una chiave API e configurarla.

1. Vai su [Google AI Studio](https://aistudio.google.com/app/apikey) per generare la tua API Key.
2. Crea un file `.env` nella directory principale del tuo progetto (`DebateLens/`) e aggiungi la tua chiave API in questo formato:

```
GOOGLE_API_KEY=LaTuaChiaveAPIQui
```

Se non configuri la chiave API o la imposti su `your-google-api-key-here`, l'applicazione funzionerà in modalità di analisi euristica.

### 5. Avvia l'Applicazione

Una volta configurato tutto, puoi avviare l'applicazione Flask:

```bash
python app.py
```

Vedrai un output nel terminale simile a questo:

```
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
🎯 DebateLens - Craicek's Version
🚀 Rizzo AI Academy
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
✅ Google Gemini AI configurato
🤖 Modalità: Analisi AI completa
🌐 Server: http://localhost:5000
============================================================
```

L'applicazione sarà accessibile tramite il tuo browser all'indirizzo: `http://localhost:5000`.

---

## 💻 Utilizzo dell'API

L'endpoint principale per l'analisi è `/api/analyze`. Accetta richieste `POST` con un corpo JSON.

**Endpoint:** `POST /api/analyze`  
**Corpo della Richiesta (JSON):**

```json
{
  "participants": [
    {
      "name": "Nome Partecipante 1",
      "text": "Il testo del discorso o del commento del primo partecipante."
    },
    {
      "name": "Nome Partecipante 2",
      "text": "Il testo del discorso o del commento del secondo partecipante."
    }
  ]
}
```

**Esempio di Risposta (JSON):**

La risposta includerà un timestamp, il conteggio dei partecipanti, i dati del grafico a radar (codificati in base64), il report comparativo e informazioni sulla versione e la modalità AI.

```json
{
  "timestamp": "YYYYMMDD_HHMMSS",
  "participants_count": 2,
  "chart_data": "base64_encoded_png_image",
  "report": {
    "summary": {
      "rigorosita_tecnica": { ... }
      // ...altri criteri
    },
    "detailed_comparison": {
      "Nome Partecipante 1": { ... },
      "Nome Partecipante 2": { ... }
    },
    "insights": [
      "🔥 Insight 1",
      "⚡ Insight 2"
    ]
  },
  "version": "DebateLens Craicek's Version",
  "ai_mode": true
}
```

---

## 🛠️ Contributi

Questo progetto è sviluppato come "Craicek's Version ".

---
