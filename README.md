# DebateLens AI - Sistema Avanzato di Analisi Comunicativa

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![Google Gemini AI](https://img.shields.io/badge/Google%20Gemini-AI-orange.svg)](https://ai.google.dev/gemini)
[![Rizzo AI Academy](https://img.shields.io/badge/Rizzo%20AI-Academy-red.svg)](https://rizzo.ai)

**Craicek's Version** - Sviluppato per Rizzo AI Academy

---

## 🎯 Panoramica del Progetto

**DebateLens AI** è un'applicazione web avanzata basata su Flask progettata per eseguire **analisi comparative della comunicazione** utilizzando l'intelligenza artificiale di Google Gemini. Sviluppata come "Craicek's Version" per la Rizzo AI Academy, l'applicazione valuta automaticamente i testi di diversi partecipanti su 6 criteri comunicativi specifici.

L'obiettivo principale è fornire un'**analisi obiettiva e dettagliata** di come i diversi oratori comunicano, evidenziando punti di forza e debolezza attraverso metriche scientificamente validate e visualizzazioni grafiche avanzate.

## ✨ Funzionalità Principali

### 🤖 **Analisi AI Avanzata**
- Utilizza **Google Gemini AI** (`gemini-1.5-flash`) per analisi contestuale approfondita
- Assegnazione automatica di punteggi da 1-10 per ogni criterio
- Generazione di spiegazioni dettagliate per ogni valutazione

### 📊 **Metriche di Valutazione**
Il sistema analizza la comunicazione su **6 criteri chiave**:

1. **Rigorosità Tecnica** - Precisione terminologica e concetti specialistici
2. **Uso di Dati Oggettivi** - Statistiche, ricerche, fonti verificabili
3. **Approccio Divulgativo** - Accessibilità, esempi, analogie
4. **Stile Comunicativo** - Fluidità e capacità di coinvolgimento
5. **Focalizzazione Argomento** - Aderenza al tema, coerenza logica
6. **Orientamento Pratico** - Soluzioni concrete, applicabilità

### 🎨 **Visualizzazione Avanzata**
- **Radar Chart comparativo** con palette di 20+ colori distintivi
- **Design tech-inspired** con gradiente rosso-oro
- **Gestione automatica** di layout per 2-20+ partecipanti
- **Distinzione visiva** tramite colori, linee e marker diversi

### 📈 **Report Intelligenti**
- **Statistiche comparative** automatiche per ogni criterio
- **Insights AI** con identificazione automatica di punti di forza
- **Analisi di equilibrio** tra partecipanti
- **Export timestamp** per tracciabilità

### 🌐 **API RESTful**
- Endpoint `/api/analyze` per analisi batch
- Formato JSON standardizzato per integrazione
- Health check su `/api/health`
- Gestione errori robusta con diagnostica dettagliata

## 🚀 Installazione e Configurazione

### Prerequisiti
- **Python 3.9+**
- **Git**
- **Account Google AI Studio** per API key

### 1. Clone del Repository
```bash
git clone https://github.com/tuousername/DebateLens.git
cd DebateLens
```

### 2. Ambiente Virtuale
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Dipendenze
```bash
pip install -r requirements.txt
```

### 4. Configurazione API Key
1. Vai su [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Genera la tua API Key
3. Crea file `.env` nella root del progetto:

```env
GOOGLE_API_KEY=la_tua_api_key_qui
```

### 5. Avvio Applicazione
```bash
python app.py
```

L'applicazione sarà disponibile su `http://localhost:5000`

## 💻 Utilizzo dell'API

### Endpoint Principale
**POST** `/api/analyze`

#### Request Body
```json
{
  "participants": [
    {
      "name": "Mario Rossi",
      "text": "Il testo del discorso del primo partecipante..."
    },
    {
      "name": "Anna Verdi", 
      "text": "Il testo del discorso del secondo partecipante..."
    }
  ]
}
```

#### Response
```json
{
  "success": true,
  "timestamp": "20241222_144530",
  "participants_count": 2,
  "chart_data": "base64_encoded_png_image",
  "report": {
    "summary": {
      "rigorosita_tecnica": {
        "average": 7.5,
        "max_participant": "Mario Rossi",
        "max_score": 8
      }
    },
    "detailed_comparison": {
      "Mario Rossi": {
        "scores": { "rigorosita_tecnica": 8, ... },
        "explanations": { ... }
      }
    },
    "insights": [
      "🔥 Mario Rossi eccelle in Rigorosità Tecnica (8/10)",
      "⚡ Anna Verdi si distingue per: approccio divulgativo"
    ]
  },
  "version": "DebateLens Craicek's Version",
  "powered_by": "Google Gemini AI"
}
```

## 🛠️ Architettura Tecnica

### Backend (Flask)
- **Gestione robusta dei tipi** con validazione automatica
- **Conversione sicura** da stringhe AI a numeri interi
- **Error handling** completo con logging dettagliato
- **Matplotlib** per generazione grafici server-side

### Frontend (Vanilla JS)
- **Validazione real-time** con feedback visivo
- **Interfaccia responsive** ottimizzata per mobile
- **Animazioni fluide** con CSS3 e JavaScript
- **Easter egg** nascosto (Konami Code)

### AI Integration
- **Google Gemini 1.5 Flash** per analisi testuale
- **Prompt engineering** ottimizzato per consistenza
- **Temperature 0.2** per output deterministici
- **JSON schema validation** per struttura dati

## 📋 Requisiti di Sistema

### Dipendenze Principali
```
Flask==3.1.1
google-generativeai==0.8.5
matplotlib==3.10.3
numpy==2.3.1
python-dotenv==1.1.0
flask-cors==6.0.1
```

### Specifiche Tecniche
- **Memoria**: 512MB+ disponibili
- **CPU**: Supporto multi-core raccomandato
- **Storage**: 100MB per installazione base
- **Network**: Connessione internet per API Google

## 🔧 Configurazione Avanzata

### Variabili Ambiente (.env)
```env
# Google AI API Key (obbligatoria)
GOOGLE_API_KEY=your_api_key_here

# Flask Configuration
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Gemini Configuration
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.2
GEMINI_MAX_TOKENS=1200

# Logging
LOG_LEVEL=INFO
LOG_FILE=debatelens.log
```

### Personalizzazione Palette Colori
Modifica il file `app.py`, funzione `create_radar_chart()`:

```python
colors = [
    '#dc2626',  # Rosso primario
    '#fbbf24',  # Oro tech
    '#custom',  # Il tuo colore personalizzato
    # ... altri colori
]
```

## 🧪 Testing e Validazione

### Test di Sistema
```bash
# Health check
curl http://localhost:5000/api/health

# Test analisi semplice
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"participants":[{"name":"Test","text":"Testo di prova lungo almeno cinquanta caratteri per superare la validazione minima"}]}'
```

### Validazioni Implementate
- ✅ **Lunghezza minima testo**: 50 caratteri
- ✅ **Campi obbligatori**: nome e testo per ogni partecipante
- ✅ **Range punteggi**: 1-10 per tutti i criteri
- ✅ **Formato JSON**: validazione struttura response
- ✅ **Gestione errori**: timeout, connessione, parsing

## 🐛 Troubleshooting

### Problemi Comuni

#### API Key non funziona
```bash
# Verifica configurazione
echo $GOOGLE_API_KEY

# Test diretto API
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$GOOGLE_API_KEY
```

#### Errori matplotlib
```bash
# Installa dipendenze sistema (Ubuntu/Debian)
sudo apt-get install python3-dev python3-matplotlib

# Windows: reinstalla matplotlib
pip uninstall matplotlib
pip install matplotlib
```

#### Port già in uso
```bash
# Trova processo su porta 5000
lsof -i :5000
kill -9 <PID>

# Oppure cambia porta
export FLASK_PORT=8080
python app.py
```

## 📈 Performance e Scalabilità

### Ottimizzazioni Implementate
- **Caching matplotlib** con cleanup automatico
- **Validazione input** pre-processing
- **Error handling** senza blocking
- **Memory management** per grafici

### Limiti Attuali
- **Concorrenza**: Single-threaded Flask dev server
- **Rate limiting**: Dipende da quota Google API
- **Memoria**: ~50MB per grafico complesso
- **Timeout**: 30s per analisi AI

### Scaling Suggestions
```python
# Per produzione, usa Gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app

# O uvicorn per async
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

## 🤝 Contributi e Sviluppo

### Struttura Progetto
```
DebateLens/
├── app.py              # Flask application principale
├── index.html          # Frontend SPA
├── requirements.txt    # Dipendenze Python
├── .env.example       # Template configurazione
├── .gitignore         # Git ignore rules
└── README.md          # Questa documentazione
```

### Workflow di Sviluppo
1. **Fork** del repository
2. **Feature branch**: `git checkout -b feature/nome-feature`
3. **Commit** con messaggi descrittivi
4. **Test** locale completo
5. **Pull Request** con descrizione dettagliata

### Code Style
- **PEP 8** per Python
- **ESLint** per JavaScript
- **Docstrings** per tutte le funzioni
- **Type hints** quando possibile

## 📄 Licenza e Credits

### Sviluppo
- **Craicek** - Sviluppatore principale
- **Rizzo AI Academy** - Sponsor e testing

### Tecnologie
- **Google Gemini AI** - Motore di analisi
- **Flask** - Web framework
- **Matplotlib** - Visualizzazione grafici
- **Bootstrap styling** - CSS framework base

---

## 🔥 Versioni e Changelog

### v1.0.0 (Craicek's Version)
- ✅ **Analisi AI completa** con Google Gemini
- ✅ **Radar chart** con 20+ colori distintivi
- ✅ **Sistema di validazione** robusto
- ✅ **Interface responsive** con animazioni
- ✅ **API RESTful** per integrazioni
- ✅ **Error handling** completo

### Roadmap Futura
- 🔲 **Supporto multi-lingua** per analisi
- 🔲 **Export PDF** dei report
- 🔲 **Dashboard analytics** avanzata
- 🔲 **Integration webhook** per notifiche
- 🔲 **Batch processing** per grandi dataset

---

<div align="center">

**⚡ Sviluppato da CRAICEK per Rizzo AI Academy**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/tuousername/DebateLens)
[![Documentation](https://img.shields.io/badge/Docs-Latest-blue.svg)](https://github.com/tuousername/DebateLens/wiki)

</div>
