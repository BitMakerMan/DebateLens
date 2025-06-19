# Specifica di Progetto Didattico: "Polis-Analyzer"

**Progetto a cura di:** Rizzo AI Academy
**Data:** 20 Giugno 2025
**Versione:** 1.0

---

## 1. Introduzione e Visione del Progetto

"Polis-Analyzer" è un progetto didattico finalizzato alla creazione di una web application in grado di eseguire un'analisi retorica e strategica comparativa di due video politici su YouTube. L'applicazione rappresenta un'eccellente opportunità per gli studenti della Rizzo AI Academy di applicare concretamente le proprie competenze nell'integrazione di sistemi, nell'ingegneria dei prompt e nell'automazione, utilizzando uno stack tecnologico moderno (**N8N**, **Railway**, **Python**, **LLM APIs**).

Il nome "Polis-Analyzer" richiama l'antica agorà greca, il cuore del dibattito politico, riletto attraverso la lente dell'intelligenza artificiale.

## 2. Obiettivi Didattici

Al termine del progetto, i partecipanti saranno in grado di:

* **Integrare API Esterne:** Connettere e orchestrare servizi diversi (YouTube, OpenAI/Google, Database) all'interno di un unico flusso logico.
* **Progettare Workflow Complessi:** Padroneggiare **N8N** per creare flussi di automazione che gestiscano logica condizionale, trasformazione dei dati ed esecuzione di script.
* **Sviluppare Prompt Efficaci (Prompt Engineering):** Creare prompt avanzati per modelli di linguaggio generativo (LLM), istruendoli a eseguire analisi strutturate e a restituire output formattati (JSON).
* **Modellare Dati Semplici:** Progettare e implementare uno schema di database per persistere i risultati delle analisi.
* **Collegare Frontend e Backend:** Comprendere come una semplice interfaccia utente possa innescare complessi processi di backend tramite webhook.
* **Lavorare in Gruppo su un Progetto Reale:** Suddividere i compiti, gestire l'integrazione tra i moduli e presentare un prodotto funzionante.

## 3. Architettura della Soluzione e Stack Tecnologico

Il progetto sarà interamente ospitato su **Railway** e sarà composto dai seguenti moduli:

### 1. Frontend (Landing Page)
* **Tecnologia:** HTML, CSS, JavaScript (senza framework complessi per semplicità didattica).
* **Funzione:** Presenta un form semplice dove l'utente può inserire gli URL di due video YouTube. Al submit, invia i dati a un webhook di N8N.
* **Hosting:** Servito come sito statico su **Railway**.

### 2. Backend (Core Logic)
* **Tecnologia:** **N8N**.
* **Funzione:** È il cuore del sistema. Un workflow N8N riceve i dati dalla landing page, orchestra le chiamate alle API per la trascrizione e l'analisi, esegue script Python, e salva i risultati nel database.
* **Hosting:** Deployato come servizio su **Railway**.

### 3. Database
* **Tecnologia:** **PostgreSQL** (o un altro DB offerto da Railway).
* **Funzione:** Persiste i risultati di ogni analisi, consentendo di creare uno storico e di recuperare le analisi passate.
* **Hosting:** Aggiunto come plugin al progetto **Railway**.

### 4. Servizi Ausiliari (Scripting)
* **Tecnologia:** **Python**.
* **Funzione:** Eseguito da N8N per compiti specifici che sono più semplici da realizzare in codice, come la generazione del grafico radar.
* **Hosting:** Il codice Python può essere incluso nel container N8N o eseguito tramite un nodo `Execute Command`.

### 5. API Esterne
* **Google APIs:** Potenzialmente `YouTube Data API v3` per ottenere metadati dei video.
* **OpenAI / Google Gemini API:** Per l'analisi del testo tramite LLM.
* **Servizi di Trascrizione:** API di OpenAI (Whisper), AssemblyAI, o estrazione diretta da YouTube.

## 4. Flusso Dati (End-to-End)

1.  L'utente visita la landing page di "Polis-Analyzer".
2.  Inserisce i due URL dei video politici e clicca "Analizza".
3.  Il JavaScript della pagina invia una richiesta `POST` con i due URL all'endpoint del **Webhook di N8N**.
4.  Il workflow N8N si avvia.
5.  **Step 1 (N8N):** Ottiene le trascrizioni dei due video.
6.  **Step 2 (N8N):** Per ogni trascrizione, invia una richiesta all'API di **OpenAI/Gemini** usando il prompt politico strutturato per l'analisi.
7.  **Step 3 (N8N):** Riceve le due analisi in formato JSON e le unifica.
8.  **Step 4 (N8N):** Esegue un **nodo `Execute Command`** che lancia uno script Python per generare il grafico radar.
9.  **Step 5 (N8N):** Si connette al **database PostgreSQL** e inserisce una nuova riga nella tabella `analyses` con tutti i dati.
10. **Step 6 (N8N):** Genera un URL univoco per la pagina dei risultati (es. `https://polis-analyzer.app/results/ANALISI_ID`) e lo restituisce alla landing page iniziale.
11. La landing page reindirizza l'utente alla pagina dei risultati, che recupererà i dati dal database per visualizzare l'analisi.

## 5. Definizione dei Ruoli e dei Task (Attività di Gruppo)

* **Team Frontend (1-2 persone):**
    * **Task:** Progettare e sviluppare la landing page (HTML/CSS/JS), la chiamata al webhook e la pagina di visualizzazione dei risultati.
* **Team Backend / N8N Specialist (2 persone):**
    * **Task:** Progettare e implementare il workflow principale in N8N, configurare il webhook, i nodi HTTP e la gestione degli errori.
* **Team AI & Prompt Engineering (1 persona):**
    * **Task:** Raffinare e testare iterativamente il prompt politico per massimizzare l'accuratezza e l'affidabilità dell'analisi dell'LLM.
* **Team Dati & Python (1 persona):**
    * **Task:** Progettare lo schema del database PostgreSQL e scrivere lo script Python (`matplotlib`) per generare il grafico radar.

## 6. Schema del Database (Esempio PostgreSQL)

```sql
CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    video1_url VARCHAR(255) NOT NULL,
    video2_url VARCHAR(255) NOT NULL,
    video1_title VARCHAR(255),
    video2_title VARCHAR(255),
    analysis_data_json JSONB, -- Qui salviamo l'intero JSON di risposta dell'AI
    radar_chart_url VARCHAR(255), -- URL del grafico generato
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
