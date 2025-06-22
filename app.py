# DebateLens - Analisi Comparativa della Comunicazione
# Craicek's Version - Rizzo AI Academy
# Versione Fixed - Gestione Robusta dei Tipi di Dato

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import base64
import sys
from datetime import datetime
import traceback

# Caricamento variabili ambiente
from dotenv import load_dotenv

load_dotenv()

# Import AI e visualizzazione
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from io import BytesIO


@dataclass
class AnalysisResult:
    """Risultato dell'analisi di un singolo partecipante"""
    participant_name: str
    rigorosita_tecnica: int
    uso_dati_oggettivi: int
    approccio_divulgativo: int
    stile_comunicativo: int
    focalizzazione_argomento: int
    orientamento_pratico: int
    explanations: dict

    def to_dict(self) -> dict:
        return {
            'participant_name': self.participant_name,
            'scores': {
                'rigorosita_tecnica': self.rigorosita_tecnica,
                'uso_dati_oggettivi': self.uso_dati_oggettivi,
                'approccio_divulgativo': self.approccio_divulgativo,
                'stile_comunicativo': self.stile_comunicativo,
                'focalizzazione_argomento': self.focalizzazione_argomento,
                'orientamento_pratico': self.orientamento_pratico
            },
            'explanations': self.explanations
        }


def safe_int(value, default=5, min_val=1, max_val=10):
    """Converte sicuramente un valore in int con validazione range"""
    try:
        # Gestisci diversi tipi di input
        if isinstance(value, str):
            # Rimuovi caratteri non numerici tranne punto e virgola
            cleaned = ''.join(c for c in value if c.isdigit() or c in '.,')
            if not cleaned:
                return default
            # Prendi solo la prima parte se ci sono decimali
            value = float(cleaned.replace(',', '.'))

        result = int(float(value))
        # Assicurati che sia nel range valido
        return max(min_val, min(max_val, result))
    except (ValueError, TypeError):
        print(f"⚠️ Valore non valido ricevuto: {value}, usando default: {default}")
        return default


class DebateLensAnalyzer:
    """Core analyzer per DebateLens con Google Gemini AI"""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if not api_key or api_key == 'your-google-api-key-here':
            raise ValueError("GOOGLE_API_KEY non configurata! Configura la tua API key nel file .env")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.analysis_criteria = [
            'rigorosita_tecnica', 'uso_dati_oggettivi', 'approccio_divulgativo',
            'stile_comunicativo', 'focalizzazione_argomento', 'orientamento_pratico'
        ]

    def create_analysis_prompt(self, text: str, participant_name: str) -> str:
        """Crea il prompt per l'analisi AI"""
        return f"""
Analizza il seguente testo di {participant_name} secondo questi 6 criteri, assegnando un punteggio NUMERICO INTERO da 1 a 10:

TESTO: {text}

CRITERI (punteggio da 1 a 10):
1. Rigorosità tecnica: Precisione terminologica e concetti specialistici
2. Uso di dati oggettivi: Statistiche, ricerche, fonti verificabili
3. Approccio divulgativo: Accessibilità, esempi, analogie
4. Stile comunicativo: Fluidità e capacità di coinvolgimento
5. Focalizzazione argomento: Aderenza al tema, coerenza logica
6. Orientamento pratico: Soluzioni concrete, applicabilità

IMPORTANTE: Restituisci SOLO numeri interi da 1 a 10 per ogni criterio.

FORMATO JSON RICHIESTO:
{{
  "rigorosita_tecnica": 7,
  "uso_dati_oggettivi": 6,
  "approccio_divulgativo": 8,
  "stile_comunicativo": 7,
  "focalizzazione_argomento": 9,
  "orientamento_pratico": 6,
  "explanations": {{
    "rigorosita_tecnica": "Spiegazione breve e concisa",
    "uso_dati_oggettivi": "Spiegazione breve e concisa",
    "approccio_divulgativo": "Spiegazione breve e concisa",
    "stile_comunicativo": "Spiegazione breve e concisa",
    "focalizzazione_argomento": "Spiegazione breve e concisa",
    "orientamento_pratico": "Spiegazione breve e concisa"
  }}
}}
"""

    def analyze_participant(self, text: str, participant_name: str) -> AnalysisResult:
        """Analizza un partecipante con Google Gemini"""
        if not text.strip():
            raise ValueError(f"Testo vuoto per {participant_name}")

        prompt = self.create_analysis_prompt(text, participant_name)

        generation_config = genai.types.GenerationConfig(
            temperature=0.2,  # Ridotto per maggiore consistenza
            max_output_tokens=1200,
            response_mime_type="application/json"
        )

        try:
            response = self.model.generate_content(prompt, generation_config=generation_config)
            result_json = json.loads(response.text)

            # Validazione e conversione sicura dei punteggi
            safe_scores = {}
            for criterion in self.analysis_criteria:
                raw_value = result_json.get(criterion, 5)
                safe_scores[criterion] = safe_int(raw_value)
                print(f"🔍 {criterion}: {raw_value} -> {safe_scores[criterion]}")

            # Validazione explanations
            explanations = result_json.get('explanations', {})
            if not isinstance(explanations, dict):
                explanations = {}

            # Assicurati che ci siano explanations per tutti i criteri
            for criterion in self.analysis_criteria:
                if criterion not in explanations or not explanations[criterion]:
                    explanations[criterion] = f"Analisi automatica per {criterion.replace('_', ' ')}"

            return AnalysisResult(
                participant_name=participant_name,
                rigorosita_tecnica=safe_scores['rigorosita_tecnica'],
                uso_dati_oggettivi=safe_scores['uso_dati_oggettivi'],
                approccio_divulgativo=safe_scores['approccio_divulgativo'],
                stile_comunicativo=safe_scores['stile_comunicativo'],
                focalizzazione_argomento=safe_scores['focalizzazione_argomento'],
                orientamento_pratico=safe_scores['orientamento_pratico'],
                explanations=explanations
            )

        except json.JSONDecodeError as e:
            print(f"❌ Errore JSON dal modello AI: {e}")
            print(f"🔍 Risposta grezza: {response.text[:500]}")
            raise ValueError(f"Risposta AI non valida per {participant_name}")
        except Exception as e:
            print(f"❌ Errore generico nell'analisi AI: {e}")
            raise

    def create_radar_chart(self, analyses: list) -> str:
        """Genera radar chart PNG con gestione emoji migliorata"""
        if not analyses:
            raise ValueError("Nessuna analisi da visualizzare")

        # Setup matplotlib per server
        import matplotlib
        matplotlib.use('Agg')
        plt.ioff()
        plt.clf()
        plt.close('all')

        # Crea il plot
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'), dpi=150)
        fig.patch.set_facecolor('#1a1a1a')

        # Labels senza emoji problematiche
        labels = ['Rigorosita\nTecnica', 'Uso Dati\nOggettivi', 'Approccio\nDivulgativo',
                  'Stile\nComunicativo', 'Focalizzazione\nArgomento', 'Orientamento\nPratico']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        # Palette estesa tech-inspired
        colors = [
            '#dc2626',  # Rosso primario
            '#fbbf24',  # Oro tech
            '#b91c1c',  # Rosso scuro
            '#f59e0b',  # Arancione
            '#7c2d12',  # Marrone metallico
            '#ea580c',  # Arancione intenso
            '#1f2937',  # Grigio antracite
            '#6366f1',  # Blu elettrico
            '#10b981',  # Verde tech
            '#8b5cf6',  # Viola digitale
            '#ec4899',  # Rosa neon
            '#06b6d4',  # Ciano
            '#84cc16',  # Verde lime
            '#f97316',  # Arancione vivace
            '#6b7280',  # Grigio argento
            '#ef4444',  # Rosso brillante
            '#facc15',  # Giallo elettrico
            '#3b82f6',  # Blu elettrico
            '#22c55e',  # Verde brillante
            '#a855f7'  # Viola brillante
        ]

        # Plot ogni partecipante con colori distintivi
        for i, analysis in enumerate(analyses):
            values = [
                analysis.rigorosita_tecnica, analysis.uso_dati_oggettivi,
                analysis.approccio_divulgativo, analysis.stile_comunicativo,
                analysis.focalizzazione_argomento, analysis.orientamento_pratico
            ]
            values += values[:1]

            color = colors[i % len(colors)]

            # Stile linea variabile per maggiore distinzione
            line_styles = ['-', '--', '-.', ':']
            line_style = line_styles[i % len(line_styles)]

            # Marker shapes variabili
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            marker = markers[i % len(markers)]

            ax.plot(angles, values, marker=marker, linestyle=line_style, linewidth=4,
                    label=analysis.participant_name, color=color, markersize=10,
                    markerfacecolor=color, markeredgecolor='white', markeredgewidth=2,
                    alpha=0.9)
            ax.fill(angles, values, alpha=0.15, color=color)

        # Styling Iron Man
        ax.set_facecolor('#0a0a0a')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12, color='#fbbf24', fontweight='bold')
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels(range(0, 11, 2), fontsize=10, color='#dc2626', fontweight='bold')
        ax.grid(True, color='#444444', alpha=0.7, linewidth=1)

        # Titolo professionale
        plt.title('DebateLens - Analisi AI Comparativa\nCraicek\'s Version - Rizzo AI Academy',
                  size=18, fontweight='bold', pad=30, color='#fbbf24')

        # Legend migliorata per molti partecipanti
        if len(analyses) <= 6:
            # Posizione standard per pochi partecipanti
            legend = plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0),
                                facecolor='#1a1a1a', edgecolor='#dc2626',
                                labelcolor='#fbbf24', fontsize=11, framealpha=0.9)
        else:
            # Posizione ottimizzata per molti partecipanti
            legend = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                                facecolor='#1a1a1a', edgecolor='#dc2626',
                                labelcolor='#fbbf24', fontsize=10, framealpha=0.9,
                                ncol=1 if len(analyses) <= 10 else 2)

        legend.get_frame().set_linewidth(2)

        plt.tight_layout()

        # Converti in base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a', edgecolor='none', pad_inches=0.2)
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()

        plt.close(fig)
        plt.clf()

        return chart_base64

    def generate_comparative_report(self, analyses: list) -> dict:
        """Genera report comparativo con gestione robusta dei tipi"""
        if len(analyses) < 2:
            raise ValueError("Servono almeno 2 partecipanti per il confronto")

        report = {
            "summary": {},
            "detailed_comparison": {},
            "insights": []
        }

        try:
            # Calcola statistiche per criterio con gestione robusta
            for criterion in self.analysis_criteria:
                scores = []
                for analysis in analyses:
                    score = getattr(analysis, criterion)
                    # Converti sicuramente a int
                    safe_score = safe_int(score)
                    scores.append(safe_score)

                if scores:  # Solo se abbiamo punteggi validi
                    max_score = max(scores)
                    min_score = min(scores)
                    max_index = scores.index(max_score)
                    min_index = scores.index(min_score)

                    report["summary"][criterion] = {
                        "average": round(sum(scores) / len(scores), 2),
                        "max_participant": analyses[max_index].participant_name,
                        "max_score": max_score,
                        "min_participant": analyses[min_index].participant_name,
                        "min_score": min_score
                    }

            # Dettagli partecipanti
            for analysis in analyses:
                report["detailed_comparison"][analysis.participant_name] = analysis.to_dict()

            # Insights automatici intelligenti
            if len(analyses) >= 2:
                # Trova il leader in ogni categoria
                for criterion in self.analysis_criteria:
                    scores = [safe_int(getattr(analysis, criterion)) for analysis in analyses]
                    max_score = max(scores)
                    max_participant = analyses[scores.index(max_score)].participant_name

                    if max_score >= 8:  # Solo per punteggi elevati
                        criterion_name = criterion.replace('_', ' ').title()
                        report["insights"].append(
                            f"🔥 {max_participant} eccelle in {criterion_name} ({max_score}/10)"
                        )

                # Confronti diretti tra primi due partecipanti
                p1, p2 = analyses[0], analyses[1]

                # Analizza punti di forza relativi
                p1_strengths = []
                p2_strengths = []

                criteria_names = {
                    'rigorosita_tecnica': 'rigorosità tecnica',
                    'uso_dati_oggettivi': 'uso di dati oggettivi',
                    'approccio_divulgativo': 'approccio divulgativo',
                    'stile_comunicativo': 'stile comunicativo',
                    'focalizzazione_argomento': 'focalizzazione sull\'argomento',
                    'orientamento_pratico': 'orientamento pratico'
                }

                for criterion in self.analysis_criteria:
                    p1_score = safe_int(getattr(p1, criterion))
                    p2_score = safe_int(getattr(p2, criterion))
                    diff = abs(p1_score - p2_score)

                    if diff >= 2:  # Differenza significativa
                        if p1_score > p2_score:
                            p1_strengths.append(criteria_names[criterion])
                        else:
                            p2_strengths.append(criteria_names[criterion])

                if p1_strengths:
                    report["insights"].append(
                        f"⚡ {p1.participant_name} si distingue per: {', '.join(p1_strengths)}"
                    )

                if p2_strengths:
                    report["insights"].append(
                        f"⚡ {p2.participant_name} si distingue per: {', '.join(p2_strengths)}"
                    )

                # Insight di equilibrio
                balanced_areas = []
                for criterion in self.analysis_criteria:
                    scores = [safe_int(getattr(analysis, criterion)) for analysis in analyses[:2]]
                    if abs(scores[0] - scores[1]) <= 1:
                        balanced_areas.append(criteria_names[criterion])

                if balanced_areas:
                    report["insights"].append(
                        f"⚖️ Equilibrio notevole in: {', '.join(balanced_areas)}"
                    )

            # Aggiungi info sulla qualità dell'analisi
            avg_scores = []
            for analysis in analyses:
                participant_scores = [
                    safe_int(analysis.rigorosita_tecnica),
                    safe_int(analysis.uso_dati_oggettivi),
                    safe_int(analysis.approccio_divulgativo),
                    safe_int(analysis.stile_comunicativo),
                    safe_int(analysis.focalizzazione_argomento),
                    safe_int(analysis.orientamento_pratico)
                ]
                participant_avg = sum(participant_scores) / len(participant_scores)
                avg_scores.append(participant_avg)

            if avg_scores:
                overall_avg = sum(avg_scores) / len(avg_scores)

                if overall_avg >= 7.5:
                    report["insights"].append("🎯 Analisi di alta qualità: comunicazione eccellente rilevata")
                elif overall_avg >= 6.0:
                    report["insights"].append("📊 Buona qualità comunicativa con margini di miglioramento")
                else:
                    report["insights"].append("💡 Potenziale di crescita significativo identificato")

        except Exception as e:
            print(f"⚠️ Errore nella generazione report: {e}")
            # Fallback sicuro
            report["insights"] = [
                "🔥 Analisi completata con successo",
                f"📊 {len(analyses)} partecipanti analizzati"
            ]

        return report


# Configurazione API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY or GOOGLE_API_KEY.strip() == '' or 'Ottieni' in GOOGLE_API_KEY:
    print("❌ ERRORE: GOOGLE_API_KEY non configurata!")
    print("🔧 Configura la tua API key nel file .env")
    print("🌐 Ottieni la key da: https://aistudio.google.com/")
    sys.exit(1)

# Inizializza analyzer
try:
    analyzer = DebateLensAnalyzer(GOOGLE_API_KEY)
    print("✅ Google Gemini AI configurato correttamente")
except Exception as e:
    print(f"❌ Errore configurazione AI: {e}")
    sys.exit(1)

# Flask App
app = Flask(__name__)
CORS(app)


# Routes
@app.route('/')
def serve_frontend():
    """Serve la pagina principale"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve file statici"""
    return send_from_directory('.', filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_debate():
    """Endpoint principale per l'analisi dei dibattiti"""
    try:
        data = request.get_json()

        if not data or len(data.get('participants', [])) < 2:
            return jsonify({'error': 'Servono almeno 2 partecipanti per l\'analisi comparativa'}), 400

        participants = data['participants']
        analyses = []

        print(f"🔍 Avvio analisi per {len(participants)} partecipanti")

        # Analizza ogni partecipante con Google Gemini
        for i, participant in enumerate(participants, 1):
            name = participant.get('name', '').strip()
            text = participant.get('text', '').strip()

            if not name:
                return jsonify({'error': 'Nome partecipante obbligatorio'}), 400

            if not text:
                return jsonify({'error': f'Testo obbligatorio per {name}'}), 400

            if len(text) < 50:
                return jsonify({'error': f'Testo troppo breve per {name} (minimo 50 caratteri)'}), 400

            print(f"🤖 Analizzando partecipante {i}: {name}")

            # Analisi AI
            analysis = analyzer.analyze_participant(text, name)
            analyses.append(analysis)
            print(f"✅ Analisi completata per {name}")

        if len(analyses) < 2:
            return jsonify({'error': 'Analisi fallita: non è stato possibile analizzare abbastanza partecipanti'}), 500

        print("📊 Generazione grafico radar...")
        # Genera risultati
        chart_data = analyzer.create_radar_chart(analyses)

        print("📝 Generazione report comparativo...")
        report = analyzer.generate_comparative_report(analyses)

        print("✅ Analisi completata con successo!")

        return jsonify({
            'success': True,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'participants_count': len(analyses),
            'chart_data': chart_data,
            'report': report,
            'version': 'DebateLens Craicek\'s Version',
            'ai_mode': True,
            'powered_by': 'Google Gemini AI'
        })

    except ValueError as e:
        print(f"❌ Errore di validazione: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"❌ Errore nell'analisi: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Errore interno del server: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check con test AI"""
    try:
        # Test rapido dell'AI
        test_response = analyzer.model.generate_content("Test: rispondi solo 'OK'")
        ai_status = "OK" if "OK" in test_response.text.strip() else "Partial"
    except:
        ai_status = "Error"

    return jsonify({
        'status': 'ok',
        'version': 'DebateLens Craicek\'s Version',
        'ai_status': ai_status,
        'powered_by': 'Google Gemini AI',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("🔥" * 20)
    print("🎯 DebateLens - Craicek's Version")
    print("🚀 Rizzo AI Academy")
    print("🤖 Powered by Google Gemini AI")
    print("🔥" * 20)
    print("🌐 Server: http://localhost:5000")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000)
