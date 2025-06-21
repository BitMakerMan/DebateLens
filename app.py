# DebateLens - Analisi Comparativa della Comunicazione
# Craicek's Version - Rizzo AI Academy
# Versione Finale Produzione

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


class DebateLensAnalyzer:
    """Core analyzer per DebateLens con Google Gemini AI"""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.last_error = None
        self.analysis_criteria = [
            'rigorosita_tecnica', 'uso_dati_oggettivi', 'approccio_divulgativo',
            'stile_comunicativo', 'focalizzazione_argomento', 'orientamento_pratico'
        ]

    def create_analysis_prompt(self, text: str, participant_name: str) -> str:
        """Crea il prompt per l'analisi AI"""
        return f"""
Analizza il seguente testo di {participant_name} secondo questi 6 criteri, assegnando un punteggio da 1 a 10:

TESTO: {text}

CRITERI:
1. Rigorosità tecnica (1-10): Precisione terminologica e concetti specialistici
2. Uso di dati oggettivi (1-10): Statistiche, ricerche, fonti verificabili
3. Approccio divulgativo (1-10): Accessibilità, esempi, analogie
4. Stile comunicativo (1-10): Fluidità e capacità di coinvolgimento
5. Focalizzazione argomento (1-10): Aderenza al tema, coerenza logica
6. Orientamento pratico (1-10): Soluzioni concrete, applicabilità

FORMATO JSON:
{{
  "rigorosita_tecnica": X,
  "uso_dati_oggettivi": X,
  "approccio_divulgativo": X,
  "stile_comunicativo": X,
  "focalizzazione_argomento": X,
  "orientamento_pratico": X,
  "explanations": {{
    "rigorosita_tecnica": "Breve spiegazione",
    "uso_dati_oggettivi": "Breve spiegazione",
    "approccio_divulgativo": "Breve spiegazione",
    "stile_comunicativo": "Breve spiegazione",
    "focalizzazione_argomento": "Breve spiegazione",
    "orientamento_pratico": "Breve spiegazione"
  }}
}}
"""

    def analyze_participant(self, text: str, participant_name: str) -> AnalysisResult:
        """Analizza un partecipante con Google Gemini"""
        try:
            prompt = self.create_analysis_prompt(text, participant_name)

            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,
                response_mime_type="application/json"
            )

            response = self.model.generate_content(prompt, generation_config=generation_config)
            result_json = json.loads(response.text)

            return AnalysisResult(
                participant_name=participant_name,
                rigorosita_tecnica=result_json['rigorosita_tecnica'],
                uso_dati_oggettivi=result_json['uso_dati_oggettivi'],
                approccio_divulgativo=result_json['approccio_divulgativo'],
                stile_comunicativo=result_json['stile_comunicativo'],
                focalizzazione_argomento=result_json['focalizzazione_argomento'],
                orientamento_pratico=result_json['orientamento_pratico'],
                explanations=result_json['explanations']
            )

        except Exception as e:
            self.last_error = str(e)
            return None

    def create_radar_chart(self, analyses: list) -> str:
        """Genera radar chart PNG Iron Man style"""
        if not analyses:
            return None

        try:
            # Setup matplotlib per server
            import matplotlib
            matplotlib.use('Agg')
            plt.ioff()
            plt.clf()
            plt.close('all')

            # Crea il plot
            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'), dpi=150)
            fig.patch.set_facecolor('#1a1a1a')

            # Labels e angoli
            labels = ['Rigorosità\nTecnica', 'Uso Dati\nOggettivi', 'Approccio\nDivulgativo',
                      'Stile\nComunicativo', 'Focalizzazione\nArgomento', 'Orientamento\nPratico']
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            # Colori Iron Man
            colors = ['#dc2626', '#fbbf24', '#b91c1c', '#f59e0b']

            # Plot ogni partecipante
            for i, analysis in enumerate(analyses):
                values = [
                    analysis.rigorosita_tecnica, analysis.uso_dati_oggettivi,
                    analysis.approccio_divulgativo, analysis.stile_comunicativo,
                    analysis.focalizzazione_argomento, analysis.orientamento_pratico
                ]
                values += values[:1]

                color = colors[i % len(colors)]
                ax.plot(angles, values, 'o-', linewidth=4, label=analysis.participant_name,
                        color=color, markersize=10, markerfacecolor=color,
                        markeredgecolor='white', markeredgewidth=2)
                ax.fill(angles, values, alpha=0.25, color=color)

            # Styling Iron Man
            ax.set_facecolor('#0a0a0a')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=12, color='#fbbf24', fontweight='bold')
            ax.set_ylim(0, 10)
            ax.set_yticks(range(0, 11, 2))
            ax.set_yticklabels(range(0, 11, 2), fontsize=10, color='#dc2626', fontweight='bold')
            ax.grid(True, color='#444444', alpha=0.7, linewidth=1)

            plt.title('🔥 DebateLens - Analisi Comparativa\nCraicek\'s Version',
                      size=18, fontweight='bold', pad=30, color='#fbbf24')

            legend = plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0),
                                facecolor='#1a1a1a', edgecolor='#dc2626',
                                labelcolor='#fbbf24', fontsize=11, framealpha=0.9)
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

        except Exception as e:
            return self.create_html_fallback(analyses)

    def create_html_fallback(self, analyses):
        """Fallback HTML quando matplotlib fallisce"""
        try:
            colors = ['#dc2626', '#fbbf24', '#b91c1c', '#f59e0b']

            html = '<div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); padding: 40px; border-radius: 16px; text-align: center; border: 2px solid #dc2626;">'
            html += '<h3 style="color: #fbbf24; margin-bottom: 30px;">📊 Radar Chart - Iron Man Style</h3>'

            # Radar visuale
            html += '<div style="position: relative; width: 400px; height: 400px; margin: 0 auto; border: 3px solid #dc2626; border-radius: 50%; background: radial-gradient(circle, rgba(220, 38, 38, 0.1) 0%, rgba(0, 0, 0, 0.4) 100%);">'

            # Cerchi concentrici
            for i, radius in enumerate([20, 35, 50, 65, 80]):
                opacity = 0.6 - (i * 0.1)
                html += f'<div style="position: absolute; top: {50 - radius / 2}%; left: {50 - radius / 2}%; width: {radius}%; height: {radius}%; border: 1px solid #666; border-radius: 50%; opacity: {opacity};"></div>'

            # Assi
            html += '<div style="position: absolute; top: 0; left: 50%; width: 2px; height: 100%; background: linear-gradient(to bottom, #dc2626, #fbbf24, #dc2626); opacity: 0.8;"></div>'
            html += '<div style="position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background: linear-gradient(to right, #dc2626, #fbbf24, #dc2626); opacity: 0.8;"></div>'

            # Punti dati
            for i, analysis in enumerate(analyses[:4]):
                color = colors[i % len(colors)]
                html += f'<div style="position: absolute; top: {20 + i * 5}%; left: {60 + i * 5}%; width: 12px; height: 12px; background: {color}; border-radius: 50%; border: 2px solid white; z-index: 10;" title="{analysis.participant_name}"></div>'

            html += '<div style="position: absolute; top: 50%; left: 50%; width: 8px; height: 8px; background: #fbbf24; border-radius: 50%; transform: translate(-50%, -50%); border: 2px solid white;"></div>'
            html += '</div>'

            # Legenda
            html += '<div style="margin-top: 25px; display: flex; justify-content: center; gap: 20px;">'
            for i, analysis in enumerate(analyses[:4]):
                color = colors[i % len(colors)]
                html += f'<div style="display: flex; align-items: center; gap: 8px;"><div style="width: 16px; height: 16px; background: {color}; border-radius: 50%; border: 2px solid white;"></div><span style="color: #fbbf24; font-weight: bold;">{analysis.participant_name}</span></div>'
            html += '</div></div>'

            return html

        except Exception as e:
            return None

    def generate_comparative_report(self, analyses: list) -> dict:
        """Genera report comparativo con insights"""
        if len(analyses) < 2:
            return {"error": "Servono almeno 2 partecipanti per il confronto"}

        report = {
            "summary": {},
            "detailed_comparison": {},
            "insights": []
        }

        try:
            # Calcola statistiche per criterio
            for criterion in self.analysis_criteria:
                scores = [getattr(analysis, criterion) for analysis in analyses]
                report["summary"][criterion] = {
                    "average": round(sum(scores) / len(scores), 2),
                    "max_participant": analyses[scores.index(max(scores))].participant_name,
                    "max_score": max(scores),
                    "min_participant": analyses[scores.index(min(scores))].participant_name,
                    "min_score": min(scores)
                }

            # Dettagli partecipanti
            for analysis in analyses:
                report["detailed_comparison"][analysis.participant_name] = analysis.to_dict()

            # Insights automatici
            if len(analyses) >= 2:
                p1, p2 = analyses[0], analyses[1]

                # Converti a int per sicurezza
                p1_tech = int(p1.rigorosita_tecnica)
                p2_tech = int(p2.rigorosita_tecnica)
                p1_div = int(p1.approccio_divulgativo)
                p2_div = int(p2.approccio_divulgativo)

                if p1_tech > p2_tech:
                    diff = p1_tech - p2_tech
                    report["insights"].append(f"🔥 {p1.participant_name} eccelle in rigorosità tecnica (+{diff} punti)")
                elif p2_tech > p1_tech:
                    diff = p2_tech - p1_tech
                    report["insights"].append(f"🔥 {p2.participant_name} eccelle in rigorosità tecnica (+{diff} punti)")

                if p1_div > p2_div:
                    diff = p1_div - p2_div
                    report["insights"].append(f"⚡ {p1.participant_name} è più divulgativo (+{diff} punti)")
                elif p2_div > p1_div:
                    diff = p2_div - p1_div
                    report["insights"].append(f"⚡ {p2.participant_name} è più divulgativo (+{diff} punti)")

                # Insights aggiuntivi
                if abs(p1_tech - p2_tech) <= 1:
                    report["insights"].append("🎯 Livello tecnico equilibrato tra i partecipanti")
                if abs(p1_div - p2_div) <= 1:
                    report["insights"].append("⚖️ Approccio divulgativo molto simile")

        except Exception as e:
            report["insights"] = [
                "🔥 Analisi comparativa completata",
                f"📊 {len(analyses)} partecipanti analizzati con successo"
            ]

        return report


def analyze_text_heuristic(text):
    """Analisi euristica per fallback senza AI"""
    if not text:
        return {criterion: 5 for criterion in ['rigorosita_tecnica', 'uso_dati_oggettivi',
                                               'approccio_divulgativo', 'stile_comunicativo',
                                               'focalizzazione_argomento', 'orientamento_pratico']}

    text_lower = text.lower()
    word_count = len(text.split())

    # Parole chiave per analisi
    technical_words = ['analisi', 'sistema', 'processo', 'metodologia', 'implementazione']
    data_words = ['percentuale', 'statistica', 'dati', 'ricerca', 'studio', '%']
    divulgative_words = ['semplice', 'facile', 'esempio', 'immaginate', 'praticamente']
    practical_words = ['utilizzare', 'applicare', 'soluzione', 'problema', 'pratico']

    # Calcola scores
    tech_score = min(10, 4 + sum(1 for word in technical_words if word in text_lower))
    data_score = min(10, 3 + sum(1 for word in data_words if word in text_lower) * 2)
    divulgative_score = min(10, 4 + sum(1 for word in divulgative_words if word in text_lower))
    style_score = min(10, 5 + min(3, word_count // 50))
    focus_score = max(3, min(10, 8 - text.count('?') - text.count('...')))
    practical_score = min(10, 4 + sum(1 for word in practical_words if word in text_lower))

    return {
        'rigorosita_tecnica': tech_score,
        'uso_dati_oggettivi': data_score,
        'approccio_divulgativo': divulgative_score,
        'stile_comunicativo': style_score,
        'focalizzazione_argomento': focus_score,
        'orientamento_pratico': practical_score
    }


def create_heuristic_analysis(name, text):
    """Crea AnalysisResult usando analisi euristica"""
    scores = analyze_text_heuristic(text)

    explanations = {
        'rigorosita_tecnica': f'{name} {"usa terminologia tecnica appropriata" if scores["rigorosita_tecnica"] >= 7 else "utilizza un linguaggio più generale"}',
        'uso_dati_oggettivi': f'{"Presenta riferimenti a dati" if scores["uso_dati_oggettivi"] >= 6 else "Basato più su opinioni"}',
        'approccio_divulgativo': f'{"Stile molto accessibile" if scores["approccio_divulgativo"] >= 7 else "Approccio più tecnico"}',
        'stile_comunicativo': f'Comunicazione {["essenziale", "buona", "efficace", "eccellente"][min(3, scores["stile_comunicativo"] // 3)]}',
        'focalizzazione_argomento': f'Mantiene {"buon" if scores["focalizzazione_argomento"] >= 6 else "discreto"} focus',
        'orientamento_pratico': f'Orientamento {"molto" if scores["orientamento_pratico"] >= 7 else "moderatamente"} pratico'
    }

    return AnalysisResult(
        participant_name=name,
        rigorosita_tecnica=scores['rigorosita_tecnica'],
        uso_dati_oggettivi=scores['uso_dati_oggettivi'],
        approccio_divulgativo=scores['approccio_divulgativo'],
        stile_comunicativo=scores['stile_comunicativo'],
        focalizzazione_argomento=scores['focalizzazione_argomento'],
        orientamento_pratico=scores['orientamento_pratico'],
        explanations=explanations
    )


# Configurazione
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your-google-api-key-here')

# Inizializza analyzer
if GOOGLE_API_KEY and GOOGLE_API_KEY != 'your-google-api-key-here':
    analyzer = DebateLensAnalyzer(GOOGLE_API_KEY)
    AI_MODE = True
else:
    analyzer = DebateLensAnalyzer("fake-key")
    AI_MODE = False

# Flask App
app = Flask(__name__)
CORS(app)


# Routes
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_debate():
    """Endpoint principale per l'analisi dei dibattiti"""
    try:
        data = request.get_json()

        if not data or len(data.get('participants', [])) < 2:
            return jsonify({'error': 'Servono almeno 2 partecipanti'}), 400

        participants = data['participants']
        analyses = []

        # Analizza ogni partecipante
        for participant in participants:
            name = participant.get('name', 'Partecipante')
            text = participant.get('text', '')

            if not text.strip():
                continue

            if AI_MODE:
                try:
                    analysis = analyzer.analyze_participant(text, name)
                    if analysis:
                        analyses.append(analysis)
                    else:
                        analyses.append(create_heuristic_analysis(name, text))
                except:
                    analyses.append(create_heuristic_analysis(name, text))
            else:
                analyses.append(create_heuristic_analysis(name, text))

        if len(analyses) < 2:
            return jsonify({'error': 'Analisi fallita'}), 500

        # Genera risultati
        chart_data = analyzer.create_radar_chart(analyses)
        report = analyzer.generate_comparative_report(analyses)

        return jsonify({
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'participants_count': len(analyses),
            'chart_data': chart_data,
            'report': report,
            'version': 'DebateLens Craicek\'s Version',
            'ai_mode': AI_MODE
        })

    except Exception as e:
        return jsonify({'error': f'Errore: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'version': 'DebateLens Craicek\'s Version',
        'ai_mode': AI_MODE,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("🔥" * 20)
    print("🎯 DebateLens - Craicek's Version")
    print("🚀 Rizzo AI Academy")
    print("🔥" * 20)

    if AI_MODE:
        print("✅ Google Gemini AI configurato")
        print("🤖 Modalità: Analisi AI completa")
    else:
        print("🧠 Modalità: Analisi euristica intelligente")
        print("💡 Configura GOOGLE_API_KEY nel file .env per AI completa")

    print("🌐 Server: http://localhost:5000")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000)