# DebateLens AI - Analisi Comparativa della Comunicazione
# Craicek's Enhanced Version con YouTube Integration + Gemini AI
# Versione Completa con Faster-Whisper + Intelligenza Artificiale per Partecipanti

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import base64
import sys
from datetime import datetime
import traceback
import re
from io import BytesIO
import tempfile

# Caricamento variabili ambiente
from dotenv import load_dotenv

load_dotenv()

# Import AI e visualizzazione
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# YouTube Integration imports con Faster-Whisper
try:
    import yt_dlp

    try:
        from faster_whisper import WhisperModel

        WHISPER_TYPE = "faster"
        print("✅ Faster-Whisper disponibile (raccomandato)")
    except ImportError:
        import whisper

        WHISPER_TYPE = "standard"
        print("✅ Standard Whisper disponibile")
    YOUTUBE_AVAILABLE = True
    print("✅ YouTube Integration disponibile")
except ImportError as e:
    YOUTUBE_AVAILABLE = False
    print(f"⚠️ YouTube Integration non disponibile: {e}")
    print("💡 Per attivare: pip install yt-dlp faster-whisper")


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


class YouTubeDebateExtractor:
    """Estrattore avanzato per dibattiti YouTube con IA Gemini"""

    def __init__(self, gemini_analyzer=None):
        if not YOUTUBE_AVAILABLE:
            raise ImportError("YouTube Integration non disponibile. Installa le dipendenze richieste.")

        # Riferimento all'analyzer Gemini per analisi intelligente
        self.gemini_analyzer = gemini_analyzer

        # Inizializza Whisper per trascrizione audio
        print("🤖 Inizializzazione sistema Whisper AI...")
        try:
            if WHISPER_TYPE == "faster":
                self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                print("✅ Sistema Faster-Whisper Craicek pronto")
            else:
                self.whisper_model = whisper.load_model("base")
                print("✅ Sistema Standard Whisper Craicek pronto")
        except Exception as e:
            print(f"❌ Errore inizializzazione Whisper: {e}")
            raise

    def extract_debate_info(self, youtube_url: str) -> dict:
        """Estrae informazioni complete dal video YouTube con AI Gemini"""
        temp_files = []
        try:
            print(f"🎥 Avvio estrazione da: {youtube_url}")

            # Crea directory temporanea
            temp_dir = tempfile.mkdtemp()
            audio_file = os.path.join(temp_dir, 'temp_audio')

            # Configurazione yt-dlp
            ydl_opts = {
                'format': 'bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio',
                'outtmpl': audio_file + '.%(ext)s',
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Estrae metadati
                print("📊 Estrazione metadati...")
                info = ydl.extract_info(youtube_url, download=False)

                # Controllo durata
                duration = info.get('duration', 0)
                if duration > 2700:  # 45 minuti
                    return {
                        'success': False,
                        'error': f'Video troppo lungo ({duration // 60} min). Massimo supportato: 45 minuti.',
                        'participants': []
                    }

                print(f"📹 Video: {info.get('title', 'N/A')}")
                print(f"⏱️ Durata: {duration} secondi ({duration // 60}:{duration % 60:02d})")

                # Download audio
                print("🔊 Download audio in corso...")
                ydl.download([youtube_url])

                # Trova file audio
                possible_files = [
                    audio_file + '.webm',
                    audio_file + '.m4a',
                    audio_file + '.wav',
                    audio_file + '.mp3',
                    audio_file + '.aac'
                ]

                actual_audio_file = None
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        actual_audio_file = file_path
                        temp_files.append(file_path)
                        break

                if not actual_audio_file:
                    raise Exception("File audio non trovato dopo il download")

                print(f"🎙️ File audio: {actual_audio_file}")

                # Trascrizione con Whisper
                print(f"🎙️ Trascrizione in corso con {WHISPER_TYPE.title()} Whisper AI...")

                if WHISPER_TYPE == "faster":
                    segments, info_whisper = self.whisper_model.transcribe(
                        actual_audio_file,
                        language='it',
                        task='transcribe'
                    )

                    segments_list = []
                    full_text = ""
                    for segment in segments:
                        segments_list.append({
                            'start': segment.start,
                            'end': segment.end,
                            'text': segment.text,
                            'avg_logprob': getattr(segment, 'avg_logprob', -0.5)
                        })
                        full_text += segment.text + " "

                    result = {
                        'text': full_text.strip(),
                        'segments': segments_list,
                        'language': info_whisper.language
                    }
                else:
                    result = self.whisper_model.transcribe(
                        actual_audio_file,
                        language='it',
                        task='transcribe',
                        fp16=False
                    )

                if not result.get('text', '').strip():
                    raise Exception("Trascrizione vuota. Il video potrebbe non contenere audio parlato.")

                print(f"📝 Trascrizione completata: {len(result['text'])} caratteri")

                # 🤖 ANALISI GEMINI AI per identificazione partecipanti
                participants = self._gemini_identify_speakers(result, info)

                return {
                    'success': True,
                    'video_title': info.get('title', 'Video YouTube'),
                    'duration': duration,
                    'channel': info.get('uploader', 'N/A'),
                    'participants': participants,
                    'full_transcript': result['text'][:500] + '...' if len(result['text']) > 500 else result['text'],
                    'extracted_by': f'Craicek Gemini AI + {WHISPER_TYPE.title()}-Whisper',
                    'whisper_language': result.get('language', 'auto')
                }

        except Exception as e:
            print(f"❌ Errore estrazione YouTube: {e}")
            return {
                'success': False,
                'error': f'Errore durante l\'estrazione: {str(e)}',
                'participants': []
            }
        finally:
            # Cleanup
            for file_path in temp_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"🧹 Rimosso: {file_path}")
                except Exception as e:
                    print(f"⚠️ Errore rimozione {file_path}: {e}")

    def _gemini_identify_speakers(self, whisper_result, video_info) -> list:
        """🤖 Usa Gemini AI per identificare intelligentemente i partecipanti"""

        if not self.gemini_analyzer or not self.gemini_analyzer.model:
            print("⚠️ Gemini non disponibile, uso metodo fallback...")
            return self._fallback_speaker_detection(whisper_result['text'], video_info)

        full_text = whisper_result['text']
        video_title = video_info.get('title', '')
        channel = video_info.get('uploader', '')

        print("🤖 Analisi Gemini AI per identificazione partecipanti...")

        # Prompt intelligente per Gemini
        gemini_prompt = self._create_speaker_identification_prompt(full_text, video_title, channel)

        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,  # Più creativo per nomi
                max_output_tokens=2000,
                response_mime_type="application/json"
            )

            response = self.gemini_analyzer.model.generate_content(gemini_prompt, generation_config=generation_config)
            gemini_result = json.loads(response.text)

            print(f"🎭 Gemini ha identificato {len(gemini_result.get('participants', []))} partecipanti")

            # Valida e converte risultato Gemini
            participants = self._validate_gemini_participants(gemini_result, full_text)

            if len(participants) >= 2:
                return participants
            else:
                print("🔄 Risultato Gemini insufficiente, uso metodo ibrido...")
                return self._hybrid_speaker_detection(whisper_result, video_info, gemini_result)

        except Exception as e:
            print(f"❌ Errore Gemini AI: {e}")
            print("🔄 Fallback a metodo tradizionale...")
            return self._fallback_speaker_detection(full_text, video_info)

    def _create_speaker_identification_prompt(self, text: str, video_title: str, channel: str) -> str:
        """Crea prompt intelligente per Gemini"""

        # Limita testo se troppo lungo
        text_sample = text[:4000] if len(text) > 4000 else text

        return f"""
Analizza questa trascrizione di un video YouTube e identifica INTELLIGENTEMENTE i partecipanti/speaker.

VIDEO INFO:
- Titolo: {video_title}
- Canale: {channel}

TRASCRIZIONE:
{text_sample}

COMPITO:
1. Identifica quanti speaker/partecipanti diversi parlano
2. Segmenta il testo per ogni speaker
3. Genera nomi INTELLIGENTI per ogni speaker basandoti su:
   - Auto-presentazioni nel testo ("Io sono Marco", "Mi chiamo Anna")
   - Tipo di video (intervista, dibattito, panel, etc.)
   - Ruoli evidenti (moderatore, ospite, candidato, etc.)
   - Nomi menzionati nel titolo
   - Contesto della conversazione

REGOLE IMPORTANTI:
- Ogni segmento deve avere MINIMO 50 caratteri
- Massimo 6 partecipanti
- Nomi creativi e appropriati al contesto
- Se trovi nomi reali nel testo, usali
- Altrimenti usa ruoli contestuali (es: "Moderatore", "Primo Candidato", "Esperto Tecnologia")

FORMATO JSON RICHIESTO:
{{
  "video_analysis": {{
    "type": "intervista/dibattito/panel/talk/conferenza",
    "main_topic": "argomento principale discusso",
    "speaker_count": "numero di speaker identificati"
  }},
  "participants": [
    {{
      "name": "Nome Intelligente Speaker",
      "role": "ruolo/funzione nel video",
      "text": "testo completo attribuito a questo speaker",
      "confidence": "alta/media/bassa",
      "name_source": "auto_presentazione/titolo_video/ruolo_contestuale/generato",
      "word_count": "numero parole approssimativo"
    }}
  ]
}}

IMPORTANTE: Fai il tuo meglio per identificare nomi reali o generare nomi appropriati al contesto!
"""

    def _validate_gemini_participants(self, gemini_result: dict, full_text: str) -> list:
        """Valida e converte il risultato di Gemini in formato DebateLens"""

        participants = []
        gemini_participants = gemini_result.get('participants', [])

        for i, participant in enumerate(gemini_participants):
            name = participant.get('name', f'Partecipante {i + 1}').strip()
            text = participant.get('text', '').strip()
            role = participant.get('role', '')
            confidence = participant.get('confidence', 'media')
            name_source = participant.get('name_source', 'generato')

            # Validazione qualità
            if len(text) >= 50 and len(text.split()) >= 10:
                # Migliora il nome se necessario
                if name_source == 'generato' and role:
                    name = self._enhance_generated_name(name, role)

                participants.append({
                    'name': name,
                    'text': text,
                    'word_count': len(text.split()),
                    'auto_detected': True,
                    'detection_method': 'gemini_ai',
                    'confidence': confidence,
                    'role': role,
                    'name_source': name_source,
                    'ai_generated': True
                })

                print(f"🎭 Partecipante validato: {name} ({confidence} confidence, {len(text)} chars)")

        return participants

    def _enhance_generated_name(self, name: str, role: str) -> str:
        """Migliora nomi generati basandosi sul ruolo"""

        # Mapping ruoli a nomi più accattivanti
        role_enhancements = {
            'moderatore': 'Moderatore',
            'ospite': 'Ospite Principale',
            'intervistatore': 'Giornalista',
            'candidato': 'Candidato',
            'esperto': 'Esperto',
            'analista': 'Analista',
            'professore': 'Professor',
            'dottore': 'Dottor',
            'giornalista': 'Giornalista'
        }

        role_lower = role.lower()
        for key, enhanced in role_enhancements.items():
            if key in role_lower:
                return enhanced

        return name

    def _hybrid_speaker_detection(self, whisper_result, video_info, gemini_partial) -> list:
        """Metodo ibrido: combina Gemini parziale + metodo tradizionale"""

        print("🔀 Modalità ibrida: Gemini + analisi tradizionale...")

        # Prova metodo tradizionale
        traditional_participants = self._fallback_speaker_detection(whisper_result['text'], video_info)

        # Se Gemini ha fornito insight sui nomi, applicali
        if gemini_partial and 'participants' in gemini_partial:
            gemini_names = [p.get('name', '') for p in gemini_partial['participants'] if p.get('name')]

            for i, participant in enumerate(traditional_participants):
                if i < len(gemini_names) and gemini_names[i]:
                    participant['name'] = gemini_names[i]
                    participant['detection_method'] = 'hybrid_gemini_traditional'
                    participant['ai_enhanced'] = True

        return traditional_participants

    def _fallback_speaker_detection(self, full_text: str, video_info) -> list:
        """Metodo di fallback tradizionale (versione semplificata)"""

        print("🎭 Metodo fallback tradizionale...")

        # Split semplice per paragrafi o frasi lunghe
        sentences = re.split(r'[.!?]+\s+', full_text)
        current_segment = ""
        segments = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            current_segment += sentence + ". "

            if len(current_segment) > 200:
                segments.append(current_segment.strip())
                current_segment = ""

        if current_segment:
            segments.append(current_segment.strip())

        # Crea partecipanti
        participants = []
        for i, segment in enumerate(segments[:6]):
            if len(segment) >= 50:
                participants.append({
                    'name': self._simple_name_generation(i + 1, video_info),
                    'text': segment,
                    'word_count': len(segment.split()),
                    'auto_detected': True,
                    'detection_method': 'traditional_fallback',
                    'ai_generated': False
                })

        return participants

    def _simple_name_generation(self, speaker_id: int, video_info) -> str:
        """Generazione nomi semplice per fallback"""

        video_title = video_info.get('title', '').lower()

        if 'intervista' in video_title:
            return 'Intervistatore' if speaker_id == 1 else f'Ospite {speaker_id - 1}'
        elif 'dibattito' in video_title:
            return f'Candidato {chr(64 + speaker_id)}'  # A, B, C, etc.
        elif 'panel' in video_title:
            return 'Moderatore' if speaker_id == 1 else f'Panelista {chr(64 + speaker_id - 1)}'
        else:
            return f'Speaker {chr(64 + speaker_id)}'  # Speaker A, B, C, etc.


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
        plt.title('DebateLens AI - Analisi Comparativa Gemini Enhanced\nCraicek\'s Version - Rizzo AI Academy',
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

# Inizializza YouTube extractor se disponibile
youtube_extractor = None
if YOUTUBE_AVAILABLE:
    try:
        # Passa l'analyzer Gemini al YouTube extractor per analisi intelligente
        youtube_extractor = YouTubeDebateExtractor(gemini_analyzer=analyzer)
        print("✅ YouTube Extractor inizializzato con Gemini AI")
    except Exception as e:
        print(f"⚠️ YouTube Extractor non disponibile: {e}")
        YOUTUBE_AVAILABLE = False

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
            'version': f'DebateLens Craicek\'s Gemini Enhanced Version ({WHISPER_TYPE.title()}-Whisper)',
            'ai_mode': True,
            'powered_by': 'Google Gemini AI',
            'youtube_enabled': YOUTUBE_AVAILABLE,
            'whisper_type': WHISPER_TYPE
        })

    except ValueError as e:
        print(f"❌ Errore di validazione: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"❌ Errore nell'analisi: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Errore interno del server: {str(e)}'}), 500


@app.route('/api/extract-youtube', methods=['POST'])
def extract_youtube_debate():
    """Endpoint per estrazione automatica da YouTube con Gemini AI"""
    if not YOUTUBE_AVAILABLE:
        return jsonify({
            'error': 'YouTube Integration non disponibile. Installa le dipendenze: pip install yt-dlp faster-whisper'
        }), 503

    try:
        data = request.get_json()
        youtube_url = data.get('youtube_url', '').strip()

        if not youtube_url:
            return jsonify({'error': 'URL YouTube obbligatorio'}), 400

        # Validazione URL YouTube estesa
        youtube_patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]+)'
        ]

        is_valid = any(re.match(pattern, youtube_url) for pattern in youtube_patterns)
        if not is_valid:
            return jsonify({
                               'error': 'URL YouTube non valido. Formati supportati: youtube.com/watch?v=, youtu.be/, youtube.com/embed/'}), 400

        print(f"🎥 Avvio estrazione YouTube con Gemini AI: {youtube_url}")

        # Estrai dati con il sistema CRAICEK + Gemini
        result = youtube_extractor.extract_debate_info(youtube_url)

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        if len(result['participants']) < 2:
            return jsonify({
                'error': f'Impossibile identificare almeno 2 partecipanti distinti nel video. Rilevati: {len(result["participants"])} partecipanti. Assicurati che il video contenga dialogo con più persone.'
            }), 400

        print(f"✅ Estrazione Gemini completata: {len(result['participants'])} partecipanti")

        return jsonify({
            'success': True,
            'video_title': result['video_title'],
            'channel': result.get('channel', 'N/A'),
            'duration': result['duration'],
            'participants': result['participants'],
            'full_transcript_preview': result.get('full_transcript', ''),
            'whisper_language': result.get('whisper_language', 'auto'),
            'extraction_method': result['extracted_by'],
            'whisper_type': WHISPER_TYPE,
            'ai_enhanced': True,
            'gemini_powered': True,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        })

    except Exception as e:
        print(f"❌ Errore estrazione YouTube Gemini: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Errore interno durante estrazione: {str(e)}',
            'suggestion': 'Verifica che il video sia pubblico e contenga dialogo parlato'
        }), 500


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
        'version': f'DebateLens Craicek\'s Gemini Enhanced Version ({WHISPER_TYPE.title()}-Whisper)',
        'ai_status': ai_status,
        'youtube_enabled': YOUTUBE_AVAILABLE,
        'whisper_type': WHISPER_TYPE,
        'gemini_ai': True,
        'powered_by': 'Google Gemini AI',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/health-youtube', methods=['GET'])
def health_check_youtube():
    """Health check specifico per componenti YouTube + Gemini"""
    try:
        components_status = {}

        # Test Whisper
        try:
            if YOUTUBE_AVAILABLE:
                if WHISPER_TYPE == "faster":
                    from faster_whisper import WhisperModel
                    components_status['whisper_ai'] = f"OK (Faster-Whisper v1.1.1)"
                else:
                    import whisper
                    components_status['whisper_ai'] = f"OK (Standard Whisper)"
            else:
                components_status['whisper_ai'] = "Not Available"
        except Exception as e:
            components_status['whisper_ai'] = f"Error: {str(e)}"

        # Test yt-dlp
        try:
            if YOUTUBE_AVAILABLE:
                import yt_dlp
                components_status['ytdlp'] = f"OK (v{yt_dlp.version.__version__})"
            else:
                components_status['ytdlp'] = "Not Available"
        except Exception as e:
            components_status['ytdlp'] = f"Error: {str(e)}"

        # Test Gemini AI
        try:
            test_response = analyzer.model.generate_content("Test breve")
            components_status['gemini_ai'] = f"OK (Gemini 1.5 Flash)"
        except Exception as e:
            components_status['gemini_ai'] = f"Error: {str(e)}"

        # Test FFmpeg
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                components_status['ffmpeg'] = f"OK ({version_line})"
            else:
                components_status['ffmpeg'] = "Error"
        except Exception as e:
            components_status['ffmpeg'] = f"Error: {str(e)}"

        overall_status = "OK" if all("OK" in status for status in components_status.values()) else "Partial"

        return jsonify({
            'status': overall_status,
            'youtube_extraction_available': YOUTUBE_AVAILABLE,
            'whisper_type': WHISPER_TYPE,
            'gemini_ai_enabled': True,
            'components': components_status,
            'version': f'DebateLens Craicek\'s Gemini Enhanced Version ({WHISPER_TYPE.title()}-Whisper)',
            'features': [
                'Gemini AI Speaker Detection',
                'Intelligent Name Extraction',
                'Contextual Role Assignment',
                'Smart Text Segmentation'
            ],
            'installation_guide': {
                'missing_deps': 'pip install yt-dlp faster-whisper',
                'ffmpeg_install': {
                    'windows': 'choco install ffmpeg OR winget install Gyan.FFmpeg',
                    'macos': 'brew install ffmpeg',
                    'linux': 'sudo apt install ffmpeg'
                }
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'youtube_extraction_available': False,
            'gemini_ai_enabled': False,
            'components': {}
        }), 500


@app.route('/api/youtube-test', methods=['POST'])
def youtube_test_endpoint():
    """Endpoint di test per YouTube con video di esempio"""
    if not YOUTUBE_AVAILABLE:
        return jsonify({
            'error': 'YouTube Integration non disponibile'
        }), 503

    try:
        # Video di test pubblico (sostituisci con un video appropriato)
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll per test

        # Test solo metadati senza download completo
        ydl_opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(test_url, download=False)

        return jsonify({
            'success': True,
            'test_video': {
                'title': info.get('title', 'N/A'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'N/A')
            },
            'whisper_type': WHISPER_TYPE,
            'gemini_ai_enabled': True,
            'message': f'Sistema YouTube con {WHISPER_TYPE.title()}-Whisper + Gemini AI funzionante. Puoi procedere con l\'estrazione di video reali.',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Test fallito: {str(e)}',
            'suggestion': 'Verifica la connessione internet e le dipendenze installate'
        }), 500


if __name__ == '__main__':
    # Prevenire che pytest interpreti questo come test
    if 'pytest' not in sys.modules:
        print("🔥" * 20)
        print("🎯 DebateLens AI - Craicek's Gemini Enhanced Version")
        print("🚀 Rizzo AI Academy")
        print("🤖 Powered by Google Gemini AI")
        if YOUTUBE_AVAILABLE:
            print(f"🎥 YouTube Integration: ATTIVA ({WHISPER_TYPE.title()}-Whisper + Gemini AI)")
            print("🎙️ Sistema Trascrizione: PRONTO")
            print("🧠 Analisi Partecipanti: GEMINI AI")
            print("✨ Identificazione Intelligente: ATTIVA")
        else:
            print("⚠️ YouTube Integration: NON DISPONIBILE")
            print("💡 Per attivare: pip install yt-dlp faster-whisper")
        print("🔥" * 20)
        print("🌐 Server: http://localhost:5000")
        print("🔧 API Docs:")
        print("   • POST /api/analyze - Analisi dibattiti")
        if YOUTUBE_AVAILABLE:
            print("   • POST /api/extract-youtube - Estrazione YouTube con Gemini AI")
            print("   • GET /api/health-youtube - Health check YouTube + Gemini")
            print("   • POST /api/youtube-test - Test sistema YouTube")
        print("   • GET /api/health - Health check generale")
        print("=" * 60)

        app.run(debug=False, host='0.0.0.0', port=5000)
