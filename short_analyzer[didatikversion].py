import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import sys
from flask import Flask, request, jsonify, url_for, send_from_directory, render_template_string
import datetime
import shutil
import pandas as pd
import plotly.graph_objects as go
import html
from pathlib import Path
import re
import tempfile
import os
import json
import traceback

# Função para converter tipos do NumPy em tipos Python nativos
# Isso é importante porque o NumPy tem seus próprios tipos que nem sempre são compatíveis com JSON
def convert_numpy_types(obj):
    # Verifica se é um tipo escalar do NumPy (como np.int64)
    if isinstance(obj, np.generic):
        return obj.item()  # Converte para o tipo Python correspondente
    # Se for um array do NumPy, transforma em lista
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Para dicionários, aplica a conversão recursivamente em cada valor
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    # Para listas ou tuplas, converte cada elemento
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    # Converte tipos inteiros específicos do NumPy
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Converte tipos flutuantes do NumPy
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    # Converte booleanos do NumPy
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj  # Retorna o objeto inalterado se não for um tipo NumPy

# Vamos tentar importar youtube-dl ou yt-dlp para baixar vídeos
# Esses pacotes são úteis para extrair vídeos do YouTube
try:
    import yt_dlp as youtube_dl
    YDL_AVAILABLE = True  # Sucesso! Podemos baixar vídeos
except ImportError:
    try:
        import youtube_dl  # Tentativa com a biblioteca mais antiga
        YDL_AVAILABLE = True
    except ImportError:
        YDL_AVAILABLE = False
        print("AVISO: youtube-dl ou yt-dlp não está instalado. Funcionalidade de download será simulada.")
        # Se não temos nenhum dos dois, vamos simular o download mais tarde

# Tentando importar o NLTK para análise de sentimento
# Isso nos ajuda a entender o tom emocional do áudio ou texto
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('vader_lexicon')  # Verifica se o léxico está disponível
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)  # Baixa silenciosamente se necessário
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("AVISO: NLTK não está instalado. Análise de sentimento será simulada.")

# Tentando importar speech_recognition para transcrição de áudio
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("AVISO: speech_recognition não está instalado.")

# Configurando o logging para acompanhar o que está acontecendo
# Isso é como um diário para o nosso programa, registrando tudo o que fazemos
logging.basicConfig(
    level=logging.DEBUG,  # Vamos capturar todos os detalhes
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato do log com data, nível e mensagem
    handlers=[
        logging.FileHandler('app.log', mode='a', encoding='utf-8'),  # Salva logs em um arquivo
        logging.StreamHandler(sys.stdout)  # Também mostra no console
    ]
)
logger = logging.getLogger(__name__)

# Criando a aplicação Flask
# Flask é o nosso garçom, recebendo pedidos (requisições) e servindo respostas
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limitando uploads a 16MB
app.config['JSON_AS_ASCII'] = False  # Permitindo caracteres não-ASCII no JSON

# Diretório para salvar os resultados
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)  # Criamos o diretório se ele não existir

class ShortsDetector:
    """Classe para encontrar os melhores trechos de um vídeo para criar Shorts.
    Pense nisso como um editor esperto que sugere os momentos mais cativantes!"""
    
    def __init__(self, target_duration: int = 119):
        self.target_duration = target_duration  # Duração ideal do Short (em segundos)
        self.block_duration = 60  # Cada bloco de análise tem 1 minuto
        
    def calculate_engagement_score(self, blocks: List[Dict]) -> List[float]:
        # Calcula uma pontuação de engajamento para cada bloco
        # Combinamos intensidade visual e sentimento para decidir o quão interessante é
        scores = []
        for block in blocks:
            intensity = block.get('intensity', 0)  # Intensidade visual do bloco
            sentiment = block.get('sentiment', 0)  # Sentimento do áudio/texto
            intensity_norm = min(intensity / 5.0, 1.0)  # Normalizamos a intensidade (0 a 1)
            sentiment_engagement = abs(sentiment)  # Sentimentos fortes (positivos ou negativos) engajam mais
            # 70% do peso para intensidade, 30% para sentimento
            engagement_score = (intensity_norm * 0.7) + (sentiment_engagement * 0.3)
            scores.append(engagement_score)
        return scores
    
    # [CÓDIGO OCULTO PARA O PÚBLICO]
    # Esta seção contém a lógica principal para encontrar os melhores segmentos
    # Não será exibida publicamente, mas inclui a seleção de trechos otimizados para Shorts
    # com base em pontuações de engajamento, consistência e duração.
    
    def _generate_justification(self, avg_engagement: float, avg_intensity: float, 
                              peak_intensity: float, avg_sentiment: float, 
                              sentiment_variety: int, consistency: float) -> str:
        # Vamos criar uma explicação amigável sobre por que escolhemos este trecho
        reasons = []
        if avg_intensity > 3.0:
            reasons.append("🔥 <strong>Alta atividade visual</strong> - Muitas mudanças e movimento mantêm a atenção")
        elif avg_intensity > 2.0:
            reasons.append("👁️ <strong>Boa atividade visual</strong> - Nível adequado de dinamismo")
        else:
            reasons.append("😐 <strong>Baixa atividade visual</strong> - Pode precisar de elementos adicionais")
        
        if peak_intensity > 4.0:
            reasons.append("⚡ <strong>Momento de pico intenso</strong> - Contém cenas de alto impacto")
            
        if avg_sentiment > 0.3:
            reasons.append("😊 <strong>Conteúdo positivo</strong> - Tom otimista que engaja audiência")
        elif avg_sentiment < -0.3:
            reasons.append("😠 <strong>Conteúdo de forte reação</strong> - Emocional, pode gerar discussão")
        elif abs(avg_sentiment) > 0.1:
            reasons.append("🎭 <strong>Conteúdo emotivo</strong> - Desperta reações da audiência")
        else:
            reasons.append("😑 <strong>Conteúdo neutro</strong> - Pode precisar de elementos mais envolventes")
            
        if sentiment_variety > 2:
            reasons.append("🎢 <strong>Variedade emocional</strong> - Diferentes tons mantêm interesse")
            
        if consistency > 0.8:
            reasons.append("📊 <strong>Alta consistência</strong> - Qualidade mantida durante todo o trecho")
        elif consistency > 0.6:
            reasons.append("📈 <strong>Boa consistência</strong> - Qualidade equilibrada")
        else:
            reasons.append("📉 <strong>Variação de qualidade</strong> - Alguns momentos mais fortes que outros")
            
        if avg_engagement > 0.7:
            reasons.append("🏆 <strong>Excelente potencial de engajamento</strong> - Ideal para Shorts")
        elif avg_engagement > 0.5:
            reasons.append("👍 <strong>Bom potencial de engajamento</strong> - Adequado para Shorts")
        else:
            reasons.append("⚠️ <strong>Potencial limitado</strong> - Considere edições adicionais")
            
        return " • ".join(reasons)  # Junta tudo com um separador bonitinho
    
    def generate_cutting_instructions(self, best_segment: Dict) -> Dict:
        # Gera instruções detalhadas para cortar o vídeo
        # Isso é útil para editores que usam ferramentas como FFmpeg ou Premiere
        start_time = best_segment['start_time']
        end_time = best_segment['end_time']
        duration = best_segment['duration']
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)
        # Comando para FFmpeg, uma ferramenta poderosa para manipulação de vídeos
        ffmpeg_cmd = f"ffmpeg -i input_video.mp4 -ss {start_min:02d}:{start_sec:02d} -t {int(duration)} -c copy output_short.mp4"
        # Instruções para quem usa Adobe Premiere
        premiere_instructions = f"Marque o ponto de entrada em {start_min:02d}:{start_sec:02d} e ponto de saída em {end_min:02d}:{end_sec:02d}"
        
        return {
            'start_time_formatted': f"{start_min:02d}:{start_sec:02d}",
            'end_time_formatted': f"{end_min:02d}:{end_sec:02d}",
            'duration_formatted': f"{int(duration//60):02d}:{int(duration%60):02d}",
            'ffmpeg_command': ffmpeg_cmd,
            'premiere_instructions': premiere_instructions,
            'youtube_studio_instructions': f"Use a ferramenta de corte no YouTube Studio: início {start_min:02d}:{start_sec:02d}, fim {end_min:02d}:{end_sec:02d}",
            'manual_instructions': f"Corte manualmente de {start_min:02d}:{start_sec:02d} até {end_min:02d}:{end_sec:02d} (duração: {int(duration//60):02d}:{int(duration%60):02d})"
        }

class VideoAnalyzer:
    """Classe principal para analisar vídeos do YouTube.
    É como um detetive que examina o vídeo e encontra os melhores momentos!"""
    
    def __init__(self):
        try:
            # Se temos NLTK, inicializamos o analisador de sentimento
            if NLTK_AVAILABLE:
                self.analyzer = SentimentIntensityAnalyzer()
            else:
                self.analyzer = None  # Sem NLTK, vamos simular
            # Configuramos o reconhecedor de fala, se disponível
            self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
            # Criamos um detector de Shorts para encontrar trechos interessantes
            self.shorts_detector = ShortsDetector()
            logger.info("VideoAnalyzer inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar VideoAnalyzer: {str(e)}", exc_info=True)
            raise
    
    def valid_youtube_url(self, url: str) -> bool:
        """Verifica se a URL é válida para o YouTube."""
        youtube_regex = (
            r'(https?://)?(www\.)?'
            r'(youtube\.com|youtu\.be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(re.match(youtube_regex, url))  # Retorna True se a URL for válida

    def extract_video_id(self, url: str) -> str:
        """Pega o ID do vídeo a partir da URL do YouTube."""
        youtube_regex = (
            r'(https?://)?(www\.)?'
            r'(youtube\.com|youtu\.be)/'
            r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        match = re.search(youtube_regex, url)
        return match.group(5) if match and match.group(5) else 'unknown'  # Retorna o ID ou 'unknown'

    # [CÓDIGO OCULTO PARA O PÚBLICO]
    # Esta seção contém métodos relacionados ao download e análise detalhada do vídeo.
    # Não será exibida publicamente, mas inclui a lógica para baixar vídeos do YouTube
    # e realizar análises de intensidade e sentimento em blocos.

    def detect_best_shorts(self, blocks: List[Dict]) -> Dict:
        """Encontra os melhores trechos para Shorts com base nos blocos analisados."""
        try:
            # Usamos nosso detector de Shorts para escolher os melhores trechos
            best_segments = self.shorts_detector.find_best_segments(blocks, num_segments=3)
            if not best_segments:
                logger.warning("Nenhum segmento adequado encontrado")
                return {'error': 'Não foi possível encontrar segmentos adequados', 'segments': []}
            
            processed_segments = []
            for i, segment in enumerate(best_segments):
                # Geramos instruções de corte para cada segmento
                cutting_instructions = self.shorts_detector.generate_cutting_instructions(segment)
                processed_segments.append({
                    'rank': i + 1,
                    'score': float(round(segment['scores']['final_score'], 3)),
                    'start_time': float(segment['start_time']),
                    'end_time': float(segment['end_time']),
                    'duration': float(segment['duration']),
                    'timestamps': {
                        'start': segment['start_timestamp'],
                        'end': segment['end_timestamp']
                    },
                    'stats': {
                        'avg_intensity': float(round(segment['scores']['avg_intensity'], 2)),
                        'peak_intensity': float(round(segment['scores']['peak_intensity'], 2)),
                        'avg_sentiment': float(round(segment['scores']['avg_sentiment'], 3)),
                        'sentiment_variety': int(segment['scores']['sentiment_variety']),
                        'consistency': float(round(segment['scores']['consistency'], 2))
                    },
                    'justification': segment['justification'],
                    'cutting_instructions': cutting_instructions,
                    'blocks_included': len(segment['blocks'])
                })
            
            return {
                'best_segment': processed_segments[0] if processed_segments else None,
                'alternative_segments': processed_segments[1:] if len(processed_segments) > 1 else [],
                'total_segments_analyzed': len(best_segments),
                'analysis_summary': {
                    'video_suitable_for_shorts': processed_segments[0]['score'] > 0.5 if processed_segments else False,
                    'best_score': float(processed_segments[0]['score']) if processed_segments else 0,
                    'recommended_editing': processed_segments[0]['score'] < 0.7 if processed_segments else True
                }
            }
        except Exception as e:
            logger.error(f"Erro ao detectar shorts: {str(e)}", exc_info=True)
            return {'error': f'Erro na detecção de shorts: {str(e)}', 'segments': []}

    def build_combined_heatmap_with_shorts(self, intensities: List[float], sentiments: List[float], 
                                         title: str, duration: float, best_segment: Dict = None) -> str:
        """Cria um gráfico interativo mostrando intensidade e sentimento ao longo do vídeo."""
        if not intensities:
            logger.warning("Nenhum dado de intensidade fornecido para o gráfico")
            return "<div class='alert alert-warning'>Não há dados suficientes para gerar o gráfico.</div>"

        # Criamos timestamps para cada bloco com base na duração real
        block_duration = duration / len(intensities)
        times = []
        for i in range(len(intensities)):
            time_seconds = i * block_duration
            minutes = int(time_seconds // 60)
            seconds = int(time_seconds % 60)
            times.append(f"{minutes}:{seconds:02d}")
            
        x_values = list(range(len(intensities)))

        try:
            # Criamos um gráfico com Plotly
            fig = go.Figure()
            
            # Linha para intensidade visual
            fig.add_trace(go.Scatter(
                x=x_values, y=intensities, mode='lines+markers',
                line=dict(color='#1a73e8', width=3, shape='spline'),
                marker=dict(color='#1a73e8', size=8),
                name='Intensidade Visual',
                yaxis='y1',
                fill='tozeroy',
                fillcolor='rgba(26, 115, 232, 0.1)',
                hovertemplate="<b>Minuto:</b> %{customdata}<br><b>Intensidade:</b> %{y:.2f}<extra></extra>",
                customdata=times
            ))
            
            # Linha para sentimento do áudio
            fig.add_trace(go.Scatter(
                x=x_values, y=sentiments, mode='lines+markers',
                line=dict(color='#34a853', width=3, shape='spline', dash='dash'),
                marker=dict(color='#34a853', size=8),
                name='Sentimento do Áudio',
                yaxis='y2',
                hovertemplate="<b>Minuto:</b> %{customdata}<br><b>Sentimento:</b> %{y:.2f}<extra></extra>",
                customdata=times
            ))
            
            # Adicionamos uma linha vertical no pico de intensidade
            if intensities:
                peak_idx = int(np.argmax(intensities))
                peak_val = max(intensities)
                fig.add_vline(x=peak_idx, line_dash='dash', line_color='#d93025', line_width=2)
                fig.add_annotation(
                    x=peak_idx, y=peak_val + 0.1,
                    text=f"🔥 Pico ({times[peak_idx]})",
                    showarrow=True, arrowhead=2, arrowcolor="#d93025",
                    font=dict(color='#d93025', size=12, family="Arial, sans-serif")
                )
            
            # Destacamos o melhor trecho para Shorts
            if best_segment:
                start_block = int(best_segment['start_time'] // block_duration)
                end_block = int(best_segment['end_time'] // block_duration)
                
                # Garantimos que os índices não saiam dos limites
                start_block = max(0, min(start_block, len(intensities) - 1))
                end_block = max(0, min(end_block, len(intensities) - 1))
                
                # Adicionamos um retângulo destacando o trecho
                fig.add_vrect(
                    x0=start_block - 0.4, x1=end_block + 0.4, 
                    fillcolor="rgba(255, 193, 7, 0.3)", 
                    layer="below", line_width=0
                )
                
                # Linhas verticais no início e fim do trecho
                fig.add_vline(x=start_block, line_dash='solid', line_color='#ff6f00', line_width=3)
                fig.add_vline(x=end_block, line_dash='solid', line_color='#ff6f00', line_width=3)
                
                # Anotação para o melhor trecho
                mid_point = (start_block + end_block) / 2
                fig.add_annotation(
                    x=mid_point, y=max(intensities) * 0.8,
                    text=f"🎬 MELHOR TRECHO<br>({best_segment['cutting_instructions']['start_time_formatted']} - {best_segment['cutting_instructions']['end_time_formatted']})",
                    showarrow=True, arrowhead=2, arrowcolor="#ff6f00",
                    font=dict(color='#ff6f00', size=14, family="Arial, sans-serif"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#ff6f00",
                    borderwidth=2
                )
            
            # Formatamos a duração para o título
            duration_formatted = f"{int(duration//60)}:{int(duration%60):02d}"
            
            # Configuramos o layout do gráfico
            fig.update_layout(
                title=f"<b>📊 Análise + 🎬 Detecção de Shorts: {html.escape(title)}</b><br><sub>Duração total: {duration_formatted} | {len(intensities)} blocos analisados</sub>",
                height=600,
                xaxis=dict(
                    title='⏱️ Tempo (Minutos)', 
                    tickvals=x_values[::max(1, len(x_values)//10)],  # Limitamos a 10 labels
                    ticktext=[times[i] for i in x_values[::max(1, len(x_values)//10)]]
                ),
                yaxis=dict(title='📈 Intensidade Visual (0-5)', range=[0, 5.2]),
                yaxis2=dict(title='😊 Sentimento (-1 a 1)', range=[-1.1, 1.1], overlaying='y', side='right'),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode="x unified",
                legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)")
            )
            
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {str(e)}", exc_info=True)
            return f"<div class='alert alert-danger'>Erro ao gerar gráfico: {str(e)}</div>"

    # [CÓDIGO OCULTO PARA O PÚBLICO]
    # Esta seção contém a lógica para exportar resultados em CSV e HTML,
    # incluindo a geração de relatórios detalhados com gráficos e instruções de corte.
    # Não será exibida publicamente.

@app.route('/')
def index():
    """Página inicial da nossa aplicação web.
    Aqui criamos um formulário simples para o usuário inserir a URL do YouTube."""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analisador de Vídeos do YouTube</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; background-color: #f8f9fa; }
            .container { max-width: 800px; }
            .form-container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .loading { display: none; }
            .result { display: none; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">📹 Analisador de Vídeos do YouTube</h1>
            <div class="form-container">
                <form id="analyzeForm">
                    <div class="mb-3">
                        <label for="youtube_url" class="form-label">URL do Vídeo do YouTube</label>
                        <input type="url" class="form-control" id="youtube_url" name="youtube_url" 
                               placeholder="https://www.youtube.com/watch?v=..." required>
                        <div class="form-text">Cole a URL completa do vídeo do YouTube que deseja analisar</div>
                    </div>
                    <button type="submit" class="btn btn-primary">🔍 Analisar Vídeo</button>
                </form>
                <div class="loading mt-3 text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Analisando...</span>
                    </div>
                    <p class="mt-2">Analisando vídeo, isso pode levar alguns minutos...</p>
                </div>
                <div class="result mt-3"></div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            document.getElementById('analyzeForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const loading = document.querySelector('.loading');
                const result = document.querySelector('.result');
                const submitBtn = e.target.querySelector('button[type="submit"]');
                
                loading.style.display = 'block';
                result.style.display = 'none';
                submitBtn.disabled = true;
                
                const formData = new FormData(e.target);
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        result.innerHTML = `
                            <div class="alert alert-success">
                                <h5>✅ Análise Concluída!</h5>
                                <p><strong>Título:</strong> ${data.analysis_summary.title}</p>
                                <p><strong>Duração:</strong> ${Math.floor(data.analysis_summary.duration/60)}:${String(Math.floor(data.analysis_summary.duration%60)).padStart(2, '0')}</p>
                                <div class="mt-3">
                                    <a href="${data.html_url}" class="btn btn-primary" target="_blank">📊 Ver Análise Completa</a>
                                    <a href="${data.csv_url}" class="btn btn-secondary" download>📥 Baixar CSV</a>
                                </div>
                            </div>
                        `;
                    } else {
                        result.innerHTML = `
                            <div class="alert alert-danger">
                                <h5>❌ Erro na Análise</h5>
                                <p>${data.error || 'Erro desconhecido'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    result.innerHTML = `
                        <div class="alert alert-danger">
                            <h5>❌ Erro de Conexão</h5>
                            <p>${error.message}</p>
                        </div>
                    `;
                } finally {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    submitBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Rota que lida com a análise do vídeo.
    Aqui recebemos a URL, processamos o vídeo e devolvemos os resultados."""
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        return jsonify({'error': 'URL do YouTube não fornecida'}), 400

    tmp_dir = None
    try:
        analyzer = VideoAnalyzer()
        if not analyzer.valid_youtube_url(youtube_url):
            return jsonify({'error': 'URL do YouTube inválida'}), 400

        # Baixamos o vídeo e extraímos informações básicas
        video_path, title, duration, tmp_dir = analyzer.download_video(youtube_url)
        # Analisamos o vídeo para obter métricas
        blocks, avg_intensity, max_intensity, avg_sentiment, sentiment_stats = analyzer.analyze_video(video_path, duration)
        # Encontramos os melhores trechos para Shorts
        shorts_analysis = analyzer.detect_best_shorts(blocks)
        
        # Extraímos intensidades e sentimentos para o gráfico
        intensities = [b['intensity'] for b in blocks]
        sentiments = [b['sentiment'] for b in blocks]
        
        # Criamos o gráfico interativo
        plot_html = analyzer.build_combined_heatmap_with_shorts(
            intensities, sentiments, title, duration, 
            shorts_analysis.get('best_segment')
        )
        
        # Registramos quando a análise foi feita
        analysed_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_id = analyzer.extract_video_id(youtube_url)
        csv_name = f"analysis_{video_id}.csv"
        html_name = f"analysis_{video_id}.html"
        
        # Exportamos os resultados
        csv_path, html_path = analyzer.export_results_with_shorts(
            title, analysed_at, duration, avg_intensity, avg_sentiment, 
            blocks, csv_name, html_name, sentiment_stats, shorts_analysis, plot_html
        )
        
        # Preparamos a resposta com links para os arquivos gerados
        response_data = {
            'message': 'Análise concluída com sucesso',
            'csv_url': url_for('download_file', filename=csv_name),
            'html_url': url_for('download_file', filename=html_name),
            'analysis_summary': {
                'title': title,
                'duration': duration,
                'avg_intensity': float(avg_intensity),
                'max_intensity': float(max_intensity),
                'avg_sentiment': float(avg_sentiment),
                'sentiment_stats': convert_numpy_types(sentiment_stats),
                'shorts_analysis': convert_numpy_types(shorts_analysis)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Erro durante a análise: {str(e)}", exc_info=True)
        return jsonify({'error': f'Erro durante a análise: {str(e)}'}), 500
    finally:
        # Limpamos o diretório temporário para não deixar lixo
        if tmp_dir and os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as e:
                logger.warning(f"Erro ao limpar diretório temporário: {str(e)}")

@app.route('/download/<filename>')
def download_file(filename):
    """Permite que os usuários baixem os arquivos gerados (CSV ou HTML)."""
    try:
        # Validamos o nome do arquivo para evitar ataques de path traversal
        if not re.match(r'^analysis_[a-zA-Z0-9_-]+\.(csv|html)$', filename):
            return jsonify({'error': 'Nome de arquivo inválido'}), 400
            
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'Arquivo não encontrado'}), 404
            
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Erro ao baixar arquivo {filename}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Erro ao baixar arquivo: {str(e)}'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Lida com erros de upload muito grande."""
    return jsonify({'error': 'Arquivo muito grande. Tamanho máximo permitido: 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Lida com erros internos do servidor."""
    logger.error(f"Erro interno do servidor: {str(error)}", exc_info=True)
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    # Iniciamos o servidor Flask em modo produção
    app.run(debug=False, host='0.0.0.0', port=5000)