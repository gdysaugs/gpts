#!/usr/bin/env python3
"""
Gradio GPT-SoVITS 音声クローニング Web UI
FastAPI版の全機能をGradio 3系に移行
初期化1回のみ、各リクエスト2-3秒で高速応答
"""

import os
import sys
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, Tuple, Any
import uuid
import tempfile
import urllib.request
import numpy as np
import soundfile as sf

# Gradio 3系
import gradio as gr

# GPT-SoVITS
os.chdir('/app')
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')

import torch

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# グローバル変数
MODELS_LOADED = False
CUSTOM_SOVITS_PATH = None
PRELOADED_LANGDETECT = None
CACHE_DIR = "/app/cache"

# GPU排他制御用
gpu_lock = threading.Lock()

def setup_torch_optimizations():
    """Torch最適化セットアップ"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        logger.info("🚀 RTX 3050 TensorCore最適化有効")
    
    torch.set_float32_matmul_precision('medium')
    torch.set_num_threads(8)
    logger.info("✅ PyTorch最適化完了")

def comprehensive_monkey_patch():
    """モンキーパッチ適用"""
    from GPT_SoVITS import inference_webui
    
    original_load_sovits = inference_webui.load_sovits_new
    original_change_sovits = inference_webui.change_sovits_weights
    
    def custom_load_sovits_new(sovits_path):
        global CUSTOM_SOVITS_PATH
        if CUSTOM_SOVITS_PATH and os.path.exists(CUSTOM_SOVITS_PATH):
            actual_path = CUSTOM_SOVITS_PATH
        else:
            actual_path = sovits_path
        
        if actual_path.endswith('.ckpt'):
            checkpoint = torch.load(actual_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            dict_s2 = {
                'weight': state_dict,
                'config': {
                    'model': {
                        'version': 'v2',
                        'semantic_frame_rate': '25hz',
                        'inter_channels': 192,
                        'hidden_channels': 192,
                        'filter_channels': 768,
                        'n_heads': 2,
                        'n_layers': 6,
                        'kernel_size': 3,
                        'p_dropout': 0.1,
                        'ssl_dim': 768,
                        'n_speakers': 300
                    },
                    'data': {
                        'sampling_rate': 32000,
                        'filter_length': 2048,
                        'hop_length': 640,
                        'win_length': 2048,
                        'n_speakers': 300,
                        'cleaned_text': True,
                        'add_blank': True,
                        'n_symbols': 178
                    }
                }
            }
            return dict_s2
        else:
            return original_load_sovits(actual_path)
    
    def custom_change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
        global CUSTOM_SOVITS_PATH
        if CUSTOM_SOVITS_PATH:
            sovits_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        return original_change_sovits(sovits_path, prompt_language, text_language)
    
    inference_webui.load_sovits_new = custom_load_sovits_new
    inference_webui.change_sovits_weights = custom_change_sovits_weights
    logger.info("🔧 モンキーパッチ適用完了")

def apply_torch_compile_optimization():
    """Torch.compile最適化適用"""
    try:
        if not hasattr(torch, 'compile'):
            logger.warning("⚠️ PyTorch 2.0+が必要（Torch.compile非対応）")
            return
        
        from GPT_SoVITS import inference_webui
        
        # SoVITSモデル最適化
        if hasattr(inference_webui, 'vq_model') and inference_webui.vq_model is not None:
            logger.info("🔥 SoVITSモデルcompile最適化中...")
            inference_webui.vq_model = torch.compile(
                inference_webui.vq_model,
                mode="max-autotune",
                dynamic=True,
                backend="inductor"
            )
        
        # GPTモデル最適化
        if hasattr(inference_webui, 't2s_model') and inference_webui.t2s_model is not None:
            logger.info("🔥 GPTモデルcompile最適化中...")
            inference_webui.t2s_model = torch.compile(
                inference_webui.t2s_model,
                mode="max-autotune",
                dynamic=True,
                backend="inductor"
            )
        
        # HuBERTモデル最適化
        if hasattr(inference_webui, 'hubert_model') and inference_webui.hubert_model is not None:
            logger.info("🔥 HuBERTモデルcompile最適化中...")
            inference_webui.hubert_model = torch.compile(
                inference_webui.hubert_model,
                mode="reduce-overhead",
                dynamic=True,
                backend="inductor"
            )
        
        logger.info("🚀 Torch.compile最適化完了")
        
    except Exception as e:
        logger.error(f"❌ Torch.compile最適化エラー: {e}")

def initialize_models():
    """モデル初期化（サーバー起動時1回のみ）"""
    global MODELS_LOADED, CUSTOM_SOVITS_PATH
    
    if MODELS_LOADED:
        return
    
    logger.info("🚀 === サーバー初期化開始 ===")
    init_start = time.time()
    
    try:
        # カスタムモデルパス設定
        CUSTOM_SOVITS_PATH = "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt"
        
        # 最適化設定
        setup_torch_optimizations()
        comprehensive_monkey_patch()
        
        # モデルロード
        from GPT_SoVITS.inference_webui import change_sovits_weights
        default_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        change_sovits_weights(default_path)
        
        # Warm-up推論
        logger.info("🔥 Warm-up推論実行中...")
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        dummy_gen = get_tts_wav(
            ref_wav_path="/app/input/reference_5sec.wav",
            prompt_text="こんにちは",
            prompt_language="Japanese",
            text="テスト",
            text_language="Japanese",
            how_to_cut="不切",
            top_k=5,
            top_p=1.0,
            temperature=1.0,
            ref_free=True
        )
        
        # 一つだけ生成してWarm-up完了
        for i, item in enumerate(dummy_gen):
            if i == 0:
                break
        
        # Torch.compile最適化
        apply_torch_compile_optimization()
        
        # CUDAキャッシュ最適化
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        MODELS_LOADED = True
        init_time = time.time() - init_start
        
        logger.info(f"✅ === サーバー初期化完了: {init_time:.2f}秒 ===")
        logger.info("🎯 以降のリクエストは2-3秒で応答予定")
        
    except Exception as e:
        logger.error(f"❌ サーバー初期化エラー: {e}")
        raise

def preload_all_dependencies():
    """全ての依存関係を事前ダウンロード・キャッシュする"""
    global PRELOADED_LANGDETECT
    
    # キャッシュディレクトリ作成
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. 言語検出モデルの事前ダウンロード
    logger.info("🔥 言語検出モデル事前ダウンロード中...")
    try:
        langdetect_model_path = f"{CACHE_DIR}/lid.176.bin"
        if not os.path.exists(langdetect_model_path):
            logger.info("📥 言語検出モデルをダウンロード中...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                langdetect_model_path
            )
            logger.info("✅ 言語検出モデルダウンロード完了")
        else:
            logger.info("✅ 言語検出モデル既存キャッシュ使用")
            
        # 環境変数でモデルパスを指定
        os.environ["FASTTEXT_MODEL_PATH"] = langdetect_model_path
        
        # モデルをメモリにロード
        import fast_langdetect
        from fast_langdetect import detect
        
        # 強制的にモデルを初期化してキャッシュ
        detect("Hello", low_memory=False)
        detect("こんにちは", low_memory=False)
        PRELOADED_LANGDETECT = True
        
        logger.info("✅ 言語検出モデルプリロード完了")
    except Exception as e:
        logger.warning(f"⚠️ 言語検出モデルプリロード失敗: {e}")
    
    # 2. Open JTalk辞書の事前ダウンロード
    logger.info("🔥 Open JTalk辞書事前ダウンロード中...")
    try:
        jtalk_dict_path = f"{CACHE_DIR}/open_jtalk_dic_utf_8-1.11.tar.gz"
        if not os.path.exists(jtalk_dict_path):
            logger.info("📥 Open JTalk辞書をダウンロード中...")
            urllib.request.urlretrieve(
                "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz",
                jtalk_dict_path
            )
            logger.info("✅ Open JTalk辞書ダウンロード完了")
        else:
            logger.info("✅ Open JTalk辞書既存キャッシュ使用")
            
        # 環境変数でキャッシュパスを設定
        os.environ["OPEN_JTALK_DICT_PATH"] = jtalk_dict_path
        
        logger.info("✅ Open JTalk辞書プリロード完了")
    except Exception as e:
        logger.warning(f"⚠️ Open JTalk辞書プリロード失敗: {e}")
    
    # 3. その他の依存関係のプリロード
    logger.info("🔥 その他依存関係プリロード中...")
    try:
        # jieba辞書のプリロード
        import jieba
        jieba.initialize()
        
        # TorchAudio周りのプリロード
        import torchaudio
        torchaudio.set_audio_backend("sox_io")
        
        logger.info("✅ その他依存関係プリロード完了")
    except Exception as e:
        logger.warning(f"⚠️ その他依存関係プリロード失敗: {e}")

def ensure_text_length(text: str, min_length: int = 20) -> str:
    """テキストが指定文字数未満の場合、自然に延長する"""
    if len(text) >= min_length:
        return text
    
    # 短いテキストの場合の延長パターン
    extensions = [
        "とても良い音声ですね。",
        "素晴らしい結果だと思います。",
        "音声クローニングの技術は凄いです。",
        "これは高品質な音声生成です。",
        "日本語特化モデルで生成されました。"
    ]
    
    # 文末に句点がない場合は追加
    if not text.endswith(('。', '！', '？')):
        text += "。"
    
    # 20文字以上になるまで延長
    while len(text) < min_length:
        # 一番適当な延長を選択（句点の重複を避ける）
        extension = extensions[len(text) % len(extensions)]
        text += extension
    
    logger.info(f"📝 テキスト自動延長: {len(text)}文字 → {text}")
    return text

def generate_voice_with_uploaded_ref(ref_audio_path: str, ref_text: str, target_text: str, 
                                   temperature: float = 1.0) -> Tuple[str, str]:
    """アップロードされた参照音声で音声生成（Gradio用）"""
    
    if not MODELS_LOADED:
        return None, "❌ モデルが初期化されていません"
    
    # GPU排他制御
    with gpu_lock:
        try:
            # テキストを20文字以上に自動延長
            target_text = ensure_text_length(target_text, 20)
            
            generation_start = time.time()
            
            from GPT_SoVITS.inference_webui import get_tts_wav
            
            # 音声生成実行
            result_generator = get_tts_wav(
                ref_wav_path=ref_audio_path,
                prompt_text=ref_text,
                prompt_language="Japanese",
                text=target_text,
                text_language="Japanese",
                how_to_cut="不切",
                top_k=5,
                top_p=1.0,
                temperature=temperature,
                ref_free=True
            )
            
            # 結果処理
            audio_segments = []
            for i, item in enumerate(result_generator):
                if isinstance(item, tuple) and len(item) == 2:
                    sample_rate, audio_data = item
                    audio_segments.append(audio_data)
            
            if not audio_segments:
                return None, "❌ 音声生成失敗"
            
            # 音声連結
            if len(audio_segments) > 1:
                final_audio = np.concatenate(audio_segments)
            else:
                final_audio = audio_segments[0]
            
            generation_time = time.time() - generation_start
            audio_duration = len(final_audio) / 32000
            realtime_factor = audio_duration / generation_time
            
            # 音声品質統計
            audio_rms = float(np.sqrt(np.mean(final_audio ** 2)))
            non_silence_ratio = float(np.sum(np.abs(final_audio) > np.max(np.abs(final_audio)) * 0.01) / len(final_audio))
            
            # 出力ファイルに保存
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_target = target_text.replace(' ', '_').replace('！', '').replace('？', '')[:30]
            output_path = f"/app/output/gradio_{timestamp}_{safe_target}.wav"
            
            sf.write(output_path, final_audio, 32000)
            
            # 統計情報メッセージ
            stats_message = f"""
            ✅ 音声生成完了！
            
            📊 **パフォーマンス統計**
            - 生成時間: {generation_time:.2f}秒
            - 音声長: {audio_duration:.2f}秒
            - リアルタイム係数: {realtime_factor:.2f}x
            - 音声品質 (RMS): {audio_rms:.4f}
            - 非無音率: {non_silence_ratio:.2f}
            
            🎯 **生成設定**
            - 参照テキスト: {ref_text}
            - 生成テキスト: {target_text}
            - 温度パラメータ: {temperature}
            
            📁 **保存場所**: {output_path}
            """
            
            logger.info(f"✅ 音声生成完了: {generation_time:.2f}秒, リアルタイム係数: {realtime_factor:.2f}x")
            
            return output_path, stats_message
            
        except Exception as e:
            logger.error(f"❌ 音声生成エラー: {e}")
            return None, f"❌ 音声生成エラー: {str(e)}"

def generate_voice_with_default_ref(ref_text: str, target_text: str, 
                                  temperature: float = 1.0) -> Tuple[str, str]:
    """デフォルト参照音声で音声生成（Gradio用）"""
    
    # デフォルト参照音声を使用
    default_ref_path = "/app/input/reference_5sec.wav"
    
    if not os.path.exists(default_ref_path):
        return None, f"❌ デフォルト参照音声が見つかりません: {default_ref_path}"
    
    return generate_voice_with_uploaded_ref(default_ref_path, ref_text, target_text, temperature)

def create_gradio_interface():
    """Gradio インターフェース作成"""
    
    # カスタムCSS
    custom_css = """
    #main_title {
        text-align: center;
        color: #2E8B57;
        font-weight: bold;
        margin-bottom: 20px;
    }
    #stats_output {
        font-family: monospace;
        font-size: 12px;
        white-space: pre-wrap;
    }
    """
    
    # 2つのタブを持つインターフェース
    with gr.Blocks(css=custom_css, title="GPT-SoVITS 音声クローニング") as demo:
        
        gr.Markdown("# 🎤 GPT-SoVITS 音声クローニングシステム", elem_id="main_title")
        gr.Markdown("### 日本語特化モデル対応 | 高速生成 | 感情豊かな音声")
        
        with gr.Tabs():
            # タブ1: 音声アップロード版
            with gr.TabItem("📁 音声アップロード"):
                gr.Markdown("### 🎵 あなたの音声をアップロードして、その声質で任意のテキストを読み上げます")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_audio_input = gr.Audio(
                            label="📤 参照音声をアップロード",
                            type="filepath",
                            value=None
                        )
                        ref_text_input = gr.Textbox(
                            label="📝 参照音声のテキスト",
                            placeholder="アップロードした音声で話されている内容を入力してください",
                            lines=2,
                            value="こんにちは、今日は良い天気ですね。"
                        )
                        target_text_input = gr.Textbox(
                            label="🎯 生成したいテキスト",
                            placeholder="この声質で読み上げたい文章を入力してください",
                            lines=3,
                            value="音声クローニング技術は素晴らしい進歩を遂げています。"
                        )
                        temperature_input = gr.Slider(
                            label="🌡️ 温度パラメータ (創造性)",
                            minimum=0.5,
                            maximum=2.0,
                            step=0.1,
                            value=1.0
                        )
                        generate_btn = gr.Button("🚀 音声生成", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_audio = gr.Audio(
                            label="🎵 生成された音声",
                            type="filepath"
                        )
                        output_stats = gr.Textbox(
                            label="📊 生成統計",
                            lines=15,
                            elem_id="stats_output"
                        )
                
                generate_btn.click(
                    fn=generate_voice_with_uploaded_ref,
                    inputs=[ref_audio_input, ref_text_input, target_text_input, temperature_input],
                    outputs=[output_audio, output_stats]
                )
            
            # タブ2: デフォルト参照音声版
            with gr.TabItem("⚡ クイック生成"):
                gr.Markdown("### 🎯 デフォルト参照音声を使用した高速生成")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        quick_ref_text = gr.Textbox(
                            label="📝 参照音声のテキスト",
                            placeholder="デフォルト参照音声の内容",
                            lines=1,
                            value="おはようございます"
                        )
                        quick_target_text = gr.Textbox(
                            label="🎯 生成したいテキスト",
                            placeholder="この声質で読み上げたい文章を入力してください",
                            lines=3,
                            value="今日は美しい天気ですね。散歩に出かけましょう。"
                        )
                        quick_temperature = gr.Slider(
                            label="🌡️ 温度パラメータ (創造性)",
                            minimum=0.5,
                            maximum=2.0,
                            step=0.1,
                            value=1.0
                        )
                        quick_generate_btn = gr.Button("⚡ 高速生成", variant="primary")
                    
                    with gr.Column(scale=1):
                        quick_output_audio = gr.Audio(
                            label="🎵 生成された音声",
                            type="filepath"
                        )
                        quick_output_stats = gr.Textbox(
                            label="📊 生成統計",
                            lines=15,
                            elem_id="stats_output"
                        )
                
                quick_generate_btn.click(
                    fn=generate_voice_with_default_ref,
                    inputs=[quick_ref_text, quick_target_text, quick_temperature],
                    outputs=[quick_output_audio, quick_output_stats]
                )
        
        # フッター
        gr.Markdown("""
        ---
        ### 🔧 **システム情報**
        - **モデル**: AkitoP/GPT-SoVITS-JA-H (650時間学習済み)
        - **最適化**: PyTorch 2.0 + Torch.compile + TensorCore
        - **推定応答時間**: 2-7秒 (初回は事前ロード済み)
        - **対応言語**: 日本語特化 (英語技術用語対応)
        """)
    
    return demo

def main():
    """メイン関数"""
    try:
        # 初期化
        logger.info("🚀 Gradio GPT-SoVITS サーバー起動中...")
        
        # 事前ロード
        preload_all_dependencies()
        
        # モデル初期化
        initialize_models()
        
        # Gradio インターフェース作成
        demo = create_gradio_interface()
        
        # サーバー起動
        logger.info("🌐 Gradio サーバーを起動中...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            inbrowser=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"❌ サーバー起動エラー: {e}")
        raise

if __name__ == "__main__":
    main()