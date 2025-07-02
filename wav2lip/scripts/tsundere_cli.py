#!/usr/bin/env python3
"""
ツンデレWav2Lip CLI
べ、別にあなたのためじゃないんだからね！
"""

import click
import os
import sys
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
import time
import random

# 自作モジュール
sys.path.append('/app')
from scripts.wav2lip_yolo_integration import TsundereWav2LipYOLOEngine

# ツンデレコンソール設定
console = Console()
logger = logging.getLogger(__name__)

class TsundereCLI:
    """ツンデレなCLIクラス"""
    
    def __init__(self):
        self.tsundere_phrases = [
            "ふん！また口パク作ってって言うのね...",
            "べ、別にあなたのためじゃないんだからね！",
            "でも...ちゃんと作ってあげるから感謝しなさいよ！",
            "な、何よ！そんなに見つめないでよ...",
            "完璧な結果を期待してるんでしょ？仕方ないわね...",
            "ふん！私の技術力を見せてあげるわ！",
            "べ、別にすごくないけど...まあまあの出来かしら",
            "もう！素直に喜びなさいよ！",
            "ツンデレじゃないわよ！ただの親切心よ！",
            "次回からは自分でやりなさいよね...でも困ったら言いなさい"
        ]
    
    def show_tsundere_message(self, message_type: str = "random"):
        """ツンデレメッセージを表示"""
        if message_type == "start":
            phrases = [
                "ふん！また口パク作ってって言うのね...",
                "べ、別にあなたのためじゃないんだからね！",
                "でも...ちゃんと作ってあげるから感謝しなさいよ！"
            ]
        elif message_type == "processing":
            phrases = [
                "ちゃんと処理してるから待ってなさい！",
                "な、何よ！そんなにじっと見てないでよ...",
                "完璧にやってるんだから邪魔しないで！"
            ]
        elif message_type == "success":
            phrases = [
                "ふん！完璧な出来よ！感謝しなさい！",
                "べ、別にすごくないけど...まあまあの出来かしら",
                "もう！素直に喜びなさいよ！"
            ]
        elif message_type == "error":
            phrases = [
                "な、何よ！エラーなんて出ないはずなのに...",
                "ちゃんとした入力ファイルを使いなさいよ！",
                "も、もう一度やり直すからちょっと待ってなさい！"
            ]
        else:
            phrases = self.tsundere_phrases
        
        message = random.choice(phrases)
        console.print(f"💕 {message}", style="bold magenta")
    
    def show_banner(self):
        """バナー表示"""
        banner_text = """
🎭 ツンデレWav2Lip CLI 🎭
Easy-Wav2Lip + YOLO11 + Docker統合システム
RTX 3050最適化版

べ、別にあなたのためじゃないんだからね！
でも...完璧な口パク動画を作ってあげるわ！
        """
        
        panel = Panel(
            banner_text.strip(),
            title="[bold cyan]ツンデレWav2Lip[/bold cyan]",
            border_style="magenta",
            padding=(1, 2)
        )
        console.print(panel)
        
        time.sleep(1)
        self.show_tsundere_message("start")

cli_helper = TsundereCLI()

@click.group()
@click.version_option(version="1.0.0", prog_name="ツンデレWav2Lip")
def main():
    """
    🎭 ツンデレWav2Lip CLI
    
    べ、別にあなたのためじゃないんだからね！
    でも完璧な口パク動画を作ってあげるわ！
    """
    pass

@main.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True),
              help='入力動画ファイルパス')
@click.option('--audio', '-a', required=True, type=click.Path(exists=True),
              help='入力音声ファイルパス')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='出力動画ファイルパス')
@click.option('--quality', '-q', 
              type=click.Choice(['Fast', 'Improved', 'Enhanced'], case_sensitive=False),
              default='Improved', show_default=True,
              help='品質設定 (Fast: 最速, Improved: バランス, Enhanced: 最高品質)')
@click.option('--yolo-model', default='yolo11n',
              type=click.Choice(['yolo11n', 'yolo11s', 'yolo11m'], case_sensitive=False),
              help='YOLOモデルサイズ (n=最速, s=バランス, m=高精度)')
@click.option('--face-confidence', default=0.7, type=float,
              help='顔検出信頼度閾値 (0.0-1.0)')
@click.option('--tensorrt', is_flag=True, default=False,
              help='TensorRT最適化を有効化 (RTX 3050推奨)')
@click.option('--target-face', type=int, default=None,
              help='対象顔ID指定 (未指定時は自動選択)')
@click.option('--tsundere-mode', is_flag=True, default=True,
              help='ツンデレモードON/OFF')
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='詳細ログ表示')
def generate(video, audio, output, quality, yolo_model, face_confidence, 
            tensorrt, target_face, tsundere_mode, verbose):
    """
    口パク動画を生成
    
    べ、別にあなたのためじゃないけど...
    完璧な口パク動画を作ってあげるわよ！
    """
    
    # ログレベル設定
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # ツンデレモード設定
    if tsundere_mode:
        cli_helper.show_banner()
    
    try:
        # 入力検証
        if not _validate_inputs(video, audio, output, tsundere_mode):
            return
        
        # エンジン初期化
        if tsundere_mode:
            console.print("🔧 [bold yellow]システム初期化中...[/bold yellow]")
            cli_helper.show_tsundere_message("processing")
        
        # モデルパス設定
        yolo_model_path = f"/app/models/yolo/{yolo_model}.pt"
        wav2lip_model_path = "/app/models/wav2lip/wav2lip_gan.pth"
        
        # GPU確認
        import torch
        if not torch.cuda.is_available():
            if tsundere_mode:
                console.print("❌ [bold red]GPUが使えないじゃない！CPUなんて遅すぎるわよ！[/bold red]")
            else:
                console.print("❌ CUDA GPU not available")
            return
        
        # TensorRT設定
        if tensorrt:
            os.environ['TENSORRT_OPTIMIZE'] = 'true'
            if tsundere_mode:
                console.print("⚡ [bold green]TensorRT最適化有効！RTX 3050で爆速よ！[/bold green]")
        
        # エンジン初期化
        engine = TsundereWav2LipYOLOEngine(
            wav2lip_model_path=wav2lip_model_path,
            yolo_model_path=yolo_model_path,
            device="cuda",
            tsundere_mode=tsundere_mode
        )
        
        # 処理実行
        if tsundere_mode:
            console.print("🎬 [bold cyan]口パク動画生成開始！[/bold cyan]")
            cli_helper.show_tsundere_message("processing")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            
            task = progress.add_task(
                "処理中..." if tsundere_mode else "Processing...", 
                total=100
            )
            
            start_time = time.time()
            
            success = engine.process_video_with_audio(
                video_path=video,
                audio_path=audio,
                output_path=output,
                quality=quality,
                target_face_id=target_face
            )
            
            progress.update(task, completed=100)
            
            end_time = time.time()
            processing_time = end_time - start_time
        
        # 結果表示
        if success:
            if tsundere_mode:
                cli_helper.show_tsundere_message("success")
                console.print(f"✅ [bold green]完璧な口パク動画ができたわよ！[/bold green]")
                console.print(f"📁 出力: {output}")
                console.print(f"⏱️  処理時間: {processing_time:.1f}秒")
                console.print("💕 [bold magenta]感謝しなさいよね！[/bold magenta]")
            else:
                console.print(f"✅ Video generated successfully!")
                console.print(f"📁 Output: {output}")
                console.print(f"⏱️ Processing time: {processing_time:.1f}s")
        else:
            if tsundere_mode:
                cli_helper.show_tsundere_message("error")
                console.print("❌ [bold red]な、何かエラーが出たじゃない！ログを確認しなさい！[/bold red]")
            else:
                console.print("❌ Processing failed. Check logs for details.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        if tsundere_mode:
            console.print("\n💔 [bold yellow]中断されたじゃない...せっかく頑張ってたのに...[/bold yellow]")
        else:
            console.print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        if tsundere_mode:
            cli_helper.show_tsundere_message("error")
            console.print(f"❌ [bold red]予期しないエラーが出たじゃない！: {e}[/bold red]")
        else:
            console.print(f"❌ Unexpected error: {e}")
        sys.exit(1)

@main.command()
@click.option('--input-dir', required=True, type=click.Path(exists=True),
              help='入力ディレクトリ（動画ファイル）')
@click.option('--audio-file', required=True, type=click.Path(exists=True),
              help='音声ファイル（全動画共通）')
@click.option('--output-dir', required=True, type=click.Path(),
              help='出力ディレクトリ')
@click.option('--quality', default='Improved',
              type=click.Choice(['Fast', 'Improved', 'Enhanced'], case_sensitive=False),
              help='品質設定')
@click.option('--parallel', default=1, type=int,
              help='並列処理数 (RTX 3050では1-2推奨)')
@click.option('--tsundere-mode', is_flag=True, default=True,
              help='ツンデレモードON/OFF')
def batch(input_dir, audio_file, output_dir, quality, parallel, tsundere_mode):
    """
    バッチ処理で複数動画を一括変換
    
    ふん！まとめて処理してほしいのね...
    仕方ないから一気にやってあげるわ！
    """
    
    if tsundere_mode:
        cli_helper.show_banner()
        console.print("📂 [bold cyan]バッチ処理モード起動！[/bold cyan]")
        console.print("ふん！まとめて処理なんて楽勝よ！")
    
    try:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 動画ファイル検索
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            if tsundere_mode:
                console.print("❌ [bold red]動画ファイルが見つからないじゃない！[/bold red]")
            else:
                console.print("❌ No video files found")
            return
        
        if tsundere_mode:
            console.print(f"📹 {len(video_files)}個の動画を発見したわよ！")
        
        # バッチ処理実行
        success_count = 0
        error_count = 0
        
        with Progress(console=console) as progress:
            main_task = progress.add_task("バッチ処理中...", total=len(video_files))
            
            for video_file in video_files:
                try:
                    output_file = output_path / f"{video_file.stem}_lipsync.mp4"
                    
                    if tsundere_mode:
                        progress.console.print(f"🎬 処理中: {video_file.name}")
                    
                    # エンジン初期化（各ファイルごと）
                    engine = TsundereWav2LipYOLOEngine(
                        device="cuda",
                        tsundere_mode=False  # バッチ時は静音
                    )
                    
                    success = engine.process_video_with_audio(
                        video_path=str(video_file),
                        audio_path=audio_file,
                        output_path=str(output_file),
                        quality=quality
                    )
                    
                    if success:
                        success_count += 1
                        if tsundere_mode:
                            progress.console.print(f"✅ {video_file.name} 完了！")
                    else:
                        error_count += 1
                        if tsundere_mode:
                            progress.console.print(f"❌ {video_file.name} 失敗...")
                
                except Exception as e:
                    error_count += 1
                    if tsundere_mode:
                        progress.console.print(f"❌ {video_file.name} エラー: {e}")
                
                progress.update(main_task, advance=1)
        
        # 結果サマリー
        if tsundere_mode:
            console.print(f"\n📊 [bold cyan]バッチ処理完了！[/bold cyan]")
            console.print(f"✅ 成功: {success_count}件")
            console.print(f"❌ 失敗: {error_count}件")
            
            if success_count > 0:
                cli_helper.show_tsundere_message("success")
            if error_count > 0:
                console.print("💔 失敗したファイルもあるけど...まあ仕方ないわね")
        else:
            console.print(f"Batch processing completed: {success_count} success, {error_count} errors")
    
    except Exception as e:
        if tsundere_mode:
            cli_helper.show_tsundere_message("error")
            console.print(f"❌ バッチ処理でエラーが出たじゃない！: {e}")
        else:
            console.print(f"❌ Batch processing error: {e}")
        sys.exit(1)

@main.command()
@click.option('--tsundere-mode', is_flag=True, default=True,
              help='ツンデレモードON/OFF')
def test(tsundere_mode):
    """
    システムテスト実行
    
    ふん！システムのテストなんて当然でしょ？
    完璧に動くか確認してあげるわ！
    """
    
    if tsundere_mode:
        cli_helper.show_banner()
        console.print("🧪 [bold cyan]システムテスト開始！[/bold cyan]")
    
    tests = [
        ("GPU確認", _test_gpu),
        ("YOLOモデル確認", _test_yolo_models),
        ("Wav2Lipモデル確認", _test_wav2lip_models),
        ("依存関係確認", _test_dependencies)
    ]
    
    results = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("テスト実行中...", total=len(tests))
        
        for test_name, test_func in tests:
            if tsundere_mode:
                progress.console.print(f"🔍 {test_name}...")
            
            try:
                result = test_func()
                results.append((test_name, result, None))
                
                if result:
                    progress.console.print(f"✅ {test_name}: OK")
                else:
                    progress.console.print(f"❌ {test_name}: NG")
                    
            except Exception as e:
                results.append((test_name, False, str(e)))
                progress.console.print(f"❌ {test_name}: エラー - {e}")
            
            progress.update(task, advance=1)
    
    # 結果サマリー
    success_count = sum(1 for _, result, _ in results if result)
    total_count = len(results)
    
    if tsundere_mode:
        console.print(f"\n📊 [bold cyan]テスト結果[/bold cyan]")
        console.print(f"✅ 成功: {success_count}/{total_count}")
        
        if success_count == total_count:
            cli_helper.show_tsundere_message("success")
            console.print("🎉 [bold green]全テストパス！完璧なシステムよ！[/bold green]")
        else:
            console.print("⚠️ [bold yellow]一部テストが失敗してるわね...確認しなさい[/bold yellow]")
    else:
        console.print(f"Test results: {success_count}/{total_count} passed")

def _validate_inputs(video, audio, output, tsundere_mode):
    """入力検証"""
    errors = []
    
    # ファイル存在確認
    if not os.path.exists(video):
        errors.append("動画ファイルが見つからないじゃない！")
    
    if not os.path.exists(audio):
        errors.append("音声ファイルが見つからないじゃない！")
    
    # 出力ディレクトリ確認
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except:
            errors.append("出力ディレクトリが作れないじゃない！")
    
    # 拡張子確認
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    audio_exts = ['.wav', '.mp3', '.m4a']
    
    if not any(video.lower().endswith(ext) for ext in video_exts):
        errors.append("対応していない動画形式よ！MP4/AVI/MOV/MKVを使いなさい！")
    
    if not any(audio.lower().endswith(ext) for ext in audio_exts):
        errors.append("対応していない音声形式よ！WAV/MP3/M4Aを使いなさい！")
    
    if errors:
        if tsundere_mode:
            for error in errors:
                console.print(f"❌ [bold red]{error}[/bold red]")
            cli_helper.show_tsundere_message("error")
        else:
            for error in errors:
                console.print(f"❌ {error}")
        return False
    
    return True

def _test_gpu():
    """GPU確認"""
    import torch
    return torch.cuda.is_available()

def _test_yolo_models():
    """YOLOモデル確認"""
    models = ['yolo11n.pt', 'yolo11s.pt']
    for model in models:
        path = f"/app/models/yolo/{model}"
        if not os.path.exists(path):
            return False
    return True

def _test_wav2lip_models():
    """Wav2Lipモデル確認"""
    models = ['wav2lip.pth', 'wav2lip_gan.pth']
    for model in models:
        path = f"/app/models/wav2lip/{model}"
        if not os.path.exists(path):
            return False
    return True

def _test_dependencies():
    """依存関係確認"""
    try:
        import torch
        import cv2
        import librosa
        import ultralytics
        return True
    except ImportError:
        return False

if __name__ == '__main__':
    main()