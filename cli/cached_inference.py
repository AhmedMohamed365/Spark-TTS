import argparse
import logging
import os
import platform
import time
from datetime import datetime

import soundfile as sf
import torch

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_cache import TokenCache


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU benchmark for word-level semantic token cache (no audio caching)."
    )
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Spark-TTS-0.5B")
    parser.add_argument("--save_dir", type=str, default="example/results")
    parser.add_argument("--cache_path", type=str, default="example/token_cache/semantic_tokens.json")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--second_text",
        type=str,
        default=None,
        help="Optional second inference text. Defaults to --text for cache hit validation.",
    )
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument("--prompt_speech_path", type=str, default=None)
    parser.add_argument("--gender", choices=["male", "female"], default=None)
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"], default=None
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"], default=None
    )
    return parser.parse_args()


def choose_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps:0")
    return torch.device("cpu")


def run_once(model, token_cache, text, args):
    start = time.perf_counter()
    wav = model.inference(
        text=text,
        prompt_speech_path=args.prompt_speech_path,
        prompt_text=args.prompt_text,
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed,
        use_word_cache=True,
        token_cache=token_cache,
    )
    elapsed = time.perf_counter() - start
    return wav, elapsed, model.last_inference_stats


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    device = choose_device()
    logging.info("Running benchmark on device: %s", device)
    model = SparkTTS(args.model_dir, device=device)
    token_cache = TokenCache(args.cache_path)

    text2 = args.second_text if args.second_text is not None else args.text

    wav1, t1, stats1 = run_once(model, token_cache, args.text, args)
    wav2, t2, stats2 = run_once(model, token_cache, text2, args)

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out1 = os.path.join(args.save_dir, f"{ts}_first.wav")
    out2 = os.path.join(args.save_dir, f"{ts}_second.wav")
    sf.write(out1, wav1, samplerate=16000)
    sf.write(out2, wav2, samplerate=16000)

    logging.info("First inference latency: %.3fs | stats=%s", t1, stats1)
    logging.info("Second inference latency: %.3fs | stats=%s", t2, stats2)
    if t1 > 0:
        logging.info("Speedup (first/second): %.2fx", t1 / t2)
    logging.info("Audio outputs: %s, %s", out1, out2)
    logging.info("Token cache: %s", args.cache_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
