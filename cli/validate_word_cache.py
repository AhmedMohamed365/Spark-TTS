import argparse
import json
import os
import tempfile
import time
from types import MethodType

import torch

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_cache import TokenCache


class DummyAudioTokenizer:
    def detokenize(self, global_ids, semantic_ids):
        length = semantic_ids.shape[-1]
        return torch.zeros(length, dtype=torch.float32)


class DummyTokenizerOutput:
    def __init__(self):
        self.input_ids = torch.tensor([[1, 2, 3]])

    def to(self, _device):
        return self


class DummyTokenizer:
    def __call__(self, *_args, **_kwargs):
        return DummyTokenizerOutput()


def build_mock_spark_tts(simulated_llm_latency: float = 0.08):
    spark = SparkTTS.__new__(SparkTTS)
    spark.device = torch.device("cpu")
    spark.audio_tokenizer = DummyAudioTokenizer()
    spark.tokenizer = DummyTokenizer()
    spark.last_inference_stats = {}

    def process_prompt(self, text, prompt_speech_path, prompt_text=None):
        _ = (prompt_speech_path, prompt_text)
        return text, torch.tensor([[11, 22, 33]])

    def run_llm(self, prompt, top_k, top_p, temperature):
        _ = (top_k, top_p, temperature)
        time.sleep(simulated_llm_latency)
        semantic_values = [(ord(ch) % 50) + 1 for ch in prompt.strip()[:6]]
        if not semantic_values:
            semantic_values = [1]
        semantic_str = "".join([f"<|bicodec_semantic_{v}|>" for v in semantic_values])
        return semantic_str

    spark.process_prompt = MethodType(process_prompt, spark)
    spark._run_llm_from_prompt = MethodType(run_llm, spark)
    return spark


def run_validation(results_dir: str, simulated_llm_latency: float):
    os.makedirs(results_dir, exist_ok=True)
    spark = build_mock_spark_tts(simulated_llm_latency=simulated_llm_latency)

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = os.path.join(tmp_dir, "semantic_cache.json")
        cache = TokenCache(cache_path)

        text = "hello world hello"

        t1_start = time.perf_counter()
        _wav1 = spark.inference(
            text=text,
            prompt_speech_path="example/prompt_audio.wav",
            prompt_text="hello",
            use_word_cache=True,
            token_cache=cache,
        )
        first_latency = time.perf_counter() - t1_start
        first_stats = dict(spark.last_inference_stats)

        t2_start = time.perf_counter()
        _wav2 = spark.inference(
            text=text,
            prompt_speech_path="example/prompt_audio.wav",
            prompt_text="hello",
            use_word_cache=True,
            token_cache=cache,
        )
        second_latency = time.perf_counter() - t2_start
        second_stats = dict(spark.last_inference_stats)

    passed = second_latency < first_latency and second_stats["llm_calls"] == 0
    speedup = first_latency / second_latency if second_latency > 0 else float("inf")

    data = {
        "first_latency_sec": first_latency,
        "second_latency_sec": second_latency,
        "speedup_x": speedup,
        "first_stats": first_stats,
        "second_stats": second_stats,
        "passed": passed,
    }

    json_path = os.path.join(results_dir, "cache_validation_results.json")
    md_path = os.path.join(results_dir, "cache_validation_results.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Word Cache Validation\n\n")
        f.write(f"- Passed: `{passed}`\n")
        f.write(f"- First latency (s): `{first_latency:.6f}`\n")
        f.write(f"- Second latency (s): `{second_latency:.6f}`\n")
        f.write(f"- Speedup (x): `{speedup:.2f}`\n")
        f.write(f"- First stats: `{first_stats}`\n")
        f.write(f"- Second stats: `{second_stats}`\n")

    return data, json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Validate word-level token cache speedup.")
    parser.add_argument("--results_dir", default="RESULTS")
    parser.add_argument("--simulated_llm_latency", type=float, default=0.08)
    args = parser.parse_args()

    data, json_path, md_path = run_validation(args.results_dir, args.simulated_llm_latency)
    print(f"Validation passed: {data['passed']}")
    print(f"First latency: {data['first_latency_sec']:.6f}s")
    print(f"Second latency: {data['second_latency_sec']:.6f}s")
    print(f"Speedup: {data['speedup_x']:.2f}x")
    print(f"Results written to: {json_path}")
    print(f"Results written to: {md_path}")


if __name__ == "__main__":
    main()
