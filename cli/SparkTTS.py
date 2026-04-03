# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
from sparktts.utils.token_cache import TokenCache


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self.last_inference_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "segments": 0,
        }
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    @staticmethod
    def _split_word_segments(text: str):
        return re.findall(r"\S+\s*", text)

    @staticmethod
    def _extract_semantic_ids(predicts: str) -> torch.Tensor:
        semantic_ids = [int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)]
        return torch.tensor(semantic_ids).long().unsqueeze(0)

    @staticmethod
    def _extract_global_ids(predicts: str) -> torch.Tensor:
        global_ids = [int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)]
        return torch.tensor(global_ids).long().unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _segment_cache_key(
        segment: str,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
    ) -> str:
        prompt_text = prompt_text or ""
        gender = gender or ""
        pitch = pitch or ""
        speed = speed or ""
        return f"segment={segment}|prompt_text={prompt_text}|gender={gender}|pitch={pitch}|speed={speed}"

    def _run_llm_from_prompt(
        self,
        prompt: str,
        top_k: float,
        top_p: float,
        temperature: float,
    ) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        use_word_cache: bool = False,
        token_cache: TokenCache = None,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        self.last_inference_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "segments": 1,
        }

        if use_word_cache and token_cache is not None:
            segments = self._split_word_segments(text)
            self.last_inference_stats["segments"] = len(segments)
            all_semantic_ids = []
            global_token_ids = None

            for segment in segments:
                cache_key = self._segment_cache_key(
                    segment=segment,
                    prompt_text=prompt_text,
                    gender=gender,
                    pitch=pitch,
                    speed=speed,
                )
                cached = token_cache.get(cache_key)
                if cached is not None:
                    all_semantic_ids.extend(cached.semantic_ids)
                    if gender is not None and global_token_ids is None and cached.global_ids is not None:
                        global_token_ids = torch.tensor(cached.global_ids).long().unsqueeze(0).unsqueeze(0)
                    self.last_inference_stats["cache_hits"] += 1
                    continue

                self.last_inference_stats["cache_misses"] += 1
                if gender is not None:
                    prompt = self.process_prompt_control(gender, pitch, speed, segment)
                    predicts = self._run_llm_from_prompt(prompt, top_k, top_p, temperature)
                    segment_semantic_ids = self._extract_semantic_ids(predicts).squeeze(0).tolist()
                    if global_token_ids is None:
                        global_token_ids = self._extract_global_ids(predicts)
                    cached_global_ids = global_token_ids.squeeze(0).squeeze(0).tolist()
                else:
                    prompt, segment_global_token_ids = self.process_prompt(
                        segment, prompt_speech_path, prompt_text
                    )
                    predicts = self._run_llm_from_prompt(prompt, top_k, top_p, temperature)
                    segment_semantic_ids = self._extract_semantic_ids(predicts).squeeze(0).tolist()
                    global_token_ids = segment_global_token_ids
                    cached_global_ids = None

                token_cache.set(cache_key, segment_semantic_ids, cached_global_ids)
                all_semantic_ids.extend(segment_semantic_ids)
                self.last_inference_stats["llm_calls"] += 1

            pred_semantic_ids = torch.tensor(all_semantic_ids).long().unsqueeze(0)
            token_cache.save()
        else:
            if gender is not None:
                prompt = self.process_prompt_control(gender, pitch, speed, text)
                predicts = self._run_llm_from_prompt(prompt, top_k, top_p, temperature)
                pred_semantic_ids = self._extract_semantic_ids(predicts)
            else:
                prompt, global_token_ids = self.process_prompt(
                    text, prompt_speech_path, prompt_text
                )
                predicts = self._run_llm_from_prompt(prompt, top_k, top_p, temperature)
                pred_semantic_ids = self._extract_semantic_ids(predicts)
            self.last_inference_stats["llm_calls"] = 1

        if gender is not None:
            if global_token_ids is None:
                global_token_ids = self._extract_global_ids(predicts)

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        return wav
