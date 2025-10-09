# Copyright (C) 2025 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import argparse
import csv
import os
import re

import librosa
import numpy as np
import soundfile as sf
import torch
from netease.utils import load_config

from model import create_model


def apply_fade(audio, fade_samples, in_out="in"):
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
        single_channel = True
    else:
        single_channel = False

    fade_curve = (
        np.linspace(0, 1, fade_samples) if in_out == "in" else np.linspace(1, 0, fade_samples)
    )

    for ch in range(audio.shape[0]):
        if fade_samples > audio.shape[1]:
            fade_samples = audio.shape[1]
            fade_curve = (
                np.linspace(0, 1, fade_samples)
                if in_out == "in"
                else np.linspace(1, 0, fade_samples)
            )
        if in_out == "in":
            audio[ch, :fade_samples] *= fade_curve
        else:
            audio[ch, -fade_samples:] *= fade_curve

    if single_channel:
        audio = audio.squeeze(0)
    return audio


def is_silent(audio, sr=16000, threshold=-40, frame_duration_ms=30):
    if len(audio) == 0:
        return True

    frame_size = int(sr * frame_duration_ms / 1000)
    energy_threshold = 10 ** (threshold / 10.0)

    voiced_frames = 0
    total_frames = 0

    for i in range(0, len(audio), frame_size):
        frame = audio[i : i + frame_size]
        if len(frame) < frame_size:
            continue

        rms = np.sqrt(np.mean(frame**2))

        if rms > energy_threshold:
            voiced_frames += 1
        total_frames += 1

    return voiced_frames / total_frames < 0.1 if total_frames > 0 else True


def process_audio_file(y, sr, model):
    try:
        original_shape = y.shape
        is_stereo = len(original_shape) > 1 and original_shape[0] > 1

        if is_stereo:
            left_channel = y[0]
            right_channel = y[1]
            y_mono = np.mean(y, axis=0)
        else:
            y_mono = y

        y_mono = y_mono.astype(np.float32)
        y_16k = librosa.resample(y_mono, orig_sr=sr, target_sr=model_sr)

        num_chunks = len(y_16k) // chunk_samples_model
        if num_chunks < 1:
            return 0, None

        segments = []
        for i in range(0, num_chunks, batch_size):
            batch_end = min(i + batch_size, num_chunks)
            batch_data = []

            for j in range(i, batch_end):
                start = j * chunk_samples_model
                end = start + chunk_samples_model
                chunk = y_16k[start:end]
                batch_data.append(chunk)

            batch_tensor = torch.from_numpy(np.array(batch_data)).float().to(device)
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            for k, prob in enumerate(probs):
                idx = i + k
                if prob >= args.threshold:
                    start_time = idx * chunk_duration
                    end_time = (idx + 1) * chunk_duration

                    chunk_start = idx * chunk_samples_model
                    chunk_end = chunk_start + chunk_samples_model
                    if not is_silent(y_16k[chunk_start:chunk_end]):
                        segments.append((start_time, end_time))

        if not segments:
            return 0, None

        merged_segments = []
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            if start <= current_end:
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end))

        if is_stereo:
            left_48k = librosa.resample(left_channel, orig_sr=sr, target_sr=target_sr)
            right_48k = librosa.resample(right_channel, orig_sr=sr, target_sr=target_sr)
        else:
            mono_48k = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        left_output = []
        right_output = []
        fade_samples = int(fade_duration * target_sr)

        for start, end in merged_segments:
            if end - start >= min_segment_duration:
                start_sample = int(start * target_sr)
                end_sample = int(end * target_sr)

                if is_stereo:
                    left_segment = left_48k[start_sample:end_sample].copy()
                    right_segment = right_48k[start_sample:end_sample].copy()
                else:
                    mono_segment = mono_48k[start_sample:end_sample].copy()
                    left_segment = mono_segment.copy()
                    right_segment = mono_segment.copy()

                left_segment = apply_fade(left_segment, fade_samples, "in")
                left_segment = apply_fade(left_segment, fade_samples, "out")
                right_segment = apply_fade(right_segment, fade_samples, "in")
                right_segment = apply_fade(right_segment, fade_samples, "out")

                left_output.append(left_segment)
                right_output.append(right_segment)

        if not left_output:
            return 0, None

        final_left = np.concatenate(left_output)
        final_right = np.concatenate(right_output)
        final_audio = np.stack([final_left, final_right], axis=0)

    except Exception as e:
        print(f"Error in process_audio_file: {str(e)}")
        print(f"Audio shape: {y.shape if 'y' in locals() else 'N/A'}")
        print(f"Sample rate: {sr if 'sr' in locals() else 'N/A'}")
        return 0, None

    return final_audio.shape[1], final_audio


def clean_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    cleaned = re.sub(r"[^\w\s-]", "", name)
    return cleaned + ".flac"


def process_directory(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    total_cleaned_duration = 0.0
    summary = []

    audio_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".opus"))
    ]

    for filename in audio_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = clean_filename(filename)
        output_path = os.path.join(output_dir, output_filename)

        original_duration = 0.0
        cleaned_duration = 0.0
        y = None
        sr = None

        try:
            try:
                y, sr = sf.read(input_path)
                if y.ndim == 1:
                    y = y.reshape(1, -1)
                else:
                    y = y.T
                original_duration = y.shape[1] / sr
            except Exception as sf_error:
                print(f"Soundfile failed for {filename}: {str(sf_error)}, trying librosa...")
                y, sr = librosa.load(input_path, sr=None, mono=False)
                if y.ndim == 1:
                    y = y.reshape(1, -1)
                original_duration = y.shape[1] / sr
        except Exception as load_error:
            print(f"Error loading {filename}: {str(load_error)}")
            continue

        if os.path.exists(output_path):
            try:
                info = sf.info(output_path)
                cleaned_duration = info.duration
            except Exception:
                try:
                    size_bytes = os.path.getsize(output_path)
                    cleaned_duration = size_bytes / (target_sr * 2)
                except:
                    cleaned_duration = 0.0
            print(f"Skipping already processed: {filename}")
            summary.append(
                {
                    "original_filename": filename,
                    "original_duration": original_duration,
                    "cleaned_duration": cleaned_duration,
                }
            )
            continue

        try:
            print(f"Processing: {filename}")
            length, processed_audio = process_audio_file(y, sr, model)

            if length > 0:
                sf.write(output_path, processed_audio.T, target_sr, format="FLAC")
                cleaned_duration = length / target_sr
                total_cleaned_duration += cleaned_duration
                print(f"Saved: {output_path} ({cleaned_duration:.2f}s)")
            else:
                print(f"No cleaned segments found in: {filename}")

            summary.append(
                {
                    "original_filename": filename,
                    "original_duration": original_duration,
                    "cleaned_duration": cleaned_duration,
                }
            )

        except Exception as process_error:
            print(f"Error processing {filename}: {str(process_error)}")
            summary.append(
                {
                    "original_filename": filename,
                    "original_duration": original_duration,
                    "cleaned_duration": 0.0,
                }
            )

    csv_path = os.path.join(output_dir, "processed_files.csv")
    markdown_path = os.path.join(output_dir, "processed_files.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile, open(
        markdown_path, "w", encoding="utf-8"
    ) as mdfile:

        writer = csv.writer(csvfile)
        writer.writerow(["original_filename", "original_duration_sec", "cleaned_duration_sec"])

        mdfile.write("| original_filename | original_duration_sec | cleaned_duration_sec |\n")
        mdfile.write("|-------------------|------------------------|-----------------------|\n")

        total_original = 0
        total_cleaned = 0

        for item in summary:
            total_original += item["original_duration"]
            total_cleaned += item["cleaned_duration"]

            writer.writerow(
                [
                    item["original_filename"],
                    f"{item['original_duration']:.2f}",
                    f"{item['cleaned_duration']:.2f}",
                ]
            )

            escaped_filename = item["original_filename"].replace("|", "\\|").replace("\n", " ")
            mdfile.write(
                f"| {escaped_filename} | {item['original_duration']:.2f} | {item['cleaned_duration']:.2f} |\n"
            )

        total_original_hours = total_original / 3600
        total_cleaned_hours = total_cleaned / 3600

        mdfile.write(
            "| **Total (seconds)** | **{:.2f}** | **{:.2f}** |\n".format(
                total_original, total_cleaned
            )
        )
        mdfile.write(
            "| **Total (hours)** | **{:.2f}** | **{:.2f}** |\n".format(
                total_original_hours, total_cleaned_hours
            )
        )

    print(f"\nTotal original audio duration: {total_original_hours:.2f} hours")
    print(f"Total cleaned audio duration: {total_cleaned_hours:.2f} hours")
    print(f"CSV report saved to: {csv_path}")
    print(f"Markdown table saved to: {markdown_path}")

    return csv_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser(description="Instrument processing parameters")

    parser.add_argument("--instr", type=str, default="piano", help="Instrument name")
    parser.add_argument("--threshold", type=float, default=0.995, help="Threshold value")
    parser.add_argument("--model_name", type=str, default="dasheng_base")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    return parser.parse_args()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_sr = 48000
    model_sr = 16000
    chunk_duration = 3.0
    chunk_samples_model = int(chunk_duration * model_sr)
    batch_size = 32
    min_segment_duration = 3.0
    fade_duration = 3

    config_path = "./Instruments_cleaner_config.yaml"
    config = load_config(config_path)

    args = parse_args()

    print(f"Instrument: {args.instr}")
    print(f"Model name: {args.model_name}")
    print(f"Threshold: {args.threshold}")

    os.makedirs(output_path, exist_ok=True)

    models = []
    fine_tune = False
    hidden_dims = config["train"]["hidden_dims"]
    use_dropout = config["train"]["use_dropout"]
    use_batchnorm = config["train"]["use_batchnorm"]

    model = create_model(
        model_name=args.model_name,
        hidden_size=hidden_dims,
        fine_tune=fine_tune,
        use_dropout=use_dropout,
        use_batchnorm=use_batchnorm,
    )

    model_path = f"..model/Instruments_cleaner_dasheng_{args.instr}.pth"
    checkpoint = torch.load(model_path, map_location=device)

    print(f"Using device: {device}")
    model = model.to(device)
    model_state_dict = checkpoint["model_state_dict"]
    binary_classifier_state_dict = checkpoint["model_state_dict"]
    if not fine_tune:
        model.binary_classifier.load_state_dict(binary_classifier_state_dict)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    has_subfolders = any(
        os.path.isdir(os.path.join(input_path, name)) for name in os.listdir(input_path)
    )

    if has_subfolders:
        for folder_name in os.listdir(input_path):
            full_input_folder = os.path.join(input_path, folder_name)
            full_output_folder = os.path.join(output_path, folder_name)

            if os.path.isdir(full_input_folder):
                os.makedirs(full_output_folder, exist_ok=True)
                print(f"Processing folder: {full_input_folder} -> {full_output_folder}")

                process_directory(full_input_folder, full_output_folder, model)
    else:
        print(f"No subfolders found. Processing files directly in: {input_path}")
        print(f"Processing folder: {input_path} -> {output_path}")
        process_directory(input_path, output_path, model)
