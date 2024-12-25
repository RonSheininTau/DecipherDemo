import argparse
import os
import shutil
from pathlib import Path
from tempfile import mktemp

import torch
from transformers import pipeline
from dataclasses import dataclass

from ffutils import ffprog

root = Path(__file__).parent


def seconds_to_srt_time_format(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"


def audio_to_srt(audio_file, temp_srt, task="transcribe", language=None, batch_size=24):
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"{device.upper()} is being used for this transcription, this process may take a while.")

    
    model_name = 'ivrit-ai/whisper-large-v2-tuned' if language is not None and language == 'he' else "openai/whisper-large-v3"

    pipe = pipeline(
        "automatic-speech-recognition",
        # model=f"openai/whisper-large-v3",
        model=model_name,
        torch_dtype=dtype,
        device=device,
        model_kwargs={"attn_implementation": "sdpa", "cache_dir": "/home/milo/users/TauDigital/TDT/cache"}
    )

    if device == "mps":
        torch.mps.empty_cache()

    outputs = pipe(
        audio_file,
        chunk_length_s=30,
        batch_size=batch_size,
        generate_kwargs={"task": task, "language": language},
        return_timestamps=True,
    )

    with open(temp_srt, "w", encoding="utf-8") as f:
        for index, chunk in enumerate(outputs['chunks']):
            if chunk['timestamp'][0] is None:
                break
            start_time = seconds_to_srt_time_format(chunk['timestamp'][0])
            end_time = start_time if chunk['timestamp'][1] is None else seconds_to_srt_time_format(
                chunk['timestamp'][1])
            f.write(f"{index + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{chunk['text'].strip()}\n\n")


@dataclass
class ResultFiles:
    output_dir: str
    subtitle_file: str


def transcribe(video_in, output_dir=None, language=None, task="transcribe",
               batch_size=24) -> ResultFiles:
    if output_dir:
        output_dir = Path(output_dir).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(os.getcwd())

    audio_file = mktemp(suffix=".aac", dir=output_dir)
    input_file = Path(video_in).absolute()
    assert input_file.exists(), f"File {input_file} does not exist"

    ffprog(
        ["ffmpeg", "-y", "-i", str(input_file), "-vn", "-c:a", "aac", audio_file],
        desc=f"Extracting audio from video",
    )

    temp_srt = mktemp(suffix=".srt", dir=output_dir)
    audio_to_srt(audio_file, temp_srt, task, language, batch_size)
    os.remove(audio_file)
    srt_filename =  f"{output_dir}/{input_file.stem}.srt"
    shutil.move(temp_srt, srt_filename)

    assert os.path.exists(srt_filename), f"SRT file not generated?"

    return ResultFiles(
        str(output_dir),
        str(srt_filename),
    )


def cli():
    t = argparse.ArgumentParser()

    t.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="input video file path e.g. video.mp4",
    )
    t.add_argument(
        "-o", "--output_dir", type=str, default=None, help="output directory path"
    )
    t.add_argument(
        "--language", type=str, default='en', help="language spoken in the audio"
    )
    t.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    t.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=24,
        help="Number of parallel batches reduce if you face out of memory errors",
    )

    return t.parse_args()

if __name__ == '__main__':
    args = cli()
    output = transcribe(
        args.input,
        args.output_dir,
        args.language,
        args.task,
        args.batch_size,
    )
