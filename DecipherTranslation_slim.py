import os
import argparse
import subprocess as sp
import openai
import multiprocessing
from multiprocessing import Semaphore, Process
import subprocess
from tqdm import tqdm
from decipher import action
from decipher.utils import fix_srt_errors, srt_to_txt
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import urllib.request
import tarfile
import shutil

openai.api_key = os.getenv('OPENAI_API_KEY')

def translate_text(course, text, target_language="en"):
    """Function to translate text using OpenAI's GPT API."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a professional translator for {course} lectures. You will get transcription texts where there might be some mistakes in the transcription, for example Or gates are turned into light gates or half-adder written as L-Fedder etc. Translate the following Hebrew text to English without adding any additional notes or explanations while fixing those little issues. Avoid redundant repetitions. Maintain the original formatting."},
            {"role": "user", "content": f"Translate the following Hebrew text to English\n{text}"},
        ]
    )
    return response['choices'][0]['message']['content']

def batch_translate_srt(course, file_path, output_path, batch_size=10):
    """Translate an SRT file in batches and save the result to a new file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    translated_lines = []
    batch = []
    count = 0
    
    for line in tqdm(lines):
        batch.append(line)
        if line.strip().isdigit():  
            count += 1

        if count == batch_size:
            batch_text = "".join(batch)
            translated_batch = translate_text(course,batch_text)
            translated_lines.append(translated_batch)
            batch = []
            count = 0

    if batch:
        batch_text = "".join(batch)
        translated_batch = translate_text(course,batch_text)
        translated_lines.append(translated_batch)
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(translated_lines))

def translate(course):
    path = f'./results/{course}/subtitles'
    subs = os.listdir(path)
   
    if not os.path.exists(f"{path}/translated"):
        os.makedirs(f"{path}/translated", exist_ok=True)

    def process_file(sub):
        if sub.endswith('.srt') and not os.path.exists(f"{path}/translated/{sub}"):
            print(f"Processing: {sub}")
            file_path, output_path = f"{path}/{sub}", f"{path}/translated/{sub}"
            batch_translate_srt(course, file_path, output_path)

    #max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    max_workers = 1
    print(max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, subs)

def process_video(vid, path, output_path, language, taskid, num_gpus):
    gpu_id = taskid % num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Task {taskid} assigned to GPU {gpu_id}")

    video_path = f'{path}/{vid}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print(f"Processing: {vid}")
    action.transcribe(video_path, output_path, language, "transcribe", 24)
    print(f"Completed {vid}")

def decipher_vids(course, language="he"):
    multiprocessing.set_start_method("spawn")
    path = f'data/{course}'
    vids = os.listdir(path)
    vids = [vid for vid in vids if Path(vid).suffix.lower() == ".mp3"]

    output_path = f'./results/{course}/subtitles'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir_vids = os.listdir(output_path)
    dir_vids = [vid.split(".")[0] for vid in dir_vids if vid.endswith(".srt")]
    vids_to_process = [vid for vid in vids if vid.split(".")[0] not in dir_vids]

    #num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))  
    num_gpus = 1
    print(num_gpus)
    with multiprocessing.Pool(processes=num_gpus) as pool:
        pool.starmap(process_video, [(vid, path, output_path, language, i, num_gpus) for  i ,vid in enumerate(vids_to_process)])

def download_ffmpeg(ffmpeg_dir):
    """Downloads and extracts FFmpeg if not already present."""
    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg")
    if os.path.exists(ffmpeg_path):
        print("FFmpeg already installed.")
        return ffmpeg_path

    print("Downloading FFmpeg...")
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    tar_file = os.path.join(ffmpeg_dir, "ffmpeg.tar.xz")

    # Create ffmpeg directory if it doesn't exist
    os.makedirs(ffmpeg_dir, exist_ok=True)

    # Download the tar file
    urllib.request.urlretrieve(url, tar_file)
    print("Download complete. Extracting...")

    # Extract tar file
    with tarfile.open(tar_file, "r:xz") as tar:
        tar.extractall(path=ffmpeg_dir, members=[
            m for m in tar.getmembers() if m.name.endswith("ffmpeg")
        ])

    # Cleanup
    os.remove(tar_file)
    print(f"FFmpeg installed at {ffmpeg_path}")
    return ffmpeg_path

def extract_mp3_from_mp4(folder):
    download_ffmpeg(os.path.expanduser("~/ffmpeg"))
    for file in os.listdir(folder):
        if file.lower().endswith(".mp4"):
            mp4_file = os.path.join(folder, file)
            mp3_file = os.path.join(folder, os.path.splitext(file)[0] + ".mp3")

            cmd = f"ffmpeg -i '{mp4_file}' -q:a 0 -map a '{mp3_file}' -y"
            subprocess.run(cmd, shell=True, check=True)

            print(f"Extracted MP3: {mp3_file}")

    for file in os.listdir(folder):
        if file.lower().endswith(".mp4"):
            mp4_file = os.path.join(folder, file)
            os.remove(mp4_file)
            print(f"Deleted MP3: {mp3_file}")

def decipher_translation_pipeline(course, language):    
    
    extract_mp3_from_mp4(f'data/{course}')
    decipher_vids(course, language)

    if language == "he":
        translate(course)

    courses = os.listdir('results')
    for course in courses:
        subtitles_out_path = f'final_results/{course}/subtitles'
        texts_out_path = f'final_results/{course}/texts'
        os.makedirs(f'final_results/{course}', exist_ok=True)
        os.makedirs(subtitles_out_path, exist_ok=True)
        os.makedirs(texts_out_path, exist_ok=True)
        files_path = f'results/{course}/subtitles/translated/'

        if not os.path.exists(files_path):
            files_path =  f'results/{course}/subtitles'

        files = os.listdir(files_path)
        for fname in files:
            if fname.endswith('.srt'):
                fix_srt_errors(f'{files_path}/{fname}', f'{subtitles_out_path}/{fname}')
                srt_to_txt(f'{subtitles_out_path}/{fname}', f"{texts_out_path}/{fname.replace('.srt', '.txt')}")
    
    sp.run(["zip", "-r", "new_data.zip", "final_results"])

    shutil.rmtree(r'./results')

if __name__ == "__main__":
    decipher_translation_pipeline("demo", "he")