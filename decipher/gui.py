import os

import gradio as gr
from decipher import action
from decipher import utils

from tempfile import mktemp, gettempdir


def __transcribe(video_in, audio_in, language, task, batch_size):
    result = action.transcribe(
        video_in,
        audio_in,
        gettempdir(),
        language if language else None,
        task.lower(),
        batch_size,
    )
    with open(result.subtitle_file, "r", encoding='utf-8') as f:
        subtitles = f.read()

    utils.srt_to_txt(result.subtitle_file, result.subtitle_file.replace("subtitles", "texts").replace('.srt', '.txt'))


    return subtitles


MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                ti_video = gr.Video(label="Video", sources=["upload"])
                ti_audio = gr.Audio(
                    label="Audio", sources=["upload"], type="filepath")
                ti_language = gr.Textbox(
                    label="Language", placeholder="English",
                    info="Language spoken in the audio leave empty for detection"
                )
                ti_task = gr.Radio(
                    choices=["Transcribe", "Translate"], value="Transcribe", label="Task",
                    info="Whether to perform X->X speech recognition or X->English translation"
                )
                ti_batch_size = gr.Slider(
                    0, 24, value=24, step=1, label="Batch Size",
                    info="Number of parallel batches reduce if you face out of memory errors"
                )
            with gr.Column():
                to_subtitles = gr.Textbox(label="Subtitles", lines=25, show_copy_button=True, autoscroll=False)
        transcribe_btn = gr.Button("Transcribe")
        transcribe_btn.click(fn=__transcribe,
                             inputs=[ti_video, ti_audio, ti_language, ti_task, ti_batch_size],
                             outputs=[to_subtitles])

    return demo
