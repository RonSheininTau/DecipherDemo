import os
import subprocess
import re


def convert_videos_to_audios(input_dir, output_dir):
    # Create the destination directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.m4a'):
            base_name = os.path.splitext(file_name)[0]
            source_file = os.path.join(input_dir, file_name)
            dest_file = os.path.join(output_dir, base_name + '.mov')

            # Run the ffmpeg command
            ffmpeg_command = f"ffmpeg -i {source_file} -c copy -movflags +faststart {dest_file}"
            subprocess.run(ffmpeg_command, shell=True, check=True)

    print("Conversion completed.")


def srt_to_txt(input_srt, output_txt):
    with open(input_srt, 'r', encoding='utf-8') as srt_file:
        lines = srt_file.readlines()

    # Extract only the subtitle text
    subtitles = [line.strip() for line in lines if any(char.isalpha() for char in line)]

    # texts = " ".join(subtitles).split('. ')
    try:
        with open(output_txt, 'w') as txt_file:
            for i, text in enumerate(subtitles):
                print(i)
                txt_file.write(text + '.\n')
    except Exception as e:
        print(f"Error:", e)


def get_lecture_num(title: str):
    return re.search(r'^(\d+)', title).group()


def fix_srt_errors(input_path, output_path):
    # Read the input SRT file
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    blocks = []
    current_block = {}
    last_id = 0

    timestamp_pattern = re.compile(r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '':
            # Blank line indicates end of a block
            if 'id' in current_block and 'timestamp' in current_block and 'text' in current_block:
                blocks.append(current_block)
                current_block = {}
            i += 1
            continue

        if 'id' not in current_block:
            # Try to parse the line as an ID
            try:
                current_block['id'] = int(line)
                last_id = current_block['id']
                i += 1
            except ValueError:
                # Missing ID
                if timestamp_pattern.match(line):
                    # Assign next ID since ID is missing
                    last_id += 1
                    current_block['id'] = last_id
                    # Do not increment i to process the line again as timestamp
                else:
                    # Line is text without ID and timestamp
                    last_id += 1
                    current_block['id'] = last_id
                    current_block['text'] = line
                    i += 1
        elif 'timestamp' not in current_block:
            if timestamp_pattern.match(line):
                current_block['timestamp'] = line
                current_block['text'] = ''
                i += 1
            else:
                # Missing timestamp, remove last ID and merge text with previous block
                if blocks:
                    blocks[-1]['text'] += line
                current_block = {}
                i += 1
        elif 'text' in current_block:
            if timestamp_pattern.match(line):
                # Timestamp detected without preceding ID, start a new block
                blocks.append(current_block)
                current_block = {}
                last_id += 1
                current_block['id'] = last_id
                current_block['timestamp'] = line
                current_block['text'] = ''
                i += 1
            else:
                # Append line to current block's text
                current_block['text'] += line
                i += 1
        else:
            # Start collecting text
            current_block['text'] = line
            i += 1

    # Add the last block if it's complete
    if 'id' in current_block and 'timestamp' in current_block and 'text' in current_block:
        blocks.append(current_block)

    # Reconstruct the SRT content
    fixed_srt_lines = []
    for block in blocks:
        fixed_srt_lines.append(str(block['id']))
        fixed_srt_lines.append(block['timestamp'])
        fixed_srt_lines.append(block['text'])
        fixed_srt_lines.append('')  # Blank line between blocks
        # Write the fixed SRT lines to the output file
    with open(output_path, 'w', encoding='utf-8') as file:
        for line in fixed_srt_lines:
            file.write(line + '\n')
