import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Hinglish Subtitle Studio",
    page_icon="🎬",
    layout="centered",
)

MODEL_SIZE = "medium"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
CPU_THREADS = max(1, min(2, os.cpu_count() or 2))

st.title("🎬 Hinglish Subtitle Studio")
st.write("High-accuracy English + Hindi mixed speech subtitle generator")
st.caption("Video → FFmpeg MP3 → multilingual Whisper → SRT/TXT")


def find_ffmpeg():
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/opt/conda/bin/ffmpeg",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=CPU_THREADS,
        num_workers=1,
    )


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def format_time(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def extract_mp3(video_path, mp3_path):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg was not found on the server. Make sure packages.txt contains ffmpeg and redeploy."
        )

    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-map", "0:a:0?",
        "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "64k",
        str(mp3_path),
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError(
            "FFmpeg could not extract the audio.\n" +
            (detail[-1500:] if detail else "Unknown FFmpeg error.")
        )

    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
        raise RuntimeError("The video does not contain a readable audio track.")


def make_srt(segments):
    lines = []
    index = 1
    for segment in segments:
        text = clean_text(segment.text)
        if not text:
            continue
        lines.append(
            f"{index}\n"
            f"{format_time(segment.start)} --> {format_time(segment.end)}\n"
            f"{text}\n"
        )
        index += 1
    return "\n".join(lines)


def make_txt(segments):
    return "\n".join(
        f"[{format_time(segment.start)} --> {format_time(segment.end)}] {clean_text(segment.text)}"
        for segment in segments
        if clean_text(segment.text)
    )


uploaded_file = st.file_uploader(
    "Upload a video",
    type=SUPPORTED_TYPES,
    help="MP4, MOV, MKV, WebM, AVI and M4V are supported.",
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix.lower()
    file_size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024

    st.info(f"File: {uploaded_file.name} · {file_size_mb:.1f} MB · {suffix[1:].upper()}")

    try:
        st.video(uploaded_file)
    except Exception:
        st.caption("Video preview is unavailable, but the file can still be processed.")

    if st.button("🚀 Generate High-Accuracy Subtitles", type="primary", use_container_width=True):
        work_dir = tempfile.mkdtemp(prefix="hinglish_")
        video_path = os.path.join(work_dir, "input" + suffix)
        mp3_path = os.path.join(work_dir, "audio.mp3")

        try:
            with st.status("Processing video…", expanded=True) as status:
                with open(video_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                status.write("📥 Video saved.")
                status.write("🎵 Extracting 16 kHz mono MP3 with FFmpeg…")
                extract_mp3(video_path, mp3_path)
                status.write("✅ MP3 extraction completed.")

                try:
                    os.remove(video_path)
                except OSError:
                    pass

                status.write(f"🧠 Loading multilingual Whisper {MODEL_SIZE} model…")
                model = load_model()

                status.write("🔎 Recognizing Hindi + English mixed speech…")
                segments, info = model.transcribe(
                    mp3_path,
                    language=None,
                    task="transcribe",
                    beam_size=5,
                    best_of=5,
                    patience=1,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200,
                    },
                    condition_on_previous_text=True,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    initial_prompt=(
                        "This is Hinglish speech, a natural mixture of Hindi and English. "
                        "Keep English words in English and Hindi speech in Devanagari when recognized."
                    ),
                )

                segment_list = list(segments)

                if not segment_list:
                    raise RuntimeError("No speech was detected in the uploaded video.")

                srt_content = make_srt(segment_list)
                txt_content = make_txt(segment_list)

                status.update(label="✅ High-accuracy subtitle generation completed", state="complete")

            st.success("🎉 Done!")
            detected_language = getattr(info, "language", "unknown")
            language_probability = getattr(info, "language_probability", None)
            st.write(f"Detected language: **{detected_language}**")
            if language_probability is not None:
                st.write(f"Language confidence: **{language_probability * 100:.1f}%**")
            st.write(f"Subtitle segments: **{len(segment_list)}**")

            st.subheader("📝 Transcript")
            st.text_area("Recognition result", txt_content, height=450)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download SRT",
                    srt_content,
                    "hinglish_subtitle.srt",
                    "application/x-subrip",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "⬇️ Download TXT",
                    txt_content,
                    "hinglish_subtitle.txt",
                    "text/plain",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(f"❌ Processing failed: {error}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

st.divider()
st.caption("Hinglish Subtitle Studio · Multilingual Whisper medium · FFmpeg + faster-whisper")
