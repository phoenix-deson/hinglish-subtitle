import os
import re
import shutil
import subprocess
import tempfile
import time
from difflib import SequenceMatcher
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
st.caption("Video → FFmpeg WAV → multilingual Whisper → hallucination-safe SRT/TXT")


def find_binary(name):
    candidates = [
        shutil.which(name),
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
        f"/opt/conda/bin/{name}",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def find_ffmpeg():
    return find_binary("ffmpeg")


def find_ffprobe():
    return find_binary("ffprobe")


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


def normalize_for_comparison(text):
    text = clean_text(text).lower()
    return re.sub(r"[^\w\u0900-\u097f]+", "", text)


def format_time(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def get_media_duration(media_path):
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None

    command = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        duration = float(result.stdout.strip())
        return duration if duration > 0 else None
    except (TypeError, ValueError):
        return None


def extract_wav(video_path, wav_path):
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
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError(
            "FFmpeg could not extract the audio.\n" +
            (detail[-1500:] if detail else "Unknown FFmpeg error.")
        )

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("The video does not contain a readable audio track.")


def filter_hallucinations(raw_segments, media_duration=None):
    """Remove Whisper failure loops, obvious silence hallucinations and bad timestamps."""
    accepted = []
    recent_texts = []
    repeat_streak = 0
    last_normalized = ""

    for segment in raw_segments:
        text = clean_text(segment.text)
        if not text:
            continue

        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))

        if media_duration is not None:
            if start >= media_duration:
                continue
            end = min(end, media_duration)

        if end <= start:
            continue

        avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))
        compression_ratio = float(getattr(segment, "compression_ratio", 0.0))

        if no_speech_prob >= 0.75 and avg_logprob < -0.8:
            continue

        normalized = normalize_for_comparison(text)
        if normalized:
            similarity = SequenceMatcher(None, normalized, last_normalized).ratio()
            if normalized == last_normalized or similarity >= 0.92:
                repeat_streak += 1
            else:
                repeat_streak = 0
            last_normalized = normalized

            if repeat_streak >= 2:
                break

        if compression_ratio >= 3.0 and len(normalized) > 20:
            continue

        accepted.append((start, end, text))
        recent_texts.append(normalized)
        if len(recent_texts) > 5:
            recent_texts.pop(0)

    if len(accepted) >= 3:
        cleaned = []
        for item in accepted:
            if cleaned:
                previous = normalize_for_comparison(cleaned[-1][2])
                current = normalize_for_comparison(item[2])
                if current and previous and SequenceMatcher(None, previous, current).ratio() >= 0.92:
                    continue
            cleaned.append(item)
        accepted = cleaned

    return accepted


def make_srt(segments):
    lines = []
    for index, (start, end, text) in enumerate(segments, start=1):
        lines.append(
            f"{index}\n"
            f"{format_time(start)} --> {format_time(end)}\n"
            f"{text}\n"
        )
    return "\n".join(lines)


def make_txt(segments):
    return "\n".join(
        f"[{format_time(start)} --> {format_time(end)}] {text}"
        for start, end, text in segments
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
        wav_path = os.path.join(work_dir, "audio.wav")

        try:
            with st.status("Processing video…", expanded=True) as status:
                with open(video_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                status.write("📥 Video saved.")

                media_duration = get_media_duration(video_path)
                if media_duration is not None:
                    status.write(f"⏱️ Video duration: {format_time(media_duration)}")
                else:
                    status.write("⏱️ Video duration could not be read; final timestamps will still be protected when possible.")

                status.write("🎵 Extracting 16 kHz mono PCM WAV with FFmpeg…")
                extract_wav(video_path, wav_path)
                status.write("✅ WAV extraction completed.")

                try:
                    os.remove(video_path)
                except OSError:
                    pass

                status.write(f"🧠 Loading multilingual Whisper {MODEL_SIZE} model…")
                model_load_start = time.time()
                model = load_model()
                model_load_seconds = time.time() - model_load_start
                status.write(f"✅ Whisper {MODEL_SIZE} model loaded in {model_load_seconds:.1f}s.")

                status.write("🔎 Recognition started. Hinglish-preserving transcription is enabled.")
                progress = st.progress(0, text="Preparing recognition…")
                progress_detail = st.empty()
                preview_box = st.empty()

                recognition_start = time.time()
                segments, info = model.transcribe(
                    wav_path,
                    language=None,
                    task="transcribe",
                    beam_size=5,
                    best_of=5,
                    patience=1,
                    temperature=(0.0, 0.2, 0.4),
                    initial_prompt=(
                        "This is natural Hinglish speech: Hindi and English are mixed in the same sentence. "
                        "Keep Hindi words in Devanagari and keep English words, names, technical terms, "
                        "and common English expressions in Latin/English script. Do not transliterate English "
                        "words into Devanagari when the speaker is speaking English. Preserve natural code-switching."
                    ),
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 1000,
                        "speech_pad_ms": 400,
                    },
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                    word_timestamps=False,
                    multilingual=False,
                )

                raw_segments = []
                last_ui_update = 0.0
                last_preview_update = 0.0

                for segment in segments:
                    raw_segments.append(segment)

                    now = time.time()
                    current_end = max(0.0, float(segment.end))
                    elapsed = now - recognition_start

                    if media_duration:
                        ratio = min(1.0, current_end / media_duration)
                        progress.progress(
                            ratio,
                            text=f"Recognizing… {format_time(current_end)} / {format_time(media_duration)} ({ratio * 100:.1f}%)",
                        )
                    else:
                        progress.progress(
                            min(0.99, len(raw_segments) / max(1, len(raw_segments) + 20)),
                            text=f"Recognizing… {format_time(current_end)} processed",
                        )

                    if now - last_ui_update >= 2.0:
                        speed = current_end / elapsed if elapsed > 0 else 0
                        eta = ((media_duration - current_end) / speed) if media_duration and speed > 0 else None
                        eta_text = f" · ETA ~{int(eta // 60)}m {int(eta % 60)}s" if eta is not None else ""
                        progress_detail.info(
                            f"🎙️ Processed **{format_time(current_end)}** of **{format_time(media_duration) if media_duration else 'unknown'}** · "
                            f"{len(raw_segments)} segments · {speed:.2f}× real-time{eta_text}"
                        )
                        last_ui_update = now

                    if now - last_preview_update >= 4.0:
                        preview_text = clean_text(segment.text)
                        if preview_text:
                            preview_box.caption(f"Latest recognition: {preview_text}")
                        last_preview_update = now

                progress.progress(1.0, text="Recognition finished. Validating transcript…")
                progress_detail.info(f"✅ Whisper returned {len(raw_segments)} raw segments. Running hallucination and timestamp checks…")

                segment_list = filter_hallucinations(raw_segments, media_duration)

                if not segment_list:
                    raise RuntimeError("No reliable speech was detected in the uploaded video.")

                if media_duration is not None:
                    segment_list = [
                        (start, min(end, media_duration), text)
                        for start, end, text in segment_list
                        if start < media_duration
                    ]

                if not segment_list:
                    raise RuntimeError("No reliable subtitle segments remain after validation.")

                srt_content = make_srt(segment_list)
                txt_content = make_txt(segment_list)

                status.write(f"🛡️ Removed/blocked {max(0, len(raw_segments) - len(segment_list))} unreliable or repeated segments.")
                status.update(label="✅ High-accuracy subtitle generation completed", state="complete")

            st.success("🎉 Done!")
            detected_language = getattr(info, "language", "unknown")
            language_probability = getattr(info, "language_probability", None)
            st.write(f"Detected language: **{detected_language}**")
            if language_probability is not None:
                st.write(f"Initial language confidence: **{language_probability * 100:.1f}%**")
            st.caption("Hinglish mode keeps Hindi in Devanagari while encouraging English words and expressions to remain in Latin script.")
            st.write(f"Subtitle segments: **{len(segment_list)}**")
            if media_duration is not None:
                st.write(f"Final subtitle time: **{format_time(segment_list[-1][1])} / {format_time(media_duration)}**")

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
