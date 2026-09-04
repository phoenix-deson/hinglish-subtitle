import os
import re
import shutil
import subprocess
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Hinglish Subtitle Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_SIZE = "medium"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
CPU_THREADS = max(1, min(2, os.cpu_count() or 2))
TRANSLATE_ENDPOINTS = [
    "https://translate.argosopentech.com",
    "https://translate.terraprint.co",
    "https://lt.vern.cc",
]
TARGET_LANGUAGES = {
    "Simplified Chinese": "zh",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Indonesian": "id",
    "Hindi": "hi",
}

st.markdown(
    """
    <style>
    .hero { padding: 1.2rem 1.4rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: .35rem 0 0; opacity: .75; }
    .feature { padding: .9rem 1rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.2); min-height: 90px; }
    .feature b { font-size: 1.05rem; }
    </style>
    <div class="hero">
      <h1>🎬 Hinglish Subtitle Studio</h1>
      <p>Accurate Hinglish transcription + free multilingual subtitle translation</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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


def parse_srt_time(value):
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt(content):
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    segments = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None or timing_index + 1 >= len(lines):
            continue
        start_text, end_text = [x.strip() for x in lines[timing_index].split("-->", 1)]
        try:
            start = parse_srt_time(start_text)
            end = parse_srt_time(end_text)
        except ValueError:
            continue
        text = clean_text(" ".join(lines[timing_index + 1:]))
        if text and end > start:
            segments.append((start, end, text))
    return segments


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


def get_media_duration(media_path):
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    command = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        raise RuntimeError("FFmpeg was not found on the server. Make sure packages.txt contains ffmpeg and redeploy.")
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-map", "0:a:0?", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError("FFmpeg could not extract the audio.\n" + (detail[-1500:] if detail else "Unknown FFmpeg error."))
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise RuntimeError("The video does not contain a readable audio track.")


def filter_hallucinations(raw_segments, media_duration=None):
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


def detect_script_language(text):
    """Lightweight fallback when the online detector is unavailable."""
    devanagari = len(re.findall(r"[\u0900-\u097f]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if devanagari > latin * 1.2:
        return "hi"
    if latin > 0:
        return "en"
    return "unknown"


def detect_language(text):
    sample = clean_text(text)[:1800]
    if not sample:
        return "unknown", 0.0
    for endpoint in TRANSLATE_ENDPOINTS:
        try:
            response = requests.post(
                endpoint + "/detect",
                data={"q": sample},
                timeout=12,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data:
                best = max(data, key=lambda item: float(item.get("confidence", 0)))
                return best.get("language", "unknown"), float(best.get("confidence", 0))
        except Exception:
            continue
    return detect_script_language(sample), 0.0


def translate_text(text, target_language):
    text = clean_text(text)
    if not text:
        return ""
    last_error = None
    for endpoint in TRANSLATE_ENDPOINTS:
        try:
            response = requests.post(
                endpoint + "/translate",
                data={"q": text[:5000], "source": "auto", "target": target_language, "format": "text"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            translated = clean_text(data.get("translatedText", ""))
            if translated:
                return translated
            last_error = "Empty translation returned"
        except Exception as error:
            last_error = error
    raise RuntimeError(str(last_error or "All free translation servers failed."))


def translate_segments(segments, target_language, progress_callback=None):
    translated = []
    total = len(segments)
    for index, (start, end, text) in enumerate(segments, start=1):
        translated_text = translate_text(text, target_language)
        translated.append((start, end, translated_text))
        if progress_callback:
            progress_callback(index, total, translated_text)
        time.sleep(0.15)
    return translated


def recognition_pipeline(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    work_dir = tempfile.mkdtemp(prefix="hinglish_")
    video_path = os.path.join(work_dir, "input" + suffix)
    wav_path = os.path.join(work_dir, "audio.wav")
    try:
        with open(video_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        media_duration = get_media_duration(video_path)
        extract_wav(video_path, wav_path)
        model = load_model()
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
            vad_parameters={"min_silence_duration_ms": 1000, "speech_pad_ms": 400},
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
        for segment in segments:
            raw_segments.append(segment)
            yield "segment", segment, media_duration, info, len(raw_segments)
        segment_list = filter_hallucinations(raw_segments, media_duration)
        if media_duration is not None:
            segment_list = [
                (start, min(end, media_duration), text)
                for start, end, text in segment_list
                if start < media_duration
            ]
        if not segment_list:
            raise RuntimeError("No reliable speech was detected in the uploaded video.")
        yield "done", segment_list, media_duration, info, len(raw_segments)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def render_translation_ui(source_segments=None, source_srt=None):
    st.subheader("🌐 Subtitle Translation")
    st.caption("Automatically detects the subtitle language, then translates the timed subtitles without changing their timestamps.")

    uploaded_srt = st.file_uploader(
        "Or upload an existing SRT subtitle file",
        type=["srt"],
        key="translation_srt_upload",
    )

    if uploaded_srt is not None:
        try:
            active_segments = parse_srt(uploaded_srt.getvalue().decode("utf-8-sig", errors="replace"))
            source_name = uploaded_srt.name
        except Exception as error:
            st.error(f"Could not read SRT: {error}")
            return
    elif source_segments:
        active_segments = source_segments
        source_name = "Generated transcript"
    else:
        st.info("Generate subtitles first, or upload an SRT file above.")
        return

    if not active_segments:
        st.warning("No subtitle entries were found.")
        return

    preview = " ".join(text for _, _, text in active_segments[:12])
    detected, confidence = detect_language(preview)
    language_names = {"en": "English", "hi": "Hindi", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "id": "Indonesian"}
    detected_label = language_names.get(detected, detected.upper() if detected != "unknown" else "Unknown")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Subtitle entries", len(active_segments))
    with col2:
        st.metric("Detected language", detected_label)
    with col3:
        st.metric("Detection confidence", f"{confidence * 100:.1f}%" if confidence else "Fallback")

    target_label = st.selectbox("Translate subtitles into", list(TARGET_LANGUAGES.keys()), index=0, key="target_language")
    target_code = TARGET_LANGUAGES[target_label]

    if detected == target_code:
        st.warning("The detected source language is the same as the selected target language. Translation may make little or no change.")

    st.caption("Engine: LibreTranslate / Argos Translate · open-source neural machine translation · free public endpoints with automatic fallback")

    if st.button("🌐 Translate Subtitles", type="primary", use_container_width=True, key="translate_button"):
        progress = st.progress(0, text="Starting translation…")
        detail = st.empty()
        preview_box = st.empty()
        started = time.time()

        def on_progress(index, total, latest):
            ratio = index / total if total else 1
            progress.progress(ratio, text=f"Translating… {index}/{total} ({ratio * 100:.1f}%)")
            elapsed = max(0.1, time.time() - started)
            speed = index / elapsed
            eta = (total - index) / speed if speed > 0 else 0
            detail.info(f"⏱️ {speed:.2f} subtitles/s · ETA ~{int(eta // 60)}m {int(eta % 60)}s")
            preview_box.caption(f"Latest translation: {latest}")

        try:
            translated_segments = translate_segments(active_segments, target_code, on_progress)
            translated_srt = make_srt(translated_segments)
            progress.progress(1.0, text="Translation completed")
            detail.success(f"Translated {len(translated_segments)} subtitle entries from {source_name}.")

            st.subheader("Translated subtitles")
            st.text_area("Translation preview", "\n".join(text for _, _, text in translated_segments), height=420)
            st.download_button(
                "⬇️ Download translated SRT",
                translated_srt,
                f"translated_{Path(source_name).stem}.srt",
                "application/x-subrip",
                use_container_width=True,
            )
        except Exception as error:
            st.error(f"Translation failed: {error}")
            st.caption("The free public translation endpoints may be temporarily busy. You can retry without changing the recognition result.")


if "segments" not in st.session_state:
    st.session_state.segments = None
if "srt_content" not in st.session_state:
    st.session_state.srt_content = None
if "txt_content" not in st.session_state:
    st.session_state.txt_content = None

with st.sidebar:
    st.markdown("### ✨ Studio")
    st.markdown("**1. Speech Recognition**\n\nMedium Whisper · Hinglish optimized")
    st.markdown("**2. Subtitle Translation**\n\nFree Argos/LibreTranslate engine")
    st.markdown("**3. Export**\n\nSRT · TXT")
    st.divider()
    st.caption("Recognition model is fixed for this version. Translation is a separate step, so it never changes your transcription result.")

recognition_tab, translation_tab = st.tabs(["🎙️ Speech Recognition", "🌐 Subtitle Translation"])

with recognition_tab:
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown('<div class="feature">🎙️ <b>Hinglish-aware</b><br><small>Hindi + English code-switching</small></div>', unsafe_allow_html=True)
    with feature_cols[1]:
        st.markdown('<div class="feature">🛡️ <b>Hallucination protection</b><br><small>Repeated-output filtering</small></div>', unsafe_allow_html=True)
    with feature_cols[2]:
        st.markdown('<div class="feature">⚡ <b>Resource controlled</b><br><small>CPU int8 · 2 threads</small></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a video",
        type=SUPPORTED_TYPES,
        help="MP4, MOV, MKV, WebM, AVI and M4V are supported.",
        key="video_upload",
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        file_size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024
        st.info(f"📁 {uploaded_file.name} · {file_size_mb:.1f} MB · {suffix[1:].upper()}")
        try:
            st.video(uploaded_file)
        except Exception:
            st.caption("Video preview is unavailable, but the file can still be processed.")

        if st.button("🚀 Generate High-Accuracy Subtitles", type="primary", use_container_width=True, key="recognize_button"):
            try:
                with st.status("Processing video…", expanded=True) as status:
                    status.write("📥 Video saved.")
                    status.write("🎵 Extracting 16 kHz mono PCM WAV with FFmpeg…")

                    progress = st.progress(0, text="Preparing recognition…")
                    progress_detail = st.empty()
                    preview_box = st.empty()
                    recognition_start = time.time()
                    media_duration = None
                    raw_count = 0
                    info = None

                    for event, payload, duration, model_info, count in recognition_pipeline(uploaded_file):
                        if event == "segment":
                            segment = payload
                            media_duration = duration
                            info = model_info
                            raw_count = count
                            current_end = max(0.0, float(segment.end))
                            elapsed = max(0.1, time.time() - recognition_start)
                            if media_duration:
                                ratio = min(1.0, current_end / media_duration)
                                progress.progress(ratio, text=f"Recognizing… {format_time(current_end)} / {format_time(media_duration)} ({ratio * 100:.1f}%)")
                                speed = current_end / elapsed
                                eta = (media_duration - current_end) / speed if speed > 0 else 0
                                progress_detail.info(f"🎙️ {raw_count} segments · {speed:.2f}× real-time · ETA ~{int(eta // 60)}m {int(eta % 60)}s")
                            else:
                                progress.progress(min(0.99, raw_count / max(1, raw_count + 20)), text=f"Recognizing… {format_time(current_end)} processed")
                            preview_box.caption(f"Latest recognition: {clean_text(segment.text)}")
                        else:
                            segment_list = payload
                            media_duration = duration
                            info = model_info
                            raw_count = count

                    if media_duration is not None:
                        status.write(f"⏱️ Video duration: {format_time(media_duration)}")
                    status.write("✅ WAV extraction completed.")
                    status.write(f"🧠 Whisper {MODEL_SIZE} model loaded and recognition completed.")
                    progress.progress(1.0, text="Recognition finished. Validating transcript…")
                    progress_detail.success(f"Whisper returned {raw_count} raw segments. Hallucination and timestamp checks completed.")

                    srt_content = make_srt(segment_list)
                    txt_content = make_txt(segment_list)
                    removed = max(0, raw_count - len(segment_list))
                    status.write(f"🛡️ Removed/blocked {removed} unreliable or repeated segments.")
                    status.update(label="✅ High-accuracy subtitle generation completed", state="complete")

                st.session_state.segments = segment_list
                st.session_state.srt_content = srt_content
                st.session_state.txt_content = txt_content
                st.success("🎉 Done! Your recognition result is ready for translation.")

            except Exception as error:
                st.error(f"❌ Processing failed: {error}")

    if st.session_state.srt_content:
        st.subheader("📝 Recognition Result")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.metric("Subtitle segments", len(st.session_state.segments))
        with info_cols[1]:
            st.metric("Model", "Whisper medium")
        with info_cols[2]:
            st.metric("Output", "SRT + TXT")
        st.text_area("Transcript", st.session_state.txt_content, height=420, key="recognition_preview")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download SRT", st.session_state.srt_content, "hinglish_subtitle.srt", "application/x-subrip", use_container_width=True)
        with col2:
            st.download_button("⬇️ Download TXT", st.session_state.txt_content, "hinglish_subtitle.txt", "text/plain", use_container_width=True)
        st.info("💡 Want another language? Open the **Subtitle Translation** tab. Translation is completely separate from recognition.")

with translation_tab:
    render_translation_ui(st.session_state.segments, st.session_state.srt_content)

st.divider()
st.caption("Hinglish Subtitle Studio · Whisper medium + FFmpeg + faster-whisper · Translation: LibreTranslate / Argos Translate")
