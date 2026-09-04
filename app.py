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

# Recognition is intentionally frozen in this version.
MODEL_SIZE = "medium"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
CPU_THREADS = max(1, min(2, os.cpu_count() or 2))

# Translation is independent from recognition. Free public services are tried in order.
# Google GTX is used as the first lightweight engine; LibreTranslate and MyMemory are fallbacks.
TRANSLATION_ENGINES = ["Google Translate", "LibreTranslate", "MyMemory"]
LIBRETRANSLATE_ENDPOINTS = [
    "https://translate.argosopentech.com",
    "https://translate.terraprint.co",
    "https://lt.vern.cc",
]
TARGET_LANGUAGES = {
    "Simplified Chinese": "zh-CN",
    "Traditional Chinese": "zh-TW",
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
    "Italian": "it",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
}

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "zh": "Chinese", "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish",
    "fr": "French", "de": "German", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "id": "Indonesian", "it": "Italian", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
}

st.markdown(
    """
    <style>
    .hero { padding: 1.4rem 1.6rem; border-radius: 20px; border: 1px solid rgba(128,128,128,.22); margin-bottom: 1rem; background: linear-gradient(135deg, rgba(128,128,128,.10), rgba(128,128,128,.03)); }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -.02em; }
    .hero p { margin: .4rem 0 0; opacity: .72; font-size: 1.02rem; }
    .feature { padding: 1rem 1.05rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.18); min-height: 92px; background: rgba(128,128,128,.035); }
    .feature b { font-size: 1.03rem; }
    .mini-note { opacity: .68; font-size: .88rem; }
    </style>
    <div class="hero">
      <h1>🎬 Hinglish Subtitle Studio</h1>
      <p>Accurate Hinglish transcription · multilingual translation · clean subtitle export</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def find_binary(name):
    candidates = [
        shutil.which(name), f"/usr/bin/{name}", f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}", f"/opt/conda/bin/{name}",
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
        MODEL_SIZE, device="cpu", compute_type="int8",
        cpu_threads=CPU_THREADS, num_workers=1,
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
        lines.append(f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n")
    return "\n".join(lines)


def make_bilingual_srt(source_segments, translated_segments):
    lines = []
    for index, ((start, end, source), (_, _, translated)) in enumerate(zip(source_segments, translated_segments), start=1):
        lines.append(
            f"{index}\n{format_time(start)} --> {format_time(end)}\n"
            f"{source}\n{translated}\n"
        )
    return "\n".join(lines)


def make_txt(segments):
    return "\n".join(f"[{format_time(start)} --> {format_time(end)}] {text}" for start, end, text in segments)


def make_bilingual_txt(source_segments, translated_segments):
    lines = []
    for (start, end, source), (_, _, translated) in zip(source_segments, translated_segments):
        lines.append(f"[{format_time(start)} --> {format_time(end)}]\n{source}\n{translated}\n")
    return "\n".join(lines)


def get_media_duration(media_path):
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    command = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)]
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
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vn", "-map", "0:a:0?", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)]
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
            segment_list = [(start, min(end, media_duration), text) for start, end, text in segment_list if start < media_duration]
        if not segment_list:
            raise RuntimeError("No reliable speech was detected in the uploaded video.")
        yield "done", segment_list, media_duration, info, len(raw_segments)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------- Translation ----------
# Language detection is performed per subtitle entry. This is important for mixed-language subtitles.
# A whole SRT is therefore never assumed to have one source language.
def detect_script_language(text):
    counts = {
        "hi": len(re.findall(r"[\u0900-\u097f]", text)),
        "zh": len(re.findall(r"[\u4e00-\u9fff]", text)),
        "ja": len(re.findall(r"[\u3040-\u30ff]", text)),
        "ko": len(re.findall(r"[\uac00-\ud7af]", text)),
        "ar": len(re.findall(r"[\u0600-\u06ff]", text)),
        "th": len(re.findall(r"[\u0e00-\u0e7f]", text)),
        "ru": len(re.findall(r"[\u0400-\u04ff]", text)),
    }
    if not text.strip():
        return "unknown"
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"


def google_translate(text, target_language):
    """Free Google Translate web endpoint; no API key is required. It is used only as a lightweight fallback engine."""
    response = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "auto", "tl": target_language, "dt": "t", "q": text[:4500]},
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    parts = data[0] if isinstance(data, list) and data else []
    translated = "".join(part[0] for part in parts if isinstance(part, list) and part and part[0])
    if not translated:
        raise RuntimeError("Google Translate returned an empty result")
    return clean_text(translated)


def libre_translate(text, target_language):
    last_error = None
    # LibreTranslate uses ISO language codes; zh-CN/zh-TW are normalized to zh.
    target = "zh" if target_language.startswith("zh-") else target_language
    for endpoint in LIBRETRANSLATE_ENDPOINTS:
        try:
            response = requests.post(
                endpoint + "/translate",
                data={"q": text[:4500], "source": "auto", "target": target, "format": "text"},
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
            translated = clean_text(data.get("translatedText", ""))
            if translated:
                return translated
            last_error = "Empty translation returned"
        except Exception as error:
            last_error = error
    raise RuntimeError(str(last_error or "LibreTranslate endpoints unavailable"))


def mymemory_translate(text, target_language):
    """Free MyMemory endpoint. It needs a source language, so script detection is used per subtitle."""
    source = detect_script_language(text)
    if source == "unknown":
        source = "en"
    target = "zh-CN" if target_language == "zh-CN" else ("zh-TW" if target_language == "zh-TW" else target_language)
    if source == target or (source == "zh" and target.startswith("zh-")):
        return text
    response = requests.get(
        "https://api.mymemory.translated.net/get",
        params={"q": text[:4500], "langpair": f"{source}|{target}"},
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    translated = clean_text((data.get("responseData") or {}).get("translatedText", ""))
    if not translated:
        raise RuntimeError("MyMemory returned an empty result")
    return translated


def translate_one(text, target_language, preferred_engine):
    text = clean_text(text)
    if not text:
        return ""

    # Do not waste network calls when the entry is already entirely in the target script/language.
    if target_language == "en" and detect_script_language(text) == "en":
        return text
    if target_language == "hi" and detect_script_language(text) == "hi":
        return text

    engines = {
        "Google Translate": google_translate,
        "LibreTranslate": libre_translate,
        "MyMemory": mymemory_translate,
    }
    ordered = [preferred_engine] + [name for name in TRANSLATION_ENGINES if name != preferred_engine]
    errors = []
    for name in ordered:
        try:
            result = engines[name](text, target_language)
            if result:
                return result
        except Exception as error:
            errors.append(f"{name}: {error}")
    raise RuntimeError("All free translation engines failed. " + " | ".join(errors[-3:]))


def translate_segments(segments, target_language, preferred_engine, progress_callback=None):
    translated = []
    total = len(segments)
    for index, (start, end, text) in enumerate(segments, start=1):
        translated_text = translate_one(text, target_language, preferred_engine)
        translated.append((start, end, translated_text))
        if progress_callback:
            progress_callback(index, total, translated_text)
        # Small pause prevents a free endpoint from being hammered by a long SRT.
        time.sleep(0.05)
    return translated


def detect_mixed_languages(segments):
    counts = {}
    for _, _, text in segments:
        code = detect_script_language(text)
        counts[code] = counts.get(code, 0) + 1
    meaningful = [(code, count) for code, count in counts.items() if code != "unknown"]
    meaningful.sort(key=lambda item: item[1], reverse=True)
    return meaningful


def render_translation_ui(source_segments=None, source_srt=None):
    st.subheader("🌐 Subtitle Translation")
    st.caption("Each subtitle entry is treated independently, so a video can contain Hindi, English, Spanish, Japanese or other languages in the same file.")

    uploaded_srt = st.file_uploader("Or upload an existing SRT subtitle file", type=["srt"], key="translation_srt_upload")
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

    language_counts = detect_mixed_languages(active_segments)
    detected_label = "Mixed / multilingual"
    if language_counts:
        labels = [f"{LANGUAGE_NAMES.get(code, code.upper())} ({count})" for code, count in language_counts[:4]]
        detected_label = " · ".join(labels)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Subtitle entries", len(active_segments))
    with col2:
        st.metric("Detected subtitle languages", detected_label)

    st.markdown("### 🎯 Translation settings")
    target_label = st.selectbox(
        "Translate every subtitle into",
        list(TARGET_LANGUAGES.keys()),
        index=0,
        key="target_language",
    )
    target_code = TARGET_LANGUAGES[target_label]

    preferred_engine = st.selectbox(
        "Translation engine",
        TRANSLATION_ENGINES,
        index=0,
        key="translation_engine",
        help="All three choices are free to use. If the selected engine fails, the app automatically falls back to the other free engines.",
    )

    st.info("💡 Source language is detected per subtitle entry. The whole video is NOT forced into one source language.")
    st.caption("Free engines: Google Translate web endpoint + LibreTranslate + MyMemory, with automatic fallback.")

    export_mode = st.radio(
        "Export format",
        ["Translated only", "Original + translated (two lines)"],
        horizontal=True,
        key="export_mode",
    )

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
            translated_segments = translate_segments(active_segments, target_code, preferred_engine, on_progress)
            translated_srt = make_srt(translated_segments)
            bilingual_srt = make_bilingual_srt(active_segments, translated_segments)
            translated_txt = make_txt(translated_segments)
            bilingual_txt = make_bilingual_txt(active_segments, translated_segments)
            progress.progress(1.0, text="Translation completed")
            detail.success(f"Translated {len(translated_segments)} subtitle entries. Mixed-language detection was applied per entry.")

            st.subheader("✨ Translation preview")
            if export_mode == "Original + translated (two lines)":
                preview_text = "\n".join(f"{source}\n{translation}" for (_, _, source), (_, _, translation) in zip(active_segments, translated_segments))
            else:
                preview_text = "\n".join(text for _, _, text in translated_segments)
            st.text_area("Preview", preview_text, height=420)

            st.markdown("### 📦 Export")
            if export_mode == "Translated only":
                st.download_button(
                    "⬇️ Download translated SRT",
                    translated_srt,
                    f"translated_{Path(source_name).stem}.srt",
                    "application/x-subrip",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Download translated TXT",
                    translated_txt,
                    f"translated_{Path(source_name).stem}.txt",
                    "text/plain",
                    use_container_width=True,
                )
            else:
                st.download_button(
                    "⬇️ Download original + translated SRT",
                    bilingual_srt,
                    f"bilingual_{Path(source_name).stem}.srt",
                    "application/x-subrip",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Download original + translated TXT",
                    bilingual_txt,
                    f"bilingual_{Path(source_name).stem}.txt",
                    "text/plain",
                    use_container_width=True,
                )

        except Exception as error:
            progress.empty()
            detail.empty()
            preview_box.empty()
            st.error(f"Translation failed: {error}")
            st.warning("The selected free engine and its fallbacks were unavailable. Your recognition result is unchanged; try the translation again or choose another engine.")


if "segments" not in st.session_state:
    st.session_state.segments = None
if "srt_content" not in st.session_state:
    st.session_state.srt_content = None
if "txt_content" not in st.session_state:
    st.session_state.txt_content = None

with st.sidebar:
    st.markdown("### ✨ Studio")
    st.markdown("**1. Speech Recognition**\n\nWhisper medium · Hinglish optimized")
    st.markdown("**2. Subtitle Translation**\n\nMixed-language, per-entry detection")
    st.markdown("**3. Export**\n\nTranslated only · Original + translated")
    st.divider()
    st.caption("Recognition model and parameters are frozen for this version. Translation is completely separate from recognition.")

recognition_tab, translation_tab = st.tabs(["🎙️ Speech Recognition", "🌐 Subtitle Translation"])

with recognition_tab:
    feature_cols = st.columns(3)
    with feature_cols[0]:
        st.markdown('<div class="feature">🎙️ <b>Hinglish-aware</b><br><small>Hindi + English code-switching</small></div>', unsafe_allow_html=True)
    with feature_cols[1]:
        st.markdown('<div class="feature">🛡️ <b>Hallucination protection</b><br><small>Repeated-output filtering</small></div>', unsafe_allow_html=True)
    with feature_cols[2]:
        st.markdown('<div class="feature">⚡ <b>Resource controlled</b><br><small>CPU int8 · 2 threads</small></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a video", type=SUPPORTED_TYPES, help="MP4, MOV, MKV, WebM, AVI and M4V are supported.", key="video_upload")

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
st.caption("Hinglish Subtitle Studio · Whisper medium + FFmpeg + faster-whisper · Translation: Google Translate + LibreTranslate + MyMemory")
