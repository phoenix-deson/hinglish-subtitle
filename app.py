import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from difflib import SequenceMatcher

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Hinglish Subtitle Studio", page_icon="🎬", layout="wide")

# -------------------- Configuration --------------------
MODEL_SIZE = "distil-large-v3"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v", "flv"]
MAX_FILE_MB = 1024
MAX_DURATION_MINUTES = 180
CPU_THREADS = max(1, min(2, os.cpu_count() or 2))

st.markdown("""
<style>
.block-container{max-width:1200px;padding-top:1.5rem;padding-bottom:3rem}
.hero{padding:1.6rem 1.8rem;border:1px solid rgba(128,128,128,.22);border-radius:20px;margin-bottom:1.2rem;background:linear-gradient(135deg,rgba(120,120,120,.08),rgba(120,120,120,.02))}
.hero h1{margin-bottom:.3rem}
.small-note{opacity:.72;font-size:.9rem}
.stButton>button{border-radius:12px;font-weight:600}
</style>
<div class="hero">
<h1>🎬 Hinglish Subtitle Studio</h1>
<p>High-accuracy English + Hindi speech recognition for MP4, MOV, MKV, WebM and more.</p>
<p class="small-note">Video is converted to lightweight 16 kHz mono MP3 before transcription to reduce processing load.</p>
</div>
""", unsafe_allow_html=True)


def find_binary(name):
    candidates = [
        shutil.which(name),
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
        f"/opt/conda/bin/{name}",
        f"/home/runner/.local/bin/{name}",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def run_command(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def get_duration(src):
    ffprobe = find_binary("ffprobe")
    if ffprobe:
        result = run_command([
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(src)
        ])
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    ffmpeg = find_binary("ffmpeg")
    if ffmpeg:
        result = run_command([ffmpeg, "-i", str(src)])
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
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
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def convert_to_mp3(src, dst):
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is not available in the deployment environment. "
            "Please redeploy the Streamlit app so packages.txt is installed."
        )

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn",
        "-map", "0:a:0?",
        "-ac", "1",
        "-ar", "16000",
        "-codec:a", "libmp3lame",
        "-b:a", "64k",
        str(dst),
    ]
    result = run_command(cmd)
    if result.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError("Audio extraction failed. The uploaded video may not contain a readable audio track.")


def make_srt(segments):
    out = []
    index = 1
    for s in segments:
        text = clean_text(s["text"])
        if not text:
            continue
        out.append(f"{index}\n{format_time(s['start'])} --> {format_time(s['end'])}\n{text}\n")
        index += 1
    return "\n".join(out)


def make_txt(segments):
    return "\n".join(
        f"[{format_time(s['start'])} --> {format_time(s['end'])}] {s['text']}"
        for s in segments if clean_text(s["text"])
    )


def avg(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


with st.sidebar:
    st.header("⚙️ Recognition Settings")
    st.info(f"Model: **{MODEL_SIZE}**")
    st.caption("Accuracy-first configuration optimized for free CPU hosting.")

    beam_size = st.slider("Beam Size", 1, 5, 5, help="Higher values can improve recognition quality but require more CPU time.")
    language_mode = st.selectbox("Language", ["Auto Detect", "Hindi", "English"])
    vad_filter = st.checkbox("VAD silence filtering", True)
    word_timestamps = st.checkbox("Word-level timestamps", False)
    context = st.checkbox("Use previous-text context", True)

    st.divider()
    st.subheader("🧪 Accuracy Test")
    reference_text = st.text_area(
        "Optional reference transcript",
        placeholder="Paste the manually corrected transcript here to compare recognition quality.",
        height=140,
    )

    st.divider()
    st.caption(f"Server protection: up to {MAX_FILE_MB} MB / {MAX_DURATION_MINUTES} minutes")

left, right = st.columns([1.55, 1])
with left:
    uploaded_file = st.file_uploader(
        "📤 Upload Video",
        type=SUPPORTED_TYPES,
        help="Supported: MP4, MOV, MKV, WebM, AVI, M4V and FLV.",
    )
with right:
    st.markdown("### 📌 Processing Pipeline")
    st.markdown(
        "Upload video → validate → extract **MP3** → "
        f"**{MODEL_SIZE}** transcription → SRT/TXT"
    )
    st.caption("The original video is never passed into Whisper. Only the extracted audio is transcribed.")


if uploaded_file:
    file_size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024
    suffix = Path(uploaded_file.name).suffix.lower()

    if file_size_mb > MAX_FILE_MB:
        st.error(f"❌ File is too large ({file_size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_MB} MB.")
        st.stop()

    preview, info = st.columns([1.6, 1])
    with preview:
        st.subheader("🎞️ Preview")
        try:
            st.video(uploaded_file)
        except Exception:
            st.info("Video preview is unavailable in this browser, but the file can still be processed.")
    with info:
        st.metric("File Size", f"{file_size_mb:.1f} MB")
        st.metric("Format", suffix[1:].upper())
        st.metric("Model", MODEL_SIZE)

    if st.button("🚀 Start High-Accuracy Transcription", type="primary", use_container_width=True):
        work_dir = tempfile.mkdtemp(prefix="hinglish_")
        video_path = os.path.join(work_dir, "input" + suffix)
        mp3_path = os.path.join(work_dir, "audio.mp3")
        status = st.status("Processing…", expanded=True)
        progress = st.progress(0)

        try:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            status.write("📥 Video saved to temporary storage.")
            progress.progress(8)

            status.write("🔎 Checking video duration…")
            duration = get_duration(video_path)
            if duration is None:
                raise RuntimeError("Unable to read the video duration. Please check that the file contains a valid video/audio stream.")

            duration_min = duration / 60
            if duration_min > MAX_DURATION_MINUTES:
                raise RuntimeError(
                    f"Video is {duration_min:.1f} minutes long. The current free-server limit is {MAX_DURATION_MINUTES} minutes."
                )
            status.write(f"⏱️ Duration: {duration_min:.1f} minutes")
            progress.progress(15)

            status.write("🎵 Extracting 16 kHz mono MP3…")
            convert_to_mp3(video_path, mp3_path)
            original_size = os.path.getsize(video_path) / 1024 / 1024
            audio_size = os.path.getsize(mp3_path) / 1024 / 1024
            status.write(f"✅ Audio extracted: {audio_size:.1f} MB from {original_size:.1f} MB video")
            progress.progress(35)

            # Release the large uploaded video as early as possible.
            try:
                os.remove(video_path)
            except OSError:
                pass

            status.write(f"🧠 Loading {MODEL_SIZE}…")
            model = load_model()
            progress.progress(50)

            language = None
            if language_mode == "Hindi":
                language = "hi"
            elif language_mode == "English":
                language = "en"

            status.write("🔎 Transcribing English + Hindi speech…")
            stream, info = model.transcribe(
                mp3_path,
                language=language,
                task="transcribe",
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                condition_on_previous_text=context,
                temperature=0.0,
            )

            segments = []
            for s in stream:
                text = clean_text(s.text)
                if text:
                    segments.append({
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": text,
                        "avg_logprob": getattr(s, "avg_logprob", None),
                        "no_speech_prob": getattr(s, "no_speech_prob", None),
                        "compression_ratio": getattr(s, "compression_ratio", None),
                    })

            if not segments:
                raise RuntimeError("No speech was detected in the audio.")

            progress.progress(90)
            quality = {
                "logprob": avg([s["avg_logprob"] for s in segments]),
                "no_speech": avg([s["no_speech_prob"] for s in segments]),
                "compression": avg([s["compression_ratio"] for s in segments]),
            }

            srt = make_srt(segments)
            txt = make_txt(segments)
            st.session_state.update(
                segments=segments,
                srt=srt,
                txt=txt,
                quality=quality,
                language=getattr(info, "language", "unknown"),
                language_probability=getattr(info, "language_probability", None),
                duration=duration,
            )

            progress.progress(100)
            status.update(label="✅ Transcription complete", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Processing failed", state="error", expanded=True)
            st.error(str(e))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


if "segments" in st.session_state:
    segments = st.session_state["segments"]
    quality = st.session_state["quality"]

    st.divider()
    st.header("📊 Transcription Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subtitle Segments", len(segments))
    c2.metric("Detected Language", st.session_state.get("language", "unknown"))
    p = st.session_state.get("language_probability")
    c3.metric("Language Confidence", f"{p * 100:.1f}%" if p is not None else "N/A")
    lp = quality.get("logprob")
    c4.metric("Avg. Log Probability", f"{lp:.3f}" if lp is not None else "N/A")

    tab1, tab2, tab3 = st.tabs(["📝 Transcript", "🧪 Quality & Accuracy", "⬇️ Download"])

    with tab1:
        st.text_area("Transcript", st.session_state["txt"], height=520)

    with tab2:
        q1, q2, q3 = st.columns(3)
        q1.metric("Avg. Log Probability", f"{lp:.3f}" if lp is not None else "N/A")
        ns = quality.get("no_speech")
        cr = quality.get("compression")
        q2.metric("Avg. No-Speech", f"{ns:.3f}" if ns is not None else "N/A")
        q3.metric("Avg. Compression", f"{cr:.3f}" if cr is not None else "N/A")

        if reference_text.strip():
            predicted = clean_text(" ".join(s["text"] for s in segments)).lower()
            reference = clean_text(reference_text).lower()
            score = SequenceMatcher(None, reference, predicted).ratio()
            st.subheader("📐 Reference Similarity")
            st.progress(score)
            st.markdown(f"### {score * 100:.2f}%")
            st.caption("This is a quick character-sequence similarity score, not a formal WER/CER measurement.")
        else:
            st.info("Paste a manually corrected reference transcript in the sidebar to compare recognition quality.")

    with tab3:
        st.download_button(
            "⬇️ Download SRT",
            st.session_state["srt"],
            "hinglish_subtitle.srt",
            "application/x-subrip",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download TXT",
            st.session_state["txt"],
            "hinglish_subtitle.txt",
            "text/plain",
            use_container_width=True,
        )

st.divider()
st.caption("Hinglish Subtitle Studio · Free CPU transcription · Video → lightweight MP3 → Whisper → SRT/TXT")
