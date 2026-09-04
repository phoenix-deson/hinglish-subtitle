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

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem}
.hero{padding:1.4rem 1.6rem;border:1px solid rgba(128,128,128,.2);border-radius:18px;margin-bottom:1.2rem}
</style>
<div class="hero">
<h1>🎬 Hinglish Subtitle Studio</h1>
<p>English + Hindi 混合语音识别 · MP4 / MOV / MKV / WebM · 自动提取 MP3 · SRT 导出</p>
</div>
""", unsafe_allow_html=True)

MODEL_SIZE = "large-v3-turbo"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]


def find_ffmpeg():
    return shutil.which("ffmpeg")


@st.cache_resource(show_spinner=False)
def load_model():
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(1, min(4, os.cpu_count() or 2)),
        num_workers=1,
    )


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms = 0
    if s >= 60:
        m += 1
        s = 0
    if m >= 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def convert_to_mp3(src, dst):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg。请先安装：brew install ffmpeg")
    cmd = [
        ffmpeg, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-b:a", "64k", str(dst)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("MP3 转换失败：\n" + (result.stderr[-2000:] or "未知 FFmpeg 错误"))


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
    st.header("⚙️ 识别设置")
    st.info(f"当前模型：**{MODEL_SIZE}**")
    st.caption("比 small 更高精度，同时比完整 large-v3 更节省资源。CPU 使用 int8，限制线程避免服务器过载。")
    beam_size = st.slider("Beam Size", 1, 5, 3)
    vad_filter = st.checkbox("VAD 静音过滤", True)
    word_timestamps = st.checkbox("词级时间戳", False)
    context = st.checkbox("使用前文上下文", True)
    st.divider()
    st.subheader("🧪 准确率测试")
    reference_text = st.text_area(
        "可选：粘贴人工标准答案",
        placeholder="用于计算字符级相似度，方便比较不同模型的识别效果。",
        height=130,
    )

left, right = st.columns([1.5, 1])
with left:
    uploaded_file = st.file_uploader(
        "📤 上传视频",
        type=SUPPORTED_TYPES,
        help="支持 MP4、MOV、MKV、WebM、AVI；识别前会先提取为 16kHz 单声道 MP3。",
    )
with right:
    st.markdown("### 📌 处理流程")
    st.markdown("上传视频 → FFmpeg 提取 MP3 → **large-v3-turbo** 识别 → 质量测试 → SRT/TXT")

if uploaded_file:
    suffix = Path(uploaded_file.name).suffix.lower()
    preview, info = st.columns([1.6, 1])
    with preview:
        st.subheader("🎞️ 预览")
        st.video(uploaded_file)
    with info:
        size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024
        st.metric("文件大小", f"{size_mb:.1f} MB")
        st.metric("格式", suffix[1:].upper())
        st.caption("不会让 Whisper 直接处理视频。先只提取音频，可明显减少解码与内存压力。")

    if st.button("🚀 开始高精度识别", type="primary", use_container_width=True):
        work_dir = tempfile.mkdtemp(prefix="hinglish_")
        video_path = os.path.join(work_dir, "input" + suffix)
        mp3_path = os.path.join(work_dir, "audio.mp3")
        status = st.status("正在处理…", expanded=True)
        progress = st.progress(0)
        try:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            status.write("📥 视频已保存")
            progress.progress(10)

            status.write("🎵 正在提取 16kHz 单声道 MP3…")
            convert_to_mp3(video_path, mp3_path)
            progress.progress(30)

            status.write(f"🧠 正在加载 {MODEL_SIZE}…")
            model = load_model()
            progress.progress(45)

            status.write("🔎 正在识别 English + Hindi 混合语音…")
            stream, info = model.transcribe(
                mp3_path,
                language=None,
                task="transcribe",
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                condition_on_previous_text=context,
                temperature=0.0,
            )

            segments = []
            for s in stream:
                segments.append({
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": clean_text(s.text),
                    "avg_logprob": getattr(s, "avg_logprob", None),
                    "no_speech_prob": getattr(s, "no_speech_prob", None),
                    "compression_ratio": getattr(s, "compression_ratio", None),
                })

            progress.progress(85)
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
            )
            progress.progress(100)
            status.update(label="✅ 识别完成", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ 处理失败", state="error", expanded=True)
            st.error(f"处理过程中发生错误：{e}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

if "segments" in st.session_state:
    segments = st.session_state["segments"]
    quality = st.session_state["quality"]
    st.divider()
    st.header("📊 识别结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("字幕段数", len(segments))
    c2.metric("识别语言", st.session_state.get("language", "unknown"))
    p = st.session_state.get("language_probability")
    c3.metric("语言判断概率", f"{p * 100:.1f}%" if p is not None else "N/A")
    lp = quality.get("logprob")
    c4.metric("平均 Log Probability", f"{lp:.3f}" if lp is not None else "N/A")

    tab1, tab2, tab3 = st.tabs(["📝 字幕文本", "🧪 识别质量", "⬇️ 下载"])
    with tab1:
        st.text_area("识别结果", st.session_state["txt"], height=500)
    with tab2:
        q1, q2, q3 = st.columns(3)
        q1.metric("Avg Log Probability", f"{lp:.3f}" if lp is not None else "N/A")
        ns = quality.get("no_speech")
        cr = quality.get("compression")
        q2.metric("Avg No-Speech", f"{ns:.3f}" if ns is not None else "N/A")
        q3.metric("Avg Compression", f"{cr:.3f}" if cr is not None else "N/A")
        if reference_text.strip():
            predicted = clean_text(" ".join(s["text"] for s in segments)).lower()
            reference = clean_text(reference_text).lower()
            score = SequenceMatcher(None, reference, predicted).ratio()
            st.subheader("📐 与标准答案的字符级相似度")
            st.progress(score)
            st.write(f"### {score * 100:.2f}%")
            st.caption("该数值用于模型版本快速对比，不等同于专业 CER/WER。")
        else:
            st.info("在左侧输入人工标准答案后，可进行简单准确率对比。")
    with tab3:
        st.download_button("⬇️ 下载 SRT", st.session_state["srt"], "hinglish_subtitle.srt", "application/x-subrip", use_container_width=True)
        st.download_button("⬇️ 下载 TXT", st.session_state["txt"], "hinglish_subtitle.txt", "text/plain", use_container_width=True)

st.divider()
st.caption("Hinglish Subtitle Studio · WebM 等视频统一先经 FFmpeg 提取 MP3，再交给 Whisper 识别。")
