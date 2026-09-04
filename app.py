import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from difflib import SequenceMatcher

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Hinglish Subtitle Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container{max-width:1180px;padding-top:1.8rem;padding-bottom:3rem}
.hero{padding:1.5rem 1.7rem;border:1px solid rgba(128,128,128,.22);border-radius:20px;margin-bottom:1.25rem;background:rgba(128,128,128,.045)}
.hero h1{margin:0 0 .35rem 0;font-size:2.15rem}
.hero p{margin:0;opacity:.78}
.small-note{font-size:.88rem;opacity:.72}
</style>
<div class="hero">
<h1>🎬 Hinglish Subtitle Studio</h1>
<p>高准确率 Hindi + English 混合语音识别 · MP4 / MOV / MKV / WebM / AVI · 免费本地模型 · SRT / TXT</p>
</div>
""",
    unsafe_allow_html=True,
)

# Streamlit Community Cloud 上更适合使用蒸馏版 large 模型：
# 准确率明显高于 small，同时比完整 large-v3 更节省内存和计算资源。
MODEL_SIZE = "distil-large-v3"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi"]
MAX_FILE_MB = 180
MAX_DURATION_SECONDS = 30 * 60


def find_binary(name):
    return shutil.which(name)


@st.cache_resource(show_spinner=False)
def load_model():
    cpu_count = os.cpu_count() or 2
    threads = max(1, min(2, cpu_count))
    return WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
        num_workers=1,
    )


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def format_time(seconds):
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def probe_duration(src):
    ffprobe = find_binary("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def convert_to_mp3(src, dst):
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg。Streamlit Cloud 请确认 packages.txt 包含 ffmpeg。")

    # 只保留语音相关信息：16 kHz、单声道，降低后续 ASR 的解码和内存压力。
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(dst),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("MP3 转换失败：\n" + (result.stderr[-2500:] or "未知 FFmpeg 错误"))


def make_srt(segments):
    out = []
    index = 1
    for s in segments:
        text = clean_text(s["text"])
        if not text:
            continue
        out.append(
            f"{index}\n{format_time(s['start'])} --> {format_time(s['end'])}\n{text}\n"
        )
        index += 1
    return "\n".join(out)


def make_txt(segments):
    return "\n".join(
        f"[{format_time(s['start'])} --> {format_time(s['end'])}] {s['text']}"
        for s in segments
        if clean_text(s["text"])
    )


def avg(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def normalize_for_score(text):
    return re.sub(r"\s+", "", clean_text(text or "")).lower()


def similarity_score(reference, predicted):
    reference = normalize_for_score(reference)
    predicted = normalize_for_score(predicted)
    if not reference or not predicted:
        return 0.0
    return SequenceMatcher(None, reference, predicted).ratio()


with st.sidebar:
    st.header("⚙️ 识别设置")
    st.success(f"当前模型：**{MODEL_SIZE}**")
    st.caption(
        "准确率优先的免费方案。使用蒸馏 large 模型 + CPU int8，并限制线程/并发，降低 Streamlit Cloud 内存压力。"
    )

    language_mode = st.selectbox(
        "识别语言",
        ["自动检测（推荐）", "Hindi", "English"],
        index=0,
        help="Hinglish 建议使用自动检测；如果视频几乎全部是 Hindi，可手动选择 Hindi。",
    )
    beam_size = st.slider(
        "Beam Size",
        min_value=1,
        max_value=5,
        value=5,
        help="准确率优先时使用 5；数值越高通常计算量越大。",
    )
    vad_filter = st.checkbox(
        "VAD 静音过滤",
        value=True,
        help="过滤长时间无语音片段，减少无效识别。",
    )
    word_timestamps = st.checkbox("词级时间戳", value=False)
    context = st.checkbox(
        "使用前文上下文",
        value=True,
        help="对连续对话通常有利于上下文一致性。",
    )

    st.divider()
    st.subheader("🛡️ 免费服务器保护")
    st.caption(f"单文件上限：{MAX_FILE_MB} MB")
    st.caption("单个视频最长：30 分钟")
    st.caption("单次只处理一个任务，处理结束自动删除临时文件。")

    st.divider()
    st.subheader("🧪 准确率测试")
    reference_text = st.text_area(
        "可选：粘贴人工标准答案",
        placeholder="输入同一段视频的人工转写结果，用于快速比较识别效果。",
        height=130,
    )

language = None
if language_mode == "Hindi":
    language = "hi"
elif language_mode == "English":
    language = "en"

left, right = st.columns([1.55, 1], gap="large")
with left:
    uploaded_file = st.file_uploader(
        "📤 上传视频",
        type=SUPPORTED_TYPES,
        help="支持 MP4、MOV、MKV、WebM、AVI。识别前会统一提取成 16kHz 单声道 MP3。",
    )
with right:
    st.markdown("### 📌 免费处理流程")
    st.markdown(
        "上传视频  →  FFmpeg 提取 MP3  →  **distil-large-v3** 识别  →  质量分析  →  SRT / TXT"
    )
    st.caption("不调用 OpenAI、DeepSeek、SiliconFlow 等付费 API。")

if uploaded_file:
    suffix = Path(uploaded_file.name).suffix.lower()
    size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("文件大小", f"{size_mb:.1f} MB")
    info_col2.metric("格式", suffix[1:].upper())
    info_col3.metric("时长限制", "≤ 30 分钟")

    if size_mb > MAX_FILE_MB:
        st.error(
            f"文件为 {size_mb:.1f} MB，超过免费服务器保护上限 {MAX_FILE_MB} MB。请压缩视频后再上传。"
        )
        st.stop()

    preview_col, detail_col = st.columns([1.65, 1], gap="large")
    with preview_col:
        st.subheader("🎞️ 视频预览")
        try:
            st.video(uploaded_file)
        except Exception:
            st.info("当前浏览器无法预览该视频，但仍可以继续进行识别。")

    with detail_col:
        st.subheader("🚦 资源策略")
        st.markdown(
            "- CPU 推理 + int8\n"
            "- 最大 2 个 CPU threads\n"
            "- 仅 1 个模型 worker\n"
            "- VAD 默认开启\n"
            "- 临时文件自动清理"
        )
        st.caption("这些限制是为了在免费 Streamlit Cloud 环境中尽可能保持稳定。")

    if st.button("🚀 开始高准确率识别", type="primary", use_container_width=True):
        work_dir = tempfile.mkdtemp(prefix="hinglish_")
        video_path = os.path.join(work_dir, "input" + suffix)
        mp3_path = os.path.join(work_dir, "audio.mp3")
        status = st.status("正在处理…", expanded=True)
        progress = st.progress(0)
        started_at = time.time()

        try:
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            status.write("📥 视频已保存到临时目录")
            progress.progress(8)

            status.write("🔎 正在检查视频时长…")
            duration = probe_duration(video_path)
            if duration is not None:
                if duration > MAX_DURATION_SECONDS:
                    raise RuntimeError(
                        f"视频时长为 {duration / 60:.1f} 分钟，超过免费服务器保护上限 30 分钟。"
                    )
                status.write(f"⏱️ 视频时长：{duration / 60:.1f} 分钟")
            else:
                status.write("⚠️ 无法读取视频时长，将继续处理。")
            progress.progress(15)

            status.write("🎵 正在提取 16kHz 单声道 MP3…")
            convert_to_mp3(video_path, mp3_path)
            progress.progress(32)

            status.write(f"🧠 正在加载 {MODEL_SIZE} 模型…")
            model = load_model()
            progress.progress(45)

            status.write("🔎 正在进行 Hindi + English 混合语音识别…")
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
            segment_count = 0
            for s in stream:
                text = clean_text(s.text)
                if not text:
                    continue
                segments.append(
                    {
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": text,
                        "avg_logprob": getattr(s, "avg_logprob", None),
                        "no_speech_prob": getattr(s, "no_speech_prob", None),
                        "compression_ratio": getattr(s, "compression_ratio", None),
                    }
                )
                segment_count += 1
                if segment_count % 10 == 0:
                    progress.progress(min(84, 45 + segment_count // 10))

            if not segments:
                raise RuntimeError("没有识别到有效语音。请确认视频中存在清晰的人声。")

            quality = {
                "logprob": avg([s["avg_logprob"] for s in segments]),
                "no_speech": avg([s["no_speech_prob"] for s in segments]),
                "compression": avg([s["compression_ratio"] for s in segments]),
            }
            srt = make_srt(segments)
            txt = make_txt(segments)
            elapsed = time.time() - started_at

            st.session_state.update(
                segments=segments,
                srt=srt,
                txt=txt,
                quality=quality,
                language=getattr(info, "language", "unknown"),
                language_probability=getattr(info, "language_probability", None),
                processing_seconds=elapsed,
                video_duration=duration,
                model_name=MODEL_SIZE,
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
    elapsed = st.session_state.get("processing_seconds")
    c4.metric("处理时间", f"{elapsed / 60:.1f} min" if elapsed is not None else "N/A")

    tab1, tab2, tab3 = st.tabs(["📝 字幕文本", "🧪 识别质量", "⬇️ 下载"])

    with tab1:
        st.text_area("识别结果", st.session_state["txt"], height=520)

    with tab2:
        q1, q2, q3 = st.columns(3)
        lp = quality.get("logprob")
        ns = quality.get("no_speech")
        cr = quality.get("compression")
        q1.metric("Avg Log Probability", f"{lp:.3f}" if lp is not None else "N/A")
        q2.metric("Avg No-Speech", f"{ns:.3f}" if ns is not None else "N/A")
        q3.metric("Avg Compression", f"{cr:.3f}" if cr is not None else "N/A")

        if reference_text.strip():
            predicted = " ".join(s["text"] for s in segments)
            score = similarity_score(reference_text, predicted)
            st.subheader("📐 与人工标准答案的字符级相似度")
            st.progress(score)
            st.write(f"### {score * 100:.2f}%")
            st.caption(
                "这是用于模型快速对比的序列相似度，不等同于专业 CER/WER。后续可继续加入真正的 CER/WER 评测。"
            )
        else:
            st.info("在左侧输入人工标准答案后，可以对当前识别结果进行快速准确率对比。")

    with tab3:
        st.download_button(
            "⬇️ 下载 SRT",
            st.session_state["srt"],
            "hinglish_subtitle.srt",
            "application/x-subrip",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ 下载 TXT",
            st.session_state["txt"],
            "hinglish_subtitle.txt",
            "text/plain",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Hinglish Subtitle Studio · 免费部署方案：视频 → FFmpeg 16kHz Mono MP3 → distil-large-v3 → SRT/TXT。"
)
