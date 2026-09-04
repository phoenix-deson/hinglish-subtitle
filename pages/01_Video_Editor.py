import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Video Subtitle Editor", page_icon="🎬", layout="wide")

VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
FONT_OPTIONS = [
    "Noto Sans", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR",
    "Noto Sans Devanagari", "Noto Sans Arabic", "Noto Sans Thai", "Noto Sans Hebrew",
    "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica",
]
POSITIONS = {"Bottom": 2, "Middle": 5, "Top": 8}
TIMING_RE = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?$"
)


def find_binary(name):
    candidates = [
        shutil.which(name),
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
        f"/opt/conda/bin/{name}",
    ]
    return next((p for p in candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)


def decode_srt(data):
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030"):
        try:
            text = data.decode(encoding)
            if "-->" in text:
                return text
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_time(value):
    value = value.strip().replace(".", ",")
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}),(\d{1,3})", value)
    if not match:
        raise ValueError(f"Invalid timestamp: {value}")
    h, minute, sec, ms = match.groups()
    return int(h) * 3600 + int(minute) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000


def fmt_time(seconds):
    total = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(total, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(content):
    """Parse real-world SRT by locating timing lines, not by requiring blank blocks.

    Supports single-language, bilingual/multilingual subtitles, CRLF/LF, BOM,
    comma/dot milliseconds, extra timing metadata, and missing subtitle numbers.
    """
    content = content.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    lines = content.split("\n")
    result = []
    invalid = 0
    i = 0

    while i < len(lines):
        timing_match = TIMING_RE.match(lines[i])
        if not timing_match:
            i += 1
            continue

        try:
            start = parse_time(timing_match.group(1))
            end = parse_time(timing_match.group(2))
        except ValueError:
            invalid += 1
            i += 1
            continue

        i += 1
        text_lines = []
        while i < len(lines) and not TIMING_RE.match(lines[i]):
            line = lines[i].rstrip()
            if line.strip():
                text_lines.append(line)
            i += 1

        text = "\n".join(text_lines).strip()
        if text and end > start:
            result.append({"start": start, "end": end, "text": text})
        else:
            invalid += 1

    return result, invalid


def probe_video(path):
    ffprobe = find_binary("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,r_frame_rate",
        "-of", "default=noprint_wrappers=1", path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    info = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value
    try:
        info["width"] = int(info.get("width", 0))
        info["height"] = int(info.get("height", 0))
        info["duration"] = float(info.get("duration", 0))
    except (TypeError, ValueError):
        return None
    return info


def ass_escape(text):
    return str(text).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def ass_color(value):
    value = str(value).strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H00{b}{g}{r}"


def ass_time(seconds):
    total = max(0, int(round(float(seconds) * 100)))
    h, rem = divmod(total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def char_width(ch):
    code = ord(ch)
    if (
        0x1100 <= code <= 0x11FF or 0x2E80 <= code <= 0x9FFF or
        0xAC00 <= code <= 0xD7AF or 0xF900 <= code <= 0xFAFF or
        0x20000 <= code <= 0x3FFFF or 0x3040 <= code <= 0x30FF or
        0x0E00 <= code <= 0x0E7F or 0x0600 <= code <= 0x06FF or
        0x0750 <= code <= 0x077F or 0x0590 <= code <= 0x05FF or
        0x0900 <= code <= 0x097F
    ):
        return 2.0
    if ch.isspace() or ch in "ilI.,'`:;!|":
        return 0.45
    if ch in "MW@#%&":
        return 1.35
    return 1.0


def wrap_text(text, max_units):
    text = str(text).strip()
    if not text:
        return ""
    output = []
    for original_line in text.splitlines():
        line = original_line.strip()
        if not line:
            continue
        if sum(char_width(c) for c in line) <= max_units:
            output.append(line)
            continue
        if re.search(r"\s", line):
            tokens = re.findall(r"\S+", line)
            current, width = "", 0.0
            for token in tokens:
                tw = sum(char_width(c) for c in token)
                if current and width + tw + 0.5 > max_units:
                    output.append(current)
                    current, width = "", 0.0
                if tw > max_units:
                    chunk, cw = "", 0.0
                    for ch in token:
                        w = char_width(ch)
                        if chunk and cw + w > max_units:
                            output.append(chunk)
                            chunk, cw = "", 0.0
                        chunk += ch
                        cw += w
                    if current:
                        output.append(current)
                    current, width = chunk, cw
                else:
                    if current:
                        current += " "
                    current += token
                    width += tw + 0.5
            if current:
                output.append(current)
        else:
            current, width = "", 0.0
            for ch in line:
                w = char_width(ch)
                if current and width + w > max_units:
                    output.append(current)
                    current, width = "", 0.0
                current += ch
                width += w
            if current:
                output.append(current)
    return "\n".join(output)


def font_for_char(ch, fallback):
    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "Noto Sans CJK SC"
    if 0x3040 <= code <= 0x30FF:
        return "Noto Sans CJK JP"
    if 0xAC00 <= code <= 0xD7AF:
        return "Noto Sans CJK KR"
    if 0x0900 <= code <= 0x097F:
        return "Noto Sans Devanagari"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
        return "Noto Sans Arabic"
    if 0x0E00 <= code <= 0x0E7F:
        return "Noto Sans Thai"
    if 0x0590 <= code <= 0x05FF:
        return "Noto Sans Hebrew"
    return fallback


def ass_runs(text, fallback):
    chunks = []
    current_font = None
    current = []
    for ch in str(text):
        font = font_for_char(ch, fallback)
        if font != current_font:
            if current:
                chunks.append((current_font, "".join(current)))
            current_font = font
            current = []
        current.append(ch)
    if current:
        chunks.append((current_font, "".join(current)))
    return "".join("{\\fn" + ass_escape(font) + "}" + ass_escape(value) for font, value in chunks)


def build_ass(subtitles, settings, width, height):
    base_w = max(width, 320)
    base_h = max(height, 180)
    scale = base_w / 1920.0
    font_size = max(16, int(round(settings["font_size"] * scale)))
    margin_v = max(10, int(round(settings["margin_v"] * scale)))
    outline = max(0, int(round(settings["outline_width"] * scale)))
    primary = ass_color(settings["text_color"])
    outline_color = ass_color(settings["outline_color"])
    bold = -1 if settings["bold"] else 0
    italic = -1 if settings["italic"] else 0
    underline = -1 if settings["underline"] else 0
    alignment = POSITIONS[settings["position"]]

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {base_w}\nPlayResY: {base_h}\n"
        "ScaledBorderAndShadow: yes\nWrapStyle: 2\nCollisions: Normal\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{settings['font']},{font_size},{primary},&H00000000,{outline_color},&HFF000000,"
        f"{bold},{italic},{underline},0,100,100,0,0,1,{outline},0,{alignment},90,90,{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for item in subtitles:
        text = wrap_text(item["text"], settings["wrap_width"])
        text = ass_runs(text.replace("\r", ""), settings["font"]).replace("\n", r"\N")
        if settings["bold"]:
            text = "{\\b1}" + text
        if settings["italic"]:
            text = "{\\i1}" + text
        if settings["underline"]:
            text = "{\\u1}" + text
        lines.append(
            f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def render_video(input_path, ass_path, output_path):
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Please install FFmpeg.")
    filter_path = ass_path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = "subtitles='" + filter_path + "'"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-shortest", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "FFmpeg subtitle rendering failed.")[-5000:])


st.title("🎬 Video Subtitle Editor")
st.caption("支持单语、双语和多行 SRT；只负责把现有字幕烧录到视频，不重新识别或翻译。")

video = st.file_uploader("① 原始视频", type=VIDEO_TYPES, key="fixed_editor_video")
srt = st.file_uploader("② SRT 字幕", type=["srt"], key="fixed_editor_srt")

if not video or not srt:
    st.info("请同时上传原始视频和 SRT 字幕。")
    st.stop()

work_dir = tempfile.mkdtemp(prefix="video_editor_")
try:
    input_path = os.path.join(work_dir, "input" + (Path(video.name).suffix.lower() or ".mp4"))
    with open(input_path, "wb") as f:
        f.write(video.getbuffer())

    info = probe_video(input_path)
    if not info or not info.get("width") or not info.get("height"):
        st.error("无法读取视频尺寸，请确认视频文件完整且包含视频轨道。")
        st.stop()

    content = decode_srt(srt.getvalue())
    parsed, invalid_count = parse_srt(content)
    if not parsed:
        st.error("没有找到有效的 SRT 字幕条目。请确认字幕包含类似“00:00:00,000 --> 00:00:03,000”的时间轴。")
        st.stop()

    source_key = f"{video.name}:{srt.name}:{len(parsed)}:{hash(content)}"
    if st.session_state.get("fixed_editor_source") != source_key:
        st.session_state.fixed_editor_source = source_key
        st.session_state.fixed_editor_subtitles = parsed

    subtitles = st.session_state.fixed_editor_subtitles
    duration = info.get("duration", 0)
    last_end = max((item["end"] for item in subtitles), default=0)

    if invalid_count:
        st.warning(f"已读取 {len(subtitles)} 条有效字幕；跳过 {invalid_count} 个无效条目。")
    else:
        st.success(f"已读取 {len(subtitles)} 条字幕 · {info['width']}×{info['height']} · 视频 {duration:.2f}s")

    if last_end > duration > 0:
        st.warning(f"字幕最后时间点为 {fmt_time(last_end)}，超过视频时长 {fmt_time(duration)}。渲染时会自动截断到视频结束。")

    st.video(video)

    with st.expander("⏱️ 整体同步校正", expanded=True):
        offset = st.slider(
            "字幕整体偏移（秒）", -10.0, 10.0, 0.0, 0.05,
            help="正数 = 字幕晚出现；负数 = 字幕提前出现。",
        )

    with st.expander("🎨 字幕样式", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            font = st.selectbox("字体", FONT_OPTIONS, index=0)
            font_size = st.slider("字号", 18, 72, 42)
        with c2:
            position = st.selectbox("位置", list(POSITIONS), index=0)
            margin_v = st.slider("垂直边距", 20, 180, 60)
        with c3:
            text_color = st.color_picker("文字颜色", "#FFFFFF")
            outline_color = st.color_picker("描边颜色", "#000000")
            outline_width = st.slider("描边宽度", 0, 8, 3)
        c4, c5, c6 = st.columns(3)
        with c4:
            bold = st.checkbox("粗体", True)
        with c5:
            italic = st.checkbox("斜体", False)
        with c6:
            underline = st.checkbox("下划线", False)
        wrap_width = st.slider("最大字幕宽度", 20, 90, 62)

    with st.expander("✏️ 字幕内容与时间轴", expanded=False):
        edited = []
        for index, item in enumerate(subtitles, start=1):
            st.markdown(f"**#{index} · {fmt_time(item['start'])} → {fmt_time(item['end'])}**")
            c1, c2 = st.columns(2)
            with c1:
                start = st.number_input("开始时间", min_value=0.0, value=float(item["start"]), step=0.01, key=f"start_{index}")
            with c2:
                end = st.number_input("结束时间", min_value=0.0, value=float(item["end"]), step=0.01, key=f"end_{index}")
            text = st.text_area("字幕", value=item["text"], height=90, key=f"text_{index}")
            edited.append({"start": start, "end": end, "text": text})

    settings = {
        "font": font,
        "font_size": font_size,
        "position": position,
        "margin_v": margin_v,
        "text_color": text_color,
        "base_text_color": text_color,
        "outline_color": outline_color,
        "outline_width": outline_width,
        "bold": bold,
        "italic": italic,
        "underline": underline,
        "wrap_width": wrap_width,
    }

    if st.button("🔥 合成字幕视频", type="primary", use_container_width=True):
        final_subtitles = []
        skipped = 0
        for item in edited:
            start = max(0.0, float(item["start"]) + offset)
            end = float(item["end"]) + offset
            if duration > 0:
                start = min(start, duration)
                end = min(end, duration)
            if end > start and str(item["text"]).strip():
                final_subtitles.append({"start": start, "end": end, "text": str(item["text"]).strip()})
            else:
                skipped += 1

        if not final_subtitles:
            st.error("没有可用于合成的有效字幕。")
            st.stop()

        ass_path = os.path.join(work_dir, "subtitles.ass")
        output_path = os.path.join(work_dir, "subtitle_video.mp4")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(build_ass(final_subtitles, settings, info["width"], info["height"]))

        progress = st.progress(0, text="正在将字幕烧录到视频……")
        try:
            render_video(input_path, ass_path, output_path)
            progress.progress(100, text="合成完成")
            st.success(f"视频字幕合成完成，共 {len(final_subtitles)} 条字幕。" + (f" 跳过 {skipped} 条空/无效字幕。" if skipped else ""))
            with open(output_path, "rb") as f:
                output_bytes = f.read()
            st.video(output_bytes)
            st.download_button(
                "⬇️ 下载合成后的视频",
                data=output_bytes,
                file_name=f"{Path(video.name).stem}_subtitled.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
        except Exception as exc:
            progress.empty()
            st.error(f"视频合成失败：{exc}")
finally:
    shutil.rmtree(work_dir, ignore_errors=True)
