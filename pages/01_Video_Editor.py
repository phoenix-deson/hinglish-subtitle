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


def find_binary(name):
    candidates = [
        shutil.which(name), f"/usr/bin/{name}", f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}", f"/opt/conda/bin/{name}",
    ]
    return next((p for p in candidates if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)


def parse_time(value):
    value = value.strip().replace(".", ",")
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", value)
    if not m:
        raise ValueError(f"Invalid timestamp: {value}")
    h, minute, sec, ms = m.groups()
    return int(h) * 3600 + int(minute) * 60 + int(sec) + int(ms.ljust(3, "0")) / 1000


def fmt_time(seconds):
    total = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(total, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(content):
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    result = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.splitlines()
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None or timing + 1 >= len(lines):
            continue
        try:
            left, right = [x.strip().split()[0] for x in lines[timing].split("-->", 1)]
            start, end = parse_time(left), parse_time(right)
        except (ValueError, IndexError):
            continue
        text = "\n".join(x.rstrip() for x in lines[timing + 1:]).strip()
        if text and end > start:
            result.append({"start": start, "end": end, "text": text})
    return result


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
            k, v = line.split("=", 1)
            info[k] = v
    try:
        info["width"] = int(info.get("width", 0))
        info["height"] = int(info.get("height", 0))
        info["duration"] = float(info.get("duration", 0))
    except ValueError:
        return None
    return info


def ass_escape(text):
    return str(text).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def ass_color(value):
    value = str(value).strip().lstrip("#")
    if len(value) != 6:
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


def wrap_line(text, max_units):
    text = str(text).strip()
    if not text:
        return ""
    if "\n" in text:
        return "\n".join(wrap_line(x, max_units) for x in text.splitlines() if x.strip())
    if sum(char_width(c) for c in text) <= max_units:
        return text
    if re.search(r"\s", text):
        tokens = re.findall(r"\S+\s*", text)
        lines, current, width = [], "", 0.0
        for token in tokens:
            token = token.rstrip()
            tw = sum(char_width(c) for c in token)
            if current and width + tw > max_units:
                lines.append(current.rstrip())
                current, width = "", 0.0
            if tw > max_units:
                if current:
                    lines.append(current.rstrip())
                    current, width = "", 0.0
                chunk, cw = "", 0.0
                for ch in token:
                    w = char_width(ch)
                    if chunk and cw + w > max_units:
                        lines.append(chunk)
                        chunk, cw = "", 0.0
                    chunk += ch
                    cw += w
                current, width = chunk, cw
            else:
                current += token + " "
                width += tw
        if current.strip():
            lines.append(current.rstrip())
        return "\n".join(lines)
    lines, current, width = [], "", 0.0
    for ch in text:
        w = char_width(ch)
        if current and width + w > max_units:
            lines.append(current)
            current, width = "", 0.0
        current += ch
        width += w
    if current:
        lines.append(current)
    return "\n".join(lines)


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
    parts, current_font, current = [], None, []
    for ch in str(text):
        font = font_for_char(ch, fallback)
        if font != current_font:
            if current:
                parts.append((current_font, "".join(current)))
            current_font, current = font, []
        current.append(ch)
    if current:
        parts.append((current_font, "".join(current)))
    return "".join("{\\fn" + ass_escape(font) + "}" + ass_escape(value) for font, value in parts)


def build_ass(subtitles, settings, width, height):
    # Keep ASS coordinates proportional to the actual source video instead of assuming 1920x1080.
    base_w, base_h = max(width, 320), max(height, 180)
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
        text = wrap_line(item["text"], settings["wrap_width"])
        text = ass_runs(text.replace("\r", ""), settings["font"]).replace("\n", r"\N")
        if settings["text_color"] != settings["base_text_color"]:
            text = "{\\c" + primary + "}" + text
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
        raise RuntimeError("FFmpeg was not found. packages.txt must contain ffmpeg.")
    # Escape the path for the FFmpeg subtitles filter without changing the real filesystem path.
    filter_path = ass_path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = "subtitles='" + filter_path + "'"
    env = os.environ.copy()
    env.setdefault("FONTCONFIG_PATH", "/etc/fonts")
    env.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or "FFmpeg subtitle rendering failed.")[-4000:])


st.title("🎬 Video Subtitle Editor")
st.caption("专门用于把现有 SRT 与原视频精确结合。原有识别、翻译等模块保持不变。")

video = st.file_uploader("① 原始视频", type=VIDEO_TYPES, key="fixed_editor_video")
srt = st.file_uploader("② SRT 字幕", type=["srt"], key="fixed_editor_srt")

if not video or not srt:
    st.info("请同时上传原始视频和 SRT 字幕。这里不会重新识别或翻译字幕。")
    st.stop()

work_dir = tempfile.mkdtemp(prefix="video_editor_")
try:
    suffix = Path(video.name).suffix.lower() or ".mp4"
    input_path = os.path.join(work_dir, "input" + suffix)
    with open(input_path, "wb") as f:
        f.write(video.getbuffer())
    info = probe_video(input_path)
    if not info or not info.get("width") or not info.get("height"):
        st.error("无法读取视频尺寸，请确认视频文件完整且包含视频轨道。")
        st.stop()

    try:
        parsed = parse_srt(srt.getvalue().decode("utf-8-sig", errors="replace"))
    except Exception as exc:
        st.error(f"SRT 读取失败：{exc}")
        st.stop()
    if not parsed:
        st.error("没有找到有效的 SRT 字幕条目。")
        st.stop()

    source_key = f"{video.name}:{srt.name}:{len(parsed)}"
    if st.session_state.get("fixed_editor_source") != source_key:
        st.session_state.fixed_editor_source = source_key
        st.session_state.fixed_editor_subtitles = parsed

    subtitles = st.session_state.fixed_editor_subtitles
    st.success(f"已载入 {len(subtitles)} 条字幕 · 视频 {info['width']}×{info['height']} · {info.get('duration', 0):.2f}s")

    st.video(video)

    with st.expander("⏱️ 整体同步校正", expanded=True):
        offset = st.slider(
            "字幕整体偏移（秒）",
            min_value=-10.0, max_value=10.0, value=0.0, step=0.05,
            help="正数 = 字幕整体晚一点出现；负数 = 字幕整体提前出现。",
        )
        if offset:
            st.caption(f"当前渲染会将全部字幕整体偏移 {offset:+.2f} 秒。不会修改原始 SRT 文件。")

    st.subheader("✏️ 字幕时间与内容")
    edited = []
    for i, item in enumerate(subtitles, 1):
        with st.expander(f"#{i} · {fmt_time(item['start'] + offset)} → {fmt_time(item['end'] + offset)}", expanded=(i == 1)):
            c1, c2 = st.columns(2)
            start_text = c1.text_input("Start", fmt_time(item["start"]), key=f"fixed_start_{i}")
            end_text = c2.text_input("End", fmt_time(item["end"]), key=f"fixed_end_{i}")
            text = st.text_area("Subtitle text", item["text"], height=90, key=f"fixed_text_{i}")
            try:
                start = parse_time(start_text)
                end = parse_time(end_text)
                if end <= start:
                    st.warning("End 必须晚于 Start。")
                edited.append({"start": start, "end": end, "text": text.strip() or item["text"]})
            except ValueError as exc:
                st.warning(str(exc))
                edited.append(item.copy())
    st.session_state.fixed_editor_subtitles = edited

    st.subheader("🎨 字幕样式")
    a, b, c = st.columns(3)
    with a:
        font = st.selectbox("Font", FONT_OPTIONS, index=0)
        font_size = st.slider("Font size", 18, 96, 42, 2)
        text_color = st.color_picker("Text color", "#FFFFFF")
    with b:
        bold = st.checkbox("Bold", value=True)
        italic = st.checkbox("Italic", value=False)
        underline = st.checkbox("Underline", value=False)
        wrap_width = st.slider("Auto-wrap width", 20, 90, 46, 2)
        highlight = st.checkbox("Text highlight", value=False)
        highlight_color = st.color_picker("Highlight color", "#FFD84D", disabled=not highlight)
    with c:
        position = st.selectbox("Position", list(POSITIONS.keys()), index=0)
        margin_v = st.slider("Vertical margin", 20, 180, 55, 5)
        outline_width = st.slider("Text outline", 0, 8, 2, 1)
        outline_color = st.color_picker("Outline color", "#000000")

    settings = {
        "font": font,
        "font_size": font_size,
        "text_color": highlight_color if highlight else text_color,
        "base_text_color": text_color,
        "outline_color": outline_color,
        "outline_width": outline_width,
        "bold": bold,
        "italic": italic,
        "underline": underline,
        "position": position,
        "margin_v": margin_v,
        "wrap_width": wrap_width,
    }

    st.subheader("🔍 渲染前检查")
    final_subtitles = []
    warnings = []
    duration = float(info.get("duration") or 0)
    for item in edited:
        start = item["start"] + offset
        end = item["end"] + offset
        if duration:
            if end <= 0 or start >= duration:
                warnings.append(f"字幕 {fmt_time(item['start'])} 完全位于视频范围之外，已跳过。")
                continue
            start = max(0, start)
            end = min(duration, end)
        if end <= start:
            continue
        final_subtitles.append({"start": start, "end": end, "text": item["text"]})
    if warnings:
        st.warning("\n".join(warnings[:10]))
    st.write(f"将渲染 **{len(final_subtitles)}** 条字幕。字幕时间会被限制在视频实际时长内。")

    if st.button("🎬 渲染并合成字幕视频", type="primary", use_container_width=True):
        if not final_subtitles:
            st.error("没有可渲染的字幕，请检查时间轴。")
        else:
            ass_path = os.path.join(work_dir, "subtitles.ass")
            output_path = os.path.join(work_dir, "subtitle_edited.mp4")
            try:
                ass = build_ass(final_subtitles, settings, info["width"], info["height"])
                with open(ass_path, "w", encoding="utf-8") as f:
                    f.write(ass)
                progress = st.progress(0, text="准备字幕时间轴…")
                progress.progress(0.25, text="根据原视频分辨率建立字幕画布…")
                progress.progress(0.45, text="使用 FFmpeg 将字幕烧录到视频…")
                render_video(input_path, ass_path, output_path)
                progress.progress(1.0, text="合成完成")
                with open(output_path, "rb") as f:
                    rendered = f.read()
                st.success("🎉 字幕已经与视频合成完成。")
                st.video(rendered)
                st.download_button(
                    "⬇️ 下载合成后的视频",
                    rendered,
                    file_name="subtitle_edited.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"视频合成失败：{exc}")
finally:
    shutil.rmtree(work_dir, ignore_errors=True)
