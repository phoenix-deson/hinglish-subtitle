import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Video Subtitle Editor", page_icon="🎬", layout="wide")

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
FONT_OPTIONS = [
    "Noto Sans",
    "Noto Sans CJK SC",
    "Noto Sans Devanagari",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
    "Helvetica",
]
POSITION_OPTIONS = {
    "Bottom": 2,
    "Middle": 5,
    "Top": 8,
}


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


def format_time(seconds):
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_time(value):
    value = value.strip().replace(".", ",")
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})", value)
    if not match:
        raise ValueError(f"Invalid timestamp: {value}")
    h, m, s, ms = match.groups()
    ms = int(ms.ljust(3, "0"))
    return int(h) * 3600 + int(m) * 60 + int(s) + ms / 1000


def parse_srt(content):
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    blocks = re.split(r"\n\s*\n", content)
    result = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None or timing_index + 1 >= len(lines):
            continue
        try:
            start_text, end_text = [x.strip().split(" ")[0] for x in lines[timing_index].split("-->", 1)]
            start = parse_time(start_text)
            end = parse_time(end_text)
        except Exception:
            continue
        text = " ".join(x.strip() for x in lines[timing_index + 1:] if x.strip())
        if text and end > start:
            result.append({"start": start, "end": end, "text": text})
    return result


def ass_escape(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\n", "\\N")
    )


def ass_time(seconds):
    total_cs = max(0, int(round(float(seconds) * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_color(hex_color):
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H00{b}{g}{r}"


def build_ass(subtitles, settings):
    primary = ass_color(settings["text_color"])
    outline = ass_color(settings["outline_color"])
    font_name = settings["font"]
    font_size = int(settings["font_size"])
    bold = -1 if settings["bold"] else 0
    italic = -1 if settings["italic"] else 0
    underline = -1 if settings["underline"] else 0
    alignment = POSITION_OPTIONS[settings["position"]]
    margin_v = int(settings["margin_v"])
    outline_width = int(settings["outline_width"])

    header = f"""[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\nWrapStyle: 2\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,{font_name},{font_size},{primary},&H00000000,{outline},&H00000000,{bold},{italic},{underline},0,100,100,0,0,1,{outline_width},0,{alignment},70,70,{margin_v},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"""

    lines = [header]
    for item in subtitles:
        text = ass_escape(item["text"])
        if settings["highlight"]:
            text = "{\\c" + ass_color(settings["highlight_color"]) + "}" + text
        if settings["bold"]:
            text = "{\\b1}" + text
        if settings["underline"]:
            text = "{\\u1}" + text
        lines.append(
            f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def run_ffmpeg(video_path, ass_path, output_path):
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Make sure packages.txt contains ffmpeg.")
    video_filter = "subtitles=" + ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "FFmpeg subtitle rendering failed.")[-2500:])


st.markdown(
    """
    <style>
    .editor-hero{padding:1.5rem 1.7rem;border-radius:22px;border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;background:linear-gradient(135deg,rgba(128,128,128,.10),rgba(128,128,128,.03))}
    .editor-hero h1{margin:0;font-size:2.25rem;letter-spacing:-.03em}.editor-hero p{margin:.4rem 0 0;opacity:.7}
    .tip{padding:1rem 1.1rem;border-radius:16px;border:1px solid rgba(128,128,128,.18);background:rgba(128,128,128,.035);margin:.5rem 0 1rem}
    </style>
    <div class="editor-hero"><h1>🎬 Video Subtitle Editor</h1><p>Fine-tune subtitles, style them, and permanently burn them into your original video.</p></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="tip">✨ <b>Clean subtitle rendering</b><br>Subtitles use transparent background with optional text outline only — no black subtitle box. Edit tiny recognition errors before exporting the final video.</div>',
    unsafe_allow_html=True,
)

video = st.file_uploader("Choose original video", type=SUPPORTED_VIDEO_TYPES, key="editor_video")
srt_file = st.file_uploader("Choose prepared SRT subtitle", type=["srt"], key="editor_srt")

if video:
    st.video(video)

if video and srt_file:
    try:
        subtitles = parse_srt(srt_file.getvalue().decode("utf-8-sig", errors="replace"))
    except Exception as exc:
        subtitles = []
        st.error(f"Could not read subtitle file: {exc}")

    if not subtitles:
        st.error("No valid subtitle entries were found in the SRT file.")
    else:
        st.success(f"Loaded {len(subtitles)} subtitle entries. You can edit the text and timing below.")

        if "editor_subtitles" not in st.session_state or st.session_state.get("editor_source_name") != srt_file.name:
            st.session_state.editor_subtitles = subtitles
            st.session_state.editor_source_name = srt_file.name

        edited = []
        st.subheader("✏️ Subtitle editing")
        for i, item in enumerate(st.session_state.editor_subtitles, 1):
            with st.expander(f"#{i}  ·  {format_time(item['start'])} → {format_time(item['end'])}", expanded=(i == 1)):
                a, b = st.columns(2)
                start_text = a.text_input("Start", format_time(item["start"]), key=f"start_{i}")
                end_text = b.text_input("End", format_time(item["end"]), key=f"end_{i}")
                text = st.text_area("Subtitle text", item["text"], height=90, key=f"text_{i}")
                try:
                    start = parse_time(start_text)
                    end = parse_time(end_text)
                    if end <= start:
                        st.warning("End time must be later than start time.")
                except ValueError as exc:
                    start, end = item["start"], item["end"]
                    st.warning(str(exc))
                edited.append({"start": start, "end": end, "text": text.strip() or item["text"]})

        st.session_state.editor_subtitles = edited

        st.subheader("🎨 Subtitle style")
        c1, c2, c3 = st.columns(3)
        with c1:
            font = st.selectbox("Font", FONT_OPTIONS, index=0)
            font_size = st.slider("Font size", 18, 96, 42, 2)
            text_color = st.color_picker("Text color", "#FFFFFF")
        with c2:
            bold = st.checkbox("Bold", value=True)
            italic = st.checkbox("Italic", value=False)
            underline = st.checkbox("Underline", value=False)
            highlight = st.checkbox("Text highlight", value=False, help="Changes text color only; the background remains transparent.")
            highlight_color = st.color_picker("Highlight color", "#FFD84D", disabled=not highlight)
        with c3:
            position = st.selectbox("Position", list(POSITION_OPTIONS.keys()), index=0)
            margin_v = st.slider("Vertical margin", 20, 180, 55, 5)
            outline_width = st.slider("Text outline", 0, 8, 2, 1)
            outline_color = st.color_picker("Outline color", "#000000")

        st.caption("The outline is around the letters only. There is no solid subtitle background box.")

        settings = {
            "font": font,
            "font_size": font_size,
            "text_color": highlight_color if highlight else text_color,
            "outline_color": outline_color,
            "outline_width": outline_width,
            "bold": bold,
            "italic": italic,
            "underline": underline,
            "highlight": False,
            "highlight_color": highlight_color,
            "position": position,
            "margin_v": margin_v,
        }

        st.subheader("🔍 Final preview")
        preview_text = "\n".join(x["text"] for x in edited[:8] if x["text"])
        st.text_area("Edited subtitle preview", preview_text, height=180, disabled=True)

        if st.button("🎬 Burn subtitles into video", type="primary", use_container_width=True):
            work_dir = tempfile.mkdtemp(prefix="subtitle_editor_")
            try:
                suffix = Path(video.name).suffix.lower() or ".mp4"
                input_path = os.path.join(work_dir, "input" + suffix)
                ass_path = os.path.join(work_dir, "subtitles.ass")
                output_path = os.path.join(work_dir, "subtitle_video.mp4")
                with open(input_path, "wb") as f:
                    f.write(video.getbuffer())
                with open(ass_path, "w", encoding="utf-8") as f:
                    f.write(build_ass(edited, settings))

                progress = st.progress(0, text="Preparing video editor…")
                progress.progress(0.25, text="Rendering transparent-background subtitles…")
                run_ffmpeg(input_path, ass_path, output_path)
                progress.progress(1.0, text="Video rendering completed.")

                with open(output_path, "rb") as f:
                    rendered = f.read()
                st.success("🎉 Subtitle video is ready.")
                st.video(rendered)
                st.download_button(
                    "⬇️ Download edited video",
                    rendered,
                    "subtitle_edited_video.mp4",
                    "video/mp4",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Video rendering failed: {exc}")
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
else:
    st.info("Upload both the original video and a prepared SRT file to start editing.")
