import os
import streamlit as st
from faster_whisper import WhisperModel

# 页面基础配置
st.set_page_config(
    page_title="Hinglish Subtitle Generator",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 英印混合（Hinglish）视频自动字幕生成器")
st.markdown("上传你的英语与印地语混合视频，系统将自动识别并生成带有时间轴的字幕。")

# 加载 Whisper 模型（为了适应免费服务器的 CPU 环境，推荐使用 base 或 small 模型，若需更高精度可尝试 medium）
@st.cache_resource
def load_model():
    # 使用 faster-whisper，在 CPU 上运行速度更快
    model_size = "small" 
    return WhisperModel(model_size, device="cpu", compute_type="int8")

with st.spinner("正在加载 AI 模型，请稍候..."):
    model = load_model()

# 文件上传组件
uploaded_file = st.file_uploader("请上传视频文件 (支持 MP4, MOV, MKV)", type=["mp4", "mov", "mkv"])

if uploaded_file is not None:
    # 保存上传的视频到临时文件
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)
    
    if st.button("开始生成字幕 🚀", type="primary"):
        with st.status("正在处理中，请耐心等待...", expanded=True) as status:
            st.write("正在提取音频并进行智能多语种识别（Hinglish）...")
            
            try:
                # 执行转写：language设为 'hi'（印地语模型对印地语/英语混合语种的适应性极强）
                segments, info = model.transcribe(temp_video_path, beam_size=5, language="hi")
                
                subtitle_result = ""
                segment_list = list(segments)
                
                for segment in segment_list:
                    line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    subtitle_result += line
                
                status.update(label="字幕生成成功！", state="complete", expanded=False)
                
                st.success("🎉 识别完成！")
                
                # 展示文本结果
                st.subheader("识别文本结果：")
                st.text_area("字幕内容", subtitle_result, height=300)
                
                # 提供下载 SRT 字幕文件的功能
                srt_content = ""
                for i, segment in enumerate(segment_list, start=1):
                    # 简单转换成 SRT 格式
                    def format_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        millisecs = int((seconds - int(seconds)) * 1000)
                        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
                    
                    start_str = format_time(segment.start)
                    end_str = format_time(segment.end)
                    srt_content += f"{i}\n{start_str} --> {end_str}\n{segment.text.strip()}\n\n"
                
                st.download_button(
                    label="下载 SRT 字幕文件",
                    data=srt_content,
                    file_name="subtitle.srt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"处理过程中出错: {e}")
            
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
