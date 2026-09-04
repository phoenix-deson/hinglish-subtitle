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

st.set_page_config(page_title="Hinglish Subtitle Studio", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# Recognition is frozen in this version. Do not change these settings without a dedicated accuracy test.
MODEL_SIZE = "medium"
SUPPORTED_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
CPU_THREADS = max(1, min(2, os.cpu_count() or 2))

TARGET_LANGUAGES = {
    "Simplified Chinese": "zh-CN", "Traditional Chinese": "zh-TW", "English": "en",
    "Japanese": "ja", "Korean": "ko", "Spanish": "es", "French": "fr", "German": "de",
    "Portuguese": "pt", "Russian": "ru", "Arabic": "ar", "Indonesian": "id", "Hindi": "hi",
    "Italian": "it", "Turkish": "tr", "Vietnamese": "vi", "Thai": "th",
}
LANGUAGE_NAMES = {"en":"English","hi":"Hindi","zh":"Chinese","zh-CN":"Simplified Chinese","zh-TW":"Traditional Chinese","ja":"Japanese","ko":"Korean","es":"Spanish","fr":"French","de":"German","pt":"Portuguese","ru":"Russian","ar":"Arabic","id":"Indonesian","it":"Italian","tr":"Turkish","vi":"Vietnamese","th":"Thai"}
LIBRETRANSLATE_ENDPOINTS = ["https://translate.argosopentech.com", "https://translate.terraprint.co", "https://lt.vern.cc"]
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

st.markdown("""
<style>
.hero{padding:1.5rem 1.7rem;border-radius:22px;border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;background:linear-gradient(135deg,rgba(128,128,128,.10),rgba(128,128,128,.03))}.hero h1{margin:0;font-size:2.3rem;letter-spacing:-.03em}.hero p{margin:.4rem 0 0;opacity:.7}.feature{padding:1rem 1.05rem;border-radius:16px;border:1px solid rgba(128,128,128,.18);min-height:95px;background:rgba(128,128,128,.035)}.feature b{font-size:1.03rem}.ai-card{padding:1rem;border-radius:16px;border:1px solid rgba(128,128,128,.22);background:rgba(128,128,128,.035);margin:.5rem 0 1rem}
</style>
<div class="hero"><h1>🎬 Hinglish Subtitle Studio</h1><p>Accurate Hinglish transcription · AI correction · multilingual translation · clean subtitle export</p></div>
""", unsafe_allow_html=True)


def find_binary(name):
    candidates=[shutil.which(name),f"/usr/bin/{name}",f"/usr/local/bin/{name}",f"/opt/homebrew/bin/{name}",f"/opt/conda/bin/{name}"]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path,os.X_OK): return path
    return None

def find_ffmpeg(): return find_binary("ffmpeg")
def find_ffprobe(): return find_binary("ffprobe")

@st.cache_resource(show_spinner=False)
def load_model():
    return WhisperModel(MODEL_SIZE,device="cpu",compute_type="int8",cpu_threads=CPU_THREADS,num_workers=1)

def clean_text(text): return re.sub(r"\s+"," ",text or "").strip()
def normalize_for_comparison(text): return re.sub(r"[^\w\u0900-\u097f]+","",clean_text(text).lower())
def format_time(seconds):
    total_ms=max(0,int(round(float(seconds)*1000))); hours,rem=divmod(total_ms,3600000); minutes,rem=divmod(rem,60000); seconds,millis=divmod(rem,1000); return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
def parse_srt_time(value):
    match=re.match(r"(\d+):(\d+):(\d+),(\d+)",value.strip())
    if not match: raise ValueError(f"Invalid SRT timestamp: {value}")
    h,m,s,ms=map(int,match.groups()); return h*3600+m*60+s+ms/1000

def parse_srt(content):
    blocks=re.split(r"\n\s*\n",content.replace("\r\n","\n").strip()); segments=[]
    for block in blocks:
        lines=block.splitlines()
        if len(lines)<3: continue
        timing_index=next((i for i,line in enumerate(lines) if "-->" in line),None)
        if timing_index is None or timing_index+1>=len(lines): continue
        try: start_text,end_text=[x.strip() for x in lines[timing_index].split("-->",1)]; start=parse_srt_time(start_text); end=parse_srt_time(end_text)
        except ValueError: continue
        text=clean_text(" ".join(lines[timing_index+1:]))
        if text and end>start: segments.append((start,end,text))
    return segments

def make_srt(segments):
    return "\n".join(f"{i}\n{format_time(s)} --> {format_time(e)}\n{t}\n" for i,(s,e,t) in enumerate(segments,1))
def make_bilingual_srt(source,translated):
    return "\n".join(f"{i}\n{format_time(s)} --> {format_time(e)}\n{src}\n{dst}\n" for i,((s,e,src),(_,_,dst)) in enumerate(zip(source,translated),1))
def make_txt(segments): return "\n".join(f"[{format_time(s)} --> {format_time(e)}] {t}" for s,e,t in segments)
def make_bilingual_txt(source,translated): return "\n".join(f"[{format_time(s)} --> {format_time(e)}]\n{src}\n{dst}\n" for (s,e,src),(_,_,dst) in zip(source,translated))

def get_media_duration(media_path):
    ffprobe=find_ffprobe()
    if not ffprobe:return None
    r=subprocess.run([ffprobe,"-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(media_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if r.returncode!=0:return None
    try:
        d=float(r.stdout.strip()); return d if d>0 else None
    except (TypeError,ValueError): return None

def extract_wav(video_path,wav_path):
    ffmpeg=find_ffmpeg()
    if not ffmpeg: raise RuntimeError("FFmpeg was not found on the server. Make sure packages.txt contains ffmpeg and redeploy.")
    r=subprocess.run([ffmpeg,"-y","-hide_banner","-loglevel","error","-i",str(video_path),"-vn","-map","0:a:0?","-ac","1","-ar","16000","-c:a","pcm_s16le",str(wav_path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if r.returncode!=0: raise RuntimeError("FFmpeg could not extract the audio.\n"+(r.stderr or "Unknown FFmpeg error.")[-1500:])
    if not os.path.exists(wav_path) or os.path.getsize(wav_path)==0: raise RuntimeError("The video does not contain a readable audio track.")

def filter_hallucinations(raw_segments,media_duration=None):
    accepted=[]; recent_texts=[]; repeat_streak=0; last_normalized=""
    for segment in raw_segments:
        text=clean_text(segment.text)
        if not text: continue
        start=max(0.0,float(segment.start)); end=max(start,float(segment.end))
        if media_duration is not None:
            if start>=media_duration: continue
            end=min(end,media_duration)
        if end<=start: continue
        avg_logprob=float(getattr(segment,"avg_logprob",0.0)); no_speech_prob=float(getattr(segment,"no_speech_prob",0.0)); compression_ratio=float(getattr(segment,"compression_ratio",0.0))
        if no_speech_prob>=0.75 and avg_logprob<-0.8: continue
        normalized=normalize_for_comparison(text)
        if normalized:
            similarity=SequenceMatcher(None,normalized,last_normalized).ratio()
            if normalized==last_normalized or similarity>=0.92: repeat_streak+=1
            else: repeat_streak=0
            last_normalized=normalized
            if repeat_streak>=2: break
        if compression_ratio>=3.0 and len(normalized)>20: continue
        accepted.append((start,end,text)); recent_texts.append(normalized)
        if len(recent_texts)>5: recent_texts.pop(0)
    if len(accepted)>=3:
        cleaned=[]
        for item in accepted:
            if cleaned:
                p=normalize_for_comparison(cleaned[-1][2]); c=normalize_for_comparison(item[2])
                if c and p and SequenceMatcher(None,p,c).ratio()>=0.92: continue
            cleaned.append(item)
        accepted=cleaned
    return accepted

def recognition_pipeline(uploaded_file):
    suffix=Path(uploaded_file.name).suffix.lower(); work_dir=tempfile.mkdtemp(prefix="hinglish_"); video_path=os.path.join(work_dir,"input"+suffix); wav_path=os.path.join(work_dir,"audio.wav")
    try:
        with open(video_path,"wb") as f:f.write(uploaded_file.getbuffer())
        media_duration=get_media_duration(video_path); extract_wav(video_path,wav_path); model=load_model()
        segments,info=model.transcribe(wav_path,language=None,task="transcribe",beam_size=5,best_of=5,patience=1,temperature=(0.0,0.2,0.4),initial_prompt=("This is natural Hinglish speech: Hindi and English are mixed in the same sentence. Keep Hindi words in Devanagari and keep English words, names, technical terms, and common English expressions in Latin/English script. Do not transliterate English words into Devanagari when the speaker is speaking English. Preserve natural code-switching."),vad_filter=True,vad_parameters={"min_silence_duration_ms":1000,"speech_pad_ms":400},condition_on_previous_text=False,compression_ratio_threshold=2.4,log_prob_threshold=-1.0,no_speech_threshold=0.6,repetition_penalty=1.05,no_repeat_ngram_size=3,word_timestamps=False,multilingual=False)
        raw=[]
        for segment in segments:
            raw.append(segment); yield "segment",segment,media_duration,info,len(raw)
        final=filter_hallucinations(raw,media_duration)
        if media_duration is not None: final=[(s,min(e,media_duration),t) for s,e,t in final if s<media_duration]
        if not final: raise RuntimeError("No reliable speech was detected in the uploaded video.")
        yield "done",final,media_duration,info,len(raw)
    finally: shutil.rmtree(work_dir,ignore_errors=True)

# ---------------- AI / translation helpers ----------------
def get_deepseek_key():
    try:
        key=st.secrets.get("DEEPSEEK_API_KEY")
        if key:return str(key).strip()
    except Exception:
        pass
    key=os.environ.get("DEEPSEEK_API_KEY")
    return str(key).strip() if key else None

def deepseek_request(messages,max_tokens=3000,temperature=0.1):
    key=get_deepseek_key()
    if not key: raise RuntimeError("DeepSeek API key is not configured. Add DEEPSEEK_API_KEY in Streamlit Cloud Secrets.")
    r=requests.post(DEEPSEEK_URL,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},json={"model":DEEPSEEK_MODEL,"messages":messages,"temperature":temperature,"max_tokens":max_tokens},timeout=90)
    if r.status_code>=400: raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text[:800]}")
    data=r.json(); choices=data.get("choices") or []
    if not choices: raise RuntimeError("DeepSeek returned no response.")
    return choices[0]["message"]["content"]

def extract_json_array(text):
    text=text.strip(); text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I|re.S).strip()
    start=text.find("["); end=text.rfind("]")
    if start<0 or end<start: raise RuntimeError("DeepSeek returned an invalid structured response.")
    import json
    return json.loads(text[start:end+1])

def batch_items(items,batch_size=8):
    for i in range(0,len(items),batch_size): yield i,items[i:i+batch_size]

def ai_correct_segments(segments,progress_callback=None):
    import json
    result=[]; total=len(segments)
    for offset,batch in batch_items(segments,8):
        payload=[{"id":i,"text":text} for i,(_,_,text) in enumerate(batch)]
        prompt=("You are a professional subtitle editor correcting an automatic Whisper transcript. The speech may be multilingual Hinglish: Hindi and English can appear in the same sentence. Correct obvious recognition errors, spelling, grammar and punctuation using context. Preserve the speaker's actual meaning and do not invent content. Keep Hindi words in Devanagari and English words, names, brands and common English expressions in Latin script. Do not translate. If a line is already correct, keep it nearly unchanged. Return ONLY a JSON array of objects with keys id and text, in the same order.\n\nSUBTITLES:\n"+json.dumps(payload,ensure_ascii=False))
        raw=deepseek_request([{"role":"system","content":"You are a careful multilingual subtitle proofreader."},{"role":"user","content":prompt}],max_tokens=4500,temperature=0.05)
        corrected=extract_json_array(raw)
        by_id={int(x.get("id")):clean_text(str(x.get("text",""))) for x in corrected if isinstance(x,dict) and str(x.get("id","")).isdigit()}
        for i,(s,e,text) in enumerate(batch): result.append((s,e,by_id.get(i,text)))
        if progress_callback: progress_callback(min(offset+len(batch),total),total,result[-1][2])
    return result

def detect_script_language(text):
    counts={"hi":len(re.findall(r"[\u0900-\u097f]",text)),"zh":len(re.findall(r"[\u4e00-\u9fff]",text)),"ja":len(re.findall(r"[\u3040-\u30ff]",text)),"ko":len(re.findall(r"[\uac00-\ud7af]",text)),"ar":len(re.findall(r"[\u0600-\u06ff]",text)),"th":len(re.findall(r"[\u0e00-\u0e7f]",text)),"ru":len(re.findall(r"[\u0400-\u04ff]",text))}
    if not text.strip():return "unknown"
    best=max(counts,key=counts.get)
    if counts[best]>0:return best
    if re.search(r"[A-Za-z]",text):return "en"
    return "unknown"

def detect_languages(segments):
    found={}
    for _,_,text in segments:
        code=detect_script_language(text)
        found[code]=found.get(code,0)+1
    return found

def google_translate(text,target):
    r=requests.get("https://translate.googleapis.com/translate_a/single",params={"client":"gtx","sl":"auto","tl":target,"dt":"t","q":text[:4500]},timeout=8); r.raise_for_status(); data=r.json(); parts=data[0] if isinstance(data,list) and data else []; out="".join(p[0] for p in parts if isinstance(p,list) and p and p[0]);
    if not out:raise RuntimeError("Google Translate returned an empty result")
    return clean_text(out)

def libre_translate(text,target):
    target="zh" if target.startswith("zh-") else target; last=None
    for endpoint in LIBRETRANSLATE_ENDPOINTS:
        try:
            r=requests.post(endpoint+"/translate",data={"q":text[:4500],"source":"auto","target":target,"format":"text"},timeout=8); r.raise_for_status(); out=clean_text(r.json().get("translatedText",""));
            if out:return out
            last="Empty translation returned"
        except Exception as e:last=e
    raise RuntimeError(str(last or "LibreTranslate unavailable"))

def free_translate_text(text,target):
    errors=[]
    for fn in (google_translate,libre_translate):
        try:return fn(text,target)
        except Exception as e:errors.append(str(e))
    raise RuntimeError("Free translation engines failed: "+" | ".join(errors)[-900:])

def ai_translate_segments(segments,target_label,target_code,progress_callback=None):
    import json
    result=[]; total=len(segments)
    for offset,batch in batch_items(segments,8):
        payload=[{"id":i,"text":text} for i,(_,_,text) in enumerate(batch)]
        prompt=(f"Translate these subtitle lines into {target_label}. The source may contain multiple languages and Hinglish. Translate the meaning naturally into the target language; do not require the whole subtitle file to have one source language. Preserve names and culturally important proper nouns unless a standard target-language form exists. Do not add explanations. Return ONLY a JSON array of objects with keys id and text, in the same order.\n\nSUBTITLES:\n"+json.dumps(payload,ensure_ascii=False))
        raw=deepseek_request([{ "role":"system","content":"You are an expert professional subtitle translator. Accuracy and natural spoken style are more important than literal word-for-word translation."},{"role":"user","content":prompt}],max_tokens=5000,temperature=0.2)
        translated=extract_json_array(raw); by_id={int(x.get("id")):clean_text(str(x.get("text",""))) for x in translated if isinstance(x,dict) and str(x.get("id","")).isdigit()}
        for i,(s,e,text) in enumerate(batch): result.append((s,e,by_id.get(i,text)))
        if progress_callback:progress_callback(min(offset+len(batch),total),total,result[-1][2])
    return result

def free_translate_segments(segments,target_code,progress_callback=None):
    result=[]; total=len(segments)
    for i,(s,e,text) in enumerate(segments,1):
        out=free_translate_text(text,target_code); result.append((s,e,out))
        if progress_callback:progress_callback(i,total,out)
    return result

# ---------------- Session state ----------------
for key,default in [("segments",None),("srt_content",None),("txt_content",None),("corrected_segments",None),("translated_segments",None),("translation_source",None),("translation_engine",None)]:
    if key not in st.session_state:st.session_state[key]=default

with st.sidebar:
    st.markdown("### ✨ Studio")
    st.markdown("**1. Speech Recognition**\n\nWhisper medium · fixed accuracy baseline")
    st.markdown("**2. AI Correction**\n\nDeepSeek · optional proofreading")
    st.markdown("**3. Translation**\n\nFree engines by default · DeepSeek AI optional")
    st.divider(); st.caption("Recognition is frozen. AI correction and translation are separate optional layers.")

recognition_tab,correction_tab,translation_tab,video_editor_tab=st.tabs(["🎙️ Speech Recognition","✨ AI Correction","🌐 Translation","🎬 Video Editor"])

with recognition_tab:
    cols=st.columns(3)
    for col,title,desc in zip(cols,["🎙️ Hinglish-aware","🛡️ Hallucination protection","⚡ Resource controlled"],["Hindi + English code-switching","Repeated-output filtering","CPU int8 · 2 threads"]):
        with col:st.markdown(f'<div class="feature">{title}<br><small>{desc}</small></div>',unsafe_allow_html=True)
    uploaded_file=st.file_uploader("Upload a video",type=SUPPORTED_TYPES,help="MP4, MOV, MKV, WebM, AVI and M4V are supported.",key="video_upload")
    if uploaded_file is not None:
        st.info(f"📁 {uploaded_file.name} · {len(uploaded_file.getbuffer())/1024/1024:.1f} MB")
        try:st.video(uploaded_file)
        except Exception:st.caption("Video preview is unavailable, but the file can still be processed.")
        if st.button("🚀 Generate High-Accuracy Subtitles",type="primary",use_container_width=True,key="recognize_button"):
            try:
                with st.status("Processing video…",expanded=True) as status:
                    status.write("📥 Video saved."); status.write("🎵 Extracting 16 kHz mono PCM WAV with FFmpeg…")
                    progress=st.progress(0,text="Preparing recognition…"); detail=st.empty(); preview=st.empty(); started=time.time(); duration=None; raw_count=0; info=None
                    for event,payload,dur,model_info,count in recognition_pipeline(uploaded_file):
                        if event=="segment":
                            seg=payload; duration=dur; info=model_info; raw_count=count; end=max(0,float(seg.end)); elapsed=max(.1,time.time()-started)
                            if duration:
                                ratio=min(1,end/duration); progress.progress(ratio,text=f"Recognizing… {format_time(end)} / {format_time(duration)} ({ratio*100:.1f}%)"); speed=end/elapsed; eta=(duration-end)/speed if speed>0 else 0; detail.info(f"🎙️ {raw_count} segments · {speed:.2f}× real-time · ETA ~{int(eta//60)}m {int(eta%60)}s")
                            preview.caption(f"Latest recognition: {clean_text(seg.text)}")
                        else: final=payload; duration=dur; info=model_info; raw_count=count
                    if duration:status.write(f"⏱️ Video duration: {format_time(duration)}")
                    status.write("✅ WAV extraction completed."); status.write(f"🧠 Whisper {MODEL_SIZE} model loaded and recognition completed."); progress.progress(1,text="Recognition finished. Validating transcript…"); detail.success(f"Whisper returned {raw_count} raw segments. Hallucination and timestamp checks completed."); status.update(label="✅ High-accuracy subtitle generation completed",state="complete")
                st.session_state.segments=final; st.session_state.corrected_segments=None; st.session_state.translated_segments=None; st.session_state.srt_content=make_srt(final); st.session_state.txt_content=make_txt(final); st.success("🎉 Done! The original recognition result is ready.")
            except Exception as e:st.error(f"❌ Processing failed: {e}")
    if st.session_state.srt_content:
        st.subheader("📝 Original Recognition Result"); c=st.columns(3); c[0].metric("Subtitle segments",len(st.session_state.segments)); c[1].metric("Model","Whisper medium"); c[2].metric("Output","SRT + TXT")
        st.text_area("Transcript",st.session_state.txt_content,height=360,key="recognition_preview")
        a,b=st.columns(2)
        with a:st.download_button("⬇️ Download Original SRT",st.session_state.srt_content,"hinglish_original.srt","application/x-subrip",use_container_width=True)
        with b:st.download_button("⬇️ Download Original TXT",st.session_state.txt_content,"hinglish_original.txt","text/plain",use_container_width=True)

with correction_tab:
    st.subheader("✨ AI Correction")
    st.caption("DeepSeek reviews the Whisper result without changing timestamps. The original transcript is always kept, so you can choose either version later.")
    if not st.session_state.segments:
        st.info("Generate subtitles first in Speech Recognition.")
    else:
        st.markdown('<div class="ai-card">🤖 <b>DeepSeek subtitle proofreading</b><br>Corrects obvious recognition errors, grammar, punctuation and mixed Hindi/English script while avoiding translation or invented content.</div>',unsafe_allow_html=True)
        if st.session_state.corrected_segments:
            st.success(f"AI correction is ready: {len(st.session_state.corrected_segments)} subtitle entries.")
            st.text_area("AI-corrected preview",make_txt(st.session_state.corrected_segments),height=380,key="corrected_preview")
            x,y=st.columns(2)
            with x:st.download_button("⬇️ Download AI-Corrected SRT",make_srt(st.session_state.corrected_segments),"hinglish_ai_corrected.srt","application/x-subrip",use_container_width=True)
            with y:st.download_button("⬇️ Download AI-Corrected TXT",make_txt(st.session_state.corrected_segments),"hinglish_ai_corrected.txt","text/plain",use_container_width=True)
        if st.button("✨ AI Correct Subtitles with DeepSeek",type="primary",use_container_width=True,key="ai_correct"):
            try:
                p=st.progress(0,text="AI correction starting…"); d=st.empty(); start=time.time()
                def cb(i,total,latest):p.progress(i/total,text=f"AI correcting… {i}/{total} ({i/total*100:.1f}%)");d.caption(f"Latest correction: {latest}")
                st.session_state.corrected_segments=ai_correct_segments(st.session_state.segments,cb); st.session_state.translated_segments=None; st.success(f"DeepSeek corrected {len(st.session_state.corrected_segments)} subtitle entries in {time.time()-start:.1f}s. Review it before using it for translation.")
                st.rerun()
            except Exception as e:st.error(f"AI correction failed: {e}")

with translation_tab:
    st.subheader("🌐 Subtitle Translation")
    st.caption("The subtitle can contain multiple languages. Each line is translated into one target language. Choose the original or AI-corrected version as the source.")
    uploaded_srt=st.file_uploader("Or upload an existing SRT subtitle file",type=["srt"],key="translation_srt_upload")
    active=None; source_name=""
    if uploaded_srt is not None:
        try:active=parse_srt(uploaded_srt.getvalue().decode("utf-8-sig",errors="replace"));source_name=uploaded_srt.name
        except Exception as e:st.error(f"Could not read SRT: {e}")
    elif st.session_state.segments:
        options=["Original recognition"]+(["AI-corrected"] if st.session_state.corrected_segments else [])
        choice=st.radio("Source subtitles",options,horizontal=True,key="translation_source_choice"); active=st.session_state.corrected_segments if choice=="AI-corrected" else st.session_state.segments; source_name="AI-corrected subtitles" if choice=="AI-corrected" else "Original recognition"
    if active:
        langs=detect_languages(active); parts=[f"{LANGUAGE_NAMES.get(k,k)} ({v})" for k,v in sorted(langs.items(),key=lambda x:-x[1])]; c=st.columns(2); c[0].metric("Subtitle entries",len(active)); c[1].metric("Detected languages"," · ".join(parts) if parts else "Unknown")
        target_label=st.selectbox("Translate subtitles into",list(TARGET_LANGUAGES.keys()),index=0,key="target_language"); target_code=TARGET_LANGUAGES[target_label]
        engine=st.selectbox("Translation engine",["🆓 Free — Google / LibreTranslate","🤖 DeepSeek AI"],index=0,key="translation_engine_choice")
        st.caption("Free mode is the default and does not use your DeepSeek API key. DeepSeek AI is optional and uses your own key only when selected.")
        if st.button("🌐 Translate Subtitles",type="primary",use_container_width=True,key="translate_button"):
            try:
                p=st.progress(0,text="Starting translation…");d=st.empty();started=time.time()
                def cb(i,total,latest):p.progress(i/total,text=f"Translating… {i}/{total} ({i/total*100:.1f}%)");d.caption(f"Latest translation: {latest}")
                if engine.startswith("🤖"):translated=ai_translate_segments(active,target_label,target_code,cb); engine_name="DeepSeek AI"
                else:translated=free_translate_segments(active,target_code,cb); engine_name="Free Google / LibreTranslate"
                st.session_state.translated_segments=translated;st.session_state.translation_source=active;st.session_state.translation_engine=engine_name
                p.progress(1,text="Translation completed");st.success(f"Translated {len(translated)} subtitle entries with {engine_name} in {time.time()-started:.1f}s.")
            except Exception as e:st.error(f"Translation failed: {e}")
    else:st.info("Generate subtitles first, or upload an SRT file above.")
    if st.session_state.translated_segments and st.session_state.translation_source:
        src=st.session_state.translation_source; dst=st.session_state.translated_segments
        st.divider();st.subheader("📦 Export")
        export_mode=st.radio("Choose export version",["Translated only","Original + translated (two lines)"],horizontal=True,key="export_mode")
        if export_mode=="Translated only":
            srt=make_srt(dst); txt=make_txt(dst); srt_name="translated_only.srt"
        else:
            srt=make_bilingual_srt(src,dst);txt=make_bilingual_txt(src,dst);srt_name="original_and_translated.srt"
        st.text_area("Export preview",srt,height=360)
        a,b=st.columns(2)
        with a:st.download_button("⬇️ Download SRT",srt,srt_name,"application/x-subrip",use_container_width=True)
        with b:st.download_button("⬇️ Download TXT",txt,srt_name.replace(".srt",".txt"),"text/plain",use_container_width=True)

st.divider();st.caption("Hinglish Subtitle Studio · Whisper medium + FFmpeg + faster-whisper · AI: DeepSeek · Free translation: Google / LibreTranslate")

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
FONT_OPTIONS = ["Noto Sans", "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans Devanagari", "Noto Sans Arabic", "Noto Sans Thai", "Noto Sans Hebrew", "DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"]
POSITION_OPTIONS = {"Bottom": 2, "Middle": 5, "Top": 8}


def editor_find_binary(name):
    candidates = [shutil.which(name), f"/usr/bin/{name}", f"/usr/local/bin/{name}", f"/opt/homebrew/bin/{name}", f"/opt/conda/bin/{name}"]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path,os.X_OK): return path
    return None


def editor_format_time(seconds):
    total_ms=max(0,int(round(float(seconds)*1000))); hours,rem=divmod(total_ms,3600000); minutes,rem=divmod(rem,60000); secs,millis=divmod(rem,1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def editor_parse_time(value):
    value=value.strip().replace(".",",")
    match=re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})",value)
    if not match: raise ValueError(f"Invalid timestamp: {value}")
    h,m,s,ms=match.groups(); ms=int(ms.ljust(3,"0")[:3])
    return int(h)*3600+int(m)*60+int(s)+ms/1000


def editor_parse_srt(content):
    content=content.replace("\ufeff","").replace("\r\n","\n").replace("\r","\n")
    lines=content.split("\n")
    result=[]
    i=0
    timing_re=re.compile(r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{1,3})(?:\s+.*)?$")
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1
            continue
        if "-->" not in line:
            i+=1
            continue
        match=timing_re.match(line)
        if not match:
            i+=1
            continue
        try:
            start=editor_parse_time(match.group(1)); end=editor_parse_time(match.group(2))
        except ValueError:
            i+=1
            continue
        i+=1
        text_lines=[]
        while i<len(lines):
            current=lines[i]
            if not current.strip():
                i+=1
                break
            if timing_re.match(current.strip()):
                break
            text_lines.append(current.rstrip())
            i+=1
        text="\n".join(text_lines).strip()
        if text and end>start:
            result.append({"start":start,"end":end,"text":text})
    return result


def editor_probe_video(path):
    ffprobe=editor_find_binary("ffprobe")
    if not ffprobe: return None
    command=[ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=width,height,duration,r_frame_rate","-show_entries","format=duration","-of","default=noprint_wrappers=1",path]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if result.returncode!=0:return None
    info={}
    for line in result.stdout.splitlines():
        if "=" in line:
            key,value=line.split("=",1); info[key]=value
    try:
        info["width"]=int(info.get("width",0)); info["height"]=int(info.get("height",0)); info["duration"]=float(info.get("duration",0) or 0)
    except ValueError:return None
    return info


def editor_ass_escape(text):
    return str(text).replace("\\",r"\\").replace("{",r"\{").replace("}",r"\}")


def editor_ass_time(seconds):
    total_cs=max(0,int(round(float(seconds)*100))); hours,rem=divmod(total_cs,360000); minutes,rem=divmod(rem,6000); secs,centis=divmod(rem,100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def editor_ass_color(hex_color):
    value=str(hex_color).strip().lstrip("#")
    if len(value)!=6: value="FFFFFF"
    r,g,b=value[0:2],value[2:4],value[4:6]
    return f"&H00{b}{g}{r}"


def editor_char_width(char):
    code=ord(char)
    if (0x1100<=code<=0x11FF or 0x2E80<=code<=0x9FFF or 0xAC00<=code<=0xD7AF or 0xF900<=code<=0xFAFF or 0x20000<=code<=0x3FFFF or 0x3040<=code<=0x30FF or 0x0E00<=code<=0x0E7F or 0x0600<=code<=0x06FF or 0x0750<=code<=0x077F or 0x0590<=code<=0x05FF or 0x0900<=code<=0x097F): return 2.0
    if char.isspace(): return 0.45
    if char in "ilI.,'`:;!|": return 0.45
    if char in "MW@#%&": return 1.35
    return 1.0


def editor_wrap_line(text,max_units):
    text=str(text).strip()
    if not text: return ""
    max_units=max(10,int(max_units))
    if "\n" in text: return "\n".join(editor_wrap_line(part,max_units) for part in text.split("\n"))
    if sum(editor_char_width(c) for c in text)<=max_units: return text
    if re.search(r"\s",text):
        tokens=re.findall(r"\S+\s*",text); lines=[]; current=""; width=0.0
        for token in tokens:
            token_clean=token.rstrip(); token_width=sum(editor_char_width(c) for c in token_clean)
            if current and width+token_width>max_units: lines.append(current.rstrip()); current=""; width=0.0
            if token_width>max_units:
                if current: lines.append(current.rstrip()); current=""; width=0.0
                chunk=""; chunk_width=0.0
                for char in token_clean:
                    cw=editor_char_width(char)
                    if chunk and chunk_width+cw>max_units: lines.append(chunk); chunk=""; chunk_width=0.0
                    chunk+=char; chunk_width+=cw
                current, width=chunk, chunk_width
            else: current+=token_clean+" "; width+=token_width
        if current.strip(): lines.append(current.rstrip())
        return "\n".join(lines)
    lines=[]; current=""; width=0.0
    for char in text:
        cw=editor_char_width(char)
        if current and width+cw>max_units: lines.append(current); current=""; width=0.0
        current+=char; width+=cw
    if current: lines.append(current)
    return "\n".join(lines)


def editor_wrap_bilingual(text,max_units):
    physical_lines=str(text).replace("\r\n","\n").replace("\r","\n").split("\n")
    return "\n".join(editor_wrap_line(part,max_units) for part in physical_lines if part.strip())


def editor_font_for_char(char,fallback):
    code=ord(char)
    if 0x4E00<=code<=0x9FFF or 0x3400<=code<=0x4DBF: return "Noto Sans CJK SC"
    if 0x3040<=code<=0x30FF: return "Noto Sans CJK JP"
    if 0xAC00<=code<=0xD7AF: return "Noto Sans CJK KR"
    if 0x0900<=code<=0x097F: return "Noto Sans Devanagari"
    if 0x0600<=code<=0x06FF or 0x0750<=code<=0x077F: return "Noto Sans Arabic"
    if 0x0E00<=code<=0x0E7F: return "Noto Sans Thai"
    if 0x0590<=code<=0x05FF: return "Noto Sans Hebrew"
    return fallback


def editor_ass_runs(text,base_font):
    parts=[]; current_font=None; current_text=[]
    for char in str(text):
        font=editor_font_for_char(char,base_font)
        if font!=current_font:
            if current_text: parts.append((current_font,"".join(current_text)))
            current_font=font; current_text=[]
        current_text.append(char)
    if current_text: parts.append((current_font,"".join(current_text)))
    return "".join("{\\fn"+editor_ass_escape(font)+"}"+editor_ass_escape(value) for font,value in parts)


def editor_build_ass(subtitles,settings,width,height,duration):
    canvas_w=max(320,int(width or 1920)); canvas_h=max(180,int(height or 1080)); scale=canvas_w/1920.0
    font_name=settings["font"]; font_size=max(12,int(round(settings["font_size"]*scale))); margin_v=max(8,int(round(settings["margin_v"]*scale))); outline_width=max(0,int(round(settings["outline_width"]*scale)))
    primary=editor_ass_color(settings["text_color"]); outline=editor_ass_color(settings["outline_color"]); bold=-1 if settings["bold"] else 0; italic=-1 if settings["italic"] else 0; underline=-1 if settings["underline"] else 0; alignment=POSITION_OPTIONS[settings["position"]]; max_units=int(settings["wrap_width"])
    header=("[Script Info]\nScriptType: v4.00+\n"+f"PlayResX: {canvas_w}\nPlayResY: {canvas_h}\nScaledBorderAndShadow: yes\nWrapStyle: 2\nCollisions: Normal\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"+f"Style: Default,{font_name},{font_size},{primary},&H00000000,{outline},&HFF000000,{bold},{italic},{underline},0,100,100,0,0,1,{outline_width},0,{alignment},90,90,{margin_v},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    lines=[header]
    for item in subtitles:
        start=max(0.0,float(item["start"])); end=max(start,float(item["end"]))
        if duration>0:
            if start>=duration: continue
            end=min(end,duration)
        if end<=start: continue
        text=editor_ass_runs(editor_wrap_bilingual(item["text"],max_units),font_name)
        if settings["text_color"]!=settings["base_text_color"]: text="{\\c"+editor_ass_color(settings["text_color"])+"}"+text
        if settings["bold"]: text="{\\b1}"+text
        if settings["italic"]: text="{\\i1}"+text
        if settings["underline"]: text="{\\u1}"+text
        text=text.replace("\n",r"\N")
        lines.append(f"Dialogue: 0,{editor_ass_time(start)},{editor_ass_time(end)},Default,,0,0,0,,{text}")
    return "\n".join(lines)+"\n"


def editor_run_ffmpeg(video_path,ass_path,output_path):
    ffmpeg=editor_find_binary("ffmpeg")
    if not ffmpeg: raise RuntimeError("FFmpeg was not found. Make sure packages.txt contains ffmpeg.")
    ass_filter_path=ass_path.replace("\\","/").replace(":",r"\:")
    video_filter="subtitles='"+ass_filter_path.replace("'",r"\'")+"'"
    env=os.environ.copy(); env.setdefault("FONTCONFIG_PATH","/etc/fonts"); env.setdefault("FONTCONFIG_FILE","/etc/fonts/fonts.conf")
    command=[ffmpeg,"-y","-hide_banner","-loglevel","error","-i",video_path,"-map","0:v:0","-map","0:a:0?","-vf",video_filter,"-c:v","libx264","-preset","veryfast","-crf","18","-c:a","aac","-b:a","192k","-movflags","+faststart","-shortest",output_path]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
    if result.returncode!=0: raise RuntimeError((result.stderr or "FFmpeg subtitle rendering failed.")[-3000:])


with video_editor_tab:
    st.markdown("""<style>.editor-hero{padding:1.5rem 1.7rem;border-radius:22px;border:1px solid rgba(128,128,128,.22);margin-bottom:1rem;background:linear-gradient(135deg,rgba(128,128,128,.10),rgba(128,128,128,.03))}.editor-hero h1{margin:0;font-size:2.25rem;letter-spacing:-.03em}.editor-hero p{margin:.4rem 0 0;opacity:.7}.tip{padding:1rem 1.1rem;border-radius:16px;border:1px solid rgba(128,128,128,.18);background:rgba(128,128,128,.035);margin:.5rem 0 1rem}</style><div class="editor-hero"><h1>🎬 Video Subtitle Editor</h1><p>Edit, automatically wrap, and render multilingual subtitles directly into your video.</p></div>""",unsafe_allow_html=True)
    st.markdown('<div class="tip">✨ <b>Multilingual rendering fixed</b><br>Bilingual subtitles keep their two lines, long subtitles wrap automatically, and Noto font fallback is used for Chinese, Hindi, Japanese, Korean, Arabic, Thai, Hebrew and other scripts. The subtitle background remains transparent.</div>',unsafe_allow_html=True)
    video=st.file_uploader("Choose original video",type=SUPPORTED_VIDEO_TYPES,key="editor_video")
    srt_file=st.file_uploader("Choose prepared SRT subtitle",type=["srt"],key="editor_srt")
    if video: st.video(video)
    if video and srt_file:
        try:
            subtitles=editor_parse_srt(srt_file.getvalue().decode("utf-8-sig",errors="replace"))
        except Exception as exc:
            subtitles=[]; st.error(f"Could not read subtitle file: {exc}")
        if not subtitles:
            st.error("No valid subtitle entries were found in the SRT file. Expected timestamps like 00:00:00,000 --> 00:00:03,000.")
        else:
            work_probe_dir=tempfile.mkdtemp(prefix="editor_probe_")
            probe_path=os.path.join(work_probe_dir,"probe"+Path(video.name).suffix.lower())
            try:
                with open(probe_path,"wb") as f:f.write(video.getbuffer())
                video_info=editor_probe_video(probe_path)
            finally:
                shutil.rmtree(work_probe_dir,ignore_errors=True)
            if not video_info:
                st.error("Unable to read the video dimensions. Please check that the video contains a valid video stream.")
            else:
                st.success(f"Loaded {len(subtitles)} subtitle entries · video {video_info['width']}×{video_info['height']} · {video_info.get('duration',0):.2f}s")
                source_signature=f"{srt_file.name}:{len(srt_file.getvalue())}:{hash(srt_file.getvalue())}"
                if "editor_subtitles" not in st.session_state or st.session_state.get("editor_source_signature")!=source_signature:
                    st.session_state.editor_subtitles=subtitles; st.session_state.editor_source_signature=source_signature
                edited=[]; st.subheader("✏️ Subtitle editing")
                for i,item in enumerate(st.session_state.editor_subtitles,1):
                    with st.expander(f"#{i} · {editor_format_time(item['start'])} → {editor_format_time(item['end'])}",expanded=(i==1)):
                        a,b=st.columns(2); start_text=a.text_input("Start",editor_format_time(item["start"]),key=f"start_{i}"); end_text=b.text_input("End",editor_format_time(item["end"]),key=f"end_{i}"); text=st.text_area("Subtitle text",item["text"],height=110,key=f"text_{i}")
                        try:
                            start=editor_parse_time(start_text); end=editor_parse_time(end_text)
                            if end<=start: st.warning("End time must be later than start time.")
                        except ValueError as exc: start,end=item["start"],item["end"]; st.warning(str(exc))
                        edited.append({"start":start,"end":end,"text":text.strip() or item["text"]})
                st.session_state.editor_subtitles=edited
                st.subheader("🎨 Subtitle style"); c1,c2,c3=st.columns(3)
                with c1:
                    font=st.selectbox("Font",FONT_OPTIONS,index=0); font_size=st.slider("Font size",18,96,42,2); text_color=st.color_picker("Text color","#FFFFFF")
                with c2:
                    bold=st.checkbox("Bold",value=True); italic=st.checkbox("Italic",value=False); underline=st.checkbox("Underline",value=False); wrap_width=st.slider("Auto-wrap width",20,90,46,2,help="Approximate display width. Lower values create shorter subtitle lines."); highlight=st.checkbox("Text highlight",value=False,help="Changes text color only; the subtitle background remains transparent."); highlight_color=st.color_picker("Highlight color","#FFD84D",disabled=not highlight)
                with c3:
                    position=st.selectbox("Position",list(POSITION_OPTIONS.keys()),index=0); margin_v=st.slider("Vertical margin",20,180,55,5); outline_width=st.slider("Text outline",0,8,2,1); outline_color=st.color_picker("Outline color","#000000")
                st.caption("Automatic wrapping is applied separately to each physical SRT line, so bilingual subtitles stay as Source + Translation.")
                settings={"font":font,"font_size":font_size,"text_color":highlight_color if highlight else text_color,"base_text_color":text_color,"outline_color":outline_color,"outline_width":outline_width,"bold":bold,"italic":italic,"underline":underline,"position":position,"margin_v":margin_v,"wrap_width":wrap_width}
                st.subheader("🔍 Final preview"); preview_text="\n".join(x["text"] for x in edited[:8] if x["text"]); st.text_area("Edited subtitle preview",preview_text,height=200,disabled=True)
                if st.button("🎬 Burn subtitles into video",type="primary",use_container_width=True):
                    work_dir=tempfile.mkdtemp(prefix="subtitle_editor_")
                    try:
                        suffix=Path(video.name).suffix.lower() or ".mp4"; input_path=os.path.join(work_dir,"input"+suffix); ass_path=os.path.join(work_dir,"subtitles.ass"); output_path=os.path.join(work_dir,"subtitle_video.mp4")
                        with open(input_path,"wb") as f: f.write(video.getbuffer())
                        with open(ass_path,"w",encoding="utf-8") as f: f.write(editor_build_ass(edited,settings,video_info["width"],video_info["height"],video_info.get("duration",0)))
                        progress=st.progress(0,text="Preparing video editor…"); progress.progress(.2,text="Parsing multilingual fonts and subtitle lines…"); progress.progress(.35,text="Rendering subtitles at the source video resolution…"); editor_run_ffmpeg(input_path,ass_path,output_path); progress.progress(1.0,text="Video rendering completed.")
                        with open(output_path,"rb") as f: rendered=f.read()
                        st.success("🎉 Subtitle video is ready."); st.video(rendered); st.download_button("⬇️ Download edited video",rendered,"subtitle_edited_video.mp4","video/mp4",use_container_width=True)
                    except Exception as exc: st.error(f"Video rendering failed: {exc}")
                    finally: shutil.rmtree(work_dir,ignore_errors=True)
    else: st.info("Upload both the original video and a prepared SRT file to start editing.")
