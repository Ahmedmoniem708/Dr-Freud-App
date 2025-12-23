import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import json
import base64
import asyncio
import edge_tts
from datetime import datetime
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langdetect import detect 
import time
import random
import pandas as pd

# ==========================================
# ⚙️ إعدادات المشروع والأمان
# ==========================================
BETA_MODE = False   
TRIAL_DAYS = 15     

# 🔐 قراءة المفتاح من الأسرار (هنا تم التنظيف الكامل للأمان)
if "GROQ_API_KEY" in st.secrets:
    API_KEY = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ مفتاح GROQ_API_KEY غير موجود!")
    st.info("للعمل محلياً: أضف المفتاح في ملف .streamlit/secrets.toml")
    st.info("عند النشر: أضف المفتاح في إعدادات (Secrets) على Streamlit Cloud.")
    st.stop()
# ==========================================

st.set_page_config(page_title="SYSTEM: FREUD", page_icon="👁️", layout="wide")

# ---------------------------------------------------------
# 🎨 CSS: تصميم "ميدتيرم" (Dark Sci-Fi & WhatsApp Layout)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;500;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
    }

    /* 🌌 الخلفية: الفراغ الأسود (The Void) */
    .stApp {
        background-color: #000000;
        background-image: 
            radial-gradient(circle at 50% 50%, rgba(10, 20, 30, 0.8) 0%, #000000 90%);
        color: #cfd8dc;
    }

    /* 📟 القائمة الجانبية (ملف المريض السري) */
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #1a1a1a;
        box-shadow: 10px 0 30px rgba(0,0,0,0.8);
    }
    
    /* 💬 كروت الرسائل (Minimalist Terminal Style) */
    .stChatMessage {
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid #333;
        border-radius: 4px;
        margin-bottom: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* 🤖 الدكتور (الكيان - أخضر نيون) */
    div[data-testid="chatAvatarIcon-assistant"] { 
        background-color: #000 !important; 
        border: 1px solid #00ff41; 
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid #00ff41;
        background: linear-gradient(90deg, rgba(0, 255, 65, 0.05) 0%, transparent 100%);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        font-family: 'Share Tech Mono', monospace; /* خط الآلة */
        color: #00ff41;
        text-shadow: 0 0 5px rgba(0, 255, 65, 0.3);
    }

    /* 👤 الضحية/المستخدم (أحمر تحذيري) */
    div[data-testid="chatAvatarIcon-user"] { 
        background-color: #000 !important;
        border: 1px solid #ff0055;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        border-right: 3px solid #ff0055;
        flex-direction: row-reverse;
        text-align: right;
        background: linear-gradient(-90deg, rgba(255, 0, 85, 0.05) 0%, transparent 100%);
    }

    /* 🎙️ زر المايك (العين المراقبة) - ثابت أسفل اليسار */
    .whatsapp-mic-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 99999;
        width: 60px;
        height: 60px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .whatsapp-mic-container button {
        background: #000 !important;
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        border: 2px solid #00ff41 !important; /* حدود خضراء مضيئة */
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.4) !important;
        color: #00ff41 !important;
        padding: 0 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        animation: pulse-green 3s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 65, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 65, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 65, 0); }
    }
    
    /* ⌨️ شريط الكتابة (Terminal Input) - مزاح لليمين */
    [data-testid="stChatInput"] {
        left: 90px !important; /* مساحة للمايك */
        width: calc(100% - 110px) !important; 
        padding-bottom: 20px;
        background: transparent !important;
    }
    
    .stChatInputContainer textarea {
        background-color: #0a0a0a !important;
        color: #00ff41 !important;
        border: 1px solid #333 !important;
        border-radius: 2px !important; /* زوايا حادة تقنية */
        font-family: 'Share Tech Mono', monospace !important;
    }
    .stChatInputContainer textarea:focus {
        border-color: #00ff41 !important;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2) !important;
    }
    
    .stChatInputContainer {
        background-color: transparent !important;
    }

    /* 🧠 كارت التحليل النفسي (Psycho-Metric) */
    .psycho-card {
        background: #000;
        border: 1px solid #333;
        padding: 10px;
        margin-bottom: 10px;
        font-family: 'Share Tech Mono', monospace;
        color: #00ff41;
        font-size: 12px;
        border-left: 3px solid #00ff41;
    }
    .psycho-label { color: #555; font-size: 9px; letter-spacing: 1px; }
    .psycho-value { font-weight: bold; font-size: 14px; margin-top: 2px; text-transform: uppercase; }

    /* العناوين */
    h1 { 
        font-family: 'Orbitron', sans-serif;
        color: #00ff41;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 5px;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
    }
    
    /* إخفاء الأيقونات الافتراضية */
    [data-testid="stChatMessage"] [data-testid="stImage"] { display: none; }
    
    /* ستايل التابات */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] { color: #555; }
    .stTabs [aria-selected="true"] { color: #00ff41 !important; background: rgba(0, 255, 65, 0.1); }

    /* كروت الشبكة */
    .match-card {
        background: #0a0a0a; border: 1px dashed #00ff41; padding: 15px; text-align: center; margin-bottom: 10px;
    }
    
</style>
""", unsafe_allow_html=True)

# --- 1. دوال النظام والبيانات ---
def get_user_profile(username):
    file_path = f"users_data/{username}_profile.json"
    default_profile = {
        "start_date": str(datetime.now().date()), 
        "plan": "free",
        # بيانات الملف الشخصي 
        "display_name": username,
        "age": "",
        "location": "القاهرة",
        "profession": "", 
        "bio": "", 
        "interests": "",
        "psycho_type": "مجهول", 
        "defense_mechanism": "SCANNING...",
        "core_trauma": "HIDDEN",
        "mood_history": [],
        "memory_summary": "" 
    }
    
    if not os.path.exists(file_path):
        with open(file_path, "w") as f: json.dump(default_profile, f)
        return default_profile, 0
        
    with open(file_path, "r") as f: 
        profile = json.load(f)
        for k, v in default_profile.items():
            if k not in profile: profile[k] = v
                
    days = (datetime.now().date() - datetime.strptime(profile["start_date"], "%Y-%m-%d").date()).days
    return profile, days

def update_profile(username, data):
    file_path = f"users_data/{username}_profile.json"
    with open(file_path, "w") as f: json.dump(data, f)

def save_history(username, history):
    with open(f"users_data/{username}_chat.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def load_history(username):
    path = f"users_data/{username}_chat.json"
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else []

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

# --- 2. الصوت والذكاء ---
def transcribe_audio(audio_bytes, lang_code):
    r = sr.Recognizer()
    try:
        with open("temp_rec.wav", "wb") as f: f.write(audio_bytes)
        with sr.AudioFile("temp_rec.wav") as source:
            r.adjust_for_ambient_noise(source, duration=0.5) 
            audio = r.record(source)
            lang = "ar-EG" if lang_code == "ar" else "en-US"
            return r.recognize_google(audio, language=lang)
    except: return None

async def generate_speech(text, lang_code):
    voice = "ar-EG-SalmaNeural" if lang_code == "ar" else "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice, rate="-15%", pitch="-5Hz")
    await communicate.save("reply.mp3")

def text_to_speech(text):
    try:
        try: lang = 'ar' if detect(text) == 'ar' else 'en'
        except: lang = 'ar'
        asyncio.run(generate_speech(text, lang))
        with open("reply.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""<audio autoplay style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
            st.markdown(md, unsafe_allow_html=True)
    except: pass

def check_safety(text):
    keywords = ["suicide", "kill myself", "die", "انتحار", "أموت", "اقتل نفسي", "إنهاء حياتي"]
    for word in keywords:
        if word in text.lower(): return False
    return True

# --- 3. المصادقة ---
if not os.path.exists("users_data"): os.makedirs("users_data")
if not os.path.exists('config.yaml'): st.stop()

with open('config.yaml') as file: config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days']
)

# ---------------------------------------------------------
# 🔥 الشاشة الرئيسية (The Gate)
# ---------------------------------------------------------
FREUD_IMG = "https://upload.wikimedia.org/wikipedia/commons/3/36/Sigmund_Freud%2C_by_Max_Halberstadt_%28cropped%29.jpg"

if not st.session_state.get("authentication_status"):
    st.markdown("<h1>SYSTEM: <span style='color: #fff;'>FREUD</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #555; font-family: Share Tech Mono;'>INITIALIZING NEURAL LINK...</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; position: relative;">
            <div style="position: absolute; width: 160px; height: 160px; border: 1px dashed #00ff41; border-radius: 50%; animation: spin 20s linear infinite;"></div>
            <img src="{FREUD_IMG}" style="border-radius: 50%; width: 150px; height: 150px; object-fit: cover; filter: grayscale(100%) contrast(1.5);">
        </div>
        <style>@keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}</style>
        <br><br>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["[ ACCESS ]", "[ NEW SUBJECT ]"])
    with tab1: authenticator.login(location="main")
    with tab2:
        try:
            email_reg, username_reg, name_reg = authenticator.register_user(location='main')
            if email_reg:
                st.success("SUBJECT REGISTERED.")
                with open('config.yaml', 'w') as file: yaml.dump(config, file, default_flow_style=False)
        except Exception as e: st.error(e)

# ---------------------------------------------------------
# 🔥 التطبيق الداخلي (The System)
# ---------------------------------------------------------
if st.session_state.get("authentication_status"):
    username = st.session_state["username"]
    name = st.session_state["name"]
    
    profile, days_passed = get_user_profile(username)
    locked = False if (BETA_MODE or profile["plan"] == "premium" or (TRIAL_DAYS - days_passed) > 0) else True

    llm = ChatGroq(groq_api_key=API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.7)

    # ==========================
    # 📌 SIDEBAR
    # ==========================
    with st.sidebar:
        st.markdown(f"""
        <div style="border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 20px;">
            <h3 style="margin:0; color: #fff; font-family: Share Tech Mono;">SUBJECT: {name}</h3>
            <p style="margin:0; color: #00ff41; font-size: 10px;">ID: {username.upper()}-001</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_prof = profile.get('profession', 'UNKNOWN')
        user_bio = profile.get('bio', '')
        
        st.markdown(f"<p style='color:#555; font-size:12px;'>PROFESSION: <span style='color:#fff;'>{user_prof}</span></p>", unsafe_allow_html=True)
        if user_bio:
            st.markdown(f"<p style='color:#aaa; font-size:12px; font-style:italic; border-left: 2px solid #333; padding-left: 5px;'>{user_bio}</p>", unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 🧬 PSYCHO-METRICS")
        def_mech = profile.get("defense_mechanism", "SCANNING...")
        trauma = profile.get("core_trauma", "HIDDEN")
        
        st.markdown(f"""
        <div class="psycho-card">
            <div class="psycho-label">DEFENSE MECHANISM</div>
            <div class="psycho-value" style="color: #ff0055;">{def_mech}</div>
        </div>
        <div class="psycho-card">
            <div class="psycho-label">DETECTED TRAUMA</div>
            <div class="psycho-value">{trauma}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧹 PURGE CHAT (SAVE MEMORY)", use_container_width=True):
            if st.session_state.get("messages", []):
                with st.spinner("ARCHIVING MEMORIES..."):
                    try:
                        chat_txt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages if m['role']!='system'])
                        sum_prompt = f"Summarize key psychological traits/problems from this chat:\n{chat_txt}"
                        summary = llm.invoke(sum_prompt).content
                        old_mem = profile.get("memory_summary", "")
                        profile["memory_summary"] = f"{old_mem}\n\n[SESSION]: {summary}"[-5000:]
                        update_profile(username, profile)
                        st.session_state.messages = []
                        save_history(username, [])
                        st.toast("MEMORY STORED.")
                        time.sleep(1)
                        st.rerun()
                    except: st.error("MEMORY ERROR")
            else: st.warning("NO DATA.")

        authenticator.logout(location='sidebar')

    # ==========================
    # 📌 TABS SYSTEM
    # ==========================
    tab_chat, tab_mirror, tab_network, tab_data = st.tabs([
        "💬 THE SESSION", "🪞 MIRROR", "🌐 NETWORK", "📊 DATA"
    ])

    # --------------------------
    # TAB 1: الشات
    # --------------------------
    with tab_chat:
        mode = st.radio("PROTOCOL", ["PRIVATE", "THE CIRCLE"], horizontal=True, label_visibility="collapsed")
        
        if mode == "THE CIRCLE":
            st.warning("⚠️ GROUP MODE ACTIVE.")
            if "group_members" not in st.session_state: st.session_state.group_members = []
            if st.button("🧬 AI MATCHMAKING", use_container_width=True):
                with st.spinner("SCANNING..."):
                    st.session_state.group_members = ["Shadow_Walker", "Lost_Soul_99", "Cairo_Ghost", "Nile_Siren", "Dark_Poet"]
                    st.rerun()
            if st.session_state.group_members:
                st.code(f"CONNECTED: {', '.join(st.session_state.group_members)}")

        if "messages" not in st.session_state: st.session_state.messages = load_history(username)
        
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                if "[SECRET_VAULT]" in msg["content"]: continue
                with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
                    st.write(msg["content"])

        final_input = None
        if not locked:
            st.markdown('<div class="whatsapp-mic-container">', unsafe_allow_html=True)
            c_mic = st.columns(1)[0]
            with c_mic:
                audio = mic_recorder(start_prompt="👁️", stop_prompt="⛔", key='mic_input')
            st.markdown('</div>', unsafe_allow_html=True)
            if audio:
                with st.spinner("DECODING..."):
                    transcribed = transcribe_audio(audio['bytes'], "ar")
                    if transcribed: final_input = transcribed

        vault_mode = st.checkbox("🔒 SECRET VAULT (Hide from log)")
        text_input = st.chat_input("ENTER DATA...")
        if text_input: final_input = text_input

        if final_input:
            user_msg = f"[SECRET_VAULT] {final_input}" if vault_mode else final_input
            if mode == "THE CIRCLE" and not vault_mode: user_msg = f"[{name}]: {final_input}"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            if not vault_mode:
                with st.chat_message("user", avatar="👤"): st.write(user_msg)

            with st.chat_message("assistant", avatar="🧠"):
                msg_ph = st.empty()
                if len(final_input.split()) > 3:
                    try:
                        analysis_p = f"Analyze psychological state in 2 words (Mechanism, Trauma): '{final_input}'"
                        res = llm.invoke(analysis_p).content
                        if "," in res:
                            mech, trauma = res.split(",", 1)
                            profile["defense_mechanism"], profile["core_trauma"] = mech.strip(), trauma.strip()
                            update_profile(username, profile)
                    except: pass

                sys_prompt = f"""
                SYSTEM: Dr. Freud (The Black Box).
                SUBJECT: {name}, Job: {profile.get('profession')}.
                LONG TERM MEMORY: {profile.get('memory_summary')}
                INPUT: "{final_input}"
                STYLE: Dark, Mysterious.
                Reply in Arabic.
                """
                try:
                    response_text = llm.invoke(sys_prompt).content
                    msg_ph.write_stream(stream_data(response_text))
                    if not locked: text_to_speech(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    save_history(username, st.session_state.messages)
                except: msg_ph.error("SYSTEM ERROR.")

    # --------------------------
    # TAB 2: المرآة
    # --------------------------
    with tab_mirror:
        st.header("🪞 REFLECTION")
        with st.form("mirror"):
            q1 = st.text_input("1. What do you see in the dark?")
            if st.form_submit_button("ANALYZE"):
                res = llm.invoke(f"Analyze psychology based on: {q1}").content
                st.markdown(res)
                profile['psycho_type'] = res.split("\n")[0]
                update_profile(username, profile)

    # --------------------------
    # TAB 3: الشبكة
    # --------------------------
    with tab_network:
        st.header("🌐 NETWORK")
        if st.button("SCAN"):
            cols = st.columns(3)
            for i, u in enumerate(["User_X", "Echo_1", "Void"]):
                with cols[i]: st.markdown(f"<div class='match-card'><h3>{u}</h3><p>MATCH</p></div>", unsafe_allow_html=True)

    # --------------------------
    # TAB 4: البيانات
    # --------------------------
    with tab_data:
        st.header("📊 DATA")
        with st.expander("UPDATE"):
            c1, c2 = st.columns(2)
            with c1:
                n_age = st.text_input("AGE", profile.get("age"))
                n_loc = st.text_input("LOCATION", profile.get("location"))
            with c2:
                n_prof = st.text_input("PROFESSION", profile.get("profession"))
                n_int = st.text_input("INTERESTS", profile.get("interests"))
            n_bio = st.text_area("BIO", profile.get("bio"))
            if st.button("SAVE"):
                profile.update({"age": n_age, "location": n_loc, "profession": n_prof, "interests": n_int, "bio": n_bio})
                update_profile(username, profile)
                st.success("UPDATED.")
                st.rerun()

elif st.session_state.get("authentication_status") is False:
    st.error('ACCESS DENIED')
