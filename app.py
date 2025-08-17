import streamlit as st
import pandas as pd
import time
import tempfile
import os
import zipfile
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json

# THIRD PARTY IMPORTS WITH ERROR HANDLING
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.error("⚠️ Transformers not installed. Run: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Language detection disabled. Run: pip install langdetect")
    LANGDETECT_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Audio processing disabled. Run: pip install openai-whisper")
    WHISPER_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Video processing disabled. Run: pip install moviepy")
    MOVIEPY_AVAILABLE = False

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    st.warning("⚠️ PDF processing disabled. Run: pip install pdfplumber")
    PDF_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    st.warning("⚠️ OpenAI integration disabled. Run: pip install openai")
    OPENAI_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Word document processing disabled. Run: pip install python-docx")
    DOCX_AVAILABLE = False

# PAGE CONFIG
st.set_page_config(
    page_title="AI Summarization Bot V3.0 - Enhanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SESSION STATE INIT
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = []
if 'settings' not in st.session_state:
    st.session_state.settings = {}

# LANGUAGES AND FLAGS
SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Chinese (Simplified)": "zh-cn", "Chinese (Traditional)": "zh-tw",
    "Japanese": "ja", "Korean": "ko", "Arabic": "ar", "Hindi": "hi",
    "Turkish": "tr", "Polish": "pl", "Swedish": "sv", "Norwegian": "no"
}
LANGUAGE_FLAGS = {
    "en": "🇺🇸", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹",
    
    "pt": "🇵🇹", "nl": "🇳🇱", "ru": "🇷🇺", "zh-cn": "🇨🇳", "zh-tw": "🇹🇼",
    "ja": "🇯🇵", "ko": "🇰🇷", "ar": "🇸🇦", "hi": "🇮🇳", "tr": "🇹🇷",
    "pl": "🇵🇱", "sv": "🇸🇪", "no": "🇳🇴"
}
SUPPORTED_FILE_TYPES = {
    'text': ['.txt', '.md', '.rtf'],
    'pdf': ['.pdf'],
    'word': ['.docx', '.doc'],
    'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
}

# --- FUNCTION DEFINITIONS ---

@st.cache_resource
def load_ai_models():
    """Load AI models with proper error handling"""
    models = {}
    if not TRANSFORMERS_AVAILABLE:
        return models
    try:
        with st.spinner("🤖 Loading AI models..."):
            models['bart'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU
            )
            models['t5'] = pipeline(
                "text2text-generation",
                model="t5-small",
                device=-1
            )
            if WHISPER_AVAILABLE:
                models['whisper'] = whisper.load_model("base")
        st.success("✅ AI Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
    return models

def detect_language_safe(text):
    if not LANGDETECT_AVAILABLE or not text.strip():
        return "en"
    try:
        detected = detect(text)
        return detected if detected in LANGUAGE_FLAGS else "en"
    except:
        return "en"

def extract_text_from_pdf(file_bytes):
    if not PDF_AVAILABLE:
        raise Exception("PDF processing not available. Install pdfplumber.")
    text_content = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {page_num} ---\n{page_text}")
        return "\n\n".join(text_content)
    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def extract_text_from_docx(file_bytes):
    if not DOCX_AVAILABLE:
        raise Exception("Word document processing not available. Install python-docx.")
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise Exception(f"Error extracting Word document text: {str(e)}")

def extract_audio_from_video(video_bytes, output_path):
    if not MOVIEPY_AVAILABLE:
        raise Exception("Video processing not available. Install moviepy.")
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_video.flush()
            video = VideoFileClip(temp_video.name)
            audio = video.audio
            audio.write_audiofile(output_path, verbose=False, logger=None)
            video.close()
            audio.close()
            os.unlink(temp_video.name)
        return True
    except Exception as e:
        st.error(f"Video processing error: {str(e)}")
        return False

def transcribe_audio(audio_bytes, models):
    if not WHISPER_AVAILABLE or 'whisper' not in models:
        raise Exception("Audio transcription not available. Install openai-whisper.")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        try:
            result = models['whisper'].transcribe(temp_audio.name)
            return result['text'], result.get('language', 'en')
        finally:
            os.unlink(temp_audio.name)

def generate_summary(text, model_name, models, settings):
    if not text.strip():
        return "No content to summarize."
    try:
        if model_name == 'BART':
            if 'bart' not in models:
                raise Exception("BART model not available")
            max_chunk_size = 1024
            if len(text) > max_chunk_size:
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                summaries = []
                for chunk in chunks:
                    result = models['bart'](
                        chunk,
                        max_length=settings.get('max_length', 130),
                        min_length=settings.get('min_length', 30),
                        do_sample=False
                    )
                    summaries.append(result[0]['summary_text'])
                combined = " ".join(summaries)
                if len(combined) > max_chunk_size:
                    final_result = models['bart'](
                        combined,
                        max_length=settings.get('max_length', 130),
                        min_length=settings.get('min_length', 30),
                        do_sample=False
                    )
                    return final_result['summary_text']
                return combined
            else:
                result = models['bart'](
                    text,
                    max_length=settings.get('max_length', 130),
                    min_length=settings.get('min_length', 30),
                    do_sample=False
                )
                return result['summary_text']
        elif model_name == 'T5':
            if 't5' not in models:
                raise Exception("T5 model not available")
            prompt = f"summarize: {text[:1024]}"
            result = models['t5'](
                prompt,
                max_length=settings.get('max_length', 130),
                min_length=settings.get('min_length', 30)
            )
            return result['generated_text']
        elif model_name == 'GPT-4':
            return generate_openai_summary(text, settings)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def generate_openai_summary(text, settings):
    if not OPENAI_AVAILABLE:
        raise Exception("OpenAI integration not available. Install openai package.")
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OpenAI API key not found. Set OPENAI_API_KEY in secrets or environment.")
    try:
        openai.api_key = api_key
        style = settings.get('summary_style', 'concise')
        prompt_templates = {
            'concise': "Provide a concise summary of the following text in 2-3 sentences:\n\n{text}",
            'detailed': "Provide a detailed summary of the following text, highlighting key points and main themes:\n\n{text}",
            'bullet_points': "Summarize the following text as bullet points covering the main topics:\n\n{text}"
        }
        prompt = prompt_templates.get(style, prompt_templates['concise']).format(
            text=text[:15000]
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.get('max_length', 150),
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def process_single_file(file_info, models, settings):
    start_time = time.time()
    result = {
        'filename': file_info['name'],
        'file_type': file_info['type'],
        'status': 'processing',
        'error': None,
        'transcript': '',
        'summary': '',
        'language': 'en',
        'processing_time': 0,
        'word_count': 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        if file_info['type'] == 'text':
            text_content = file_info['content']
            if isinstance(text_content, bytes):
                text_content = text_content.decode('utf-8', errors='ignore')
        elif file_info['type'] == 'pdf':
            text_content = extract_text_from_pdf(file_info['content'])
        elif file_info['type'] == 'word':
            text_content = extract_text_from_docx(file_info['content'])
        elif file_info['type'] == 'audio':
            text_content, detected_lang = transcribe_audio(file_info['content'], models)
            result['language'] = detected_lang
        elif file_info['type'] == 'video':
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                if extract_audio_from_video(file_info['content'], temp_audio.name):
                    with open(temp_audio.name, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    text_content, detected_lang = transcribe_audio(audio_bytes, models)
                    result['language'] = detected_lang
                    os.unlink(temp_audio.name)
                else:
                    raise Exception("Failed to extract audio from video")
        else:
            raise Exception(f"Unsupported file type: {file_info['type']}")
        result['transcript'] = text_content
        result['word_count'] = len(text_content.split())
        if file_info['type'] in ['text', 'pdf', 'word']:
            result['language'] = detect_language_safe(text_content)
        if text_content.strip():
            result['summary'] = generate_summary(
                text_content,
                settings.get('model', 'BART'),
                models,
                settings
            )
        else:
            result['summary'] = "No content found to summarize."
        result['status'] = 'completed'
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    result['processing_time'] = time.time() - start_time
    return result

def create_summary_report(result):
    report = f"""
AI SUMMARIZATION REPORT
=====================

File Information:
- Original File: {result['filename']}
- File Type: {result['file_type'].upper()}
- Processing Date: {result['timestamp']}
- Language: {result['language'].upper()} {LANGUAGE_FLAGS.get(result['language'], '🌍')}
- Word Count: {result['word_count']:,}
- Processing Time: {result['processing_time']:.2f} seconds

SUMMARY
=======
{result['summary']}

FULL TRANSCRIPT
==============
{result['transcript']}
"""
    return report

def create_batch_report(results):
    total_files = len(results)
    successful = len([r for r in results if r['status'] == 'completed'])
    failed = total_files - successful
    total_time = sum(r['processing_time'] for r in results)
    total_words = sum(r['word_count'] for r in results)
    languages = {}
    for result in results:
        lang = result['language']
        languages[lang] = languages.get(lang, 0) + 1
    report = f"""
BATCH PROCESSING REPORT
======================

Summary Statistics:
- Total Files Processed: {total_files}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {(successful/total_files*100):.1f}%
- Total Processing Time: {total_time:.2f} seconds
- Total Words Processed: {total_words:,}
- Average Processing Speed: {(total_words/total_time):.0f} words/second
Language Distribution:
"""
    for lang, count in languages.items():
        flag = LANGUAGE_FLAGS.get(lang, '🌍')
        report += f"- {flag} {lang.upper()}: {count} file(s)\n"
    report += "\n\nFile Details:\n"
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        report += f"{i}. {status_icon} {result['filename']} ({result['processing_time']:.1f}s)\n"
        if result['error']:
            report += f"   Error: {result['error']}\n"
    return report

def create_download_package(results, format_type='zip'):
    if format_type == 'zip':
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for result in results:
                if result['status'] == 'completed' and result['summary']:
                    filename = f"{os.path.splitext(result['filename'])[0]}_summary.txt"
                    content = create_summary_report(result)
                    zip_file.writestr(filename, content)
            batch_report = create_batch_report(results)
            zip_file.writestr("batch_processing_report.txt", batch_report)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    elif format_type == 'json':
        json_data = json.dumps(results, indent=2, default=str)
        return json_data.encode()

# UI HELPERS
def process_uploaded_file(uploaded_file, file_type, models, settings):
    file_info = {
        'name': uploaded_file.name,
        'type': file_type,
        'content': uploaded_file.read()
    }
    with st.spinner(f"🤖 Processing {uploaded_file.name}..."):
        result = process_single_file(file_info, models, settings)
    display_single_result(result)

def display_single_result(result):
    if result['status'] == 'completed':
        st.success(f"✅ {result['filename']} processed successfully!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Language", f"{LANGUAGE_FLAGS.get(result['language'], '🌍')} {result['language'].upper()}")
        with col2:
            st.metric("Words", f"{result['word_count']:,}")
        with col3:
            st.metric("Processing Time", f"{result['processing_time']:.1f}s")
        with col4:
            efficiency = result['word_count'] / result['processing_time'] if result['processing_time'] > 0 else 0
            st.metric("Speed", f"{efficiency:.0f} w/s")
        st.markdown("### 📋 Summary")
        st.write(result['summary'])
        if result['transcript'] and st.session_state.settings.get('save_transcripts', True):
            with st.expander("📝 View Full Transcript"):
                st.text_area("", result['transcript'], height=200, key=f"transcript_{result['filename']}")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "💾 Download Summary",
                result['summary'],
                file_name=f"{os.path.splitext(result['filename'])[0]}_summary.txt",
                mime="text/plain"
            )
        with col2:
            report = create_summary_report(result)
            st.download_button(
                "📊 Download Full Report",
                report,
                file_name=f"{os.path.splitext(result['filename'])}_report.txt",
                mime="text/plain"
            )
        st.session_state.processing_history.append(result)
    else:
        st.error(f"❌ Error processing {result['filename']}: {result['error']}")

def process_batch_files(uploaded_files, models, settings):
    file_infos = []
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[1].lower()
        file_type = 'text'  # default
        for ftype, extensions in SUPPORTED_FILE_TYPES.items():
            if file_ext in extensions:
                file_type = ftype
                break
        file_infos.append({
            'name': file.name,
            'type': file_type,
            'content': file.read()
        })

    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    results = []
    for i, file_info in enumerate(file_infos):
        status_text.text(f"🤖 Processing {file_info['name']}... ({i+1}/{len(file_infos)})")
        result = process_single_file(file_info, models, settings)
        results.append(result)
        progress_bar.progress((i + 1) / len(file_infos))
        with results_container:
            if result['status'] == 'completed':
                st.success(f"✅ {result['filename']} - {result['processing_time']:.1f}s")
            else:
                st.error(f"❌ {result['filename']} - {result['error']}")
    status_text.text("✅ Batch processing completed!")
    st.session_state.current_batch = results
    display_batch_results(results)

def display_batch_results(results):
    st.markdown("### 📊 Batch Processing Summary")
    total_files = len(results)
    successful = len([r for r in results if r['status'] == 'completed'])
    failed = total_files - successful
    total_time = sum(r['processing_time'] for r in results)
    total_words = sum(r['word_count'] for r in results if r['status'] == 'completed')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Files", total_files)
    with col2:
        st.metric("Successful", successful, delta=f"{(successful/total_files*100):.0f}%" if total_files > 0 else "0%")
    with col3:
        st.metric("Failed", failed)
    with col4:
        st.metric("Total Time", f"{total_time:.1f}s")
    with col5:
        speed = total_words / total_time if total_time > 0 else 0
        st.metric("Avg Speed", f"{speed:.0f} w/s")
    if successful > 0:
        languages = {}
        for result in results:
            if result['status'] == 'completed':
                lang = result['language']
                languages[lang] = languages.get(lang, 0) + 1
        st.markdown("### 🌍 Language Distribution")
        lang_cols = st.columns(min(len(languages), 6))
        for i, (lang, count) in enumerate(languages.items()):
            with lang_cols[i % 6]:
                flag = LANGUAGE_FLAGS.get(lang, '🌍')
                st.metric(f"{flag} {lang.upper()}", count)
    st.markdown("### 📋 Individual Results")
    col1, col2 = st.columns(2)
    with col1:
        show_filter = st.selectbox("Show:", ["All Files", "Successful Only", "Failed Only"])
    with col2:
        sort_by = st.selectbox("Sort by:", ["Filename", "Processing Time", "Word Count", "Status"])
    filtered_results = results
    if show_filter == "Successful Only":
        filtered_results = [r for r in results if r['status'] == 'completed']
    elif show_filter == "Failed Only":
        filtered_results = [r for r in results if r['status'] == 'error']
    if sort_by == "Processing Time":
        filtered_results.sort(key=lambda x: x['processing_time'], reverse=True)
    elif sort_by == "Word Count":
        filtered_results.sort(key=lambda x: x['word_count'], reverse=True)
    elif sort_by == "Status":
        filtered_results.sort(key=lambda x: x['status'])
    else:
        filtered_results.sort(key=lambda x: x['filename'])
    for result in filtered_results:
        with st.expander(f"{'✅' if result['status'] == 'completed' else '❌'} {result['filename']}", expanded=False):
            if result['status'] == 'completed':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Language:** {LANGUAGE_FLAGS.get(result['language'], '🌍')} {result['language'].upper()}")
                with col2:
                    st.write(f"**Words:** {result['word_count']:,}")
                with col3:
                    st.write(f"**Time:** {result['processing_time']:.1f}s")
                st.markdown("**Summary:**")
                st.write(result['summary'])
                if result['transcript'] and st.session_state.settings.get('save_transcripts', True):
                    with st.expander("📝 Full Transcript"):
                        st.text_area("", result['transcript'], height=150, key=f"batch_transcript_{result['filename']}")
            else:
                st.error(f"**Error:** {result['error']}")
    st.markdown("### 💾 Download Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        if successful > 0:
            zip_data = create_download_package(results, 'zip')
            st.download_button(
                "📦 Download ZIP Package",
                zip_data,
                file_name=f"batch_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
    with col2:
        if successful > 0:
            json_data = create_download_package(results, 'json')
            st.download_button(
                "📊 Download JSON Data",
                json_data,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with col3:
        batch_report = create_batch_report(results)
        st.download_button(
            "📋 Download Report",
            batch_report,
            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    st.session_state.processing_history.extend([r for r in results if r['status'] == 'completed'])

# -- MAIN APP LOGIC --

st.title("🤖 AI Summarization Bot V3.0 - Enhanced")
st.markdown("**Advanced AI-powered summarization with support for text, audio, video, PDF, and Word documents**")

with st.sidebar:
    st.header("⚙️ Configuration")
    available_models = []
    models = load_ai_models() if TRANSFORMERS_AVAILABLE else {}
    if 'bart' in models:
        available_models.append("BART (Best for English)")
    if 't5' in models:
        available_models.append("T5 (Multilingual)")
    if OPENAI_AVAILABLE:
        available_models.append("GPT-4 (Premium)")
    if not available_models:
        st.error("❌ No AI models available. Check your installation.")
        st.stop()
    model_choice = st.selectbox("🤖 AI Model:", available_models)
    summary_style = st.selectbox(
        "📝 Summary Style:",
        ["concise", "detailed", "bullet_points"],
        help="Choose the format and depth of your summaries"
    )
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max Length", 50, 300, 130)
    with col2:
        min_length = st.slider("Min Length", 10, 100, 30)
    batch_size = st.slider("Batch Size", 1, 10, 3, help="Number of files to process simultaneously")
    with st.expander("🔧 Advanced Settings"):
        enable_language_detection = st.checkbox("Enable Language Detection", True)
        save_transcripts = st.checkbox("Save Full Transcripts", True)
        quality_check = st.checkbox("Enable Quality Checks", True)
    settings = {
        'model': model_choice.split()[0],
        'summary_style': summary_style,
        'max_length': max_length,
        'min_length': min_length,
        'batch_size': batch_size,
        'enable_language_detection': enable_language_detection,
        'save_transcripts': save_transcripts,
        'quality_check': quality_check
    }
    st.session_state.settings = settings

tab1, tab2, tab3, tab4 = st.tabs(["📄 Single File", "📁 Batch Processing", "📊 Results & Analytics", "⚙️ Settings & Help"])

with tab1:
    st.header("📄 Single File Processing")
    file_type = st.selectbox(
        "Select file type:",
        ["📝 Text", "📄 PDF", "📘 Word Document", "🎵 Audio", "🎬 Video"]
    )
    if file_type == "📝 Text":
        input_method = st.radio("Input method:", ["Type/Paste", "Upload File"], horizontal=True)
        if input_method == "Type/Paste":
            text_input = st.text_area("Enter your text:", height=200, placeholder="Paste your text here...")
            if text_input and st.button("🚀 Summarize Text", type="primary"):
                file_info = {
                    'name': 'direct_input.txt',
                    'type': 'text',
                    'content': text_input
                }
                with st.spinner("🤖 Processing text..."):
                    result = process_single_file(file_info, models, settings)
                display_single_result(result)
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'md', 'rtf'])
            if uploaded_file and st.button("🚀 Process File", type="primary"):
                process_uploaded_file(uploaded_file, 'text', models, settings)
    elif file_type == "📄 PDF":
        if not PDF_AVAILABLE:
            st.error("❌ PDF processing not available. Install pdfplumber: `pip install pdfplumber`")
        else:
            uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
            if uploaded_file and st.button("🚀 Process PDF", type="primary"):
                process_uploaded_file(uploaded_file, 'pdf', models, settings)
    elif file_type == "📘 Word Document":
        if not DOCX_AVAILABLE:
            st.error("❌ Word document processing not available. Install python-docx: `pip install python-docx`")
        else:
            uploaded_file = st.file_uploader("Upload Word document", type=['docx', 'doc'])
            if uploaded_file and st.button("🚀 Process Document", type="primary"):
                process_uploaded_file(uploaded_file, 'word', models, settings)
    elif file_type == "🎵 Audio":
        if not WHISPER_AVAILABLE:
            st.error("❌ Audio processing not available. Install openai-whisper: `pip install openai-whisper`")
        else:
            uploaded_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'm4a', 'ogg', 'flac'])
            if uploaded_file and st.button("🚀 Process Audio", type="primary"):
                process_uploaded_file(uploaded_file, 'audio', models, settings)
    elif file_type == "🎬 Video":
        if not MOVIEPY_AVAILABLE or not WHISPER_AVAILABLE:
            st.error("❌ Video processing not available. Install required packages: `pip install moviepy openai-whisper`")
        else:
            uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv', 'wmv'])
            if uploaded_file and st.button("🚀 Process Video", type="primary"):
                process_uploaded_file(uploaded_file, 'video', models, settings)

# ---- rest of tabs (batch, analytics, settings/help) unchanged from your code ----
# Just make sure function definitions always occur before usage

