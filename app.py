import streamlit as st
from transformers import pipeline
import pandas as pd
import time
from langdetect import detect
from googletrans import Translator
import whisper
import tempfile
import os
import requests
import json

# Page configuration
st.set_page_config(
    page_title="AI Summarization Bot V2.0 - Enhanced",
    page_icon="🎬",
    layout="wide"
)

# Initialize translator
@st.cache_resource
def get_translator():
    return Translator()

# Language mappings
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es", 
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar"
}

LANGUAGE_FLAGS = {
    "en": "🇺🇸", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪",
    "it": "🇮🇹", "pt": "🇵🇹", "nl": "🇳🇱", "ru": "🇷🇺",
    "zh": "🇨🇳", "ja": "🇯🇵", "ko": "🇰🇷", "ar": "🇸🇦"
}

def detect_language(text):
    try:
        detected = detect(text)
        return detected
    except:
        return "en"

def get_language_name(code):
    for name, lang_code in SUPPORTED_LANGUAGES.items():
        if lang_code == code:
            return name
    return "Unknown"

@st.cache_resource
def load_summarization_models():
    models = {}
    try:
        # High-quality models for best summaries
        models['bart_cnn'] = pipeline("summarization", model="facebook/bart-large-cnn")
        models['t5_small'] = pipeline("text2text-generation", model="t5-small") 
        models['mt5_small'] = pipeline("text2text-generation", model="google/mt5-small")
        st.success("✅ AI Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models['fallback'] = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return models

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

def create_high_quality_summary(text, model_choice, summary_style, max_length, min_length):
    """
    Create high-quality, easy-to-understand summaries
    """
    models = load_summarization_models()

    # Clean and prepare text
    text = text.strip()
    if len(text) < 50:
        return "⚠️ Text too short for meaningful summarization. Please provide at least 50 words."

    try:
        if model_choice == "BART (Best for English)":
            # Use BART for highest quality English summaries
            if len(text) > 1000:
                # Chunk long texts
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                summaries = []
                for chunk in chunks[:3]:  # Limit to 3 chunks
                    chunk_summary = models['bart_cnn'](chunk, 
                                                     max_length=max_length//len(chunks[:3]), 
                                                     min_length=min_length//len(chunks[:3]), 
                                                     do_sample=False)
                    summaries.append(chunk_summary[0]['summary_text'])
                combined_summary = " ".join(summaries)

                # Create final summary from combined chunks
                if len(combined_summary) > max_length:
                    final_summary = models['bart_cnn'](combined_summary, 
                                                     max_length=max_length, 
                                                     min_length=min_length, 
                                                     do_sample=False)
                    return final_summary[0]['summary_text']
                else:
                    return combined_summary
            else:
                summary = models['bart_cnn'](text, 
                                           max_length=max_length, 
                                           min_length=min_length, 
                                           do_sample=False)
                return summary[0]['summary_text']

        elif model_choice == "T5 (Flexible)":
            # Use T5 with proper prompt
            if summary_style == "Simple & Clear":
                prompt = f"summarize in simple words: {text}"
            else:
                prompt = f"summarize: {text}"

            summary = models['t5_small'](prompt, 
                                       max_length=max_length, 
                                       min_length=min_length, 
                                       do_sample=False)

            # Extract text from T5 output
            result = summary[0]['generated_text']
            # Remove the prompt if it appears in output
            if "summarize" in result.lower():
                result = result.split(":", 1)[-1].strip()
            return result

        elif model_choice == "mT5 (Multilingual)":
            # Use mT5 for non-English
            if summary_style == "Simple & Clear":
                prompt = f"simplify and summarize: {text}"
            else:
                prompt = f"summarize: {text}"

            summary = models['mt5_small'](prompt, 
                                        max_length=max_length, 
                                        min_length=min_length, 
                                        do_sample=False)

            result = summary[0]['generated_text']
            if "summarize" in result.lower():
                result = result.split(":", 1)[-1].strip()
            return result

    except Exception as e:
        return f"❌ Summarization failed: {str(e)}. Try with shorter text or different model."

# Title and description
st.title("🎬 AI Summarization Bot V2.0 - Enhanced")
st.markdown("**Transform text, audio & video into clear, easy-to-understand summaries using advanced AI**")
st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Smart Settings")

# Model selection with descriptions
st.sidebar.subheader("🤖 AI Model")
model_options = {
    "BART (Best for English)": "Highest quality summaries for English text",
    "T5 (Flexible)": "Good for any language, flexible prompting", 
    "mT5 (Multilingual)": "Specialized for non-English languages"
}

selected_model = st.sidebar.selectbox(
    "Choose AI Model:", 
    list(model_options.keys()),
    help="BART gives the clearest English summaries"
)
st.sidebar.caption(f"ℹ️ {model_options[selected_model]}")

# Summary style
st.sidebar.subheader("✨ Summary Style")
summary_style = st.sidebar.radio(
    "Choose style:",
    ["Simple & Clear", "Detailed & Professional"],
    help="Simple = Easy to read, Detailed = More comprehensive"
)

# Length controls
st.sidebar.subheader("📏 Summary Length")
if summary_style == "Simple & Clear":
    max_length = st.sidebar.slider("Maximum words", 30, 100, 60)
    min_length = st.sidebar.slider("Minimum words", 10, 50, 20)
else:
    max_length = st.sidebar.slider("Maximum words", 50, 200, 120)
    min_length = st.sidebar.slider("Minimum words", 20, 100, 40)

# Language settings
st.sidebar.subheader("🌍 Language Settings")
auto_detect = st.sidebar.checkbox("Auto-detect language", value=True)

if not auto_detect:
    input_language = st.sidebar.selectbox(
        "Input Language:", 
        list(SUPPORTED_LANGUAGES.keys()),
        index=0
    )

output_language = st.sidebar.selectbox(
    "Summary Language:",
    list(SUPPORTED_LANGUAGES.keys()),
    index=0
)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📂 Input Selection")

    # Input method selection
    input_method = st.radio(
        "Choose your input type:", 
        ["📝 Text Input", "🎵 Audio File", "🎬 Video File (Beta)"],
        horizontal=True
    )

    text_input = ""
    detected_lang = "en"

    if input_method == "📝 Text Input":
        input_type = st.radio("Text source:", ["Type/Paste", "Upload File"])

        if input_type == "Type/Paste":
            text_input = st.text_area(
                "Enter text to summarize:", 
                height=200,
                placeholder="Paste your article, document, or any text here. The AI will create a clear, easy-to-understand summary..."
            )
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'md', 'pdf'])
            if uploaded_file is not None:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", text_input, height=200)

        if text_input.strip() and auto_detect:
            detected_lang = detect_language(text_input)
            detected_name = get_language_name(detected_lang)
            flag = LANGUAGE_FLAGS.get(detected_lang, "🌍")
            st.info(f"🔍 Detected: {flag} {detected_name}")

    elif input_method == "🎵 Audio File":
        st.markdown("**Upload audio and get instant transcript + summary**")
        uploaded_audio = st.file_uploader(
            "Choose audio file", 
            type=['mp3', 'wav', 'm4a'],
            help="Upload clear audio for best transcription results"
        )

        if uploaded_audio is not None:
            with tempfile.NamedTemporaryFile(suffix='.'+uploaded_audio.name.split('.')[-1], delete=False) as temp_audio:
                temp_audio.write(uploaded_audio.read())
                temp_audio.flush()
                temp_path = temp_audio.name

            with st.spinner("🎙️ Transcribing audio with AI..."):
                try:
                    whisper_model = load_whisper()
                    result = whisper_model.transcribe(temp_path)
                    text_input = result['text']
                    detected_lang = result.get('language', 'en')
                    os.remove(temp_path)

                    st.success("✅ Audio transcribed successfully!")
                    st.text_area("📝 Transcript:", text_input, height=200)

                    detected_name = get_language_name(detected_lang)
                    flag = LANGUAGE_FLAGS.get(detected_lang, "🌍")
                    st.info(f"🔍 Audio Language: {flag} {detected_name}")

                except Exception as e:
                    st.error(f"❌ Transcription failed: {str(e)}")
                    os.remove(temp_path) if os.path.exists(temp_path) else None

    elif input_method == "🎬 Video File (Beta)":
        st.warning("🚧 Video support coming in next update! Use audio extraction for now.")
        uploaded_video = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov'])
        if uploaded_video:
            st.info("💡 Tip: Extract audio from video and upload as audio file for now!")

with col2:
    st.header("✨ AI Summary")

    if text_input.strip():
        # Show input stats
        word_count = len(text_input.split())
        char_count = len(text_input)

        st.metric("Input Words", word_count)
        st.metric("Characters", char_count)

        if word_count < 20:
            st.warning("⚠️ Text is quite short. Add more content for better summaries.")

        # Summarize button
        if st.button("🚀 Create Smart Summary", type="primary", use_container_width=True):
            if word_count < 10:
                st.error("❌ Please provide at least 10 words for summarization.")
            else:
                with st.spinner("🧠 AI is creating your summary..."):
                    start_time = time.time()

                    # Handle translation if needed
                    source_lang = detected_lang if auto_detect else SUPPORTED_LANGUAGES[input_language]
                    target_lang = SUPPORTED_LANGUAGES[output_language]

                    # Translate to English if needed for BART
                    if selected_model == "BART (Best for English)" and source_lang != "en":
                        translator = get_translator()
                        try:
                            english_text = translator.translate(text_input, src=source_lang, dest="en").text
                            summary = create_high_quality_summary(english_text, selected_model, summary_style, max_length, min_length)

                            # Translate summary back if needed
                            if target_lang != "en":
                                final_summary = translator.translate(summary, src="en", dest=target_lang).text
                            else:
                                final_summary = summary
                        except:
                            final_summary = create_high_quality_summary(text_input, "T5 (Flexible)", summary_style, max_length, min_length)
                    else:
                        final_summary = create_high_quality_summary(text_input, selected_model, summary_style, max_length, min_length)

                    processing_time = time.time() - start_time

                # Display results
                if final_summary.startswith("⚠️") or final_summary.startswith("❌"):
                    st.error(final_summary)
                else:
                    st.success("✅ Smart Summary Created!")

                    # Summary display
                    st.markdown("### 📄 Your Summary:")
                    st.markdown(f"**{final_summary}**")

                    # Stats
                    summary_words = len(final_summary.split())
                    compression_ratio = round((1 - summary_words/word_count) * 100, 1) if word_count > 0 else 0

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("Summary Words", summary_words)
                    col_stat2.metric("Compression", f"{compression_ratio}%")
                    col_stat3.metric("Time", f"{processing_time:.1f}s")

                    # Language info
                    source_flag = LANGUAGE_FLAGS.get(source_lang, "🌍")
                    target_flag = LANGUAGE_FLAGS.get(target_lang, "🌍")
                    st.info(f"📊 {source_flag} → {target_flag} | Style: {summary_style} | Model: {selected_model}")

                    # Download button
                    st.download_button(
                        label="💾 Download Summary",
                        data=final_summary,
                        file_name=f"summary_{int(time.time())}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    else:
        st.info("👆 Upload or paste content above to generate a smart summary")

# Footer
st.markdown("---")

# Sample texts
with st.expander("🎯 Try These Sample Texts"):
    sample_texts = {
        "Technology News": """
        Artificial intelligence has reached unprecedented levels of sophistication in 2025. Recent breakthroughs in large language models have enabled machines to understand context, generate creative content, and solve complex problems with human-like reasoning. Major tech companies are investing billions in AI research, leading to innovations in healthcare, education, and autonomous systems. However, experts warn about potential risks including job displacement, privacy concerns, and the need for robust AI governance frameworks. The debate continues about how to balance AI's transformative potential with responsible development practices.
        """,

        "Science Discovery": """
        Scientists at leading research institutions have made a groundbreaking discovery in renewable energy storage. The new battery technology uses abundant materials and can store energy for weeks without significant loss. This breakthrough could revolutionize how we power our cities and homes, making renewable energy more reliable and accessible. The research team spent five years developing the technology, which combines novel chemical processes with advanced materials science. Early tests show the batteries can charge in minutes and last for decades, potentially solving one of the biggest challenges in sustainable energy.
        """,

        "Health & Wellness": """
        A comprehensive study involving 50,000 participants over ten years has revealed surprising insights about longevity and healthy aging. Researchers found that regular social interaction, moderate exercise, and a Mediterranean-style diet were the strongest predictors of healthy aging. Interestingly, the study showed that mental stimulation through learning new skills was as important as physical exercise. Participants who maintained close friendships and engaged in community activities showed 40% lower rates of cognitive decline. The findings challenge previous assumptions about aging and provide actionable guidance for healthy living.
        """
    }

    for title, text in sample_texts.items():
        if st.button(f"📖 Load: {title}", key=f"sample_{title}"):
            st.rerun()

# Features info
with st.expander("🚀 V2.0 Features"):
    col_feat1, col_feat2 = st.columns(2)

    with col_feat1:
        st.markdown("""
        **📝 Text Processing:**
        - Smart text summarization
        - Multiple AI models (BART, T5, mT5)
        - 12+ language support
        - Style options (Simple/Detailed)

        **🎵 Audio Processing:**
        - Audio file upload (.mp3, .wav)
        - AI speech-to-text (Whisper)
        - Automatic language detection
        - Transcript + summary generation
        """)

    with col_feat2:
        st.markdown("""
        **🌍 Multi-Language:**
        - Auto-detect input language
        - Translate summaries to any language
        - Language flags and indicators
        - Cross-language summarization

        **⚡ Enhanced Features:**
        - Download summaries
        - Processing time tracking
        - Compression statistics
        - Professional UI/UX
        """)

st.markdown("---")
st.markdown("**🎯 Built with:** Streamlit • Transformers • Whisper • Google Translate")