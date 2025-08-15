import streamlit as st
from transformers import pipeline
import pandas as pd
import time
from langdetect import detect
from googletrans import Translator
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Text Summarization Bot V1.1",
    page_icon="🌐",
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
        return "en"  # Default to English

def get_language_name(code):
    for name, lang_code in SUPPORTED_LANGUAGES.items():
        if lang_code == code:
            return name
    return "Unknown"

@st.cache_resource
def load_multilingual_models():
    models = {}
    try:
        # Multilingual models
        models['mbart'] = pipeline("summarization", model="facebook/mbart-large-50-many-to-many-mmt")
        models['mt5'] = pipeline("text2text-generation", model="google/mt5-small") 
        models['multilingual_bart'] = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models['backup'] = pipeline("summarization", model="t5-small")
    return models

# Title and description
st.title("🌐 Text Summarization Bot V1.1 - Multi-Language")
st.markdown("**Convert text into summaries in 12+ languages with AI translation**")
st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Language settings
st.sidebar.subheader("🌍 Language Settings")
auto_detect = st.sidebar.checkbox("Auto-detect input language", value=True)

if not auto_detect:
    input_language = st.sidebar.selectbox(
        "Input Language:", 
        list(SUPPORTED_LANGUAGES.keys()),
        index=0
    )
else:
    input_language = None

output_language = st.sidebar.selectbox(
    "Summary Output Language:",
    list(SUPPORTED_LANGUAGES.keys()),
    index=0
)

# Model settings
st.sidebar.subheader("🤖 Model Settings")
model_options = {
    "mBERT + BART (Recommended)": "multilingual_bart",
    "mT5 (Google Multilingual)": "mt5",
    "mBART (Facebook Multilingual)": "mbart"
}
selected_model = st.sidebar.selectbox("Choose AI Model:", list(model_options.keys()))

# Summary settings
max_length = st.sidebar.slider("Maximum summary length", 50, 200, 100)
min_length = st.sidebar.slider("Minimum summary length", 10, 100, 30)

# Translation settings
st.sidebar.subheader("🔄 Translation Settings")
translate_mode = st.sidebar.radio(
    "Translation Strategy:",
    ["Translate → Summarize → Translate Back", "Direct Multilingual Summarization"]
)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Text")

    # Text input options
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload Text File"])

    if input_method == "Type/Paste Text":
        text_input = st.text_area(
            "Enter text to summarize (any language):", 
            height=200,
            placeholder="Paste your article, document, or text in any supported language..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'md'])
        if uploaded_file is not None:
            text_input = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", text_input, height=200)
        else:
            text_input = ""

    # Language detection display
    if text_input.strip() and auto_detect:
        detected_lang = detect_language(text_input)
        detected_name = get_language_name(detected_lang)
        flag = LANGUAGE_FLAGS.get(detected_lang, "🌍")
        st.info(f"🔍 Detected Language: {flag} {detected_name} ({detected_lang})")

with col2:
    st.header("📊 Multi-Language Summary")

    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if text_input.strip():
            try:
                with st.spinner("AI is processing your multilingual text..."):
                    translator = get_translator()

                    # Detect input language if auto-detect is enabled
                    if auto_detect:
                        source_lang = detect_language(text_input)
                        source_name = get_language_name(source_lang)
                    else:
                        source_lang = SUPPORTED_LANGUAGES[input_language]
                        source_name = input_language

                    target_lang = SUPPORTED_LANGUAGES[output_language]

                    start_time = time.time()

                    # Load models
                    models = load_multilingual_models()

                    if translate_mode == "Translate → Summarize → Translate Back":
                        # Strategy 1: Translate to English, summarize, translate back

                        # Step 1: Translate to English if not already English
                        if source_lang != "en":
                            with st.spinner("Translating to English..."):
                                english_text = translator.translate(text_input, src=source_lang, dest="en").text
                        else:
                            english_text = text_input

                        # Step 2: Summarize in English
                        with st.spinner("Generating summary..."):
                            summarizer = models.get('multilingual_bart', models.get('backup'))

                            if len(english_text) > 1000:
                                chunks = [english_text[i:i+1000] for i in range(0, len(english_text), 1000)]
                                summaries = []
                                for chunk in chunks[:3]:
                                    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                                    if isinstance(summary, list):
                                        summaries.append(summary[0]['summary_text'])
                                    else:
                                        summaries.append(summary)
                                english_summary = " ".join(summaries)
                            else:
                                summary = summarizer(english_text, max_length=max_length, min_length=min_length, do_sample=False)
                                if isinstance(summary, list):
                                    english_summary = summary[0]['summary_text']
                                else:
                                    english_summary = summary

                        # Step 3: Translate summary to target language
                        if target_lang != "en":
                            with st.spinner(f"Translating summary to {output_language}..."):
                                final_summary = translator.translate(english_summary, src="en", dest=target_lang).text
                        else:
                            final_summary = english_summary

                    else:
                        # Strategy 2: Direct multilingual summarization (experimental)
                        with st.spinner("Direct multilingual processing..."):
                            model = models.get('mt5', models.get('backup'))

                            # For mT5, we need to format the input appropriately
                            if 'mt5' in models and selected_model == "mT5 (Google Multilingual)":
                                prompt = f"summarize: {text_input}"
                                result = model(prompt, max_length=max_length, min_length=min_length, do_sample=False)
                                if isinstance(result, list):
                                    summary_text = result[0]['generated_text']
                                else:
                                    summary_text = str(result)

                                # Translate if needed
                                if target_lang != source_lang:
                                    final_summary = translator.translate(summary_text, src=source_lang, dest=target_lang).text
                                else:
                                    final_summary = summary_text
                            else:
                                # Fallback to translation method
                                if source_lang != "en":
                                    english_text = translator.translate(text_input, src=source_lang, dest="en").text
                                else:
                                    english_text = text_input

                                summarizer = models.get('multilingual_bart', models.get('backup'))
                                summary = summarizer(english_text, max_length=max_length, min_length=min_length, do_sample=False)
                                english_summary = summary[0]['summary_text'] if isinstance(summary, list) else str(summary)

                                if target_lang != "en":
                                    final_summary = translator.translate(english_summary, src="en", dest=target_lang).text
                                else:
                                    final_summary = english_summary

                    processing_time = time.time() - start_time

                # Display results
                st.success("✅ Multi-language Summary Generated!")

                # Language info
                source_flag = LANGUAGE_FLAGS.get(source_lang, "🌍")
                target_flag = LANGUAGE_FLAGS.get(target_lang, "🌍")

                st.markdown(f"### 📄 Summary:")
                st.markdown(f"**{source_flag} {source_name} → {target_flag} {output_language}**")
                st.write(final_summary)

                # Translation chain display
                if translate_mode == "Translate → Summarize → Translate Back" and source_lang != "en" and target_lang != "en":
                    with st.expander("🔄 Translation Process"):
                        st.markdown(f"**Original ({source_name}):** {text_input[:200]}...")
                        st.markdown(f"**English Translation:** {english_text[:200]}...")
                        st.markdown(f"**English Summary:** {english_summary}")
                        st.markdown(f"**Final ({output_language}):** {final_summary}")

                # Statistics
                st.markdown("### 📈 Statistics:")
                original_words = len(text_input.split())
                summary_words = len(final_summary.split())
                compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0

                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                col_stat1.metric("Original Words", original_words)
                col_stat2.metric("Summary Words", summary_words)
                col_stat3.metric("Compression", f"{compression_ratio}%")
                col_stat4.metric("Languages", f"{source_flag}→{target_flag}")

                st.info(f"⏱️ Processing time: {processing_time:.2f} seconds | Translation strategy: {translate_mode}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Try with shorter text, different language pair, or check your internet connection.")
        else:
            st.warning("⚠️ Please enter some text to summarize!")

# Footer
st.markdown("---")
st.markdown("### 🌐 About V1.1 Multi-Language")
col_about1, col_about2 = st.columns(2)

with col_about1:
    st.markdown("""
    **✨ New Features:**
    - 🌍 12+ language support
    - 🔍 Automatic language detection
    - 🔄 Smart translation strategies  
    - 🤖 Multilingual AI models
    - 📊 Enhanced statistics
    """)

with col_about2:
    st.markdown("""
    **🚀 Next Version (V2.0):**
    - 🎵 Audio file summarization
    - 🎬 Video transcript processing
    - 📁 Batch file processing
    - ⚡ Faster processing
    """)

# Multilingual sample texts
with st.expander("🌍 Try with multilingual samples"):
    sample_texts = {
        "English - Technology": """
        Artificial intelligence continues to revolutionize industries worldwide. Machine learning algorithms now power everything from recommendation systems to autonomous vehicles. The rapid advancement in natural language processing has enabled chatbots and virtual assistants to understand human communication better than ever before. However, concerns about job displacement and ethical AI development remain significant challenges.
        """,

        "Spanish - Ciencia": """
        La inteligencia artificial está transformando el mundo de maneras que nunca imaginamos. Los algoritmos de aprendizaje automático ahora impulsan desde sistemas de recomendación hasta vehículos autónomos. El rápido avance en el procesamiento de lenguaje natural ha permitido que los chatbots y asistentes virtuales entiendan la comunicación humana mejor que nunca.
        """,

        "French - Technologie": """
        L'intelligence artificielle continue de révolutionner les industries du monde entier. Les algorithmes d'apprentissage automatique alimentent désormais tout, des systèmes de recommandation aux véhicules autonomes. Les progrès rapides dans le traitement du langage naturel ont permis aux chatbots et assistants virtuels de mieux comprendre la communication humaine que jamais auparavant.
        """,

        "German - Wissenschaft": """
        Künstliche Intelligenz revolutioniert weiterhin Industrien weltweit. Algorithmen des maschinellen Lernens treiben nun alles an, von Empfehlungssystemen bis hin zu autonomen Fahrzeugen. Der schnelle Fortschritt in der natürlichen Sprachverarbeitung hat es Chatbots und virtuellen Assistenten ermöglicht, menschliche Kommunikation besser zu verstehen als je zuvor.
        """
    }

    for title, text in sample_texts.items():
        if st.button(f"📝 Load: {title}"):
            st.rerun()

# Language support info
with st.expander("🗣️ Supported Languages"):
    st.markdown("**Currently supported languages:**")

    # Create a nice grid of supported languages
    lang_cols = st.columns(4)
    for i, (lang_name, lang_code) in enumerate(SUPPORTED_LANGUAGES.items()):
        flag = LANGUAGE_FLAGS.get(lang_code, "🌍")
        lang_cols[i % 4].markdown(f"{flag} **{lang_name}** ({lang_code})")

    st.markdown("---")
    st.markdown("**Translation powered by Google Translate API**")
    st.markdown("**AI Models: mBERT, mT5, mBART for multilingual processing**")