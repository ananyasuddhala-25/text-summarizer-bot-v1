import streamlit as st
from transformers import pipeline
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="Text Summarization Bot V1.0",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Text Summarization Bot V1.0 - MVP")
st.markdown("**Convert long text into concise summaries using AI**")
st.markdown("---")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
max_length = st.sidebar.slider("Maximum summary length", 50, 200, 100)
min_length = st.sidebar.slider("Minimum summary length", 10, 100, 30)

# Model selection
model_options = {
    "BART (Recommended)": "facebook/bart-large-cnn",
    "T5 Small": "t5-small", 
    "Pegasus": "google/pegasus-cnn_dailymail"
}
selected_model = st.sidebar.selectbox("Choose AI Model:", list(model_options.keys()))

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input Text")

    # Text input options
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload Text File"])

    if input_method == "Type/Paste Text":
        text_input = st.text_area(
            "Enter text to summarize:", 
            height=200,
            placeholder="Paste your article, document, or any long text here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt', 'md'])
        if uploaded_file is not None:
            text_input = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", text_input, height=200)
        else:
            text_input = ""

with col2:
    st.header("📊 Summary")

    if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
        if text_input.strip():
            try:
                with st.spinner("AI is analyzing your text..."):
                    # Load the model
                    @st.cache_resource
                    def load_summarizer(model_name):
                        return pipeline("summarization", model=model_name)

                    summarizer = load_summarizer(model_options[selected_model])

                    # Generate summary
                    start_time = time.time()

                    # Handle long texts by chunking if needed
                    if len(text_input) > 1000:
                        # Simple chunking for very long texts
                        chunks = [text_input[i:i+1000] for i in range(0, len(text_input), 1000)]
                        summaries = []
                        for chunk in chunks[:3]:  # Limit to first 3 chunks for demo
                            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                            summaries.append(summary[0]['summary_text'])
                        final_summary = " ".join(summaries)
                    else:
                        summary = summarizer(text_input, max_length=max_length, min_length=min_length, do_sample=False)
                        final_summary = summary[0]['summary_text']

                    processing_time = time.time() - start_time

                # Display results
                st.success("✅ Summary Generated!")
                st.markdown("### 📄 Summary:")
                st.write(final_summary)

                # Statistics
                st.markdown("### 📈 Statistics:")
                original_words = len(text_input.split())
                summary_words = len(final_summary.split())
                compression_ratio = round((1 - summary_words/original_words) * 100, 1)

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("Original Words", original_words)
                col_stat2.metric("Summary Words", summary_words)
                col_stat3.metric("Compression", f"{compression_ratio}%")

                st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Try with shorter text or different model.")
        else:
            st.warning("⚠️ Please enter some text to summarize!")

# Footer
st.markdown("---")
st.markdown("### 🔧 About V1.0 MVP")
st.markdown("""
**Features:**
- ✅ Basic text summarization
- ✅ Multiple AI models (BART, T5, Pegasus)
- ✅ Adjustable summary length
- ✅ File upload support
- ✅ Processing statistics

**Next Version (V1.1):**
- 🔄 Multi-language support
- 🌐 Language detection
- 🔤 Translation capabilities
""")

# Sample texts for demo
with st.expander("📋 Try with sample texts"):
    sample_texts = {
        "News Article": """
        Artificial intelligence has made remarkable progress in recent years, with large language models like GPT-4 and BERT revolutionizing natural language processing. These models can understand context, generate human-like text, and perform complex reasoning tasks. However, they also raise concerns about bias, misinformation, and job displacement. Companies are investing billions in AI research, while governments are developing regulations to ensure responsible AI development. The future of AI promises both tremendous opportunities and significant challenges that society must address thoughtfully.
        """,
        "Scientific Text": """
        Climate change represents one of the most pressing challenges of our time. Rising global temperatures, caused primarily by greenhouse gas emissions from human activities, are leading to melting ice caps, rising sea levels, and extreme weather events. The Intergovernmental Panel on Climate Change (IPCC) has warned that without immediate action to reduce carbon emissions, global warming could exceed 1.5°C above pre-industrial levels by 2030. This would trigger irreversible changes to Earth's climate system, affecting food security, water resources, and human health worldwide.
        """
    }

    for title, text in sample_texts.items():
        if st.button(f"Use {title} Sample"):
            st.rerun()