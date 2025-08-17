import streamlit as st
import os
import sys

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from model_loader import load_ai_models
from file_processing import (
    determine_file_type, 
    process_single_file,
    process_batch_files
)
from utils import (
    get_file_icon,
    display_result_with_animation,
    display_batch_results,
    cleanup_session_memory,
    monitor_system_resources
)

# PAGE CONFIG
st.set_page_config(
    page_title="🤖 AI Summarization Bot V3.0 Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://github.com/your-repo',
        'Report a bug': 'mailto:your-email@domain.com',
        'About': "# AI Summarization Bot V3.0 Pro\nBuilt with ❤️ using Streamlit"
    }
)

# SESSION STATE INITIALIZATION
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = []
if 'settings' not in st.session_state:
    st.session_state.settings = {}

# CUSTOM CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    [data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 0.8rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .badge {
        background: linear-gradient(45deg, #1f77b4, #17a2b8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_ai_models_simple():
    """Simple model loading without advanced features"""
    models = {}
    try:
        models['bart'] = pipeline("summarization", model="facebook/bart-large-cnn")
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error: {e}")
    return models

# HEADER
def display_header():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">
            🤖 AI Summarization Bot V3.0 Pro
        </h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            ⚡ Lightning-fast AI-powered summarization with GPU acceleration
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span class="badge">📝 Text Processing</span>
            <span class="badge">🎵 Audio Analysis</span>
            <span class="badge">🎬 Video Transcription</span>
            <span class="badge">📄 Document Parsing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

display_header()

# SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load models
    models = load_ai_models()
    
    # Model selection
    available_models = []
    if models.get('bart'):
        available_models.append("BART (Best for English)")
    if models.get('t5'):
        available_models.append("T5 (Multilingual)")

    if not available_models:
        st.error("❌ No AI models available. Check your installation.")
        st.stop()
    
    model_choice = st.selectbox("🤖 AI Model:", available_models)
    
    # Settings
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
    
    batch_size = st.slider("Batch Size", 1, 10, 4, help="Number of files to process simultaneously")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        enable_language_detection = st.checkbox("Enable Language Detection", True)
        save_transcripts = st.checkbox("Save Full Transcripts", True)
        quality_check = st.checkbox("Enable Quality Checks", True)
    
    # Store settings
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
    
    # System monitoring
    monitor_system_resources()
    
    # Memory cleanup button
    if st.button("🧹 Clean Memory", help="Free up system memory"):
        cleanup_session_memory()

# MAIN INTERFACE TABS
tab1, tab2, tab3, tab4 = st.tabs(["📄 Single File", "📁 Batch Processing", "📊 Results & Analytics", "⚙️ Settings & Help"])

# SINGLE FILE PROCESSING TAB
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
                display_result_with_animation(result)
        else:
            uploaded_file = st.file_uploader("Upload text file", type=['txt', 'md', 'rtf'])
            if uploaded_file and st.button("🚀 Process File", type="primary"):
                file_info = {
                    'name': uploaded_file.name,
                    'type': 'text',
                    'content': uploaded_file.read()
                }
                with st.spinner(f"🤖 Processing {uploaded_file.name}..."):
                    result = process_single_file(file_info, models, settings)
                display_result_with_animation(result)
    
    elif file_type == "📄 PDF":
        uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
        if uploaded_file and st.button("🚀 Process PDF", type="primary"):
            file_info = {
                'name': uploaded_file.name,
                'type': 'pdf',
                'content': uploaded_file.read()
            }
            with st.spinner(f"🤖 Processing {uploaded_file.name}..."):
                result = process_single_file(file_info, models, settings)
            display_result_with_animation(result)
    
    elif file_type == "📘 Word Document":
        uploaded_file = st.file_uploader("Upload Word document", type=['docx', 'doc'])
        if uploaded_file and st.button("🚀 Process Document", type="primary"):
            file_info = {
                'name': uploaded_file.name,
                'type': 'word',
                'content': uploaded_file.read()
            }
            with st.spinner(f"🤖 Processing {uploaded_file.name}..."):
                result = process_single_file(file_info, models, settings)
            display_result_with_animation(result)
    
    elif file_type == "🎵 Audio":
        uploaded_file = st.file_uploader("Upload audio file", type=['mp3', 'wav', 'm4a', 'ogg', 'flac'])
        if uploaded_file and st.button("🚀 Process Audio", type="primary"):
            file_info = {
                'name': uploaded_file.name,
                'type': 'audio',
                'content': uploaded_file.read()
            }
            with st.spinner(f"🤖 Processing {uploaded_file.name}..."):
                result = process_single_file(file_info, models, settings)
            display_result_with_animation(result)
    
    elif file_type == "🎬 Video":
        uploaded_file = st.file_uploader("Upload video file", type=['mp4', 'avi', 'mov', 'mkv', 'wmv'])
        if uploaded_file and st.button("🚀 Process Video", type="primary"):
            file_info = {
                'name': uploaded_file.name,
                'type': 'video',
                'content': uploaded_file.read()
            }
            with st.spinner(f"🤖 Processing {uploaded_file.name}..."):
                result = process_single_file(file_info, models, settings)
            display_result_with_animation(result)

# BATCH PROCESSING TAB
with tab2:
    st.header("📁 Batch Processing")
    st.write("Upload multiple files for concurrent processing")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['txt', 'md', 'rtf', 'pdf', 'docx', 'doc', 'mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'avi', 'mov', 'mkv', 'wmv']
    )
    
    if uploaded_files and st.button("🚀 Process All Files", type="primary"):
        results = process_batch_files(uploaded_files, models, settings)
        st.session_state.current_batch = results
        display_batch_results(results)

# RESULTS & ANALYTICS TAB
with tab3:
    st.header("📊 Results & Analytics")
    
    if st.session_state.processing_history:
        st.write(f"**Total files processed:** {len(st.session_state.processing_history)}")
        
        # Display recent results
        for result in st.session_state.processing_history[-5:]:  # Last 5 results
            display_result_with_animation(result)
    else:
        st.info("No processing history yet. Process some files to see analytics.")

# SETTINGS & HELP TAB
with tab4:
    st.header("⚙️ Settings & Help")
    
    st.subheader("📋 Current Settings")
    st.json(st.session_state.settings)
    
    st.subheader("❓ How to Use")
    st.markdown("""
    1. **Single File**: Upload or paste text for individual processing
    2. **Batch Processing**: Upload multiple files for concurrent processing
    3. **Model Selection**: Choose between BART (English), T5 (Multilingual), or GPT-4
    4. **Settings**: Adjust summary length, style, and processing options
    
    **Supported Formats:**
    - Text: .txt, .md, .rtf
    - Documents: .pdf, .docx, .doc
    - Audio: .mp3, .wav, .m4a, .ogg, .flac
    - Video: .mp4, .avi, .mov, .mkv, .wmv
    """)
    
    st.subheader("🚀 Performance Tips")
    st.markdown("""
    - Use GPU for 3-5x faster processing
    - Enable batch processing for multiple files
    - Clean memory regularly for optimal performance
    - Choose appropriate model for your language needs
    """)
