import streamlit as st
import torch
from transformers import pipeline
import os

# Import availability flags
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

@st.cache_resource(show_spinner="🤖 Loading optimized AI models...")
def load_ai_models():
    """
    Load AI models with GPU acceleration and proper caching
    Returns a dictionary of loaded models
    """
    models = {}
    
    if not TRANSFORMERS_AVAILABLE:
        st.error("⚠️ Transformers not available. Please install: pip install transformers torch")
        return models
    
    # Determine device for optimization
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    try:
        with st.spinner("⚡ Loading BART model with GPU acceleration..."):
            models['bart'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                model_kwargs={"torch_dtype": torch_dtype}
            )
        
        with st.spinner("🔥 Loading T5 model with optimization..."):
            models['t5'] = pipeline(
                "text2text-generation",
                model="t5-base",
                device=device,
                model_kwargs={"torch_dtype": torch_dtype}
            )
        
        # Load Whisper model for audio processing
        if WHISPER_AVAILABLE:
            with st.spinner("🎵 Loading Whisper for audio processing..."):
                device_str = "cuda" if torch.cuda.is_available() else "cpu"
                models['whisper'] = whisper.load_model("base", device=device_str)
        
        # Success message with system info
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.success(f"✅ Models loaded with GPU acceleration!")
                st.info(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            except:
                st.success("✅ Models loaded with GPU acceleration!")
        else:
            st.success("✅ Models loaded (CPU optimized)")
        
        # Log loaded models
        model_list = ", ".join([k.upper() for k in models.keys()])
        st.info(f"🔧 Available models: {model_list}")
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        
        # Fallback: try loading CPU-only models
        try:
            st.info("🔄 Attempting CPU fallback...")
            models['bart'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Force CPU
            )
            st.success("✅ Fallback model loaded (CPU only)")
        except Exception as fallback_error:
            st.error(f"❌ Fallback failed: {str(fallback_error)}")
    
    return models

def get_model_info():
    """Get system and model information"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "whisper_available": WHISPER_AVAILABLE,
    }
    
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info["device_count"] = torch.cuda.device_count()
        except:
            pass
    
    return info

def clear_model_cache():
    """Clear model cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    st.cache_resource.clear()
    st.success("🧹 Model cache cleared!")
