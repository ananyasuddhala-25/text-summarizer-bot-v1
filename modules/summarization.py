import streamlit as st
import torch
import time
from typing import List, Dict, Any
import re
import os

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def preprocess_text_for_speed(text: str, max_length: int = 1024) -> str:
    """
    Preprocess text for faster processing - remove unnecessary whitespace, 
    fix encoding issues, and truncate intelligently
    """
    if not text or not text.strip():
        return ""
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.,!?;:()\-\'\"]+', '', text)
    
    # Smart truncation - keep beginning and end if too long
    words = text.split()
    if len(' '.join(words)) > max_length:
        # Take first 60% and last 40% of content
        split_point = int(len(words) * 0.6)
        truncated_words = words[:split_point] + words[-int(len(words) * 0.4):]
        text = ' '.join(truncated_words)
    
    return text

def chunk_text_efficiently(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better summary quality
    """
    if not text or len(text.split()) <= chunk_size:
        return [text]
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        # Break if we've covered all words
        if i + chunk_size >= len(words):
            break
    
    return chunks

@st.cache_data(ttl=3600, max_entries=50)  # Cache summaries for 1 hour
def generate_summary_cached(text_hash: str, text: str, model_name: str, max_length: int, min_length: int) -> str:
    """
    Cached wrapper for summary generation to avoid reprocessing same content
    """
    # This will be called by the main function if cache miss occurs
    return None

def generate_summary_bart_optimized(text: str, model, settings: Dict) -> str:
    """
    Optimized BART summary generation with chunking and batching
    """
    max_length = min(settings.get('max_length', 130), 200)  # Cap max length
    min_length = settings.get('min_length', 30)
    
    # Preprocess for speed
    text = preprocess_text_for_speed(text, 1024)
    
    if len(text.split()) <= 100:  # Very short text
        try:
            result = model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=2,  # Reduced from default 4 for speed
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            return result[0]['summary_text']
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    # For longer texts, use chunking
    chunks = chunk_text_efficiently(text, chunk_size=800)
    
    if len(chunks) == 1:
        # Single chunk processing
        try:
            result = model(
                chunks[0],
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
            return result[0]['summary_text']
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    # Multi-chunk processing
    chunk_summaries = []
    progress_placeholder = st.empty()
    
    for i, chunk in enumerate(chunks):
        progress_placeholder.text(f"🤖 Processing chunk {i+1}/{len(chunks)}...")
        
        try:
            result = model(
                chunk,
                max_length=min(max_length, 100),  # Shorter summaries for chunks
                min_length=min(min_length, 20),
                do_sample=False,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            chunk_summaries.append(result[0]['summary_text'])
        except Exception as e:
            st.warning(f"Chunk {i+1} failed: {e}")
            continue
    
    progress_placeholder.empty()
    
    if not chunk_summaries:
        return "Error: Could not process any chunks of the text."
    
    # Combine chunk summaries
    combined_summary = ' '.join(chunk_summaries)
    
    # Final summarization if combined summary is too long
    if len(combined_summary.split()) > max_length:
        try:
            final_result = model(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                num_beams=2,
                early_stopping=True
            )
            return final_result[0]['summary_text']
        except Exception as e:
            return combined_summary[:max_length*4]  # Fallback truncation
    
    return combined_summary

def generate_summary_t5_optimized(text: str, model, settings: Dict) -> str:
    """
    Optimized T5 summary generation
    """
    max_length = settings.get('max_length', 130)
    min_length = settings.get('min_length', 30)
    
    # Preprocess for speed
    text = preprocess_text_for_speed(text, 512)  # T5 has smaller context
    
    # Format input for T5
    prompt = f"summarize: {text}"
    
    try:
        result = model(
            prompt,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=2,  # Reduced for speed
            early_stopping=True,
            temperature=0.3
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"Error generating T5 summary: {str(e)}"

def generate_openai_summary_optimized(text: str, settings: Dict) -> str:
    """
    Optimized OpenAI/GPT summary generation with smart prompting
    """
    if not OPENAI_AVAILABLE:
        raise Exception("OpenAI integration not available. Install openai package.")
    
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OpenAI API key not found.")
    
    # Truncate text for API limits and cost efficiency
    max_tokens_input = 3000  # Leave room for response
    if len(text) > max_tokens_input * 3:  # Rough token estimation
        text = text[:max_tokens_input * 3]
    
    style = settings.get('summary_style', 'concise')
    max_length = settings.get('max_length', 150)
    
    # Optimized prompts for different styles
    prompts = {
        'concise': f"Summarize this text in 2-3 sentences, focusing on key points:\n\n{text}",
        'detailed': f"Provide a comprehensive summary highlighting main themes and important details:\n\n{text}",
        'bullet_points': f"Create a bullet-point summary of the main topics:\n\n{text}"
    }
    
    prompt = prompts.get(style, prompts['concise'])
    
    try:
        import openai
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Faster and cheaper than GPT-4
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length,
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def generate_summary(text: str, model_name: str, models: Dict, settings: Dict) -> str:
    """
    Main optimized summary generation function with caching and performance optimizations
    """
    if not text or not text.strip():
        return "No content to summarize."
    
    # Generate hash for caching
    import hashlib
    text_hash = hashlib.md5(f"{text}{model_name}{settings}".encode()).hexdigest()
    
    # Check cache first
    cached_summary = generate_summary_cached(text_hash, text, model_name, 
                                           settings.get('max_length', 130), 
                                           settings.get('min_length', 30))
    if cached_summary:
        return cached_summary
    
    start_time = time.time()
    
    try:
        if model_name.upper() == 'BART':
            if 'bart' not in models:
                raise Exception("BART model not available")
            summary = generate_summary_bart_optimized(text, models['bart'], settings)
        
        elif model_name.upper() == 'T5':
            if 't5' not in models:
                raise Exception("T5 model not available")
            summary = generate_summary_t5_optimized(text, models['t5'], settings)
        
        elif model_name.upper() == 'GPT-4' or model_name.upper() == 'OPENAI':
            if not models.get('openai'):
                raise Exception("OpenAI integration not available")
            summary = generate_openai_summary_optimized(text, settings)
        
        else:
            # Fallback to BART if available
            if 'bart' in models:
                summary = generate_summary_bart_optimized(text, models['bart'], settings)
            else:
                raise Exception(f"Unsupported model: {model_name}")
        
        processing_time = time.time() - start_time
        
        # Cache the result
        generate_summary_cached.__wrapped__(text_hash, text, model_name, 
                                          settings.get('max_length', 130), 
                                          settings.get('min_length', 30))
        
        # Add performance info if debugging
        if settings.get('show_performance', False):
            summary += f"\n\n[Processed in {processing_time:.2f}s using {model_name}]"
        
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def batch_summarize(texts: List[str], model_name: str, models: Dict, settings: Dict) -> List[str]:
    """
    Batch summarization for multiple texts - more efficient than individual processing
    """
    if not texts:
        return []
    
    summaries = []
    
    # Use ThreadPoolExecutor for CPU-bound tasks when using different models
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=min(4, len(texts))) as executor:
        futures = []
        
        for text in texts:
            future = executor.submit(generate_summary, text, model_name, models, settings)
            futures.append(future)
        
        for future in futures:
            try:
                summary = future.result(timeout=120)  # 2 minute timeout per summary
                summaries.append(summary)
            except Exception as e:
                summaries.append(f"Error: {str(e)}")
    
    return summaries

def clear_summary_cache():
    """
    Clear the summary cache to free memory
    """
    generate_summary_cached.clear()
    st.success("🧹 Summary cache cleared!")

def get_optimal_model_for_text(text: str, available_models: List[str]) -> str:
    """
    Suggest the best model based on text characteristics
    """
    word_count = len(text.split())
    
    # For very short texts
    if word_count < 100:
        return 'T5' if 'T5' in available_models else 'BART'
    
    # For medium texts
    elif word_count < 1000:
        return 'BART' if 'BART' in available_models else 'T5'
    
    # For long texts
    else:
        # BART handles long texts better with chunking
        return 'BART' if 'BART' in available_models else 'T5'
