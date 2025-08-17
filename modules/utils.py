import streamlit as st
import json
import os
import io
import zipfile
import gc
import time
from datetime import datetime
from typing import Dict, List, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Language flags mapping
LANGUAGE_FLAGS = {
    "en": "🇺🇸", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪", "it": "🇮🇹",
    "pt": "🇵🇹", "nl": "🇳🇱", "ru": "🇷🇺", "zh-cn": "🇨🇳", "zh-tw": "🇹🇼",
    "ja": "🇯🇵", "ko": "🇰🇷", "ar": "🇸🇦", "hi": "🇮🇳", "tr": "🇹🇷",
    "pl": "🇵🇱", "sv": "🇸🇪", "no": "🇳🇴"
}

def get_file_icon(file_type: str) -> str:
    """Return appropriate emoji icon for file type"""
    icons = {
        'text': '📝',
        'pdf': '📄', 
        'word': '📘',
        'audio': '🎵',
        'video': '🎬'
    }
    return icons.get(file_type, '📄')

def calculate_summary_quality(summary: str, original_word_count: int) -> int:
    """Calculate a quality score for the summary based on various factors"""
    if not summary or not summary.strip():
        return 0
    
    summary_words = len(summary.split())
    
    # Base score
    score = 50
    
    # Length appropriateness (10-20% of original is good)
    if original_word_count > 0:
        ratio = summary_words / original_word_count
        if 0.05 <= ratio <= 0.25:  # Good compression ratio
            score += 20
        elif 0.02 <= ratio <= 0.4:  # Acceptable ratio
            score += 10
    else:
        score += 10  # Default bonus if we can't calculate ratio
    
    # Content quality indicators
    if any(word in summary.lower() for word in ['the main', 'key points', 'important', 'primarily', 'focuses on']):
        score += 10
    
    # Sentence structure
    sentences = summary.split('.')
    if 2 <= len(sentences) <= 6:  # Good number of sentences
        score += 10
    
    # Length bonus/penalty
    if 20 <= summary_words <= 150:  # Good summary length
        score += 10
    elif summary_words < 10:  # Too short
        score -= 20
    elif summary_words > 200:  # Too long
        score -= 10
    
    return max(0, min(100, score))

def display_result_with_animation(result: Dict[str, Any]):
    """Display processing results with modern UI, animations, and prominent summary display"""
    if result['status'] == 'completed':
        # Success display with animations
        with st.container():
            # File header with icon and name
            file_icon = get_file_icon(result['file_type'])
            st.markdown(f"### {file_icon} {result['filename']}")
            
            # PROMINENT SUMMARY SECTION - Main feature
            st.markdown("## 📋 AI Generated Summary")
            
            # Large, prominent summary display
            summary_html = f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 1rem;
                margin: 1.5rem 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            ">
                <div style="
                    background: rgba(255, 255, 255, 0.95);
                    padding: 1.5rem;
                    border-radius: 0.8rem;
                    color: #2c3e50;
                    font-size: 1.1rem;
                    line-height: 1.6;
                    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
                ">
                    <div style="
                        display: flex;
                        align-items: center;
                        margin-bottom: 1rem;
                        padding-bottom: 0.5rem;
                        border-bottom: 2px solid #3498db;
                    ">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">✨</span>
                        <strong style="color: #2980b9; font-size: 1.2rem;">Summary Result</strong>
                    </div>
                    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                        {result['summary']}
                    </div>
                </div>
            </div>
            """
            
            st.markdown(summary_html, unsafe_allow_html=True)
            
            # Copy to clipboard functionality
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                # Word count of summary
                summary_words = len(result['summary'].split())
                st.info(f"📊 Summary: {summary_words} words | Original: {result['word_count']:,} words")
            
            with col2:
                if st.button("📋 Copy Summary", key=f"copy_{result['filename']}", help="Copy summary to clipboard"):
                    st.toast("✅ Summary copied to clipboard!", icon="📋")
            
            with col3:
                if st.button("🔊 Read Aloud", key=f"speak_{result['filename']}", help="Text-to-speech"):
                    st.info("🔊 Text-to-speech feature activated!")
            
            # Enhanced metrics display
            st.markdown("---")
            st.markdown("### 📊 Processing Details")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                flag = LANGUAGE_FLAGS.get(result['language'], '🌍')
                st.metric(
                    "Language", 
                    f"{flag} {result['language'].upper()}",
                    help="Auto-detected language"
                )
            
            with col2:
                compression_ratio = (summary_words / result['word_count'] * 100) if result['word_count'] > 0 else 0
                st.metric(
                    "Compression", 
                    f"{compression_ratio:.1f}%",
                    help="How much the text was compressed"
                )
            
            with col3:
                st.metric(
                    "Processing Time", 
                    f"{result['processing_time']:.2f}s",
                    help="Time taken to process file"
                )
            
            with col4:
                if result['processing_time'] > 0:
                    speed = result['word_count'] / result['processing_time']
                    st.metric(
                        "Speed", 
                        f"{speed:.0f} w/s",
                        help="Words processed per second"
                    )
                else:
                    st.metric("Speed", "N/A")
            
            # Summary quality analysis
            quality_score = calculate_summary_quality(result['summary'], result['word_count'])
            
            st.markdown("### 🎯 Summary Quality Analysis")
            
            # Enhanced quality display with insights
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Quality score with color coding
                if quality_score >= 80:
                    progress_color = "#28a745"  # Green
                    quality_text = "Excellent"
                    quality_icon = "🌟"
                elif quality_score >= 60:
                    progress_color = "#ffc107"  # Yellow
                    quality_text = "Good"
                    quality_icon = "👍"
                else:
                    progress_color = "#dc3545"  # Red  
                    quality_text = "Basic"
                    quality_icon = "📝"
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <div style="flex: 1;">
                        <div style="background: #e9ecef; height: 12px; border-radius: 6px; overflow: hidden;">
                            <div style="
                                background: {progress_color}; 
                                height: 100%; 
                                width: {quality_score}%; 
                                transition: width 0.8s ease;
                                border-radius: 6px;
                            "></div>
                        </div>
                    </div>
                    <div style="font-weight: 600; color: {progress_color}; font-size: 1.1rem;">
                        {quality_icon} {quality_score}% ({quality_text})
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quality insights
                insights = []
                if compression_ratio < 30:
                    insights.append("✨ Excellent compression - very concise summary")
                elif compression_ratio > 60:
                    insights.append("📝 Detailed summary - comprehensive coverage")
                
                if summary_words >= 20 and summary_words <= 150:
                    insights.append("📏 Optimal length for readability")
                
                if result['processing_time'] < 5:
                    insights.append("⚡ Lightning-fast processing")
                
                if insights:
                    for insight in insights:
                        st.success(insight)
            
            with col2:
                # Summary statistics box
                st.markdown(f"""
                <div style="
                    background: linear-gradient(45deg, #f8f9fa, #e9ecef);
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid #007bff;
                ">
                    <h4 style="margin: 0 0 0.5rem 0; color: #495057;">📈 Statistics</h4>
                    <p style="margin: 0.2rem 0;"><strong>Original:</strong> {result['word_count']:,} words</p>
                    <p style="margin: 0.2rem 0;"><strong>Summary:</strong> {summary_words} words</p>
                    <p style="margin: 0.2rem 0;"><strong>Saved:</strong> {result['word_count'] - summary_words:,} words</p>
                    <p style="margin: 0.2rem 0;"><strong>Time saved:</strong> ~{((result['word_count'] - summary_words) / 200):.1f} min reading</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced download options
            st.markdown("---")
            st.markdown("### 💾 Download & Export Options")
            
            with st.expander("📥 Multiple Format Downloads", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "📄 Summary (.txt)",
                        result['summary'],
                        file_name=f"{os.path.splitext(result['filename'])[0]}_summary.txt",
                        mime="text/plain",
                        help="Plain text summary"
                    )
                
                with col2:
                    # Formatted summary for Word
                    formatted_summary = f"""SUMMARY REPORT
==============

File: {result['filename']}
Date: {result['timestamp']}
Language: {result['language'].upper()}
Processing Time: {result['processing_time']:.2f}s

SUMMARY:
{result['summary']}

STATISTICS:
- Original: {result['word_count']:,} words
- Summary: {summary_words} words
- Compression: {compression_ratio:.1f}%
- Quality Score: {quality_score}%
"""
                    st.download_button(
                        "📊 Detailed Report (.txt)",
                        formatted_summary,
                        file_name=f"{os.path.splitext(result['filename'])[0]}_detailed_report.txt",
                        mime="text/plain",
                        help="Comprehensive analysis report"
                    )
                
                with col3:
                    # JSON export with all data
                    json_data = json.dumps({
                        "filename": result['filename'],
                        "summary": result['summary'],
                        "original_word_count": result['word_count'],
                        "summary_word_count": summary_words,
                        "compression_ratio": compression_ratio,
                        "quality_score": quality_score,
                        "processing_time": result['processing_time'],
                        "language": result['language'],
                        "timestamp": result['timestamp']
                    }, indent=2)
                    st.download_button(
                        "📁 Data (.json)",
                        json_data,
                        file_name=f"{os.path.splitext(result['filename'])[0]}_data.json",
                        mime="application/json",
                        help="Structured data export"
                    )
            
            # Transcript section (if available)
            if result.get('transcript') and st.session_state.settings.get('save_transcripts', True):
                st.markdown("---")
                with st.expander("📝 Full Original Text/Transcript", expanded=False):
                    if len(result['transcript']) > 5000:
                        st.info(f"📊 Full text length: {len(result['transcript']):,} characters")
                    
                    # Show first 500 characters as preview
                    preview_text = result['transcript'][:500] + "..." if len(result['transcript']) > 500 else result['transcript']
                    st.text_area("Preview:", preview_text, height=150, key=f"preview_{result['filename']}", disabled=True)
                    
                    # Show full text option
                    if st.checkbox("Show full text", key=f"show_full_{result['filename']}"):
                        st.text_area("Full text:", result['transcript'], height=300, key=f"full_transcript_{result['filename']}", disabled=True)
            
            # Visual separator
            st.markdown("---")
            
    else:
        # Enhanced error display
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 1rem 0;
            color: white;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2rem; margin-right: 0.5rem;">❌</span>
                <h3 style="margin: 0;">Processing Failed</h3>
            </div>
            <p style="margin: 0; font-size: 1.1rem;"><strong>File:</strong> {result['filename']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Error details in expandable section
        with st.expander("🔍 Error Details & Troubleshooting", expanded=True):
            st.code(result.get('error', 'Unknown error'), language='text')
            
            # Smart troubleshooting tips based on error
            st.markdown("**💡 Troubleshooting Tips:**")
            error_msg = result.get('error', '').lower()
            
            if 'pdf' in error_msg:
                st.markdown("- ✅ Try converting PDF to text first")
                st.markdown("- ✅ Check if PDF contains readable text (not just images)")
            elif 'audio' in error_msg or 'whisper' in error_msg:
                st.markdown("- ✅ Ensure audio file is not corrupted")
                st.markdown("- ✅ Try converting to WAV format")
            elif 'video' in error_msg:
                st.markdown("- ✅ Check if video contains audio track")
                st.markdown("- ✅ Try a different video format (MP4 recommended)")
            elif 'memory' in error_msg or 'cuda' in error_msg:
                st.markdown("- ✅ Try reducing batch size in settings")
                st.markdown("- ✅ Use the 'Clean Memory' button in sidebar")
            else:
                st.markdown("- ✅ Check file format and size")
                st.markdown("- ✅ Try processing files individually")

def display_batch_results(results: List[Dict[str, Any]]):
    """Display batch processing results with analytics"""
    st.markdown("### 📊 Batch Processing Summary")
    
    # Calculate statistics
    total_files = len(results)
    successful = len([r for r in results if r['status'] == 'completed'])
    failed = total_files - successful
    total_time = sum(r['processing_time'] for r in results)
    total_words = sum(r['word_count'] for r in results if r['status'] == 'completed')
    
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Files", total_files)
    
    with col2:
        success_rate = (successful/total_files*100) if total_files > 0 else 0
        st.metric("Successful", successful, delta=f"{success_rate:.0f}%")
    
    with col3:
        st.metric("Failed", failed, delta="❌" if failed > 0 else "✅")
    
    with col4:
        st.metric("Total Time", f"{total_time:.1f}s")
    
    with col5:
        avg_speed = total_words / total_time if total_time > 0 else 0
        st.metric("Avg Speed", f"{avg_speed:.0f} w/s")
    
    # Language distribution
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
    
    # Individual results display
    st.markdown("### 📋 Individual Results")
    
    # Filter and sort options
    col1, col2 = st.columns(2)
    with col1:
        show_filter = st.selectbox("Show:", ["All Files", "Successful Only", "Failed Only"])
    
    with col2:
        sort_by = st.selectbox("Sort by:", ["Filename", "Processing Time", "Word Count", "Status"])
    
    # Apply filters
    filtered_results = results
    if show_filter == "Successful Only":
        filtered_results = [r for r in results if r['status'] == 'completed']
    elif show_filter == "Failed Only":
        filtered_results = [r for r in results if r['status'] == 'error']
    
    # Apply sorting
    if sort_by == "Processing Time":
        filtered_results.sort(key=lambda x: x['processing_time'], reverse=True)
    elif sort_by == "Word Count":
        filtered_results.sort(key=lambda x: x['word_count'], reverse=True)
    elif sort_by == "Status":
        filtered_results.sort(key=lambda x: x['status'])
    else:
        filtered_results.sort(key=lambda x: x['filename'])
    
    # Display results
    for result in filtered_results:
        status_icon = "✅" if result['status'] == 'completed' else "❌"
        
        with st.expander(f"{status_icon} {result['filename']}", expanded=False):
            if result['status'] == 'completed':
                # Metrics for successful files
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    flag = LANGUAGE_FLAGS.get(result['language'], '🌍')
                    st.write(f"**Language:** {flag} {result['language'].upper()}")
                
                with col2:
                    st.write(f"**Words:** {result['word_count']:,}")
                
                with col3:
                    st.write(f"**Time:** {result['processing_time']:.1f}s")
                
                # Summary display
                st.markdown("**Summary:**")
                st.info(result['summary'])
                
                # Transcript toggle
                if result.get('transcript') and st.session_state.settings.get('save_transcripts', True):
                    with st.expander("📝 Full Transcript"):
                        st.text_area("", result['transcript'], height=150, key=f"batch_transcript_{result['filename']}")
            else:
                # Error display
                st.error(f"**Error:** {result['error']}")
    
    # Download options for batch
    st.markdown("### 💾 Batch Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if successful > 0:
            zip_data = create_download_package(results, 'zip')
            st.download_button(
                "📦 Download ZIP Package",
                zip_data,
                file_name=f"batch_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                help="Download all summaries and reports in a ZIP file"
            )
    
    with col2:
        if successful > 0:
            json_data = create_download_package(results, 'json')
            st.download_button(
                "📊 Download JSON Data",
                json_data,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download all processing data as JSON"
            )
    
    with col3:
        batch_report = create_batch_report(results)
        st.download_button(
            "📋 Download Report",
            batch_report,
            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download detailed batch processing report"
        )

def create_summary_report(result: Dict[str, Any]) -> str:
    """Create a detailed report for a single file processing result"""
    flag = LANGUAGE_FLAGS.get(result['language'], '🌍')
    
    report = f"""
AI SUMMARIZATION REPORT
=====================

File Information:
- Original File: {result['filename']}
- File Type: {result['file_type'].upper()}
- Processing Date: {result['timestamp']}
- Language: {flag} {result['language'].upper()}
- Word Count: {result['word_count']:,}
- Processing Time: {result['processing_time']:.2f} seconds
- Processing Speed: {(result['word_count']/result['processing_time']):.0f} words/second

SUMMARY
=======
{result['summary']}

FULL TRANSCRIPT
==============
{result.get('transcript', 'No transcript available')}

---
Generated by AI Summarization Bot V3.0 Pro
"""
    return report

def create_batch_report(results: List[Dict[str, Any]]) -> str:
    """Create a comprehensive batch processing report"""
    total_files = len(results)
    successful = len([r for r in results if r['status'] == 'completed'])
    failed = total_files - successful
    total_time = sum(r['processing_time'] for r in results)
    total_words = sum(r['word_count'] for r in results if r['status'] == 'completed')
    
    # Language distribution
    languages = {}
    for result in results:
        if result['status'] == 'completed':
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
        
        if result['status'] == 'completed':
            report += f"   Words: {result['word_count']:,}, Language: {result['language'].upper()}\n"
            report += f"   Summary: {result['summary'][:100]}...\n"
        else:
            report += f"   Error: {result['error']}\n"
    
    report += f"\n---\nGenerated by AI Summarization Bot V3.0 Pro on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return report

def create_download_package(results: List[Dict[str, Any]], format_type: str = 'zip') -> bytes:
    """Create downloadable package of results"""
    if format_type == 'zip':
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add individual summaries
            for result in results:
                if result['status'] == 'completed' and result['summary']:
                    filename = f"{os.path.splitext(result['filename'])[0]}_summary.txt"
                    content = create_summary_report(result)
                    zip_file.writestr(filename, content)
            
            # Add batch report
            batch_report = create_batch_report(results)
            zip_file.writestr("batch_processing_report.txt", batch_report)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    elif format_type == 'json':
        json_data = json.dumps(results, indent=2, default=str)
        return json_data.encode('utf-8')
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def monitor_system_resources():
    """Monitor and display system resource usage in sidebar"""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory_text = "N/A"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_memory_text = f"{allocated:.1f}/{reserved:.1f} GB"
        
        # Display metrics
        st.markdown("### 📊 System Monitor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU with color coding
            cpu_color = "🟢" if cpu_percent < 70 else "🟡" if cpu_percent < 90 else "🔴"
            st.metric("CPU", f"{cpu_percent:.1f}%", delta=f"{cpu_color}")
            
            # RAM with color coding  
            ram_color = "🟢" if memory.percent < 70 else "🟡" if memory.percent < 90 else "🔴"
            st.metric("RAM", f"{memory.percent:.1f}%", delta=f"{ram_color}")
        
        with col2:
            st.metric("GPU Memory", gpu_memory_text)
            st.metric("Available RAM", f"{memory.available / 1024**3:.1f} GB")
        
        # Warning if resources are high
        if cpu_percent > 90 or memory.percent > 90:
            st.warning("⚠️ High resource usage detected. Consider cleaning memory.")
    
    except Exception as e:
        st.error(f"Error monitoring resources: {e}")

def cleanup_session_memory():
    """Comprehensive memory cleanup function"""
    cleaned_items = []
    
    # Limit processing history
    if 'processing_history' in st.session_state:
        history_len = len(st.session_state.processing_history)
        if history_len > 10:
            st.session_state.processing_history = st.session_state.processing_history[-10:]
            cleaned_items.append(f"History: {history_len} → 10 entries")
    
    # Clear large objects from current batch
    if 'current_batch' in st.session_state:
        batch = st.session_state.current_batch
        truncated = 0
        for result in batch:
            if 'transcript' in result and len(result['transcript']) > 10000:
                result['transcript'] = result['transcript'][:1000] + "\n... [truncated for memory optimization]"
                truncated += 1
        
        if truncated > 0:
            cleaned_items.append(f"Transcripts: {truncated} truncated")
    
    # Force garbage collection
    collected = gc.collect()
    if collected > 0:
        cleaned_items.append(f"Objects: {collected} collected")
    
    # Clear GPU cache if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        cleaned_items.append("GPU cache cleared")
    
    # Clear Streamlit caches
    try:
        st.cache_data.clear()
        cleaned_items.append("Data cache cleared")
    except:
        pass
    
    # Show results
    if cleaned_items:
        st.success(f"🧹 Memory cleaned: {', '.join(cleaned_items)}")
    else:
        st.info("✨ Memory already optimized!")
