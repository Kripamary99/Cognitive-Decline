import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from audio_processor import AudioProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from pydub import AudioSegment
import speech_recognition as sr
import os
import time
from audio_recorder_streamlit import audio_recorder
import io
import wave
from nltk.tokenize import sent_tokenize
from visualizations import (
    plot_waveform, plot_spectrogram, plot_mel_spectrogram,
    plot_feature_correlation, plot_feature_boxplots,
    create_radar_chart, plot_pitch_contour, plot_energy_curve,
    save_all_visualizations
)

# Set page config
st.set_page_config(
    page_title="MemoTag - Cognitive Decline Detection",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()

# Initialize session state for recording
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = 'stopped'

def process_audio_with_google_stt(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            
            # Use Google Speech Recognition
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand the audio. Please ensure clear speech.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def save_audio_bytes_to_wav(audio_bytes):
    """Convert audio bytes to WAV format"""
    # The audio recorder gives us raw audio data at 48000Hz, 16-bit, 1 channel
    sample_rate = 48000
    channels = 1
    sample_width = 2  # 16-bit

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    
    return wav_buffer.getvalue()

# Title and description
st.title("🎤 MemoTag - Voice-Based Cognitive Decline Detection")
st.markdown("""
This tool analyzes voice samples to detect potential indicators of cognitive decline.
Upload an audio file or record directly to get started.
""")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Basic Analysis", "Detailed Analysis", "Full Cognitive Assessment"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses advanced audio processing and machine learning to analyze speech patterns
    that may indicate early signs of cognitive decline.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload Audio")
    audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
    
    if audio_file is not None:
        # First handle speech-to-text conversion
        with st.spinner("Converting speech to text..."):
            try:
                # Save the uploaded file temporarily
                temp_input_path = "temp_input_audio"
                temp_wav_path = "temp_audio.wav"
                
                # Save uploaded file
                with open(temp_input_path, "wb") as f:
                    f.write(audio_file.getvalue())
                
                try:
                    # Determine the format from the uploaded file
                    file_extension = audio_file.name.split('.')[-1].lower()
                    
                    # Load audio based on format and convert to WAV
                    if file_extension == 'mp3':
                        audio = AudioSegment.from_mp3(temp_input_path)
                    elif file_extension == 'ogg':
                        audio = AudioSegment.from_ogg(temp_input_path)
                    elif file_extension == 'wav':
                        audio = AudioSegment.from_wav(temp_input_path)
                    else:
                        st.error(f"Unsupported file format: {file_extension}")
                        raise ValueError(f"Unsupported file format: {file_extension}")
                    
                    audio = audio.set_channels(1)  
                    audio = audio.set_frame_rate(16000)  
                    
                    # Export as WAV
                    audio.export(temp_wav_path, format="wav")
                    
                    # Perform speech recognition with Google STT
                    transcribed_text = process_audio_with_google_stt(temp_wav_path)
                    
                    if transcribed_text:
                        st.subheader("Speech Recognition Results")
                        st.success("Text transcription completed successfully!")
                        
                        st.text_area("Transcribed Text", transcribed_text, height=150)
                        
                        # Basic Statistics Section
                        st.subheader("Basic Statistics")
                        words = transcribed_text.split()
                        word_count = len(words)
                        
                        stats_cols = st.columns(4)
                        with stats_cols[0]:
                            st.metric("Total Words", word_count)
                        with stats_cols[1]:
                            unique_words = len(set(words))
                            st.metric("Unique Words", unique_words)
                        with stats_cols[2]:
                            vocabulary_richness = round((unique_words / word_count) * 100, 2) if word_count > 0 else 0
                            st.metric("Vocabulary Richness", f"{vocabulary_richness}%")
                        with stats_cols[3]:
                            pauses = len([c for c in transcribed_text if c in ',.?!'])
                            st.metric("Estimated Pauses", pauses)
                        
                        # Audio Analysis Section
                        st.subheader("Audio Analysis")
                        
                        # Load and preprocess audio
                        y, sr = librosa.load(temp_wav_path)
                        
                        # Create tabs for different visualizations
                        viz_tabs = st.tabs([
                            "Waveform & Energy", 
                            "Spectrograms", 
                            "Pitch Analysis",
                            "Feature Analysis"
                        ])
                        
                        with viz_tabs[0]:
                            st.markdown("### Waveform and Energy Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Audio Waveform")
                                fig = plot_waveform(y, sr)
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown("#### Energy Curve")
                                fig = plot_energy_curve(y, sr)
                                st.pyplot(fig)
                        
                        with viz_tabs[1]:
                            st.markdown("### Spectrogram Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Standard Spectrogram")
                                fig = plot_spectrogram(y, sr)
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown("#### Mel Spectrogram")
                                fig = plot_mel_spectrogram(y, sr)
                                st.pyplot(fig)
                        
                        with viz_tabs[2]:
                            st.markdown("### Pitch Analysis")
                            fig = plot_pitch_contour(y, sr)
                            st.pyplot(fig)
                            
                            # Process audio for detailed analysis
                            results = st.session_state.audio_processor.process_audio(temp_wav_path)
                            
                            # Display pitch statistics
                            pitch_cols = st.columns(3)
                            with pitch_cols[0]:
                                st.metric("Average Pitch", f"{results['features'].get('pitch_mean', 0):.2f} Hz")
                            with pitch_cols[1]:
                                st.metric("Pitch Variation", f"{results['features'].get('pitch_std', 0):.2f}")
                            with pitch_cols[2]:
                                st.metric("Jitter", f"{results['features'].get('jitter', 0):.4f}")
                        
                        with viz_tabs[3]:
                            st.markdown("### Feature Analysis")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Feature Correlation")
                                features_df = pd.DataFrame([results['features']])
                                fig = plot_feature_correlation(features_df)
                                st.pyplot(fig)
                            
                            with col2:
                                st.markdown("#### Feature Distribution")
                                fig = plot_feature_boxplots(features_df)
                                st.pyplot(fig)
                            
                            st.markdown("#### Speech Features Overview")
                            features_dict = {
                                'pitch': results['features'].get('pitch_mean', 0),
                                'intensity': results['features'].get('energy_mean', 0),
                                'jitter': results['features'].get('jitter', 0),
                                'shimmer': results['features'].get('shimmer', 0),
                                'hnr': results['features'].get('hnr', 0),
                                'risk_score': results.get('risk_score', 0)
                            }
                            fig = create_radar_chart(features_dict)
                            st.plotly_chart(fig)
                        
                        # Save visualizations to directory
                        try:
                            if not os.path.exists("visualizations"):
                                os.makedirs("visualizations")
                            save_all_visualizations(y, sr, results['features'])
                            st.success("All visualizations have been saved to the 'visualizations' directory!")
                        except Exception as e:
                            st.warning(f"Could not save visualizations: {str(e)}")
                        
                        # Remove risk assessment from here since it will be in the side column
                
                finally:
                    if os.path.exists(temp_input_path):
                        os.remove(temp_input_path)
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)
            
            except Exception as e:
                st.error(f"""
                An error occurred during processing: {str(e)}
                
                Please ensure:
                - The audio file is not corrupted
                - The audio contains clear speech
                - The file format is WAV, MP3, or OGG
                - The audio duration is not too long
                - You have a stable internet connection
                """)

with col2:
    st.header("Recording")
    
    st.write("Click the microphone button below to start/stop recording:")
    
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e8576e",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x"
    )
    
    if audio_bytes:
        try:
            wav_bytes = save_audio_bytes_to_wav(audio_bytes)
            
            with open("temp_recording.wav", "wb") as f:
                f.write(wav_bytes)
            
            st.success("Recording saved! Processing audio...")
            
            try:
                transcribed_text = process_audio_with_google_stt("temp_recording.wav")
                
                if transcribed_text:
                    st.success("Recording processed successfully!")
                    
                    st.text_area("Transcribed Text", transcribed_text, height=150)
                    
                    st.markdown("---")
                    
                    words = transcribed_text.split()
                    word_count = len(words)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Words", word_count)
                    with col2:
                        unique_words = len(set(words))
                        st.metric("Unique Words", unique_words)
                    with col3:
                        vocabulary_richness = round((unique_words / word_count) * 100, 2) if word_count > 0 else 0
                        st.metric("Vocabulary Richness", f"{vocabulary_richness}%")
                    
                    results = st.session_state.audio_processor.process_audio("temp_recording.wav")
                    results['transcribed_text'] = transcribed_text
            
            except Exception as e:
                st.error(f"Error processing recording: {str(e)}")
            
            finally:
                if os.path.exists("temp_recording.wav"):
                    os.remove("temp_recording.wav")
        
        except Exception as e:
            st.error(f"""
            Error saving recording: {str(e)}
            
            Please try:
            1. Using a different browser (Chrome recommended)
            2. Checking your microphone permissions
            3. Ensuring your microphone is working properly
            """)
    
    with st.expander("Recording Instructions"):
        st.markdown("""
        ### How to Record:
        1. Click the microphone button to start recording
        2. Click it again to stop recording
        3. Wait for the analysis to complete
        
        ### Tips for Better Results:
        - Use a quiet environment
        - Speak clearly and at a normal pace
        - Keep the microphone at a consistent distance
        - Avoid background noise
        
        ### Troubleshooting:
        - Make sure your browser has permission to use the microphone
        - If no sound is recorded, check your microphone settings
        - Try refreshing the page if the recorder doesn't respond
        """)
    
    st.markdown("---")
    
    # Risk Assessment Section in side column
    st.header("Risk Assessment")
    if 'results' in locals() and results.get('risk_score') is not None:
        risk_score = results.get('risk_score', 0)
        
        # Display risk score with color coding
        risk_color = (
            "#e74c3c" if risk_score > 0.7 else
            "#f39c12" if risk_score > 0.4 else
            "#2ecc71"
        )
        
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {risk_color}22; border: 2px solid {risk_color}'>
            <h3 style='color: {risk_color}'>Risk Score Assessment</h3>
            <p style='font-size: 24px; font-weight: bold;'>{risk_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if risk_score > 0.7:
            st.error("High risk of cognitive decline detected")
            st.write("""
            **Key Concerns:**
            - High pause frequency
            - Reduced speech rate
            - Significant hesitation markers
            """)
        elif risk_score > 0.4:
            st.warning("Moderate risk of cognitive decline detected")
            st.write("""
            **Areas to Monitor:**
            - Speech patterns
            - Word recall
            - Sentence complexity
            """)
        else:
            st.success("Low risk of cognitive decline detected")
            st.write("""
            **Healthy Indicators:**
            - Normal speech rate
            - Good word diversity
            - Clear articulation
            """)
        
        # Feature Importance visualization
        st.subheader("Feature Importance")
        feature_importance = {
            'Speech Rate': 0.25,
            'Pause Patterns': 0.20,
            'Hesitation': 0.15,
            'Word Diversity': 0.15,
            'Sentence Structure': 0.15,
            'Voice Quality': 0.10
        }
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=list(feature_importance.values()),
                   y=list(feature_importance.keys()), ax=ax)
        plt.title("Feature Importance in Risk Assessment")
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")
st.markdown("""
### Methodology
This tool employs a comprehensive approach to assess cognitive function through speech analysis:

1. **Acoustic Analysis**
   - Speech rate and rhythm
   - Pause patterns and frequency
   - Voice quality (jitter, shimmer)
   - Pitch variation

2. **Linguistic Analysis**
   - Word diversity and complexity
   - Sentence structure
   - Hesitation markers
   - Word recall patterns

3. **Cognitive Assessment**
   - Memory indicators
   - Processing speed
   - Language coherence
   - Task completion

4. **Machine Learning Approach**
   - Feature extraction and normalization
   - Anomaly detection
   - Risk score calculation
   - Pattern recognition
""") 