# MemoTag - Speech Analysis and Visualization Tool

## Overview
MemoTag is a powerful web-based application that provides real-time speech analysis and visualization. It processes both uploaded audio files and live recordings to generate comprehensive audio analysis and visualizations. The tool combines speech-to-text capabilities with advanced audio feature analysis to provide insights into speech patterns.

## Features
- **Audio Input Options**
  - Upload audio files (WAV, MP3, OGG)
  - Live recording through browser microphone
  
- **Speech Analysis**
  - Speech-to-text transcription using Google Speech Recognition
  - Word count and vocabulary richness analysis
  - Pause pattern detection
  - Speech rate analysis

- **Visualizations**
  1. **Audio Analysis**
     - Waveform visualization
     - Standard spectrogram
     - Mel spectrogram
     - Pitch contour analysis
     - Energy curve visualization
  
  2. **Feature Analysis**
     - Feature correlation heatmap
     - Feature distribution boxplots
     - Speech features radar chart
     - Feature importance visualization

- **Risk Assessment**
  - Cognitive decline risk scoring
  - Detailed analysis of speech patterns
  - Visual risk indicators
  - Feature importance breakdown

## Setup
1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. FFmpeg is included in the project directory for audio processing

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Access the web interface through your browser (typically http://localhost:8501)
3. Choose your input method:
   - Upload an audio file
   - Use the microphone recorder
4. View the generated visualizations and analysis in real-time

## Visualization Outputs
The application generates several visualization files in the `visualizations` directory:
- `waveform.png` - Audio waveform visualization
- `spectrogram.png` - Standard spectrogram
- `mel_spectrogram.png` - Mel-scale spectrogram
- `pitch_contour.png` - Pitch analysis
- `energy_curve.png` - Energy distribution
- `correlation_heatmap.png` - Feature correlations
- `feature_boxplots.png` - Feature distributions
- `radar_chart.html` - Interactive feature overview

## Technical Requirements
- Python 3.7+
- Web browser with microphone access (for recording feature)
- Internet connection (for speech-to-text functionality)
- FFmpeg (included in the project)

## File Structure
```
memotag/
├── app.py                 # Main Streamlit application
├── audio_processor.py     # Audio processing functions
├── visualizations.py      # Visualization functions
├── requirements.txt       # Project dependencies
├── ffmpeg/               # FFmpeg binaries
└── visualizations/       # Generated visualization outputs
```

## Notes
- For best results, use a quiet environment when recording
- Supported audio formats: WAV, MP3, OGG
- Speech recognition requires an active internet connection
- Visualizations are automatically saved for each analysis

## Troubleshooting
- Ensure microphone permissions are enabled in your browser
- Check internet connectivity for speech recognition
- Verify audio file format and quality
- Make sure all dependencies are properly installed

 