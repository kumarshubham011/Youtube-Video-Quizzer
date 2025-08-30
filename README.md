# YouTube Video Quizzer

Transform YouTube videos into interactive learning experiences with AI-generated quizzes and Q&A capabilities.

## Features

- **Video Processing**: Automatic transcription of YouTube videos
- **Smart Quizzing**: AI-generated multiple choice questions
- **Interactive Q&A**: Ask questions about video content
- **Clean Interface**: Streamlined web interface

## Prerequisites

- Python 3.8+
- Groq API key ([Get one free](https://console.groq.com/keys))
- FFmpeg for audio processing

## Quick Start

1. **Clone and install dependencies**

   ```bash
   git clone <repository-url>
   cd youtube-quizzer
   pip install -r requirements.txt
   ```

2. **Install FFmpeg**

   - Windows: Download from [ffmpeg.org](https://ffmpeg.org) and add to PATH
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Setup in browser**
   - Enter your Groq API key in the sidebar
   - Paste a YouTube URL
   - Process video and generate quiz

## Usage

1. Process any public YouTube video
2. Generate 3-10 quiz questions automatically
3. Test your understanding with interactive questions
4. Ask follow-up questions about the content

## Technology

Built with Streamlit, LangChain, and Groq's LLM API. Uses OpenAI Whisper for transcription and ChromaDB for vector storage.
