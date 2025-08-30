# Fix SQLite issue before importing ChromaDB
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import tempfile
import whisper
from pytube import YouTube
import yt_dlp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
import chromadb
from typing import List, Dict, Any
import json


class YouTubeQuizzerPipeline:
    def __init__(self, groq_api_key: str | None = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Set it in .env or pass to constructor.")
        os.environ["GROQ_API_KEY"] = self.groq_api_key

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        self.transcript = ""

    def download_youtube_audio(self, url: str) -> str:
        try:
            url = self._clean_youtube_url(url)

            try:
                return self._download_with_pytube(url)
            except Exception as pytube_error:
                print(f"Pytube failed, trying yt-dlp: {pytube_error}")
                return self._download_with_ytdlp(url)

        except Exception as e:
            raise Exception(f"Error downloading YouTube video: {str(e)}")

    def _download_with_pytube(self, url: str) -> str:
        yt = YouTube(url)

        import time
        time.sleep(1)

        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            raise Exception("No audio stream found for this video")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = temp_file.name
        temp_file.close()

        audio_stream.download(filename=temp_path)
        return temp_path

    def _download_with_ytdlp(self, url: str) -> str:
        temp_dir = tempfile.mkdtemp()
        print(f"Downloading to temp directory: {temp_dir}")

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        import time
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                downloaded_files = [f for f in os.listdir(temp_dir)
                                    if f.endswith(('.m4a', '.mp3', '.mp4', '.webm'))]
                print(f"Found downloaded files: {downloaded_files}")

                if not downloaded_files:
                    time.sleep(1)
                    retry_count += 1
                    continue

                audio_path = os.path.join(temp_dir, downloaded_files[0])
                print(f"Audio file path: {audio_path}")

                if not os.path.exists(audio_path):
                    time.sleep(1)
                    retry_count += 1
                    continue

                file_size = os.path.getsize(audio_path)
                print(f"File size: {file_size} bytes")

                if file_size == 0:
                    time.sleep(1)
                    retry_count += 1
                    continue

                with open(audio_path, 'rb') as test_file:
                    test_file.read(1024)

                print(f"File is ready for transcription: {audio_path}")
                return audio_path

            except Exception as e:
                print(f"File not ready yet (attempt {retry_count + 1}): {e}")
                time.sleep(1)
                retry_count += 1

        raise Exception(
            "Failed to prepare audio file for transcription after multiple attempts")

    def _clean_youtube_url(self, url: str) -> str:
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('v=')[1].split('&')[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
            return f"https://www.youtube.com/watch?v={video_id}"
        else:
            return url

    def validate_youtube_url(self, url: str) -> bool:
        try:
            url = self._clean_youtube_url(url)
            yt = YouTube(url)
            _ = yt.title
            return True
        except Exception:
            return False

    def transcribe_audio(self, audio_path: str) -> str:
        temp_dir = None
        try:
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file not found: {audio_path}")

            print(f"Starting transcription of: {audio_path}")

            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            self.transcript = result["text"]
            print(
                f"Transcription completed. Length: {len(self.transcript)} characters")
            return self.transcript

        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise Exception(f"Error transcribing audio: {str(e)}")
        finally:
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    print(f"Cleaned up audio file: {audio_path}")

                    temp_dir = os.path.dirname(audio_path)
                    if temp_dir and os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                        try:
                            os.rmdir(temp_dir)
                            print(f"Cleaned up temp directory: {temp_dir}")
                        except:
                            pass
            except Exception as cleanup_error:
                print(f"Cleanup error (ignored): {cleanup_error}")
                pass

    def create_vector_store(self, text: str) -> Chroma:
        try:
            chunks = self.text_splitter.split_text(text)
            self.vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                collection_name="youtube_content"
            )
            return self.vector_store
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")

    def generate_quiz(self, num_questions: int = 5) -> List[Dict[str, Any]]:
        if not self.vector_store:
            raise Exception(
                "Vector store not initialized. Please process a video first.")

        quiz_prompt = PromptTemplate(
            input_variables=["context", "num_questions"],
            template="""
            You are a quiz generator. Based on the following context from a YouTube video, create exactly {num_questions} multiple-choice quiz questions.
            
            Context:
            {context}
            
            IMPORTANT: You must respond with ONLY valid JSON. No other text before or after the JSON.
            
            Create {num_questions} questions in this exact JSON format:
            [
                {{
                    "question": "What is the main topic discussed in the video?",
                    "options": {{
                        "A": "Option A text",
                        "B": "Option B text", 
                        "C": "Option C text",
                        "D": "Option D text"
                    }},
                    "correct_answer": "A",
                    "explanation": "Brief explanation of why this answer is correct"
                }}
            ]
            
            Rules:
            1. Return ONLY the JSON array, no other text
            2. Make questions specific to the video content
            3. Ensure all options are plausible but only one is correct
            4. Keep explanations concise but informative
            """
        )

        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
            docs = retriever.invoke("")
            context = "\n".join([doc.page_content for doc in docs])

            chain = quiz_prompt.format_prompt(
                context=context, num_questions=num_questions)
            response = self.llm.invoke(chain.to_string())

            response_text = response.content.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            quiz_data = json.loads(response_text)
            return quiz_data

        except json.JSONDecodeError as e:
            raise Exception(
                f"Error parsing quiz JSON: {str(e)}. Response was: {response.content[:200]}...")
        except Exception as e:
            raise Exception(f"Error generating quiz: {str(e)}")

    def answer_question(self, question: str) -> str:
        if not self.vector_store:
            raise Exception(
                "Vector store not initialized. Please process a video first.")

        try:
            prompt = PromptTemplate(
                input_variables=["context", "input"],
                template=(
                    "You are a helpful tutor. Use the context to answer the question.\n"
                    "Context:\n{context}\n\nQuestion: {input}\nAnswer:"
                ),
            )
            doc_chain = create_stuff_documents_chain(self.llm, prompt)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            chain = create_retrieval_chain(retriever, doc_chain)
            result = chain.invoke({"input": question})
            return result.get("answer", "")

        except Exception as e:
            raise Exception(f"Error answering question: {str(e)}")

    def process_video(self, youtube_url: str) -> str:
        try:
            audio_path = self.download_youtube_audio(youtube_url)
            transcript = self.transcribe_audio(audio_path)
            self.create_vector_store(transcript)
            return transcript
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

    def get_video_info(self, url: str) -> Dict[str, str]:
        try:
            url = self._clean_youtube_url(url)

            try:
                return self._get_info_with_pytube(url)
            except Exception as pytube_error:
                print(f"Pytube failed, trying yt-dlp: {pytube_error}")
                return self._get_info_with_ytdlp(url)

        except Exception as e:
            raise Exception(f"Error getting video info: {str(e)}")

    def _get_info_with_pytube(self, url: str) -> Dict[str, str]:
        yt = YouTube(url)

        import time
        time.sleep(1)

        return {
            "title": yt.title or "Unknown Title",
            "author": yt.author or "Unknown Author",
            "length": str(yt.length) + " seconds" if yt.length else "Unknown Length",
            "views": str(yt.views) if yt.views else "Unknown Views",
            "description": (yt.description[:200] + "...") if yt.description and len(yt.description) > 200 else (yt.description or "No description available")
        }

    def _get_info_with_ytdlp(self, url: str) -> Dict[str, str]:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return {
            "title": info.get('title', 'Unknown Title'),
            "author": info.get('uploader', 'Unknown Author'),
            "length": str(info.get('duration', 0)) + " seconds" if info.get('duration') else "Unknown Length",
            "views": str(info.get('view_count', 0)) if info.get('view_count') else "Unknown Views",
            "description": (info.get('description', '')[:200] + "...") if info.get('description') and len(info.get('description', '')) > 200 else (info.get('description', '') or "No description available")
        }
