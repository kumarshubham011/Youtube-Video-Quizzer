import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import YouTubeQuizzerPipeline
import json

load_dotenv()

st.set_page_config(
    page_title="YouTube Video Quizzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .quiz-question {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = []
    if 'show_answers' not in st.session_state:
        st.session_state.show_answers = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}


def main():
    initialize_session_state()

    st.markdown('<h1 class="main-header">🎓 YouTube Video Quizzer</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform any YouTube video into an interactive learning experience</p>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔑 Setup (Groq)")
        groq_key = st.text_input(
            "GROQ API Key",
            type="password",
            help="Enter your Groq API key"
        )

        if groq_key:
            if st.session_state.pipeline is None:
                try:
                    st.session_state.pipeline = YouTubeQuizzerPipeline(
                        groq_api_key=groq_key)
                    st.success("✅ Groq configured successfully!")
                except Exception as e:
                    st.error(f"❌ Error configuring Groq: {str(e)}")

        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Enter your Groq API key
        2. Paste a YouTube URL
        3. Wait for video processing
        4. Generate quiz questions
        5. Test your knowledge!
        6. Ask follow-up questions
        """)

        st.markdown("---")
        st.markdown("### ⚠️ Note")
        st.markdown("""
        - Video processing may take several minutes
        - Longer videos require more processing time
        - Ensure you have a stable internet connection
        """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎥 Video Input")

        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the full YouTube URL here"
        )

        if youtube_url and st.session_state.pipeline:
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Processing video... This may take several minutes."):
                    try:
                        video_info = st.session_state.pipeline.get_video_info(
                            youtube_url)

                        st.subheader("📺 Video Information")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write(f"**Title:** {video_info['title']}")
                            st.write(f"**Author:** {video_info['author']}")
                        with col_info2:
                            st.write(f"**Length:** {video_info['length']}")
                            st.write(f"**Views:** {video_info['views']}")

                        st.write(
                            f"**Description:** {video_info['description']}")

                        transcript = st.session_state.pipeline.process_video(
                            youtube_url)
                        st.session_state.video_processed = True

                        st.success("✅ Video processed successfully!")
                        st.session_state.chat_history = []

                        with st.expander("📝 View Transcript (First 500 characters)"):
                            st.text(
                                transcript[:500] + "..." if len(transcript) > 500 else transcript)

                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")

        if st.session_state.video_processed:
            st.header("📝 Quiz Generation")

            col_quiz1, col_quiz2 = st.columns([2, 1])
            with col_quiz1:
                num_questions = st.slider("Number of questions:", 3, 10, 5)

            with col_quiz2:
                if st.button("🎯 Generate Quiz", type="primary"):
                    with st.spinner("Generating quiz questions..."):
                        try:
                            quiz_data = st.session_state.pipeline.generate_quiz(
                                num_questions)
                            st.session_state.quiz_data = quiz_data
                            st.session_state.quiz_generated = True
                            st.session_state.show_answers = False
                            st.session_state.user_answers = {}
                            st.success(
                                f"✅ Generated {len(quiz_data)} quiz questions!")
                        except Exception as e:
                            st.error(f"❌ Error generating quiz: {str(e)}")

    with col2:
        st.header("📊 Quick Stats")
        if st.session_state.video_processed:
            st.metric("Video Processed", "✅ Yes")
            if st.session_state.quiz_generated:
                st.metric("Quiz Generated",
                          f"✅ {len(st.session_state.quiz_data)} questions")
            else:
                st.metric("Quiz Generated", "❌ No")
        else:
            st.metric("Video Processed", "❌ No")
            st.metric("Quiz Generated", "❌ No")

    if st.session_state.quiz_generated and st.session_state.quiz_data:
        st.header("🧠 Quiz Questions")

        col_toggle1, col_toggle2 = st.columns([3, 1])
        with col_toggle2:
            if st.button("👁 Show/Hide Answers"):
                st.session_state.show_answers = not st.session_state.show_answers

        for i, question_data in enumerate(st.session_state.quiz_data):
            with st.container():
                st.markdown(f'<div class="quiz-question">',
                            unsafe_allow_html=True)
                st.subheader(f"Question {i+1}")
                st.write(question_data["question"])

                options = question_data["options"]

                # Create radio options with no default selection
                option_keys = list(options.keys())
                option_labels = [
                    f"{key}: {options[key]}" for key in option_keys]

                # Use index=-1 to have no default selection, but this doesn't work in streamlit
                # Instead, we'll use a selectbox or handle it differently
                user_answer = st.radio(
                    f"Select your answer for question {i+1}:",
                    options=option_keys,
                    format_func=lambda x: f"{x}: {options[x]}",
                    key=f"question_{i}",
                    index=None,  # This sets no default selection
                    label_visibility="collapsed"
                )

                if user_answer:
                    st.session_state.user_answers[i] = user_answer

                if st.session_state.show_answers and user_answer:
                    correct_answer = question_data["correct_answer"]
                    explanation = question_data["explanation"]

                    if user_answer == correct_answer:
                        st.success(f"✅ Correct! {explanation}")
                    else:
                        st.error(
                            f"❌ Incorrect. The correct answer is {correct_answer}: {options[correct_answer]}")
                        st.info(f"💡 {explanation}")

                st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.video_processed:
        st.header("💬 Ask Questions About the Video")

        user_question = st.text_input(
            "Ask a question about the video content:",
            placeholder="What was the main topic discussed?",
            key="chat_input"
        )

        if user_question and st.button("🤖 Ask"):
            if st.session_state.pipeline:
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.pipeline.answer_question(
                            user_question)
                        st.session_state.chat_history.append({
                            "user": user_question,
                            "bot": answer
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error getting answer: {str(e)}")

        if st.session_state.chat_history:
            st.subheader("💭 Chat History")
            for message in st.session_state.chat_history:
                st.markdown(f'<div class="chat-message user-message">',
                            unsafe_allow_html=True)
                st.write(f"**You:** {message['user']}")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="chat-message bot-message">',
                            unsafe_allow_html=True)
                st.write(f"**Assistant:** {message['bot']}")
                st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


if __name__ == "__main__":
    main()
