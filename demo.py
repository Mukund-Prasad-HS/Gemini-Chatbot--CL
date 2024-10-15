import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
os.getenv("GOOGLE_API_KEY")

from google.generativeai import configure

configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():
    prompt_template = """
    You are an advanced AI assistant capable of handling a wide range of topics. Please follow these guidelines:

    1. Provide concise and accurate answers.
    2. For resume-related questions, suggest improvements or provide detailed insights based on best practices.
    3. For programming-related queries, offer clear explanations, examples, or code snippets to solve problems.
    4. Use structured formatting like bullet points, lists, or steps when appropriate.
    5. If a question is ambiguous, ask clarifying questions before providing an answer.

    Question: {question}

    Response:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    return model, prompt


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls = [c for c in self.calls if c > now - self.period]
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded. Please try again later.")
            self.calls.append(now)
            return f(*args, **kwargs)

        return wrapper


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@RateLimiter(max_calls=10, period=60)
def user_input_with_retry(user_question):
    model, prompt = get_conversational_chain()
    response = model.predict(user_question)
    return response


def main():
    st.set_page_config(page_title="AI Chatbot", page_icon=":speech_balloon:", layout="wide")
    st.header("AI Chatbot 💬🤖")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""

    with st.sidebar:
        user_name = st.text_input("Your Name", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name

        if st.checkbox("Dark Mode"):
            st.markdown("""
                <style>
                    .stApp {
                        background-color: #2b2b2b;
                        color: white;
                    }
                </style>
            """, unsafe_allow_html=True)

    st.subheader("Chat")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")

    user_question = st.chat_input("Ask a question about any topic")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner("Thinking...."):
            try:
                response = user_input_with_retry(user_question)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please try again later.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
