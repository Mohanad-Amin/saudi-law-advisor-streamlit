import streamlit as st
import os
from dotenv import load_dotenv

# Make sure you have an __init__.py file in the 'core' directory
from core.law_retriever import LawRetriever

# --- Page Configuration ---
st.set_page_config(
    page_title="المستشار القانوني الذكي",
    page_icon="⚖️",
    layout="centered"
)

# --- Environment Variables & Secrets ---
load_dotenv()
# For local development, create a .env file with your OPENAI_API_KEY
# For Streamlit Cloud, set the secret in the app settings
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("مفتاح OpenAI API غير موجود. الرجاء إضافته لملف .env أو في إعدادات التطبيق السحابية.")
    st.stop()

# --- Load Retriever (Cached for Performance) ---
@st.cache_resource
def load_retriever_system():
    """Loads the LawRetriever instance using Streamlit's caching."""
    print("Initializing LawRetriever for the first time...")
    try:
        # We will use your fine-tuned model for consistency and to showcase your skills
        retriever_instance = LawRetriever(
            model_name="TheMohanad1/Fine-Tuned-E5",
            openai_api_key=openai_api_key
        )
        print("LawRetriever loaded successfully.")
        return retriever_instance
    except Exception as e:
        st.error(f"حدث خطأ فادح أثناء تهيئة النظام: {e}")
        st.stop()

retriever = load_retriever_system()

# --- Chat Interface ---
st.title("⚖️ المستشار القانوني الذكي")
st.caption("مساعدك لفهم الأنظمة والقوانين السعودية")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("اسأل عن نظام العمل, الشركات, ..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("...أبحث في نصوص المواد القانونية"):
            
            # The retriever will access the full history from session_state
            result = retriever.search_and_answer(
                query=prompt,
                chat_history=st.session_state.messages
            )
            
            response_text = result.get("answer", "عفواً، لم أتمكن من إيجاد إجابة.")
            sources = result.get("sources", [])

            st.markdown(response_text)
            
            # Display sources in an expander
            if sources:
                with st.expander("📚 عرض المصادر المستخدمة"):
                    for i, source in enumerate(sources):
                        # The source content is now nested in 'law_text'
                        st.write(f"**المصدر رقم [{i+1}]** (المادة الأصلية رقم: {source.get('source_index', 'N/A')})")
                        st.info(source.get('law_text', 'لا يوجد نص للمصدر.'))
                        st.markdown("---")

    # Add assistant response to chat history, including the sources for future re-ranking
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources # Store sources in session state
    })

