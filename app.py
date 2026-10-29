
import streamlit as st
from src.paperlens.comparison import answer_question

st.set_page_config(
    page_title="PaperLens",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PaperLens")
st.subheader("Research Paper Question Answering")

st.write(
    "Ask questions about five diabetic retinopathy research papers "
    "and explore evidence-backed answers with page citations."
)

question = st.text_input(
    "Enter your question",
    placeholder="e.g. Compare the approaches used by the five papers.",
)

if st.button("Ask PaperLens"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching research papers and generating an answer..."):
            try:
                result = answer_question(question)

                st.subheader("Answer")
                st.markdown(result["answer"])

                if result["sources"]:
                    st.subheader("Retrieved sources")
                    for source in result["sources"]:
                        st.write(f"- {source}")

            except Exception as error:
                st.error(f"Something went wrong: {error}")

