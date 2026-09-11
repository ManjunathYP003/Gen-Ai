import streamlit as st

from Rag.RAG_BasedKnowledgeRetrievalSystemminiProj.rag_backend import process_urls, generate_answer


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="RAG URL Assistant",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🤖 RAG URL Assistant")

st.write(
    "Enter one or more URLs, process them, and ask questions "
    "based on their content."
)


# -----------------------------
# URL inputs
# -----------------------------

st.subheader("1. Enter URLs")

url1 = st.text_input(
    "URL 1",
    placeholder="https://example.com"
)

url2 = st.text_input(
    "URL 2",
    placeholder="https://example.com"
)

url3 = st.text_input(
    "URL 3",
    placeholder="https://example.com"
)


# -----------------------------
# Process URLs
# -----------------------------

if st.button("Process URLs"):

    urls = [
        url for url in [url1, url2, url3]
        if url.strip()
    ]

    if not urls:

        st.warning("Please enter at least one URL.")

    else:

        with st.spinner("Loading and processing URLs..."):

            try:

                number_of_chunks = process_urls(urls)

                st.success(
                    f"Successfully processed {number_of_chunks} chunks."
                )

            except Exception as e:

                st.error(
                    f"Error while processing URLs: {e}"
                )


# -----------------------------
# Question
# -----------------------------

st.subheader("2. Ask a Question")

question = st.text_input(
    "Question",
    placeholder="What is GPT-6 Astra?"
)


# -----------------------------
# Generate answer
# -----------------------------

if st.button("Ask Question"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating answer..."):

            try:

                answer, sources = generate_answer(question)

                st.subheader("Answer")

                st.write(answer)

                if sources:

                    st.subheader("Sources")

                    st.write(sources)

            except Exception as e:

                st.error(
                    f"Error while generating answer: {e}"
                )