import streamlit as st
from src.rag_engine import build_conversational_rag_chain

st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")
st.title("PDF Document Assistant")

if "chain" not in st.session_state:
    st.session_state.chain = build_conversational_rag_chain()

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_" + str(id(st.session_state))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your uploaded PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            response = st.session_state.chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            answer = response["answer"]
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
