import streamlit as st
from src.query import ask

st.title("Bilingual Annual Report Assistant")
st.caption("Ask questions in English or French")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Searching the reports..."):
        response = ask(question)
    st.subheader("Answer")
    st.write(response["output"]["text"])
    # TODO: show the sources under the answer (reuse your loop from show())