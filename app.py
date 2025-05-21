import streamlit as st
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load the FAQ data
faq_df = pd.read_csv("faq.csv")
faq_df["content"] = "Q: " + faq_df["Question"] + "\nA: " + faq_df["Answer"]
loader = DataFrameLoader(faq_df[["content"]], page_content_column="content")
documents = loader.load()

# Split the documents
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create retriever and QA chain
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit interface
st.title("🎧 RAG Chatbot - FAQ Assistant")

query = st.text_input("Ask me something about our service:")
if query:
    response = qa_chain.run(query)
    st.markdown(f"**Answer:** {response}")