
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Step 1: Load the FAQ CSV file
faq_df = pd.read_csv("faq.csv")

# Step 2: Convert DataFrame to Documents
faq_df["content"] = "Q: " + faq_df["Question"] + "\nA: " + faq_df["Answer"]
loader = DataFrameLoader(faq_df[["content"]], page_content_column="content")
documents = loader.load()

# Step 3: Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 4: Embed the chunks
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 5: Set up Retriever and RAG pipeline
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 6: Chat Loop
print("Welcome to the RAG Chatbot! Type 'exit' to quit.")
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        break
    response = qa_chain.run(query)
    print("Bot:", response)
