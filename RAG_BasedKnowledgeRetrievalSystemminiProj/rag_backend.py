# %%
from pathlib import Path


# %%
import sys
print(sys.executable)

# !{sys.executable} -m pip install -U langchain-huggingface

# %%
llm=None
vector_store=None

# %%
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQAWithSourcesChain

# %%
#  !pip install -U langchain-chroma

# %%
# urls = [
#    'https://www.blockchain-council.org/ai/gpt-6-astra-openai-frontier-llm/'
   
# ]

# %%
load_dotenv()

# %%
ch_size=1000
ch_overlap=10
Embeding_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
vector_dir=Path.cwd()/"resources"/"vectorstore"
Collection_name='Reviwer'
Embedding_model = "sentence-transformers/all-MiniLM-L6-v2"



# %%
ef = HuggingFaceEmbeddings(
    model_name=Embedding_model
)

def initialize_components():
    global llm,vector_store
    if llm==None:
        llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0.9,max_tokens=1000) 
    if vector_store==None:
        vector_store=Chroma(collection_name=Collection_name,persist_directory=str(vector_dir),
                                embedding_function=ef)

# %%


# %%
def process_urls(urls):
    initialize_components()

    loader=WebBaseLoader(urls)
    data=loader.load()
    textsplitter=RecursiveCharacterTextSplitter(
        chunk_size=ch_size,chunk_overlap=ch_overlap
    )
    docs=textsplitter.split_documents(data)
    vector_store.add_documents(docs)



# %%
# process_urls(urls)

# %%
def generate_answer(query):
    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
    result=chain.invoke({"question":query},return_only_output=True)
    sources=result.get("sources","")
    return result['answer'],sources


# %%
# answer,sources=generate_answer("What Is GPT 6 Astra?")
# print("Answer:{}".format(answer))
# print("Sources{}".format(sources))

# %%
# print(llm)

# %%



