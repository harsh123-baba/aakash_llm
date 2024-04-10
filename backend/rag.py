
from time import time


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp

from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
template =  """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])
n_gpu_layers = 4  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Loading model,
llm = LlamaCpp(
    model_path="f:\Aakash\llama-2-7b-chat.Q8_0.gguf",
    max_tokens=1024,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
    n_ctx=4096, # Context window
    stop = ['USER:'], # Dynamic stopping when such token is detected.
    temperature = 0.4,
)

#idhar vo submtted document ana chahiye
loader = TextLoader("f:\Aakash\DOCUMENTS\ch31.pdf",
                encoding="latin-1")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
    )
print(qa)
from time import time    
def test_rag(query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    #this output should get printed
    print("\nResult: ", result)
    return result

ans = test_rag("what is your name")
print(ans)