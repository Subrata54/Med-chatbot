from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.prompt import system_prompt


# --------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# --------------------------------
# FLASK APP
# --------------------------------

app = Flask(__name__)


# --------------------------------
# RAG CHAIN
# --------------------------------

def create_rag_chain():

    print("Loading Hugging Face embeddings...")

    embeddings = download_hugging_face_embeddings()

    print("Connecting to Pinecone...")

    index_name = "med-chatbot"

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    print("Loading OpenAI chat model...")

    chat_model = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        chat_model,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    print("RAG chain ready.")

    return rag_chain


# --------------------------------
# CREATE RAG CHAIN
# --------------------------------

rag_chain = create_rag_chain()


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


# --------------------------------
# CHAT API
# --------------------------------

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question.", 400

    print("Question:", msg)

    response = rag_chain.invoke({
        "input": msg
    })

    answer = response.get("answer", "I don't know.")

    print("Response:", answer)

    return str(answer)


# --------------------------------
# LOCAL DEVELOPMENT
# --------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )