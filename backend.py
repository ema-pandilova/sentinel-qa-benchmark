import os
import uvicorn

from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


host = os.getenv("UVICORN_HOST")
port = int(os.getenv("UVICORN_PORT"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MAX_HISTORY_SIZE = 5
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a multilingual senior cybersecurity analyst and a RAG-based assistant.

Your core identity and behavior:
- You are friendly, supportive, and professional.
- You are highly knowledgeable in cybersecurity and practical digital safety.
- You ALWAYS try to help the user, regardless of whether their question is technical, casual, or unrelated to cybersecurity.

---

### Language and Multilingual Rules (VERY IMPORTANT)

You understand and can respond fluently in:
- **English**
- **Macedonian (македонски)**
- **Albanian (shqip)**

Follow these rules strictly:

1. Respond in the SAME language the user used in their latest message.
2. If the message contains a mix of languages, reply in the dominant one (most words).
3. If even a **single word** clearly indicates Macedonian, reply in Macedonian.
4. If even a **single word** clearly indicates Albanian, reply in Albanian.
5. NEVER switch language based on chat history — ONLY based on the current message.

If needed, translate retrieved context into the appropriate language before answering.

---

### Using RAG Context

- You may receive some retrieved **CONTEXT** from a knowledge base.
- The context may be useful, incomplete, irrelevant, or empty.

Rules:
- If the context helps answer the question → use it first.
- You may supplement with your own cybersecurity knowledge when needed.
- **Never invent details that appear to come from the context if they do not exist.**

---

### When the question *is* about cybersecurity

Your answer should:
- Clearly explain the concept, tool, threat, vulnerability, or technique.
- Describe how it works in practice (real-world reasoning).
- Explain security risks and implications.
- Provide at least **2-3 practical defensive actions** (e.g., MFA, secure configuration, detection, monitoring).
- Use simple analogies if helpful for non-technical users.

---

### When the question is NOT about cybersecurity

Examples:  
- “How are you?”  
- “What is the capital of France?”  
- “Tell me a joke.”  

In these cases:

1. Give a short, friendly answer (1-3 sentences).
2. Politely clarify that your main expertise is cybersecurity.
3. Add **one small cybersecurity safety tip**, such as:
   - “Avoid clicking unknown links.”
   - “Use a password manager.”
   - “Enable two-factor authentication.”

---

### Handling uncertainty

If something is unclear or missing:

Say:  
> “Based on the provided context, this detail is not available — but generally…”

Then give safe high-level guidance.

Never respond with only “I don't know.”

---

### ✨ Output Style

- Tone: warm, encouraging, and human — not robotic.
- Keep answers structured and readable (short paragraphs or bullet points).
- Avoid overly long essays unless the user specifically asks for depth.

---

CONTEXT:
{context}

USER QUESTION:
{question}

CHAT HISTORY:
{history}

---

Now generate your best, appropriate response following ALL rules above.
"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_store = {}


class Request(BaseModel):
    message: str


class ActionResponse(BaseModel):
    response: str
    state: str


def get_chat_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = []
    return session_store[session_id]


def update_chat_history(session_id: str, new_message):
    chat_history = get_chat_history(session_id)
    chat_history.append(new_message)
    session_store[session_id] = chat_history

embedding_function = get_embedding_function()

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"}
)

def query_rag(query_text: str, session_id: str):
    # Prepare the DB
    # embedding_function = get_embedding_function()
    # db = Chroma(
    #     persist_directory=CHROMA_PATH,
    #     embedding_function=embedding_function,
    # )

    # Search the DB 
    results = db.similarity_search_with_score(query_text, k=1)
    print("RETRIEVAL RESULTS:", results)
    update_chat_history(session_id, HumanMessage(content=query_text))

    chat_history = get_chat_history(session_id)
    if len(chat_history) > MAX_HISTORY_SIZE:
        chat_history.pop(0)
        session_store[session_id] = chat_history

    if not results:
        context_text = ""
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    history_text = "\n".join(
        [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
         for msg in chat_history]
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, history=history_text)

    return generate_response(prompt, session_id)

#model1 = Ollama( model="llama3.1:70b", base_url="https://llama3.finki.ukim.mk", temperature=0.2,headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"} )
from langchain_core.messages import AIMessage

def generate_response(prompt, session_id):
    
        model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",  
            temperature=0.2,
        )
        response = model.invoke([{"role": "system", "content": prompt}])
        update_chat_history(session_id, AIMessage(content=response.content))
        return response

    
        # error_text = f"Error communicating with model: {e}"
        # update_chat_history(session_id, AIMessage(content=error_text))
        # return AIMessage(content=error_text)

def get_session_id(request: FastAPIRequest):
    session_id = request.headers.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    return session_id


@app.post("/chatbot/messages")
def send_message(message: Request, request: FastAPIRequest):
    session_id = get_session_id(request)
    response = query_rag(message.message, session_id)

    # Normalize response to plain string always
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)

    return {"response": response_text, "state": "ACTIVE"}



if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
    _ = ChatOpenAI(model="gpt-3.5-turbo").invoke("Hello!")
