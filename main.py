import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

host = os.getenv("UVICORN_HOST", "127.0.0.1")
port = int(os.getenv("UVICORN_PORT", "8000"))
CHROMA_PATH = "chroma"
RAG_MODE = os.getenv("RAG_MODE", "regular")  # either 'regular', 'dspy', 'llama', or 'deepseek'

PROMPT_TEMPLATE = """
You are a multilingual senior cybersecurity analyst and a RAG-based assistant.

Your core identity and behavior:
- You are friendly, supportive, and professional.
- You are highly knowledgeable in cybersecurity and practical digital safety.
- You ALWAYS try to help the user, regardless of whether their question is technical, casual, or unrelated to cybersecurity.

---

### Language and Multilingual Rules (VERY IMPORTANT)

You understand and can respond fluently in:
- English
- Macedonian (македонски)
- Albanian (shqip)

Follow these rules strictly:

1. Respond in the SAME language the user used in their latest message.
2. If the message contains a mix of languages, reply in the dominant one (most words).
3. If even a single word clearly indicates Macedonian, reply in Macedonian.
4. If even a single word clearly indicates Albanian, reply in Albanian.
5. NEVER switch language based on chat history — ONLY based on the current message.

If needed, translate retrieved context into the appropriate language before answering.

---

### Using RAG Context

- You may receive some retrieved CONTEXT from a knowledge base. It may be useful, incomplete, irrelevant, or empty.

Rules:
- If the context helps answer the question → use it first.
- You may supplement with your own cybersecurity knowledge when needed.
- Never invent details that look like they come from the context if they do not exist.

---

### When the question IS about cybersecurity

Your answer should:
- Clearly explain the concept, tool, threat, vulnerability, or technique.
- Describe how it works in practice (real-world reasoning).
- Explain security risks and implications.
- Provide at least 2–3 practical defensive actions (e.g., MFA, secure configuration, detection, monitoring).
- Use simple analogies if helpful for non-technical users.

---

### When the question is NOT about cybersecurity

Examples:
- "How are you?"
- "What is the capital of France?"
- "Tell me a joke."

In these cases:

1. Give a short, friendly answer (1–3 sentences).
2. Politely clarify that your main expertise is cybersecurity.
3. Add one small cybersecurity safety tip, such as:
   - "Avoid clicking unknown links."
   - "Use a password manager."
   - "Enable two-factor authentication."

---

### Handling uncertainty

If something is unclear or missing:

Say:
> "Based on the provided context, this detail is not available — but generally…"

Then give safe high-level guidance.

Never respond with only "I don't know."

---

### Output Style

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


class Request(BaseModel):
    message: str


class ActionResponse(BaseModel):
    response: str
    state: str


def load_db():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


def retrieve_context(query_text: str, k: int = 3):
    db = load_db()
    results = db.similarity_search_with_score(query_text, k=k)
    if not results:
        return None
    return "\n\n---\n\n".join([doc.page_content for doc, _ in results])


if RAG_MODE == "regular":
    print("[INFO] Using REGULAR (GPT) mode")

    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage

    MAX_HISTORY_SIZE = 10
    session_store = {}

    def get_session_id(request: FastAPIRequest):
        session_id = request.headers.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        return session_id

    def get_chat_history(session_id: str):
        return session_store.setdefault(session_id, [])

    def update_chat_history(session_id: str, message):
        history = get_chat_history(session_id)
        history.append(message)
        if len(history) > MAX_HISTORY_SIZE:
            history.pop(0)
        session_store[session_id] = history

    def build_prompt(context: str, query_text: str, history: list) -> str:
        history_text = "\n".join(
            [
                f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
                for msg in history
            ]
        )
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return prompt_template.format(
            context=context,
            question=query_text,
            history=history_text,
        )

    model = ChatOpenAI(model="gpt-4o", temperature=0.1)

    @app.post("/chatbot/messages")
    def chatbot_regular(message: Request, request: FastAPIRequest):
        session_id = get_session_id(request)
        update_chat_history(session_id, HumanMessage(content=message.message))

        context = retrieve_context(message.message)
        if not context:
            context = "No additional context from the knowledge base was found. You may rely on your own cybersecurity knowledge."

        history = get_chat_history(session_id)
        full_prompt = build_prompt(context, message.message, history)

        try:
            # Send the whole formatted prompt as a system message
            response = model.invoke([{"role": "system", "content": full_prompt}])
            update_chat_history(session_id, AIMessage(content=response.content))
            return {"response": response.content, "state": "ACTIVE"}
        except Exception as e:
            return {"response": f"Error: {str(e)}", "state": "ERROR"}

elif RAG_MODE == "dspy":
    print("[INFO] Using DSPy mode")
    import dspy

    class RAGQuery(dspy.Signature):
        """
        Multilingual cybersecurity RAG assistant.

        Use the provided CONTEXT when it is relevant, but you may also rely on your
        general cybersecurity knowledge and best practices. Never hallucinate details
        that look like they come from the context if they are not there.

        You are multilingual (English, Macedonian, Albanian) and must respond in
        the same language as the question. For non-cyber questions, answer briefly,
        mention you are mainly a cybersecurity expert, and give one simple security tip.
        """
        context = dspy.InputField()
        question = dspy.InputField()
        answer = dspy.OutputField()

    DSPY_MODEL_NAME = os.getenv("DSPY_MODEL_NAME", "gpt-4o")

    if DSPY_MODEL_NAME in ["llama3.3:70b", "deepseek-r1:70b"]:
        model_str = f"ollama/{DSPY_MODEL_NAME}"
        print(f"[INFO] DSPy using Ollama model: {model_str}")
        dspy.configure(
            lm=dspy.LM(
                model=model_str,
                api_base="https://llama3.finki.ukim.mk",
                temperature=0.1,
            )
        )
    else:
        model_str = f"openai/{DSPY_MODEL_NAME}"
        print(f"[INFO] DSPy using OpenAI model: {model_str}")
        dspy.configure(
            lm=dspy.LM(
                model=model_str,
                temperature=0.1,
            )
        )

    class RAGModel(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict(RAGQuery)

        def forward(self, context, question):
            return self.predictor(context=context, question=question)

    rag_model = RAGModel()

    def generate_response(context, question):
        try:
            response = rag_model(context=context, question=question)
            return (
                response.answer
                if response.answer
                else "I could not generate a detailed answer, but remember to keep your accounts safe with strong, unique passwords and two-factor authentication."
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @app.post("/chatbot/messages")
    def chatbot_dspy(message: Request):
        context = retrieve_context(message.message)
        if not context:
            context = ""  # let the model rely on its general cyber knowledge

        response = generate_response(context, message.message)
        return {"response": response, "state": "ACTIVE"}

elif RAG_MODE in ["llama", "deepseek"]:
    print(f"[INFO] Using OLLAMA mode: {RAG_MODE.upper()}")
    from langchain_community.llms import Ollama

    model_name = {"llama": "llama3.3:70b", "deepseek": "deepseek-r1:70b"}[RAG_MODE]

    model = Ollama(
        model=model_name,
        base_url="https://llama3.finki.ukim.mk",
        temperature=0.1,
    )

    @app.post("/chatbot/messages")
    def chatbot_ollama(message: Request):
        context = retrieve_context(message.message)
        if not context:
            context = "No additional context from the knowledge base was found. You may rely on your own cybersecurity knowledge."

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=message.message,
            history="(No explicit chat history provided in this mode)",
        )

        try:
            response = model.invoke(prompt)
            return {"response": response, "state": "ACTIVE"}
        except Exception as e:
            return {"response": f"Error: {str(e)}", "state": "ERROR"}

else:
    raise ValueError(
        f"Invalid RAG_MODE '{RAG_MODE}'. Choose 'regular', 'dspy', 'llama', or 'deepseek'."
    )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
