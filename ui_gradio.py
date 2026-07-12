import os
import gradio as gr

from rag.config import DEFAULT_K
from rag.pipelines import answer_zeroshot, answer_rag_classic, answer_rag_dspy, answer_graph_rag

DEFAULT_MODE = os.getenv("RAG_MODE", "VectorRAG - Classic")

def run_once(message: str, mode: str, k: int, show_debug: bool):
    if not message or not message.strip():
        return "", ""

    if mode == "Zeroshot LLM":
        r = answer_zeroshot("live", message)
    elif mode == "VectorRAG - Classic":
        r = answer_rag_classic("live", message, k=k)
    elif mode == "VectorRAG - DSPY":
        r = answer_rag_dspy("live", message, k=k)
    elif mode == "VectorGraphRAG - Classic":
        r = answer_graph_rag("live",message,k=k)
    elif mode == "VectorGraphRAG - DSPY (Not implemented yet)":
        print("To be implemented")    
    else:
        return f"Invalid mode: {mode}", ""

    debug = ""
    if show_debug:
        debug_lines = []
        debug_lines.append(f"mode={r.mode}")
        debug_lines.append(f"model={r.model_name}")
        debug_lines.append(f"latency_sec={r.latency_sec:.3f}")
        debug_lines.append(f"retrieved_docs={len(r.retrieved_docs)}")
        for i, d in enumerate(r.retrieved_docs[:10], start=1):
            debug_lines.append(
                f"{i}. doc_id={d.get('doc_id')} chunk_id={d.get('chunk_id')} score={d.get('score')} source={d.get('source')}"
            )
        debug = "\n".join(debug_lines)

    return r.answer, debug


with gr.Blocks(title="Cyberbot Research UI") as demo:
    with gr.Row():
        mode = gr.Dropdown(
            choices=["Zeroshot LLM", "VectorRAG - Classic", "VectorRAG - DSPY","VectorGraphRAG - Classic","VectorGraphRAG - DSPY (Not implemented yet)"],
            value=DEFAULT_MODE,
            label="Mode",
        )
        k = gr.Slider(1, 10, value=DEFAULT_K, step=1, label="Top-K retrieval (k)")
        show_debug = gr.Checkbox(value=True, label="Show debug (retrieval + latency)")

    with gr.Column():
        chatbot = gr.Chatbot(label="Chat", height=400, type="messages", value=[])
        user_msg = gr.Textbox(label="Your message", placeholder="Ask a cybersecurity question…")
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
        debug_out = gr.Textbox(label="Debug", lines=10)

    def chat_turn(message, history, mode, k, show_debug):
        # ignore empty messages
        if not message or not message.strip():
            return history or [], "", ""

        # ensure history is a list
        if history is None:
            history = []

        answer, dbg = run_once(message, mode, int(k), show_debug)

        # force answer to string
        if answer is None:
            answer = ""

        # append messages in proper format
        history.append({"role": "user", "content": str(message)})
        history.append({"role": "assistant", "content": str(answer)})

        return history, "", str(dbg)



    clear.click(lambda: ([], ""), outputs=[chatbot, debug_out])




    send.click(chat_turn, inputs=[user_msg, chatbot, mode, k, show_debug], outputs=[chatbot, user_msg, debug_out])
    user_msg.submit(chat_turn, inputs=[user_msg, chatbot, mode, k, show_debug], outputs=[chatbot, user_msg, debug_out])
    clear.click(lambda: ([], ""), outputs=[chatbot, debug_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("GRADIO_PORT", "7860")))
