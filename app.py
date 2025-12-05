import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

GGUF_MODEL_PATH = hf_hub_download(
    repo_id="coolestGuyEver/llama-3.2-3B-finetuned-GGUF",
    filename="llama-3.2-3B-finetuned.gguf",
    token=os.environ["HF_TOKEN"]
)

llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_gpu_layers=0,
    n_ctx=4096,
    verbose=False
)


def generate_response(prompt, max_tokens=200, temperature=0.7):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["\nUser:", "###"]
    )
    return output["choices"][0]["text"]


def tutor_hint(question):
    prompt = (
        "You are a helpful tutor. Give a gentle hint without revealing the full answer.\n"
        f"Question: {question}\n"
        "Hint:"
    )
    return generate_response(prompt, 150, 0.7)


def tutor_explain(question):
    prompt = (
        "You are a clear, patient tutor. Explain the problem step-by-step.\n"
        f"Question: {question}\n"
        "Explanation:"
    )
    return generate_response(prompt, 250, 0.6)


def tutor_check(answer, question):
    prompt = (
        "You are a tutor evaluating a student's attempt.\n"
        f"Question: {question}\n"
        f"Student's attempt: {answer}\n"
        "Provide feedback, correctness, and one improvement. Do NOT give the full solution."
    )
    return generate_response(prompt, 200, 0.5)


css = """
@import url('https://fonts.googleapis.com/css2?family=Kode+Mono:wght@300;400;600&display=swap');

body, .gradio-container {
    background-color: #000000;
    font-family: 'Kode Mono', monospace;
    color: #ffffff;
}

.container {
    max-width: 1200px !important;
    margin: 0 auto;
    padding-top: 4rem;
}

h1 {
    font-weight: 300;
    font-size: 5rem;
    letter-spacing: -0.06em;
    margin-bottom: 0.5rem;
    color: #ffffff;
    line-height: 1;
}

.subtitle {
    color: #888888;
    font-size: 1.2rem;
    font-weight: 300;
    margin-bottom: 3rem;
    letter-spacing: -0.02em;
}

textarea {
    background-color: #0a0a0a !important;
    border: 1px solid #222222 !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease;
}

textarea:focus {
    border-color: #444444 !important;
    box-shadow: none !important;
}

button.primary-btn {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 50px !important;
    padding: 16px 40px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border: none !important;
    transition: all 0.3s ease;
    margin-top: 1rem;
}

button.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(255,255,255,0.1);
}

.block-label span {
    color: #666666 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 600;
}

.output-box textarea {
    background-color: transparent !important;
    border: 1px solid #222222 !important;
}
"""


with gr.Blocks() as demo:
    gr.HTML(f"<style>{css}</style>")

    with gr.Column(elem_classes="container"):
        gr.Markdown("# IRIS")
        gr.Markdown("Intelligent Response & Inference System",
                    elem_classes="subtitle")

        with gr.Tabs():
            with gr.Tab("Tutor Mode"):
                with gr.Row():
                    question_box = gr.TextArea(
                        label="Question",
                        placeholder="Enter a question you'd like help with...",
                        lines=4
                    )
                    answer_box = gr.TextArea(
                        label="Your Attempt (optional)",
                        placeholder="Write your attempt here if you want feedback...",
                        lines=4
                    )

                with gr.Row():
                    btn_hint = gr.Button(
                        "Get Hint", elem_classes="primary-btn")
                    btn_explain = gr.Button(
                        "Explain Step-by-Step", elem_classes="primary-btn")
                    btn_check = gr.Button(
                        "Check My Attempt", elem_classes="primary-btn")

                tutor_output = gr.TextArea(
                    label="Tutor Output",
                    lines=10,
                    interactive=False,
                    elem_classes=["block-label", "output-box"]
                )

                btn_hint.click(
                    fn=tutor_hint,
                    inputs=question_box,
                    outputs=tutor_output
                )

                btn_explain.click(
                    fn=tutor_explain,
                    inputs=question_box,
                    outputs=tutor_output
                )

                btn_check.click(
                    fn=tutor_check,
                    inputs=[answer_box, question_box],
                    outputs=tutor_output
                )

            # FREEFORM CHAT TAB
            with gr.Tab("Freeform Chat"):
                with gr.Row():
                    prompt = gr.TextArea(
                        label="Input Prompt",
                        placeholder="Type anything here...",
                        lines=6
                    )
                    output = gr.TextArea(
                        label="Model Output",
                        lines=10,
                        interactive=False,
                        elem_classes=["block-label", "output-box"]
                    )

                with gr.Row():
                    max_tokens = gr.Slider(
                        10, 512, value=150, label="Max Tokens")
                    temperature = gr.Slider(
                        0.1, 1.0, value=0.7, label="Temperature")

                submit_btn = gr.Button(
                    "Generate Response", elem_classes="primary-btn")

                submit_btn.click(
                    fn=generate_response,
                    inputs=[prompt, max_tokens, temperature],
                    outputs=output
                )

if __name__ == "__main__":
    demo.launch()
