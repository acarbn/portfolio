
import gradio as gr
from main import generate_text, tokenizer,model 

# Gradio UI layout
with gr.Blocks() as demo:
    gr.Markdown("## GPT2 Text Generator")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(lines=4, label="Prompt")
            output = gr.Textbox(lines=8, label="Generated Text")

        with gr.Column(scale=1):
            max_length = gr.Slider(20, 200, value=100, step=10, label="Max Length")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(10, 200, value=100, step=10, label="Top-k")
            top_p = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Top-p")
            rep_penalty=gr.Slider(minimum=1.0, maximum=2.0, value=1.2, step=0.1, label="Repetition Penalty")

    with gr.Row():
        generate_btn = gr.Button("Generate")
        clear_btn = gr.Button("Clear")

    # Button logic
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_length, temperature, top_k, top_p],
        outputs=output
    )

    clear_btn.click(
        fn=lambda: ("",),
        inputs=[],
        outputs=[prompt, output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
