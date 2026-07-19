import gradio as gr
import os

def hello(name):
    return f"Hello {name}"

demo = gr.Interface(
    fn=hello,
    inputs="text",
    outputs="text"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
