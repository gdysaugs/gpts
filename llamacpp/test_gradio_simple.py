#!/usr/bin/env python3
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

# Simple Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# Simple Gradio Test")
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    greet_btn = gr.Button("Greet")
    
    greet_btn.click(greet, inputs=name, outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)