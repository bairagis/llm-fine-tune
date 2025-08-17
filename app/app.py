from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
# login to hugginface hub
import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# Load base model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # e.g., "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load PEFT adapter
peft_model_id = "/media/sudip/linux-extra/code/llm-fine-tune/model/health_assistant_model"  # path where you saved with save_model()
model_peft = PeftModel.from_pretrained(base_model, peft_model_id)

# Set to evaluation mode
model_peft.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model_peft.to(device)

print("Model and tokenizer loaded successfully.")


from transformers import pipeline
pipeline_qn_peft = pipeline("text-generation", model=model_peft, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)



def structure_output_from_result(result):
    """ Structure the output from the result of the LLM call.
    Args:
        result (list): The result from the LLM call.
    Returns:
        str: The structured output content.
    """
    for output in result[0]['generated_text'] :
        if output['role'] == 'assistant':
            return output['content'].strip()
        



def call_llm_experts(input_text, pipeline):
    """ Call the LLM with the input text and return the structured output.
    Args:
        input_text (str): The input text to send to the LLM.
        pipeline (Pipeline): The Hugging Face pipeline for text generation.
    Returns:
        str: The structured output from the LLM.
    """

    message = [{"role": "user", "content": input_text}]
    message = [ {"role": "system", "content": "You are a health expert. Answer the user's question in detail."}] + message
    print("Calling LLM with message:", message)
    result = pipeline(message, max_new_tokens=1000, do_sample=True, temperature=0.7)

    return structure_output_from_result(result)


# create a gradio interface for the health expert model
import gradio as gr 
def gradio_interface(message,history):
    answer = call_llm_experts(message, pipeline_qn_peft)
    return answer


# Create the main app 
if __name__ == "__main__":
    # Create a Hugging Face pipeline for text generation
   
    # Create a Gradio interface

    gr.ChatInterface(
        gradio_interface,
        type="messages",
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
        save_history=True,
        title="Health Expert LLM",
        description="Ask a health-related question and get answers from the fine-tuned LLM."
    ).launch(share=True)  # Set share=True to create a public link

