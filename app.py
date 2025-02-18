import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ========================================
# 1. MODEL LOADING
# ========================================
MODEL_NAME = "gpt2"  # A small model for quick local testing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# If you have a GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Create a pipeline for text generation
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# ========================================
# 2. SUGGESTION LOGIC
# ========================================
MAX_CONTEXT_TOKENS = 50     # how many tokens from the end of user text to consider
PREDICTION_TOKENS = 5       # how many tokens to predict

def suggest_next_words(user_text, history):
    """
    Given the user's current text (notes), return a short suggestion (next few words).
    - history can track previous suggestions or acceptance, if needed.
    - For a minimal example, we only focus on user_text -> suggestion.
    """

    if not user_text.strip():
        # if there's no text, no suggestion
        return "", history

    # Tokenize and truncate to the last MAX_CONTEXT_TOKENS tokens
    input_ids = tokenizer(user_text, return_tensors="pt", add_special_tokens=False).input_ids
    if input_ids.shape[1] > MAX_CONTEXT_TOKENS:
        input_ids = input_ids[:, -MAX_CONTEXT_TOKENS:]  # keep only the last N tokens

    input_ids = input_ids.to(device)

    # Generate a small number of tokens for the suggestion
    # We set no repetition penalty or top_k as a simple example. Tweak as needed.
    output_ids = model.generate(
        input_ids,
        max_new_tokens=PREDICTION_TOKENS,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    # The generated text includes the original prompt + new tokens
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Isolate the new words from the original text
    # 1) Count how many tokens in the original prompt
    original_len = input_ids.shape[1]
    # 2) Re-tokenize the entire output, then split off the last few tokens
    full_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    new_tokens = full_tokens[original_len:]
    suggestion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up whitespace
    suggestion = suggestion.strip()

    return suggestion, history

def accept_suggestion(user_text, suggestion, history):
    """
    Append the suggestion to the user text if user clicks an 'Accept' button.
    """
    if suggestion:
        # Add a space if user text doesn't already end with whitespace
        if not user_text.endswith((" ", "\n")):
            user_text += " "
        user_text += suggestion
    return user_text, history


# ========================================
# 3. BUILDING THE GRADIO UI
# ========================================
with gr.Blocks() as demo:
    gr.Markdown("# Smart Compose Prototype\nType in the textbox, and get a next-word suggestion.")

    with gr.Row():
        note_editor = gr.Textbox(
            label="Your Notes",
            placeholder="Start typing your text here...",
            lines=4,
            # live=True  # triggers the change event as you type
        )
        suggestion_box = gr.Textbox(
            label="Suggestion",
            interactive=False,
            placeholder="Predicted next words...",
        )

    # hidden state to track conversation or usage stats if needed
    state = gr.State([])

    # Buttons
    accept_button = gr.Button("Accept Suggestion")
    clear_button = gr.Button("Clear All")

    # ========== Function to update suggestions in real-time ==========
    def update_suggestion(user_text, history):
        sug, new_hist = suggest_next_words(user_text, history)
        return sug, new_hist

    # Whenever user types in note_editor, we update the suggestion
    note_editor.change(
        fn=update_suggestion,
        inputs=[note_editor, state],
        outputs=[suggestion_box, state]
    )

    # Button callback to accept the suggestion
    def accept_and_update_text(user_text, suggestion, history):
        updated_text, new_hist = accept_suggestion(user_text, suggestion, history)
        # after accepting, we might want to generate a new suggestion for the newly appended text
        new_sug, new_hist = suggest_next_words(updated_text, new_hist)
        return updated_text, new_sug, new_hist

    accept_button.click(
        fn=accept_and_update_text,
        inputs=[note_editor, suggestion_box, state],
        outputs=[note_editor, suggestion_box, state]
    )

    # Clear all text and suggestions
    def clear_all():
        return "", "", []

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[note_editor, suggestion_box, state]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
