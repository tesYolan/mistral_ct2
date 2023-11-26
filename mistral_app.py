import gradio as gr
from llm_handler import make_chat_character
import json


def setup_scenario(system_prompt):

    dialog = [{"role": "system", "content": system_prompt}]

    return_dict = {'dialog':dialog}

    gr.Info("Role Setup")

    return [], return_dict

def chat_character(msg, chatbot, prompt_state, system_prompt):

    dialog = prompt_state.get('dialog', [])
    if msg == "":
        gr.Warning("Please enter a message to chat with the bot")

    if dialog == []:
        gr.Warning("Please setup the bot")


    response = make_chat_character(msg, dialog)

    prompt_state['dialog'] = response['dialog']

    chatbot.append((msg, response['response']))


    return "", chatbot, prompt_state

def prompt_expand_fn(system_prompt):
        if system_prompt == "":
            gr.Warning("Please enter a system prompt")
    
        # if bot_name == "":
        #     gr.Warning("Please enter a bot name")
    
        # if user_name == "":
        #     gr.Warning("Please enter a user name")
    
        lazy_prompt = system_prompt
    
        system_prompt_expansion = """
            You are an expert Prompt Writer for Large Language Models.
            Your goal is to improve the prompt given below for chatting :
            Here are several tips on writing great prompts:
            -------
            Start the prompt by stating that it is an expert in the subject.
            Put instructions at the beginning of the prompt and use ### or to separate the instruction and context 
            Be specific, descriptive and as detailed as possible about the desired context, outcome, length, format, style, etc 
            ---------
            Here's an example of a great prompt:
            As a master YouTube content creator, develop an engaging script that revolves around the theme of "Exploring Ancient Ruins."
            Your script should encompass exciting discoveries, historical insights, and a sense of adventure.
            Include a mix of on-screen narration, engaging visuals, and possibly interactions with co-hosts or experts.
            The script should ideally result in a video of around 10-15 minutes, providing viewers with a captivating journey through the secrets of the past.
            Example:
            "Welcome back, fellow history enthusiasts, to our channel! Today, we embark on a thrilling expedition..."
            -----
            after you improve you return the json with the following format:
            {"Original Prompt": original_prompt,  "improved_prompt": improved_prompt}
        """
        dialog = [{"role": "system", "content": system_prompt_expansion}]
        response = make_chat_character(lazy_prompt, dialog)

        try: 
            # get json response
            full_line = response['response']
            # try json loads to load
            json_obj = json.loads(full_line)
            if 'improved_prompt' in json_obj:
                improved_prompt = json_obj['improved_prompt']
                return response['response'], improved_prompt
            else:
                gr.Warning("Couldn't parse the response, please try again")
                return response['response'], response['response']
        except:
            gr.Warning("Couldn't parse the response, please try again")
            return response['response'], response['response']




with gr.Blocks() as demo:
    gr.Markdown("## Mistral-7B-Instruct and Prompt Expansion")

    with gr.Row():

        with gr.Column():
            prompt_state = gr.State(value=dict())

            system_prompt = gr.Textbox(lines=3, label="System Prompt")
            expanded_system_prompt = gr.Textbox(lines=3, label="Expanded System Prompt Unedited")
            expanded_system_prompt_extracted = gr.Textbox(lines=3, label="Expanded System Prompt Extracted")
            prompt_expansion_btn = gr.Button("Expand Prompt", label="Expand System Prompt")
            # bot_name = gr.Textbox(lines=1, label="Bot Name")
            # user_name = gr.Textbox(lines=1, label="User Name")
            setup = gr.Button("Setup the bot", label = "Setup your pipeline")



            chatbot = gr.Chatbot()
            msg = gr.Textbox()

            clear = gr.ClearButton([msg, chatbot])

            msg.submit(chat_character, inputs=[msg, chatbot, prompt_state, expanded_system_prompt_extracted], outputs=[msg, chatbot, prompt_state])

            setup.click(setup_scenario, inputs=[system_prompt], outputs=[chatbot, prompt_state])

            prompt_expansion_btn.click(prompt_expand_fn, inputs=[system_prompt], outputs=[expanded_system_prompt, expanded_system_prompt_extracted])

if __name__ == "__main__":
    demo.launch(debug=True)