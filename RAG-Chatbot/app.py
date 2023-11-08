from dotenv import load_dotenv
import gradio as gr
import os
from pathlib import Path
import autogen
import chromadb
import multiprocessing as mp
from autogen.retrieve_utils import TEXT_FORMATS, get_file_from_url, is_url
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import (
    RetrieveUserProxyAgent,
    PROMPT_QA,
)
load_dotenv()
TIMEOUT = 60

def initialize_agents(config_list, docs_path=None):
    if isinstance(config_list, gr.State):
        _config_list = config_list.value
    else:
        _config_list = config_list
    if docs_path is None:
        docs_path = os.environ.get("DOCS_PATH", os.path.join(os.getcwd(),"ingest_data/"))

    assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
    )

    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        retrieve_config={
            "task": "QA",
            "docs_path": docs_path,
            "chunk_token_size": 2000,
            "model": _config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "embedding_model": "all-mpnet-base-v2",
            "customized_prompt": PROMPT_QA,
            "get_or_create": True,
            "collection_name": "autogen_rag",
        },
    )

    return assistant, ragproxyagent


def initiate_chat(config_list, problem, queue, n_results=3):
    global assistant, ragproxyagent
    if isinstance(config_list, gr.State):
        _config_list = config_list.value
    else:
        _config_list = config_list
    if len(_config_list[0].get("api_key", "")) < 2:
        queue.put(
            ["Hi, nice to meet you, I do not have an API key, go debug!"]
        )
        return
    else:
        llm_config = (
            {
                "request_timeout": TIMEOUT,
                # "seed": 42,
                "config_list": _config_list,
                "use_cache": False,
            },
        )
        assistant.llm_config.update(llm_config[0])
    assistant.reset()
    try:
        ragproxyagent.initiate_chat(
            assistant, problem=problem, silent=False, n_results=n_results
        )
        messages = ragproxyagent.chat_messages
        messages = [messages[k] for k in messages.keys()][0]
        messages = [m["content"] for m in messages if m["role"] == "user"]
        print("messages: ", messages)
    except Exception as e:
        messages = [str(e)]
    queue.put(messages)


def chatbot_reply(input_text):
    """Chat with the agent through terminal."""
    queue = mp.Queue()
    process = mp.Process(
        target=initiate_chat,
        args=(config_list, input_text, queue),
    )
    process.start()
    try:
        # process.join(TIMEOUT+2)
        messages = queue.get(timeout=TIMEOUT)
    except Exception as e:
        messages = [
            str(e)
            if len(str(e)) > 0
            else "Invalid Request to OpenAI, please check your API keys."
        ]
    finally:
        try:
            process.terminate()
        except:
            pass
    return messages


def get_description_text():
    return """
    # Retrieve Chat Demo
    """


global assistant, ragproxyagent

with gr.Blocks() as demo:
    config_list, assistant, ragproxyagent = (
        gr.State(
            [
                {
                    "api_key": os.environ.get("OPENAI_API_KEY", ""), 
                    "model": "gpt-3.5-turbo",
                }
            ]
        ),
        None,
        None,
    )
    assistant, ragproxyagent = initialize_agents(config_list)

    gr.Markdown(get_description_text())
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "autogen.png"))),
        # height=600,
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
    )

    with gr.Row():

        def update_config(config_list):
            global assistant, ragproxyagent
            config_list = autogen.config_list_from_models(
                model_list=[os.environ.get("MODEL", "gpt-3.5-turbo")],
            )
            if not config_list:
                config_list = [
                    {
                        "api_key": os.environ.get("OPENAI_API_KEY", ""),
                       "model": "gpt-3.5-turbo",
                    }
                ]
            config_list[0]["api_key"] = os.environ.get("OPENAI_API_KEY", "")
            llm_config = (
                {
                    "request_timeout": TIMEOUT,
                    # "seed": 42,
                    "config_list": config_list,
                },
            )
            assistant.llm_config.update(llm_config[0])
            ragproxyagent._model = config_list[0]["model"]
            return config_list

        def set_params(model):
            os.environ["MODEL"] = model
            return model

        txt_model = gr.Dropdown(
            label="Model",
            choices=[
                
                "gpt-4-1106-preview",
                "gpt-4",
                "gpt-3.5-turbo-16k-0613",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo",
            ],
            allow_custom_value=True,
            value="gpt-3.5-turbo",
            container=True,
        )
 

    clear = gr.ClearButton([txt_input, chatbot])

    # with gr.Row():

    #     def upload_file(file):
    #         return update_context_url(file.name)

    #     upload_button = gr.UploadButton(
    #         "Click to upload a context file or enter a url in the right textbox",
    #         file_types=[f".{i}" for i in TEXT_FORMATS],
    #         file_count="single",
    #     )

    #     txt_context_url = gr.Textbox(
    #         label="Enter the url to your context file and chat on the context",
    #         info=f"File must be in the format of [{', '.join(TEXT_FORMATS)}]",
    #         max_lines=1,
    #         show_label=True,
    #         value="",
    #         container=True,
    #     )

    txt_prompt = gr.Textbox(
        label="Enter your prompt for Retrieve Agent and press enter to replace the default prompt",
        max_lines=40,
        show_label=True,
        value=PROMPT_QA,
        container=True,
        show_copy_button=True,
    )
    """ This method is called when the user presses the submit button. """
    def respond(message, chat_history, model):
        global config_list
        set_params(model)
        config_list = update_config(config_list)
        messages = chatbot_reply(message)
        _msg = (
            messages[-1]
            if len(messages) > 0 and messages[-1] != "TERMINATE"
            else messages[-2]
            if len(messages) > 1
            else "Context is not enough for answering the question. Please press `enter` in the context url textbox to make sure the context is activated for the chat."
        )
        chat_history.append((message, _msg))
        return "", chat_history

    def update_prompt(prompt):
        ragproxyagent.customized_prompt = prompt
        return prompt

    # def update_context_url(context_url):
    #     global assistant, ragproxyagent

    #     file_extension = Path(context_url).suffix
    #     print("file_extension: ", file_extension)
    #     if file_extension.lower() not in [f".{i}" for i in TEXT_FORMATS]:
    #         return f"File must be in the format of {TEXT_FORMATS}"

    #     if is_url(context_url):
    #         try:
    #             file_path = get_file_from_url(
    #                 context_url,
    #                 save_path=os.path.join("/tmp", os.path.basename(context_url)),
    #             )
    #         except Exception as e:
    #             return str(e)
    #     else:
    #         file_path = context_url
    #         context_url = os.path.basename(context_url)

    #     try:
    #         chromadb.PersistentClient(path="/tmp/chromadb").delete_collection(
    #             name="autogen_rag"
    #         )
    #     except:
    #         pass
    #     assistant, ragproxyagent = initialize_agents(config_list, docs_path=file_path)
    #     return context_url

    txt_input.submit(
        respond,
        [txt_input, chatbot, txt_model],
        [txt_input, chatbot],
    )
    txt_prompt.submit(update_prompt, [txt_prompt], [txt_prompt])
    # txt_context_url.submit(update_context_url, [txt_context_url], [txt_context_url])
    # upload_button.upload(upload_file, upload_button, [txt_context_url])


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
