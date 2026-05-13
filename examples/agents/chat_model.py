import os
from langchain_openai import ChatOpenAI

def get_chat_client(
                temperature: float=None,
                streaming: bool=False,
                model_name: str=None,
                **kwargs
                ) -> ChatOpenAI:
    '''
    return chat client based on config
    temperature: What sampling temperature to use.
    streaming: Whether to stream the results or not.
    '''
    if model_name is None:
        model = os.environ.get('OPENAI_MODEL_NAME')
    else:
        model = model_name

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_BASEURL = os.environ.get('OPENAI_BASEURL')
    chat_client = ChatOpenAI(
        model=model,
        openai_api_key=OPENAI_API_KEY, 
        base_url=OPENAI_BASEURL,
        temperature=temperature,
        streaming=streaming,
        **kwargs)
    
    return chat_client