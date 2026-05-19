import os
from langchain_deepseek import ChatDeepSeek

def get_chat_client(
                temperature: float=None,
                streaming: bool=False,
                model_name: str=None,
                **kwargs
                ) -> ChatDeepSeek:
    '''
    return chat client based on config
    temperature: What sampling temperature to use.
    streaming: Whether to stream the results or not.
    '''
    if model_name is None:
        model = os.environ.get('OPENAI_MODEL_NAME')
    else:
        model = model_name

    # model = f'openai/{model}'
    
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPENAI_BASEURL = os.environ.get('OPENAI_BASEURL')
    # for reasoning content, ChatOpenAI dont support
    chat_client = ChatDeepSeek(
        model=model,
        api_key=OPENAI_API_KEY,
        api_base=OPENAI_BASEURL,
        temperature=temperature,
        streaming=streaming,
        **kwargs)
    
    return chat_client