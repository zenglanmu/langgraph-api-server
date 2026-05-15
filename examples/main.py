import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from langgraph_api import setup_api
import uvicorn

from dotenv import load_dotenv
load_dotenv(override=True)

import examples.agents

langgraph_api_lifespan = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global langgraph_api_lifespan
    async with langgraph_api_lifespan(app):
        yield


app = FastAPI(
    title='langgraph api example server',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

langgraph_api_lifespan = setup_api(  
  router=app,
  redis_url=os.environ['REDIS_URL'],
  langgraph_database_uri=os.environ['LANGGRAPH_DATABASE_URI'],  
  embeding_model_name=os.environ['LANGGRAPH_EMBED_MODEL'],
  embeding_dim=os.environ['LANGGRAPH_EMBED_DIMENSION'],
  embeding_base_url=os.environ['LANGGRAPH_EMBED_MODEL_BASE_URL'],
  embeding_api_key=os.environ['LANGGRAPH_EMBED_MODEL_API_KEY'],
  include_router_kwargs=dict(prefix="")
)


if __name__ == '__main__':
  uvicorn.run(app=app, host='127.0.0.1', port=2024, reload=False)