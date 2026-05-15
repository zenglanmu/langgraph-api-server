from fastapi import APIRouter

from . import info, runs, threads, assistants, store, crons


lg_api_router = APIRouter()
lg_api_router.include_router(info.router)
lg_api_router.include_router(runs.router)
lg_api_router.include_router(threads.router)
lg_api_router.include_router(assistants.router)
lg_api_router.include_router(store.router)
lg_api_router.include_router(crons.router)