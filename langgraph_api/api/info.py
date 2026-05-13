'''
mock info api
'''
from fastapi import APIRouter

router = APIRouter(prefix="/info", tags=["info"])

SERVER_INFO = {
    'version': '0.8.7',
    'langgraph_py_version': '1.2.0',
    'flags': {
        'assistants': True,
        'crons': True,
        'langsmith': False,
        'langsmith_tracing_replicas': False
    },
    'host': {
        'kind': 'self-hosted',
        'project_id': None,
        'host_revision_id': None,
        'revision_id': None,
        'tenant_id': None
    }
}

@router.get("")
async def info(self)->dict:
    return SERVER_INFO