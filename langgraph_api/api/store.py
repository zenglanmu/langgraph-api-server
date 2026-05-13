from typing import Annotated
from fastapi import APIRouter, Query, HTTPException

from ..registry import get_graph_store
from ..utils.models import (StorePutRequest, StoreGetRequest, StoreGetItem, 
                      StoreDeleteRequest, StoreSearchRequest, StoreSeearchResponse,
                      StoreSearchItem, StoreSeachNamespaceRequest, StoreListNamespaceResponse)


router = APIRouter(prefix="/store", tags=["store"])

@router.put("/items", status_code=200)
async def put_item(payload: StorePutRequest):
    async with get_graph_store() as store:        
        await store.aput(
            payload.namespace_ns, 
            key=payload.key, 
            value=payload.value, 
            index=payload.index,
            ttl=payload.ttl,
        )
    
@router.get("/items")
async def get_item(payload: Annotated[StoreGetRequest, Query()]) -> StoreGetItem:
    async with get_graph_store() as store:
        item = await store.aget(payload.namespace_ns, key=payload.key, refresh_ttl=payload.refresh_ttl)
        if item:
            return StoreGetItem.model_validate(item.dict())
        raise HTTPException(status_code=404, detail="item not found")
        
        
@router.delete("/items", status_code=200)
async def delete_item(payload: StoreDeleteRequest):
    async with get_graph_store() as store:
        await store.adelete(payload.namespace_ns, key=payload.key)
    
    
@router.post("/items/search")
async def search_items(payload: StoreSearchRequest) -> StoreSeearchResponse:
    async with get_graph_store() as store:
        search_items = await store.asearch(
            tuple(payload.namespace_prefix), 
            query=payload.query, 
            filter=payload.filter,
            limit=payload.limit, 
            offset=payload.offset, 
            refresh_ttl=payload.refresh_ttl,
        )
        items = [StoreSearchItem.model_validate(x.dict()) for x in search_items]
        return StoreSeearchResponse(items=items)
    
    
@router.post("/namespaces")
async def list_namespaces(payload: StoreSeachNamespaceRequest) -> StoreListNamespaceResponse:
    async with get_graph_store() as store:
        namespaces = await store.alist_namespaces(
            prefix=payload.prefix,
            suffix=payload.suffix,
            max_depth=payload.max_depth,
            limit=payload.limit,
            offset=payload.offset,
        )
        return StoreListNamespaceResponse(namespaces=namespaces)