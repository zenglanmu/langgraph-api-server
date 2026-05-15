from datetime import datetime
from typing import Generic, Literal, Sequence, TypeVar
from langgraph.checkpoint.base import CheckpointTuple
from pydantic import BaseModel, Field, ConfigDict, computed_field, model_validator
from typing import Optional, Dict, Any, List

T = TypeVar('T', bound=BaseModel)

class ItemResponse(BaseModel, Generic[T]):
    msg: Literal['ok', 'added', 'modified', 'deleted'] = Field(default='ok', description='操作结果')
    data: Optional[T] = Field(default=None, description='返回数据')
    
    
class Message(BaseModel):
    type: str
    content: str | list[dict]
    
class InputModel(BaseModel):
    '''
    for graph input, it should accept any input, which leaves to the graph input design.
    '''    
    messages: list[Message] | None = Field(default=None, description='should normally contains messages input')
    model_config = ConfigDict(extra='allow')

StreamMode = Literal[
    "values",
    "messages",
    "updates",
    "events",
    "tasks",
    "checkpoints",
    "debug",
    "custom",
    "messages-tuple",
]

CancelAction = Literal["interrupt", "rollback"]
MultitaskStrategy = Literal["reject", "interrupt", "rollback", "enqueue"]
IfNotExists = Literal["reject", "create"]
OnCompletionBehavior = Literal["delete", "keep"]
DisconnectMode = Literal["cancel", "continue"]
Durability = Literal["sync", "async", "exit"]
RunStatus = Literal["pending", "running", "success", "error", "cancelled", "timeout"]
BulkCancelRunsStatus = Literal["pending", "running", "all"]
RunSelectField = Literal[
    "run_id",
    "thread_id",
    "assistant_id",
    "created_at",
    "updated_at",
    "status",
    "metadata",
    "multitask_strategy",
    "error_message",
]


class Command(BaseModel):
    resume: Any | None = None
    update: dict[str, Any] | None = None
    goto: list[str] | str | None = None
    model_config = ConfigDict(extra="allow")


class Checkpoint(BaseModel):
    checkpoint_ns: str = ""
    checkpoint_id: str | None = None
    checkpoint_map: dict[str, Any] | None = None


class StreamRunRequest(BaseModel):
    assistant_id: str = Field(..., description="Assistant ID or graph name")
    input: InputModel | None = Field(None, description="Input data for the graph")
    command: Command | None = Field(None, description="Command to execute (cannot combine with input)")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the run")
    context: Optional[Dict[str, Any]] = Field(None, description="Static context to add to the assistant")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for the run")
    stream_mode: Sequence[StreamMode] | StreamMode | None = Field(None, description="Stream mode (values, updates, debug, etc.)")
    stream_subgraphs: bool = Field(False, description="Whether to stream output from subgraphs")
    stream_resumable: bool = Field(False, description="Whether the stream is resumable after disconnect")
    interrupt_before: Optional[List[str]] = Field(None, description="Nodes to interrupt before")
    interrupt_after: Optional[List[str]] = Field(None, description="Nodes to interrupt after")
    checkpoint: Checkpoint | None = Field(None, description="Checkpoint to resume from")
    checkpoint_id: Optional[str] = Field(None, description="Checkpoint ID for resuming")
    checkpoint_during: Optional[bool] = Field(None, description="(deprecated) Whether to checkpoint during the run")
    durability: Durability | None = Field(None, description='Durability: "sync", "async", or "exit"')
    multitask_strategy: MultitaskStrategy | None = Field(None, description="Multitask strategy")
    if_not_exists: IfNotExists | None = Field(None, description="How to handle missing thread")
    on_disconnect: DisconnectMode | None = Field(None, description="Action on disconnect")
    on_completion: OnCompletionBehavior | None = Field(None, description="Behavior for stateless run thread on completion")
    webhook: Optional[str] = Field(None, description="Webhook URL to call after run completes")
    feedback_keys: Optional[List[str]] = Field(None, description="Feedback keys to collect")
    after_seconds: Optional[int] = Field(None, description="Schedule run after N seconds")


class RunCreate(BaseModel):
    assistant_id: str = Field(..., description="Assistant ID or graph name")
    input: InputModel | None = Field(None, description="Input data for the graph")
    command: Command | None = Field(None, description="Command to execute")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration for the run")
    context: Optional[Dict[str, Any]] = Field(None, description="Static context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for the run")
    stream_mode: Sequence[StreamMode] | StreamMode | None = Field(None, description="Stream mode(s)")
    stream_subgraphs: bool = Field(False, description="Whether to stream subgraphs")
    stream_resumable: bool = Field(False, description="Whether the stream is resumable")
    interrupt_before: Optional[List[str]] = Field(None, description="Nodes to interrupt before")
    interrupt_after: Optional[List[str]] = Field(None, description="Nodes to interrupt after")
    checkpoint: Checkpoint | None = Field(None, description="Checkpoint to resume from")
    checkpoint_id: Optional[str] = Field(None, description="Checkpoint ID")
    checkpoint_during: Optional[bool] = Field(None, description="(deprecated)")
    durability: Durability | None = Field(None, description="Durability")
    multitask_strategy: MultitaskStrategy | None = Field(None, description="Multitask strategy")
    if_not_exists: IfNotExists | None = Field(None, description="How to handle missing thread")
    on_disconnect: DisconnectMode | None = Field(None, description="Action on disconnect")
    on_completion: OnCompletionBehavior | None = Field(None, description="Stateless run thread behavior")
    webhook: Optional[str] = Field(None, description="Webhook URL")
    feedback_keys: Optional[List[str]] = Field(None, description="Feedback keys")
    after_seconds: Optional[int] = Field(None, description="Schedule after N seconds")


class RunCreateMetadata(BaseModel):
    run_id: str


class Run(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str | None = None
    created_at: datetime
    updated_at: datetime
    status: RunStatus
    metadata: Dict[str, Any] = {}
    multitask_strategy: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class CancelRunRequest(BaseModel):
    action: CancelAction = Field("interrupt", description="Cancel action: interrupt or rollback")
    wait: bool = Field(False, description="Whether to wait until run has completed")


class CancelManyRunsRequest(BaseModel):
    thread_id: str | None = Field(None, description="Thread ID to cancel runs from")
    run_ids: list[str] | None = Field(None, description="List of run IDs to cancel")
    status: BulkCancelRunsStatus | None = Field(None, description="Filter by status: pending, running, or all")
    action: CancelAction = Field("interrupt", description="Cancel action: interrupt or rollback")


class Thread(BaseModel):
    """Represents a conversation thread."""

    thread_id: str
    """The ID of the thread."""
    created_at: datetime
    """The time the thread was created."""
    updated_at: datetime
    """The last time the thread was updated."""
    metadata: dict
    """The thread metadata."""
    status: str | None = None
    """The thread status (idle, busy, interrupted, error)."""
    user_id: str | None = None
    """The user ID that owns this thread."""

class ThreadCreateRequest(BaseModel):
    """Represents a conversation thread."""

    thread_id: Optional[str] = None
    """The ID of the thread."""
    metadata: dict = {}
    """The thread metadata."""
    if_exists: Literal["raise", "do_nothing"] = 'raise'
    """How to handle duplicate creation"""
    supersteps: list[dict[str, list[dict[str, Any]]]] | None = None
    '''not suported yet'''
    ttl: int | None = Field(default=None)
    '''Optional time-to-live in minutes for the thread. None, never expire'''
    
class ThreadUpdateRequest(BaseModel):
    metadata: dict = {}
    ttl: int | None = Field(default=None)
    '''Optional time-to-live in minutes for the thread. None, never expire'''


ThreadStatus = Literal["idle", "busy", "interrupted", "error"]
ThreadSortBy = Literal[
    "thread_id", "status", "created_at", "updated_at", "state_updated_at"
]
SortOrder = Literal["asc", "desc"]

ThreadSelectField = Literal[
    "thread_id",
    "created_at",
    "updated_at",
    "metadata",
    "config",
    "context",
    "status",
    "values",
    "interrupts",
]
class ThreadSearchRequest(BaseModel):
    metadata: dict[str, Any] | None = Field(
        None, 
        description="Thread metadata to filter on."
    )
    values: dict[str, Any] | None = Field(
        None, 
        description="State values to filter on."
    )
    ids: list[str] | None = Field(
        None, 
        description="List of thread IDs to filter by."
    )
    status: ThreadStatus | None = Field(
        None, 
        description="Thread status to filter on. Must be one of 'idle', 'busy', 'interrupted' or 'error'."
    )
    limit: int = Field(
        10, 
        description="Limit on number of threads to return."
    )
    offset: int = Field(
        0, 
        description="Offset in threads table to start search from."
    )
    sort_by: ThreadSortBy | None = Field(
        None, 
        description="Sort by field."
    )
    sort_order: SortOrder | None = Field(
        None, 
        description="Sort order."
    )
    select: list[ThreadSelectField] | None = Field(
        None, 
        description="List of fields to include in the response."
    )
    extract: dict[str, str] | None = Field(
        None, 
        description=(
            "Dictionary mapping aliases to JSONB paths to extract from thread data. "
            "Paths use dot notation for nested keys and bracket notation for array indices "
            "(e.g., `{'last_msg': 'values.messages[-1]'}`). "
            "Extracted values are returned in an `extracted` field on each thread. "
            "Maximum 10 paths per request."
        )
    )
    query: str | None = Field(
        None,
        description="Query string for vector similarity search against thread metadata and values."
    )

PruneStrategy = Literal["delete", "keep_latest"]
class ThreadPruneRequest(BaseModel):
    thread_ids: list[str] = Field(description='List of thread IDs to prune.')
    strategy: PruneStrategy = Field(default='delete', description='The prune strategy.')
    
class StorePutRequest(BaseModel):
    namespace: list[str]
    
    @computed_field
    def namespace_ns(self)->tuple[str, ...]:
        return tuple(self.namespace)
    
    key: str
    value: dict[str, Any]
    index: Literal[False] | list[str] | None = None
    ttl: float | None | Literal[False] = Field(default=None)
    '''
    time for store to expire, None, never expire, False, backend default value
    note langgraph_sdk would drop None value params in sending to backend, so default value should keep as None
    '''
    
class StoreGetRequest(BaseModel):
    namespace: str
    
    @computed_field
    def namespace_ns(self)->tuple[str, ...]:
        return tuple(self.namespace.split('.'))
    
    key: str
    refresh_ttl: bool | None = None
    
class StoreGetItem(BaseModel):
    """Represents a single document or data entry in the graph's Store.

    Items are used to store cross-thread memories.
    """

    namespace: list[str]
    """The namespace of the item. A namespace is analogous to a document's directory."""
    key: str
    """The unique identifier of the item within its namespace.

    In general, keys needn't be globally unique.
    """
    value: dict[str, Any]
    """The value stored in the item. This is the document itself."""
    created_at: datetime
    """The timestamp when the item was created."""
    updated_at: datetime
    """The timestamp when the item was last updated."""

class StoreDeleteRequest(BaseModel):
    namespace: list[str]
    @computed_field
    def namespace_ns(self)->tuple[str, ...]:
        return tuple(self.namespace)
    key: str    
    
class StoreSearchRequest(BaseModel):
    namespace_prefix: list[str]
    filter: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0
    query: str | None = None
    refresh_ttl: bool | None = None
    
class StoreSearchItem(StoreGetItem):
    score: float | None = None

class StoreSeearchResponse(BaseModel):
    items: list[StoreSearchItem]
    
class StoreSeachNamespaceRequest(BaseModel):
    prefix: list[str] | None = None
    suffix: list[str] | None = None
    max_depth: int | None = None
    limit: int = 100
    offset: int = 0
    
class StoreListNamespaceResponse(BaseModel):
    """Response structure for listing namespaces."""

    namespaces: list[list[str]]
    """A list of namespace paths, where each path is a list of strings."""


# ============================================
# ThreadState 相关模型定义 (用于与 langgraph_sdk 兼容)
# ============================================

class Interrupt(BaseModel):
    """Represents an interruption in the execution flow."""
    value: Any
    """The value associated with the interrupt."""
    id: str
    """The ID of the interrupt. Can be used to resume the interrupt."""


class ThreadTask(BaseModel):
    """Represents a task within a thread."""
    id: str
    """The task ID."""
    name: str
    """The task name (node name)."""
    error: str | None = None
    """Error message if task failed."""
    interrupts: list[Interrupt] = Field(default_factory=list)
    """List of interrupts raised in this task."""
    checkpoint: Checkpoint | None = None
    """Checkpoint associated with this task."""
    state: Optional["ThreadState"] = None
    """State of this task (for subgraphs)."""
    result: dict[str, Any] | None = None
    """Task execution result."""

class ThreadStateMetadata(BaseModel):
    source: Literal["input", "loop", "update", "fork"] | None
    """The source of the checkpoint.
    """
    step: int | None
    """The step number of the checkpoint.
    """    
    run_id: str | None
    """The ID of the run that created this checkpoint."""
    
    user_id: str | None = None    
    graph_id: str | None = None    
    thread_id: str | None = None    
    created_by: str | None = None    
    assistant_id: str | None = None   
     
class ThreadState(BaseModel):
    """
    Represents the state of a thread.
    Compatible with langgraph_sdk.schema.ThreadState
    """
    values: dict[str, Any]
    """The state values (from checkpoint.channel_values)."""
    next: list[str] = Field(default_factory=list)
    """The next nodes to execute. If empty, the thread is done until new input is received."""
    checkpoint: Checkpoint
    """The checkpoint information."""
    metadata: ThreadStateMetadata = None
    """Metadata for this state."""
    created_at: str | None = None
    """Timestamp of state creation (from checkpoint.ts)."""
    parent_config: Checkpoint | None = None
    """The parent checkpoint config. If missing, this is the root checkpoint."""
    tasks: list[ThreadTask] = Field(default_factory=list)
    """Tasks to execute in this step. If already attempted, may contain an error."""
    interrupts: list[Interrupt] = Field(default_factory=list)
    """Interrupts which were thrown in this thread."""


# 用于 LangGraph CheckpointTuple 转换的辅助方法
def convert_checkpoint_tuple_to_thread_state(
    checkpoint_tuple: CheckpointTuple,
    next_nodes: list[str] | None = None,
    tasks: list[ThreadTask] | None = None,
    interrupts: list[Interrupt] | None = None
) -> ThreadState:
    """
    将 checkpointer.aget_tuple() 返回的 CheckpointTuple 转换为 ThreadState 格式。
    
    Args:
        checkpoint_tuple: CheckpointTuple 对象或原始字典，包含:
            - config: RunnableConfig with configurable (thread_id, checkpoint_ns, checkpoint_id)
            - checkpoint: 包含 channel_values, ts, id 等
            - metadata: 检查点元数据
            - parent_config: 父检查点配置
        next_nodes: 下一个要执行的节点列表 (默认为空列表)
        tasks: 任务列表 (默认为空列表)
        interrupts: 中断列表 (默认为空列表)
    
    Returns:
        ThreadState: 与 langgraph_sdk 兼容的线程状态对象
    
    Example:
        ```python
        async with get_graph_checkpointer() as checkpointer:
            tuple_result = await checkpointer.aget_tuple(config)
            thread_state = convert_checkpoint_tuple_to_thread_state(tuple_result)
        ```
    """
    # Extract configurable from config
    configurable = checkpoint_tuple.config.get("configurable", {})
    thread_id = configurable.get("thread_id", "")
    checkpoint_ns = configurable.get("checkpoint_ns", "")
    checkpoint_id = configurable.get("checkpoint_id") or checkpoint_tuple.checkpoint.id
    
    # Build Checkpoint
    checkpoint = Checkpoint(
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        checkpoint_id=checkpoint_id,
        checkpoint_map=None  # Not available in CheckpointTuple
    )
    
    # Build parent_checkpoint if exists
    parent_config = None
    if checkpoint_tuple.parent_config:
        parent_configurable = checkpoint_tuple.parent_config.get("configurable", {})
        parent_config = Checkpoint(
            thread_id=parent_configurable.get("thread_id", thread_id),
            checkpoint_ns=parent_configurable.get("checkpoint_ns", checkpoint_ns),
            checkpoint_id=parent_configurable.get("checkpoint_id"),
            checkpoint_map=None
        )
    
    from langchain_core.messages import BaseMessage

    values = {}
    channel_values = checkpoint_tuple.checkpoint['channel_values']

    pending_writes = checkpoint_tuple.pending_writes or []
    pending_by_channel: dict[str, list[Any]] = {}
    for task_id, channel, value in pending_writes:
        pending_by_channel.setdefault(channel, []).append(value)

    merged_values: dict[str, Any] = {**channel_values}
    for channel, pending_vals in pending_by_channel.items():
        if channel not in merged_values:
            merged_values[channel] = pending_vals
        elif isinstance(merged_values[channel], list):
            merged_values[channel] = [*merged_values[channel], *pending_vals]

    if "messages" in merged_values:
        messages: list[BaseMessage] = merged_values["messages"]
        if len(messages) > 0:
            if isinstance(messages[0], dict):
                values['messages'] = messages
            else:
                values['messages'] = [m.model_dump() for m in messages]
        else:
            values['messages'] = []
    
    # TODO, metadata should include source, user_id, graph_id, created_by, assistant_id etc
    metadata = ThreadStateMetadata(
        source=checkpoint_tuple.metadata.get('source', None),
        step=checkpoint_tuple.metadata.get('step', None),
        run_id=checkpoint_tuple.metadata.get('run_id', None),
        user_id=configurable.get('user_id', None),
        graph_id=configurable.get('graph_id', None),
        assistant_id=configurable.get('assistant_id', None),        
        thread_id=thread_id,        
    )
    
    return ThreadState(
        values=values,
        next=next_nodes or [],
        checkpoint=checkpoint,
        metadata=metadata,
        created_at=checkpoint_tuple.checkpoint["ts"],
        parent_config=parent_config,
        tasks=tasks or [],
        interrupts=interrupts or []
    )

class ThreadUpdateStateRequest(BaseModel):
    values: dict[str, Any] | list[dict] | None = None
    as_node: str | None = None
    checkpoint_id: str | None = None
    checkpoint: Checkpoint | None = None


class ThreadUpdateStateResponse(BaseModel):
    checkpoint: Checkpoint


class ThreadGetStateRequest(BaseModel):
    checkpoint: Checkpoint | None = None
    checkpoint_id: str | None = None
    subgraphs: bool = False


class ThreadGetHistoryRequest(BaseModel):
    limit: int = 10
    before: str | None = None
    metadata: dict[str, Any] | None = None
    checkpoint: Checkpoint | None = None


class ThreadRun(Run):
    pass


# ============================================
# Cron 相关模型定义 (兼容 langgraph_sdk CronClient)
# ============================================

CronSortBy = Literal[
    "cron_id",
    "assistant_id",
    "thread_id",
    "created_at",
    "updated_at",
    "next_run_date",
    "end_time",
]

CronSelectField = Literal[
    "cron_id",
    "assistant_id",
    "thread_id",
    "end_time",
    "schedule",
    "created_at",
    "updated_at",
    "user_id",
    "payload",
    "next_run_date",
    "metadata",
    "on_run_completed",
    "enabled",
]


class CronCreate(BaseModel):
    schedule: str = Field(..., description="Cron schedule expression (UTC)")
    assistant_id: str = Field(..., description="Assistant ID or graph name")
    thread_id: str | None = Field(None, description="Thread ID (for thread-scoped crons)")
    input: InputModel | None = Field(None, description="Input data for the graph")
    metadata: dict[str, Any] | None = Field(None, description="Metadata for the cron runs")
    config: dict[str, Any] | None = Field(None, description="Configuration for the assistant")
    context: dict[str, Any] | None = Field(None, description="Static context to add")
    interrupt_before: list[str] | None = Field(None, description="Nodes to interrupt before")
    interrupt_after: list[str] | None = Field(None, description="Nodes to interrupt after")
    webhook: str | None = Field(None, description="Webhook URL to call after run completes")
    on_run_completed: OnCompletionBehavior | None = Field(None, description="Thread behavior on completion")
    multitask_strategy: MultitaskStrategy | None = Field(None, description="Multitask strategy")
    end_time: datetime | None = Field(None, description="Time to stop running the cron")
    enabled: bool | None = Field(None, description="Whether the cron is enabled")
    stream_mode: Sequence[StreamMode] | StreamMode | None = Field(None, description="Stream mode(s)")
    stream_subgraphs: bool = Field(False, description="Whether to stream subgraphs")
    stream_resumable: bool = Field(False, description="Whether the stream is resumable")
    durability: Durability | None = Field(None, description="Durability level")


class CronUpdate(BaseModel):
    schedule: str | None = Field(None, description="Cron schedule expression (UTC)")
    end_time: datetime | None = Field(None, description="Time to stop running the cron")
    input: InputModel | None = Field(None, description="Input data for the graph")
    metadata: dict[str, Any] | None = Field(None, description="Metadata for the cron runs")
    config: dict[str, Any] | None = Field(None, description="Configuration for the assistant")
    context: dict[str, Any] | None = Field(None, description="Static context")
    webhook: str | None = Field(None, description="Webhook URL")
    interrupt_before: list[str] | None = Field(None, description="Nodes to interrupt before")
    interrupt_after: list[str] | None = Field(None, description="Nodes to interrupt after")
    on_run_completed: OnCompletionBehavior | None = Field(None, description="Thread behavior on completion")
    enabled: bool | None = Field(None, description="Whether the cron is enabled")
    stream_mode: Sequence[StreamMode] | StreamMode | None = Field(None, description="Stream mode(s)")
    stream_subgraphs: bool | None = Field(None, description="Whether to stream subgraphs")
    stream_resumable: bool | None = Field(None, description="Whether the stream is resumable")
    durability: Durability | None = Field(None, description="Durability level")


class CronSearchRequest(BaseModel):
    assistant_id: str | None = Field(None, description="Filter by assistant ID")
    thread_id: str | None = Field(None, description="Filter by thread ID")
    enabled: bool | None = Field(None, description="Filter by enabled status")
    limit: int = Field(10, description="Max results to return")
    offset: int = Field(0, description="Number of results to skip")
    sort_by: CronSortBy | None = Field(None, description="Sort by field")
    sort_order: SortOrder | None = Field(None, description="Sort order")


class Cron(BaseModel):
    cron_id: str
    assistant_id: str
    thread_id: str | None = None
    on_run_completed: OnCompletionBehavior | None = None
    end_time: datetime | None = None
    schedule: str
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any] = {}
    user_id: str | None = None
    next_run_date: datetime | None = None
    metadata: dict[str, Any] = {}
    enabled: bool = True


# ============================================
# Assistant 相关模型定义 (兼容 langgraph_sdk AssistantsClient)
# ============================================

OnConflictBehavior = Literal["raise", "do_nothing"]
AssistantSortBy = Literal[
    "assistant_id", "graph_id", "name", "created_at", "updated_at"
]
AssistantSelectField = Literal[
    "assistant_id",
    "graph_id",
    "name",
    "description",
    "config",
    "context",
    "created_at",
    "updated_at",
    "metadata",
    "version",
]


class Assistant(BaseModel):
    assistant_id: str
    graph_id: str
    config: dict[str, Any] = {}
    context: dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = {}
    version: int = 1
    name: str = "Untitled"
    description: str | None = None
    user_id: str | None = None


class AssistantVersion(BaseModel):
    assistant_id: str
    version: int
    graph_id: str
    config: dict[str, Any] = {}
    context: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    name: str = "Untitled"
    description: str | None = None
    created_at: datetime


class AssistantCreateRequest(BaseModel):
    graph_id: str | None = Field(None, description="Graph ID the assistant should use")
    config: dict[str, Any] | None = Field(None, description="Configuration for the graph")
    context: dict[str, Any] | None = Field(None, description="Static context to add to the assistant")
    metadata: dict[str, Any] | None = Field(None, description="Metadata to add to the assistant")
    assistant_id: str | None = Field(None, description="Custom assistant ID (defaults to UUID)")
    if_exists: OnConflictBehavior = Field("raise", description="How to handle duplicate creation")
    name: str | None = Field(None, description="Name of the assistant")
    description: str | None = Field(None, description="Description of the assistant")


class AssistantUpdateRequest(BaseModel):
    graph_id: str | None = Field(None, description="New graph ID")
    config: dict[str, Any] | None = Field(None, description="New configuration")
    context: dict[str, Any] | None = Field(None, description="New static context")
    metadata: dict[str, Any] | None = Field(None, description="Metadata to merge")
    name: str | None = Field(None, description="New name")
    description: str | None = Field(None, description="New description")


class AssistantSearchRequest(BaseModel):
    metadata: dict[str, Any] | None = Field(None, description="Metadata to filter by")
    graph_id: str | None = Field(None, description="Graph ID to filter by")
    name: str | None = Field(None, description="Name to filter by (case-insensitive substring)")
    limit: int = Field(10, description="Max results to return")
    offset: int = Field(0, description="Number of results to skip")
    sort_by: AssistantSortBy | None = Field(None, description="Sort by field")
    sort_order: SortOrder | None = Field(None, description="Sort order")
    select: list[AssistantSelectField] | None = Field(None, description="Fields to include")


class AssistantSearchResponse(BaseModel):
    assistants: list[Assistant]
    next: str | None = None


class AssistantSetVersionRequest(BaseModel):
    version: int = Field(..., description="Version to set as latest")
