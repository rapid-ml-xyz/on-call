from abc import ABC, abstractmethod
from typing import Any, Dict, List, Generic
from .models import S, T, Agent, NodeConfig, NodeType


class BaseOrchestrator(ABC, Generic[S, T]):
    """Base class for workflow orchestration."""

    def __init__(self):
        self.tools: Dict[str, T] = {}
        self.nodes: Dict[str, NodeConfig[S, T]] = {}
        self.entry_point: str | None = None

    def get_tools_for_node(self, node_name: str) -> List[T]:
        node = self.nodes.get(node_name)
        if not node:
            raise ValueError(f"Node {node_name} not found")

        if node.node_type == NodeType.FUNCTION:
            return []

        if not node.allowed_tools:
            return list(self.tools.values())

        return [
            tool for name, tool in self.tools.items()
            if name in node.allowed_tools
        ]

    @abstractmethod
    def create_agent(self, tools: List[T], agent_config: Dict[str, Any]) -> Agent[S]:
        pass

    def add_node(self, config: NodeConfig[S, T]) -> None:
        if config.node_type == NodeType.AGENT:
            if not config.agent_config:
                raise ValueError("Agent nodes must include agent_config")
            if not config.agent:
                tools = [self.tools[name] for name in config.allowed_tools]
                config.agent = self.create_agent(tools, config.agent_config)

        self.nodes[config.name] = config
        self._add_node_to_graph(config)

    def configure_nodes(self, node_configs: List[NodeConfig[S, T]]) -> None:
        for config in node_configs:
            self.add_node(config)

    @abstractmethod
    def _add_node_to_graph(self, node: NodeConfig[S, T]) -> None:
        pass

    @abstractmethod
    def set_entry_point(self, node_name: str) -> None:
        pass

    @abstractmethod
    def run(self, initial_state: S) -> S:
        pass

    @abstractmethod
    def visualize_graph(self) -> None:
        pass
