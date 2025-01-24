from ..orchestrator.engines import LangGraphMessageState


def do_nothing(state: LangGraphMessageState) -> LangGraphMessageState:
    return state
