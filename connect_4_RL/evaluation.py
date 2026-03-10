from typing import Any, Callable
import gymnasium as gym


def has(
    o: Any,
    classinfo: Any | None = None,
    default: Any = None,
    path: list[str] | None = None,
    is_callable: bool | None = None,
) -> bool:
    if path is None:
        path = []
    try:
        cur = o
        for p in path:
            cur = cur[p]
        if classinfo is not None and not isinstance(cur, classinfo):
            raise ValueError("Not a match")
        if is_callable and not callable(cur):
            raise ValueError("Not callable")
        if not is_callable and callable(cur):
            raise ValueError("Is callable")
        return True
    except Exception:
        if default is not None and o is not None and len(path) > 0:
            cur = o
            for p in path[:-1]:
                if not has(cur, dict, path=[p]):
                    cur[p] = {}
                cur = cur[p]
            cur[path[-1]] = default
        return False


def evaluate(
    environment: str | Any,  # "Environment" → Any
    agents: list[str | Callable | Any] | None = None,  # Agent → Any
    configuration: dict[str, Any] | None = None,
    steps: list[list[dict[str, Any]]] | None = None,
    num_episodes: int = 1,
    debug: bool = False,
    state: list[dict[str, Any]] | None = None,
) -> list[list[float | None]]:
    """
    Evaluate and return the rewards of one or more episodes (environment and agents combo).

    Args:
        environment (str|Environment):
        agents (list):
        configuration (dict, optional):
        steps (list, optional):
        num_episodes (int=1, optional): How many episodes to execute (run until done).
        debug (bool=False, optional): Render print() statments to stdout
        state (optional)

    Returns:
        list of list of int: List of final rewards for all agents for all episodes.
    """
    if agents is None:
        agents = []
    if configuration is None:
        configuration = {}
    if steps is None:
        steps = []

    e = make(environment, configuration, steps=steps, debug=debug, state=state)
    rewards = [[] for i in range(num_episodes)]
    for i in range(num_episodes):
        last_state = e.run(agents)[-1]
        rewards[i] = [state.reward for state in last_state]
    return rewards


def make(
    environment: str | Any | Callable,
    configuration: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
    steps: list[list[dict[str, Any]]] | None = None,
    logs: list[list[dict[str, Any]]] | None = None,
    debug: bool = False,
    state: list[dict[str, Any]] | None = None,
) -> Any:
    if configuration is None:
        configuration = {}
    if info is None:
        info = {}
    if steps is None:
        steps = []
    if logs is None:
        logs = []

    if isinstance(environment, str):
        return gym.make(environment, **configuration)
    elif callable(environment):
        return environment(**configuration)
    elif has(environment, path=["interpreter"], is_callable=True):
        return environment
    raise ValueError("Unknown Environment Specification")
