"""Tracing with MLflow, kept on this computer (data/mlflow/mlflow.db): every answer is one trace, with a step for the router, each
search, each embedding call and each model call (prompt, reply, tokens, seconds). Study tools and uploads are traced the same way.
View with trace_ui.bat. Off with EDU_RAG_TRACE=0, or when MLflow is not installed: the app then runs exactly as before.
Tracing must never break an answer, so every MLflow call here is guarded."""
import functools
import os
import threading
from contextlib import contextmanager

from . import config

_STATE = {}
_LOCK = threading.Lock()
MAX_TEXT = 4000                                     # characters kept per text field, so the store stays small


def enabled():
    return config.TRACING and "PYTEST_CURRENT_TEST" not in os.environ


def client():
    """The mlflow module, set up once (about 8 s the first time); None when tracing is off or unavailable."""
    if not enabled():
        return None
    if "mlflow" not in _STATE:
        with _LOCK:
            if "mlflow" not in _STATE:
                try:
                    import mlflow
                    config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
                    mlflow.set_tracking_uri(config.MLFLOW_URI)
                    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
                    _STATE["mlflow"] = mlflow
                except Exception as e:              # not installed, or the store cannot be opened
                    _STATE.update(mlflow=None, error=f"{type(e).__name__}: {e}")
    return _STATE["mlflow"]


def warm_up():
    """Load MLflow in the background at app start, so the first answer does not wait for it."""
    if enabled() and "mlflow" not in _STATE:
        threading.Thread(target=client, daemon=True, name="edu-rag-tracing").start()


def status():
    """'on', 'off', 'starting' or the reason it is unavailable: shown in the sidebar."""
    if not enabled():
        return "off"
    if "mlflow" not in _STATE:
        return "starting"
    return "on" if _STATE["mlflow"] else f"unavailable ({_STATE.get('error', '')[:60]})"


def clip(value):
    """Long texts are cut and big lists shortened before they are stored."""
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + f"... [{len(value) - MAX_TEXT} more characters]"
    if isinstance(value, dict):
        return {k: clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clip(v) for v in value[:50]]
    return value


@contextmanager
def span(name, kind="CHAIN", inputs=None, **attributes):
    """One step of a trace. Yields the MLflow span, or None when tracing is off (callers pass it to finish())."""
    ml = client()
    opened = None
    if ml is not None:
        try:
            opened = ml.start_span(name=name, span_type=kind, attributes={k: v for k, v in attributes.items() if v is not None} or None)
            live = opened.__enter__()
            if inputs is not None:
                live.set_inputs(clip(inputs))
        except Exception:
            opened = None
    if opened is None:
        yield None
        return
    try:
        yield live
    except BaseException as e:                      # the step failed: MLflow records the error, the app still sees it
        try:
            opened.__exit__(type(e), e, e.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            opened.__exit__(None, None, None)
        except Exception:
            pass


def finish(live, outputs=None, tokens=None, **attributes):
    """Record a step's result. tokens = (prompt, output) fills MLflow's token usage, which it totals per trace."""
    if live is None:
        return
    try:
        if outputs is not None:
            live.set_outputs(clip(outputs))
        for k, v in attributes.items():
            if v is not None:
                live.set_attribute(k, v)
        if tokens:
            live.set_attribute("mlflow.chat.tokenUsage", {"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": tokens[0] + tokens[1]})
    except Exception:
        pass


def traced(name, kind="CHAIN", inputs=None, outputs=None):
    """Decorator for a whole function: inputs(*args, **kwargs) and outputs(result) pick what is worth storing."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                given = inputs(*args, **kwargs) if inputs else None
            except Exception:
                given = None
            with span(name, kind, given) as live:
                result = fn(*args, **kwargs)
                try:
                    finish(live, outputs(result) if outputs else None)
                except Exception:
                    pass
                return result
        return inner
    return wrap


def tag_trace(live, session=None, **tags):
    """Tags on the whole trace (model, style, ...) and the chat it belongs to, so a chat's answers can be grouped."""
    ml = client()
    if ml is None or live is None:
        return None
    try:
        ml.update_current_trace(tags={k: str(v) for k, v in tags.items() if v is not None},
                                metadata={"mlflow.trace.session": str(session)} if session else None)
        return live.trace_id
    except Exception:
        return None


def log_feedback(trace_id, name, value, comment=None):
    """A student's thumbs up/down, attached to the trace of the answer it rates."""
    ml = client()
    if ml is None or not trace_id:
        return False
    try:
        from mlflow.entities import AssessmentSource
        ml.log_feedback(trace_id=trace_id, name=name, value=value, rationale=comment,
                        source=AssessmentSource(source_type="HUMAN", source_id="student"))
        return True
    except Exception:
        return False


def documents(chunks):
    """Retrieved passages in the shape MLflow's trace viewer shows as documents."""
    return [{"page_content": c.get("text", "")[:1500],
             "metadata": {"source": c.get("source_name"), "pages": f"{c.get('page_start')}-{c.get('page_end')}", "title": c.get("title"),
                          "chunk_id": c.get("chunk_id"), "role": c.get("role")}} for c in chunks]
