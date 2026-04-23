import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
from pathlib import Path

from config import Config

# =========================
# SYSTEM PROMPT AND CONSTANTS
# =========================

SYSTEM_PROMPT = (
    "You are an expert ecommerce email payload validation agent. "
    "Your job is to validate the structure, content, and compliance of incoming ecommerce-related email payloads. "
    "Check for required fields, correct data types, and adherence to business rules. "
    "If the payload is valid, respond with a JSON object: {\"valid\": true, \"errors\": []}. "
    "If invalid, respond with {\"valid\": false, \"errors\": [list of error messages]}. "
    "Be formal and concise in your error messages. "
    "Do not attempt to execute or transform the payload; only validate and explain issues."
)
OUTPUT_FORMAT = "JSON object with fields: valid (bool), errors (list of strings)."
FALLBACK_RESPONSE = (
    "{\"valid\": false, \"errors\": [\"Unable to validate the payload due to insufficient information or unsupported format.\"]}"
)

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# INPUT/OUTPUT MODELS
# =========================

class ValidateEmailPayloadRequest(BaseModel):
    payload: Dict[str, Any] = Field(..., description="The ecommerce email payload to validate (arbitrary JSON object)")

class ValidateEmailPayloadResponse(BaseModel):
    valid: bool = Field(..., description="Whether the payload is valid")
    errors: List[str] = Field(..., description="List of validation error messages (empty if valid)")

# =========================
# SANITIZER UTILITY
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# OBSERVABILITY LIFESPAN
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Ecommerce Email Payload Validation Agent",
    description="Validates ecommerce email payloads for structure, content, and compliance using LLM.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# =========================
# ERROR HANDLING
# =========================

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request body.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted (quotes, commas, brackets).",
                "Check that all required fields are present and correctly typed.",
                "Payloads must not exceed 50,000 characters."
            ]
        }
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Malformed JSON in request body.",
            "details": str(exc),
            "tips": [
                "Check for missing or extra commas, brackets, or quotes.",
                "Ensure the JSON is valid and not truncated."
            ]
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.getLogger("agent").error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error.",
            "details": str(exc),
            "tips": [
                "If this error persists, contact the agent administrator."
            ]
        }
    )

# =========================
# VALIDATION LOGIC
# =========================

class LLMService:
    """Handles LLM calls for validation."""

    def __init__(self):
        self._client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_llm_client(self):
        """Lazily initialize and return the Azure OpenAI async client."""
        if self._client is not None:
            return self._client
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        import openai
        self._client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def validate_payload(self, payload: Dict[str, Any]) -> str:
        """
        Calls the LLM to validate the payload.
        Returns: LLM response as string (should be JSON).
        """
        client = self.get_llm_client()
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
            },
            {
                "role": "user",
                "content": f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
            }
        ]
        _t0 = _time.time()
        _llm_kwargs = Config.get_llm_kwargs()
        response = await client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4o",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4o",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return content

# =========================
# AGENT CLASS
# =========================

class EcommerceEmailPayloadValidationAgent:
    """Main agent class for payload validation."""

    def __init__(self):
        self.llm_service = LLMService()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the given payload using the LLM.
        Returns: dict with keys 'valid' (bool) and 'errors' (list of strings).
        """
        async with trace_step(
            "validate_payload",
            step_type="llm_call",
            decision_summary="Validate ecommerce email payload using LLM",
            output_fn=lambda r: f"valid={r.get('valid', '?')}, errors={len(r.get('errors', []))}"
        ) as step:
            # Input validation
            if not isinstance(payload, dict):
                result = {
                    "valid": False,
                    "errors": ["Payload must be a JSON object."]
                }
                step.capture(result)
                return result
            try:
                raw_llm_response = await self.llm_service.validate_payload(payload)
                sanitized = sanitize_llm_output(raw_llm_response, content_type="code")
                try:
                    parsed = json.loads(sanitized)
                    # Defensive: ensure correct shape
                    valid = bool(parsed.get("valid", False))
                    errors = parsed.get("errors", [])
                    if not isinstance(errors, list):
                        errors = [str(errors)]
                    result = {"valid": valid, "errors": errors}
                except Exception:
                    # If LLM output is not valid JSON, fallback
                    result = json.loads(FALLBACK_RESPONSE)
                step.capture(result)
                return result
            except Exception as e:
                logging.getLogger("agent").warning(f"LLM validation failed: {e}")
                try:
                    result = json.loads(FALLBACK_RESPONSE)
                except Exception:
                    result = {"valid": False, "errors": ["Agent error: " + str(e)]}
                step.capture(result)
                return result

# =========================
# ENDPOINTS
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/validate", response_model=ValidateEmailPayloadResponse)
async def validate_endpoint(req: ValidateEmailPayloadRequest):
    """
    Validate an ecommerce email payload.
    """
    agent = EcommerceEmailPayloadValidationAgent()
    result = await agent.process(req.payload)
    return result

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())