"""
AI Misinformation Detection — FastAPI REST Layer
Endpoints: submit content, get queue, moderator actions, appeals, stats.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid

from src.ml.classifier import MisinfoClassifier, ContentItem, ContentCategory
from src.ml.audit import AuditLog, ModeratorAction

# ── App & shared state ───────────────────────────────────────────────────────

app = FastAPI(
    title="AI-Based Misinformation Detection API",
    description=(
        "Socio-technical content moderation system. "
        "Combines NLP classification, human-in-the-loop moderation, "
        "audit logging, and bias monitoring."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = MisinfoClassifier()
audit_log  = AuditLog()

# ── Request / response schemas ───────────────────────────────────────────────

class SubmitContentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    category: ContentCategory = ContentCategory.GENERAL
    user_history_flags: int = Field(0, ge=0, le=100)
    source_reputation: float = Field(1.0, ge=0.0, le=1.0)
    metadata: dict = {}

class ModeratorDecisionRequest(BaseModel):
    moderator_id: str
    action: ModeratorAction
    justification_code: Optional[str] = None

class AppealRequest(BaseModel):
    content_id: str

class AppealResolveRequest(BaseModel):
    content_id: str
    outcome: str  # "upheld" | "overturned"

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "system": "AI-Based Misinformation Detection",
        "status": "operational",
        "version": "1.0.0",
    }


@app.post("/api/v1/content/submit", tags=["detection"])
def submit_content(req: SubmitContentRequest):
    """
    Submit user-generated content for misinformation scoring.
    Returns confidence score and routing decision.
    """
    content_id = f"#{str(uuid.uuid4())[:4].upper()}"
    item = ContentItem(
        content_id        = content_id,
        text              = req.text,
        category          = req.category,
        user_history_flags= req.user_history_flags,
        source_reputation = req.source_reputation,
        metadata          = req.metadata,
    )
    result = classifier.score(item)

    # Auto-log automated actions immediately
    if result.routing_decision == "auto_remove":
        audit_log.record(result, moderator_action=ModeratorAction.CONFIRM_REMOVE)
    elif result.routing_decision == "pass_through":
        audit_log.record(result, moderator_action=ModeratorAction.LABEL_ONLY)
    else:
        audit_log.record(result)   # stays PENDING for human review

    return {
        "content_id":       result.content_id,
        "confidence_score": result.confidence_score,
        "routing_decision": result.routing_decision,
        "category":         result.category,
        "risk_signals":     result.risk_signals,
        "rule_violations":  result.rule_violations,
        "fingerprint":      result.content_fingerprint,
        "processed_at":     result.processed_at,
    }


@app.get("/api/v1/moderation/queue", tags=["moderation"])
def get_review_queue():
    """Return all items pending human moderator review."""
    return {"queue": audit_log.pending_human_review()}


@app.post("/api/v1/moderation/{content_id}/decide", tags=["moderation"])
def moderator_decision(content_id: str, req: ModeratorDecisionRequest):
    """Submit a moderator decision for a queued content item."""
    events = audit_log.all_events()
    match = next((e for e in events if e["content_id"] == content_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Content ID not found")

    from src.ml.classifier import ScoringResult, RoutingDecision
    # Reconstruct a minimal ScoringResult to pass to record()
    sr = ScoringResult(
        content_id          = match["content_id"],
        confidence_score    = match["confidence_score"],
        routing_decision    = RoutingDecision(match["routing_decision"]),
        category            = ContentCategory(match["category"]),
        risk_signals        = match["risk_signals"],
        rule_violations     = match["rule_violations"],
        content_fingerprint = match["content_fingerprint"],
    )
    audit_log.record(
        sr,
        moderator_id      = req.moderator_id,
        moderator_action  = req.action,
        justification_code= req.justification_code,
    )
    return {"status": "decision_recorded", "content_id": content_id, "action": req.action}


@app.post("/api/v1/appeals/submit", tags=["appeals"])
def submit_appeal(req: AppealRequest):
    """User submits an appeal for a moderation action."""
    ok = audit_log.submit_appeal(req.content_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Content ID not found")
    return {"status": "appeal_submitted", "content_id": req.content_id}


@app.post("/api/v1/appeals/resolve", tags=["appeals"])
def resolve_appeal(req: AppealResolveRequest):
    """Senior moderator resolves an appeal."""
    ok = audit_log.resolve_appeal(req.content_id, req.outcome)
    if not ok:
        raise HTTPException(status_code=404, detail="No pending appeal for that content ID")
    return {"status": "appeal_resolved", "outcome": req.outcome}


@app.get("/api/v1/stats", tags=["governance"])
def get_stats():
    """Platform-level moderation statistics for transparency reporting."""
    return audit_log.stats()


@app.get("/api/v1/governance/bias-report", tags=["governance"])
def get_bias_report():
    """Category-level override rates — proxy for bias monitoring."""
    return audit_log.bias_report()


@app.get("/api/v1/audit/events", tags=["governance"])
def get_audit_events():
    """Full immutable audit log (all moderation events)."""
    return {"events": audit_log.all_events()}
