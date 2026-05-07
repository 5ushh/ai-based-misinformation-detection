"""
Audit Logging & Moderator Action Store
Immutable append-only event log for every moderation decision.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class ModeratorAction(str, Enum):
    CONFIRM_REMOVE  = "confirm_remove"
    OVERRIDE_PASS   = "override_pass"
    ESCALATE        = "escalate_senior"
    LABEL_ONLY      = "label_only"
    PENDING         = "pending"


@dataclass
class AuditEvent:
    event_id:          str
    content_id:        str
    content_fingerprint: str
    confidence_score:  float
    routing_decision:  str
    category:          str
    risk_signals:      list[str]
    rule_violations:   list[str]
    moderator_id:      Optional[str]
    moderator_action:  str
    justification_code: Optional[str]
    ai_recommendation: str
    override:          bool               # True if human overrode AI
    timestamp:         str
    appeal_submitted:  bool = False
    appeal_outcome:    Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    """
    In-memory append-only audit log.
    Production: backed by PostgreSQL with immutable partition.
    """

    def __init__(self):
        self._events: list[AuditEvent] = []

    def record(
        self,
        scoring_result,
        moderator_id: Optional[str] = None,
        moderator_action: ModeratorAction = ModeratorAction.PENDING,
        justification_code: Optional[str] = None,
    ) -> AuditEvent:
        override = (
            moderator_action == ModeratorAction.OVERRIDE_PASS
            and scoring_result.routing_decision in ("auto_remove", "human_review")
        )

        event = AuditEvent(
            event_id            = str(uuid.uuid4()),
            content_id          = scoring_result.content_id,
            content_fingerprint = scoring_result.content_fingerprint,
            confidence_score    = scoring_result.confidence_score,
            routing_decision    = scoring_result.routing_decision,
            category            = scoring_result.category,
            risk_signals        = scoring_result.risk_signals,
            rule_violations     = scoring_result.rule_violations,
            moderator_id        = moderator_id,
            moderator_action    = moderator_action,
            justification_code  = justification_code,
            ai_recommendation   = scoring_result.routing_decision,
            override            = override,
            timestamp           = datetime.utcnow().isoformat(),
        )
        self._events.append(event)
        return event

    def submit_appeal(self, content_id: str) -> bool:
        for ev in self._events:
            if ev.content_id == content_id:
                ev.appeal_submitted = True
                return True
        return False

    def resolve_appeal(self, content_id: str, outcome: str) -> bool:
        for ev in self._events:
            if ev.content_id == content_id and ev.appeal_submitted:
                ev.appeal_outcome = outcome
                return True
        return False

    # ── Reporting helpers ─────────────────────────────────────────────────

    def all_events(self) -> list[dict]:
        return [e.to_dict() for e in self._events]

    def stats(self) -> dict:
        total = len(self._events)
        if total == 0:
            return {"total": 0}

        auto     = sum(1 for e in self._events if e.routing_decision == "auto_remove")
        human    = sum(1 for e in self._events if e.routing_decision == "human_review")
        passed   = sum(1 for e in self._events if e.routing_decision == "pass_through")
        overrides = sum(1 for e in self._events if e.override)
        appeals  = sum(1 for e in self._events if e.appeal_submitted)

        scores = [e.confidence_score for e in self._events]
        avg_score = sum(scores) / total

        return {
            "total":          total,
            "auto_removed":   auto,
            "human_review":   human,
            "passed":         passed,
            "override_count": overrides,
            "override_rate":  round(overrides / total, 4),
            "appeal_count":   appeals,
            "avg_confidence": round(avg_score, 4),
            "auto_resolve_rate": round((auto + passed) / total, 4),
        }

    def pending_human_review(self) -> list[dict]:
        return [
            e.to_dict() for e in self._events
            if e.routing_decision == "human_review"
            and e.moderator_action == ModeratorAction.PENDING
        ]

    def bias_report(self) -> dict:
        """Category-level false-positive proxy (overrides on auto-remove decisions)."""
        from collections import defaultdict
        cat_total    = defaultdict(int)
        cat_override = defaultdict(int)

        for e in self._events:
            cat_total[e.category] += 1
            if e.override:
                cat_override[e.category] += 1

        return {
            cat: {
                "total": cat_total[cat],
                "overrides": cat_override[cat],
                "override_rate": round(cat_override[cat] / cat_total[cat], 4)
                if cat_total[cat] > 0 else 0,
            }
            for cat in cat_total
        }
