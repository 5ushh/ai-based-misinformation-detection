"""
AI-Based Misinformation Detection — NLP Classifier
Socio-Technical System | ECE-GY 5213 NYU Tandon, Spring 2026
Author: Sushmitha Vashist (sv3005)

Transformer-based NLP classifier that assigns misinformation confidence scores
to user-generated content, then routes to automated action or human review.
"""

import re
import math
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────────

class ContentCategory(str, Enum):
    HEALTH      = "health_misinfo"
    POLITICAL   = "political"
    FINANCIAL   = "financial"
    SATIRE      = "satire"
    SYNTHETIC   = "synthetic_media"
    GENERAL     = "general"


class RoutingDecision(str, Enum):
    AUTO_REMOVE   = "auto_remove"    # score > 0.90
    HUMAN_REVIEW  = "human_review"   # score 0.60–0.90
    PASS_THROUGH  = "pass_through"   # score < 0.60


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ContentItem:
    content_id: str
    text: str
    category: ContentCategory = ContentCategory.GENERAL
    user_history_flags: int = 0          # prior violations by this user
    source_reputation: float = 1.0       # 0.0 (unknown) → 1.0 (trusted)
    metadata: dict = field(default_factory=dict)
    submitted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ScoringResult:
    content_id: str
    confidence_score: float              # 0.0 → 1.0  (higher = more likely misinfo)
    routing_decision: RoutingDecision
    category: ContentCategory
    risk_signals: list[str]
    rule_violations: list[str]
    content_fingerprint: str
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Lexical signal dictionaries ─────────────────────────────────────────────

_MISINFO_SIGNALS = {
    # Health misinformation
    "cure",  "miracle cure", "doctors don't want you to know", "big pharma",
    "5g causes", "vaccine causes", "bleach cures", "100% natural remedy",
    "suppress the truth", "government is hiding",

    # Political manipulation
    "rigged election", "deep state", "fake ballots", "stolen election",
    "crisis actor", "false flag", "martial law incoming",

    # Financial manipulation
    "guaranteed returns", "get rich quick", "pump and dump", "insider tip",
    "100x your money", "secret investment",

    # General misinformation markers
    "they don't want you to know", "mainstream media lies",
    "share before deleted", "wake up sheeple", "do your own research",
    "alternative facts",
}

_SATIRE_SIGNALS = {
    "satire", "parody", "onion", "babylon bee", "the onion",
    "not real news", "obviously fake", "for entertainment",
    "fictional account", "humor", "/s",
}

_CREDIBILITY_BOOSTERS = {
    "according to", "study shows", "researchers found", "published in",
    "peer reviewed", "cdc", "who", "nih", "reuters", "ap news",
    "university of", "journal of",
}

_POLICY_VIOLATIONS = {
    "buy followers",  "click the link below to win",
    "wire transfer now", "nigerian prince", "send bitcoin",
    "leaked video", "classified document",
}


# ── Feature extraction ──────────────────────────────────────────────────────

def _extract_features(text: str) -> dict:
    """Extract lexical and structural signals from raw text."""
    lower = text.lower()
    words = re.findall(r'\b\w+\b', lower)
    word_count = max(len(words), 1)

    misinfo_hits = sum(1 for s in _MISINFO_SIGNALS if s in lower)
    satire_hits  = sum(1 for s in _SATIRE_SIGNALS  if s in lower)
    cred_hits    = sum(1 for s in _CREDIBILITY_BOOSTERS if s in lower)

    caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclaim_ratio = text.count('!') / word_count
    url_count     = len(re.findall(r'https?://\S+', text))

    return {
        "misinfo_signal_density": misinfo_hits / word_count,
        "satire_signal_count":    satire_hits,
        "credibility_signal_density": cred_hits / word_count,
        "caps_ratio":     caps_ratio,
        "exclaim_ratio":  exclaim_ratio,
        "url_count":      url_count,
        "word_count":     word_count,
        "misinfo_hits":   misinfo_hits,
    }


def _detect_rule_violations(text: str) -> list[str]:
    """Hard policy rule checks — categorical violations bypass the ML score."""
    lower = text.lower()
    return [v for v in _POLICY_VIOLATIONS if v in lower]


# ── Scoring engine ──────────────────────────────────────────────────────────

class MisinfoClassifier:
    """
    Lightweight rule-augmented NLP classifier.

    In production this wraps a fine-tuned BERT-family model.
    This implementation uses calibrated heuristics that mirror
    the scoring logic described in the system architecture report,
    making the pipeline fully runnable without GPU dependencies.
    """

    # Routing thresholds (configurable per category in production)
    THRESHOLDS = {
        RoutingDecision.AUTO_REMOVE:  0.90,
        RoutingDecision.HUMAN_REVIEW: 0.60,
    }

    def score(self, item: ContentItem) -> ScoringResult:
        features      = _extract_features(item.text)
        rule_violations = _detect_rule_violations(item.text)
        risk_signals: list[str] = []

        # ── Base score from lexical signals ────────────────────────────────
        score = 0.0

        # Misinformation signal density  (weight: 0.40)
        msd = features["misinfo_signal_density"]
        score += min(msd * 8.0, 0.40)
        if msd > 0:
            risk_signals.append(f"misinfo_signals_detected ({features['misinfo_hits']})")

        # Caps / shouting  (weight: 0.10)
        if features["caps_ratio"] > 0.25:
            score += 0.10
            risk_signals.append("excessive_caps")

        # Exclamation spam  (weight: 0.08)
        if features["exclaim_ratio"] > 0.05:
            score += 0.08
            risk_signals.append("exclamation_spam")

        # Suspicious URLs  (weight: 0.07)
        if features["url_count"] >= 2:
            score += 0.07
            risk_signals.append("multiple_urls")

        # ── Satire discount ─────────────────────────────────────────────────
        if features["satire_signal_count"] > 0:
            score *= 0.40          # heavy discount for declared satire
            risk_signals.append("satire_signals_present")

        # ── Credibility boost ───────────────────────────────────────────────
        cred_discount = min(features["credibility_signal_density"] * 3.0, 0.20)
        score -= cred_discount

        # ── User/source context modifiers ───────────────────────────────────
        # Repeat offender
        if item.user_history_flags > 0:
            bonus = min(item.user_history_flags * 0.05, 0.15)
            score += bonus
            risk_signals.append(f"user_prior_flags ({item.user_history_flags})")

        # Low-reputation source
        if item.source_reputation < 0.50:
            score += (0.50 - item.source_reputation) * 0.20
            risk_signals.append("low_source_reputation")

        # ── Hard rule violation override ────────────────────────────────────
        if rule_violations:
            score = max(score, 0.95)    # floor at 0.95 — auto-remove
            risk_signals.append("policy_rule_violation")

        # ── Category-specific adjustment ────────────────────────────────────
        category_boosts = {
            ContentCategory.HEALTH:    0.05,
            ContentCategory.FINANCIAL: 0.03,
            ContentCategory.POLITICAL: 0.02,
        }
        score += category_boosts.get(item.category, 0.0)

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # ── Routing decision ────────────────────────────────────────────────
        if score >= self.THRESHOLDS[RoutingDecision.AUTO_REMOVE]:
            decision = RoutingDecision.AUTO_REMOVE
        elif score >= self.THRESHOLDS[RoutingDecision.HUMAN_REVIEW]:
            decision = RoutingDecision.HUMAN_REVIEW
        else:
            decision = RoutingDecision.PASS_THROUGH

        fingerprint = hashlib.sha256(item.text.encode()).hexdigest()[:16]

        return ScoringResult(
            content_id        = item.content_id,
            confidence_score  = round(score, 4),
            routing_decision  = decision,
            category          = item.category,
            risk_signals      = risk_signals,
            rule_violations   = rule_violations,
            content_fingerprint = fingerprint,
        )
