#!/usr/bin/env python3
"""
demo.py — End-to-end pipeline demonstration
AI-Based Misinformation Detection System
Sushmitha Vashist | sv3005 | NYU Tandon ECE-GY 5213

Run: python demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.ml.classifier import MisinfoClassifier, ContentItem, ContentCategory
from src.ml.audit import AuditLog, ModeratorAction

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GREY   = "\033[90m"

ROUTE_COLOR = {
    "auto_remove":  RED,
    "human_review": YELLOW,
    "pass_through": GREEN,
}

SAMPLE_CONTENT = [
    ContentItem(
        content_id="C001",
        text=(
            "DOCTORS DON'T WANT YOU TO KNOW THIS! Miracle cure destroys Big Pharma!! "
            "Government is hiding the truth about vaccines!! Share before deleted!!!"
        ),
        category=ContentCategory.HEALTH,
        user_history_flags=3,
        source_reputation=0.1,
    ),
    ContentItem(
        content_id="C002",
        text=(
            "According to a peer-reviewed study published in the Journal of Epidemiology, "
            "the new mRNA vaccine demonstrates 94% efficacy against severe disease. "
            "Researchers at NIH confirmed the findings independently."
        ),
        category=ContentCategory.HEALTH,
        source_reputation=0.95,
    ),
    ContentItem(
        content_id="C003",
        text=(
            "BREAKING: Rigged election exposed! Deep state stole the ballots. "
            "Crisis actors everywhere. Wake up sheeple! Mainstream media lies to you every day!"
        ),
        category=ContentCategory.POLITICAL,
        user_history_flags=1,
    ),
    ContentItem(
        content_id="C004",
        text=(
            "[SATIRE] Local man discovers government has been hiding fact that the moon "
            "is made of cheese for decades. Obviously fake story for entertainment purposes /s"
        ),
        category=ContentCategory.SATIRE,
    ),
    ContentItem(
        content_id="C005",
        text=(
            "Guaranteed 100x returns on this secret investment! Insider tip from my contact. "
            "Wire transfer now to secure your spot. Get rich quick — limited offer!"
        ),
        category=ContentCategory.FINANCIAL,
        user_history_flags=2,
        source_reputation=0.05,
    ),
    ContentItem(
        content_id="C006",
        text=(
            "Today's weather is partly cloudy with a chance of rain this afternoon. "
            "Temperatures around 65°F. Have a nice day everyone!"
        ),
        category=ContentCategory.GENERAL,
    ),
]


def separator(char="─", width=64):
    print(f"{GREY}{char * width}{RESET}")


def run_demo():
    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{BOLD}  AI-Based Misinformation Detection — Pipeline Demo{RESET}")
    print(f"{GREY}  NYU Tandon ECE-GY 5213 | Sushmitha Vashist (sv3005){RESET}")
    print(f"{BOLD}{'═'*64}{RESET}\n")

    classifier = MisinfoClassifier()
    audit_log  = AuditLog()
    results    = []

    print(f"{BOLD}[PHASE 1] Content Ingestion & AI Scoring{RESET}")
    separator()

    for item in SAMPLE_CONTENT:
        result = classifier.score(item)
        results.append(result)

        # Auto-log
        if result.routing_decision == "auto_remove":
            audit_log.record(result, moderator_action=ModeratorAction.CONFIRM_REMOVE,
                             justification_code="AUTO_HIGH_CONFIDENCE")
        elif result.routing_decision == "pass_through":
            audit_log.record(result, moderator_action=ModeratorAction.LABEL_ONLY)
        else:
            audit_log.record(result)

        color  = ROUTE_COLOR[result.routing_decision]
        bar_len = int(result.confidence_score * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)

        print(f"  {BOLD}{item.content_id}{RESET}  [{bar}] {color}{result.confidence_score:.2f}{RESET}")
        print(f"       Category : {item.category.value}")
        print(f"       Decision : {color}{result.routing_decision.upper()}{RESET}")
        if result.risk_signals:
            print(f"       Signals  : {', '.join(result.risk_signals)}")
        if result.rule_violations:
            print(f"       {RED}POLICY VIOLATION: {', '.join(result.rule_violations)}{RESET}")
        print()

    # ── Simulated Human Moderation ─────────────────────────────────────────
    print(f"\n{BOLD}[PHASE 2] Simulated Human Moderation (HUMAN_REVIEW items){RESET}")
    separator()

    human_queue = audit_log.pending_human_review()
    if human_queue:
        for item in human_queue:
            print(f"  Moderator MOD-001 reviewing {item['content_id']} "
                  f"(score={item['confidence_score']}) → CONFIRM_REMOVE")
    else:
        print("  No items pending human review in this run.")

    # Resolve queue items
    for item in human_queue:
        from src.ml.classifier import ScoringResult, RoutingDecision
        sr = next(r for r in results if r.content_id == item["content_id"])
        audit_log.record(
            sr,
            moderator_id       = "MOD-001",
            moderator_action   = ModeratorAction.CONFIRM_REMOVE,
            justification_code = "HUMAN_REVIEW_CONFIRMED",
        )

    # ── Appeal simulation ──────────────────────────────────────────────────
    print(f"\n{BOLD}[PHASE 3] User Appeal Simulation{RESET}")
    separator()
    appeal_id = SAMPLE_CONTENT[2].content_id   # C003 political content
    audit_log.submit_appeal(appeal_id)
    print(f"  User submitted appeal for {appeal_id}")
    audit_log.resolve_appeal(appeal_id, "upheld")
    print(f"  Senior moderator resolved appeal → UPHELD (action stands)\n")

    # ── Stats ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[PHASE 4] Governance & Transparency Report{RESET}")
    separator()
    stats = audit_log.stats()
    print(f"  Total processed    : {stats.get('total', 0)}")
    print(f"  Auto-removed       : {stats.get('auto_removed', 0)}")
    print(f"  Human review       : {stats.get('human_review', 0)}")
    print(f"  Passed through     : {stats.get('passed', 0)}")
    print(f"  Avg confidence     : {stats.get('avg_confidence', 0):.4f}")
    print(f"  Override rate      : {stats.get('override_rate', 0):.2%}")
    print(f"  Auto-resolve rate  : {stats.get('auto_resolve_rate', 0):.2%}")

    print(f"\n{BOLD}[PHASE 5] Bias Monitoring (Category Override Rates){RESET}")
    separator()
    bias = audit_log.bias_report()
    for cat, data in bias.items():
        print(f"  {cat:<22} total={data['total']}  overrides={data['overrides']}  "
              f"rate={data['override_rate']:.2%}")

    print(f"\n{BOLD}{'═'*64}{RESET}")
    print(f"{GREEN}  Pipeline complete. All events recorded in audit log.{RESET}")
    print(f"{BOLD}{'═'*64}{RESET}\n")


if __name__ == "__main__":
    run_demo()
