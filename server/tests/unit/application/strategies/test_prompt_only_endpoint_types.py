from app.application.strategies.prompt_only_reasoner import (
    PromptOnlyDecision,
    PromptOnlyReasoner,
)
from app.application.use_cases.build_evidence_context import EvidenceContext
from app.application.use_cases.recover_missing_link import RecoveryStatus
from app.domain.ontology.schema import EntityType
from app.domain.reasoning.models import AbstentionReason, ReasoningStatus


class FakePromptModel:
    def __init__(self, response: PromptOnlyDecision) -> None:
        self.response = response
        self.prompt = ""

    def invoke(self, prompt: str) -> PromptOnlyDecision:
        self.prompt = prompt
        return self.response


def test_equipment_cannot_be_generated_as_consequence() -> None:
    model = FakePromptModel(
        PromptOnlyDecision(
            decision="propose",
            missing_value="hexane storage tank",
            confidence=0.9,
            reason="The equipment is mentioned in the scenario.",
        )
    )
    context = EvidenceContext(
        scenario_id="T301-004",
        predicate="co_listed_with_consequence",
        known_endpoint="OperationalError",
        missing_endpoint="object",
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        scenario_context="Hexane storage unit | T-301 hexane storage tank",
        scenario_entities=(
            ("Hexane storage unit", EntityType.PROCESS_UNIT),
            ("T-301 hexane storage tank and associated equipment", EntityType.EQUIPMENT),
        ),
    )

    result = PromptOnlyReasoner(chat_model=model).reason(context)

    assert "Required missing-endpoint type: Consequence" in model.prompt
    assert result.status == ReasoningStatus.ABSTAINED
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_SUPPORT
    assert "Equipment" in result.supporting_evidence[0]
    assert "Consequence" in result.supporting_evidence[0]


def test_unlisted_consequence_candidate_is_not_rejected_as_wrong_type() -> None:
    model = FakePromptModel(
        PromptOnlyDecision(
            decision="propose",
            missing_value="Overfilling",
            confidence=0.7,
            reason="The scenario describes a high-level storage deviation.",
        )
    )
    context = EvidenceContext(
        scenario_id="T301-004",
        predicate="co_listed_with_consequence",
        known_endpoint="OperationalError",
        missing_endpoint="object",
        status=RecoveryStatus.NO_CANDIDATES,
        evidence_items=(),
        scenario_context="Hexane storage unit | T-301 hexane storage tank",
        scenario_entities=(("T-301 hexane storage tank", EntityType.EQUIPMENT),),
    )

    result = PromptOnlyReasoner(chat_model=model).reason(context)

    assert result.status == ReasoningStatus.PROPOSED
    assert result.proposed_relation is not None
    assert result.proposed_relation.object == "Overfilling"
    assert result.proposed_relation.object_type == EntityType.CONSEQUENCE
