"""
SkillRouter: automatically matches the most suitable skill for a question.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

from .skill_types import Skill, InstructionSlot
from ..data.types import TemporalQuestion

logger = logging.getLogger(__name__)

# Blacklist persistence filename: same directory as skills/*.md/*.json
_BLACKLIST_FILENAME = "blacklist.json"


class SkillRouter:
    """Automatically matches the most suitable skill for a question.

    Matching strategy:
    1. Rule matching: check whether the question hits a skill's trigger condition keywords
    2. Success-rate weighting: prefer skills with a higher success rate

    Args:
        skills_dir: skill file directory
        route_threshold: matching score threshold
    """

    def __init__(self, skills_dir: str = "", route_threshold: float = 0.45,
                 min_downstream_samples: int = 5,
                 baseline_accuracy: float = 0.0,
                 disable_below_baseline_margin: float = 0.05,
                 hard_fail_min_samples: int = 3,
                 visual_layer_enabled: bool = False,
                 agent_mode: str = "multi_round_search"):
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.route_threshold = route_threshold
        # The downstream win rate is treated as "credible evidence" only after at least N injections
        self.min_downstream_samples = min_downstream_samples
        # Global baseline that the orchestrator may update after each batch finishes
        self.baseline_accuracy = baseline_accuracy
        # If a skill's downstream win rate < baseline - margin and samples are sufficient, disable it automatically
        self.disable_below_baseline_margin = disable_below_baseline_margin
        # Early-disable threshold: even if the sample count has not reached min_downstream_samples,
        # as long as downstream fully fails (0 wins / N losses) and N >= hard_fail_min_samples, blacklist it directly.
        self.hard_fail_min_samples = hard_fail_min_samples
        self.visual_layer_enabled = bool(visual_layer_enabled)
        self._is_control_mode = (agent_mode == "control_api_harness")
        self.skills: Dict[str, Skill] = {}
        self.last_slot_route: Optional[Dict] = None
        # Permanent blacklist: skill_id -> {"reason": str, "banned_at": iso_ts, "stats": {...}}
        self._blacklist: Dict[str, Dict] = {}
        self._load_blacklist()

    def update_baseline(self, baseline_accuracy: float) -> None:
        """Called by the orchestrator at the end of each batch to update the global baseline used for skill disabling.

        After updating the baseline, scan all registered skills and automatically add to the blacklist
        any skill that has reached the "verified negative effect" condition. This way the routing takes
        effect immediately even if the next run does not restart the process.
        """
        self.baseline_accuracy = baseline_accuracy
        self._auto_blacklist_verified_bad()

    # ------------------------------------------------------------------
    # P0: blacklist mechanism
    # ------------------------------------------------------------------
    def _load_blacklist(self) -> None:
        """Load the permanent blacklist from skills_dir/blacklist.json."""
        if self.skills_dir is None:
            return
        bl_path = self.skills_dir / _BLACKLIST_FILENAME
        if not bl_path.exists():
            return
        try:
            data = json.loads(bl_path.read_text(encoding="utf-8"))
            # Compatible with two formats:
            #   legacy list[str] (plain list of skill_id)
            #   new {skill_id: {reason, banned_at, stats}}
            if isinstance(data, list):
                self._blacklist = {sid: {"reason": "legacy"} for sid in data}
            elif isinstance(data, dict):
                # Allow either {"skills": {...}} or a direct skill_id map
                payload = data.get("skills", data)
                if isinstance(payload, dict):
                    self._blacklist = dict(payload)
            logger.info(
                f"[SkillRouter] Loaded permanent blacklist: {len(self._blacklist)} skills"
            )
        except Exception as exc:
            logger.warning(f"[SkillRouter] Failed to load blacklist: {bl_path} ({exc})")

    def _persist_blacklist(self) -> None:
        if self.skills_dir is None:
            return
        try:
            self.skills_dir.mkdir(parents=True, exist_ok=True)
            bl_path = self.skills_dir / _BLACKLIST_FILENAME
            bl_path.write_text(
                json.dumps(
                    {"skills": self._blacklist},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[SkillRouter] Failed to persist blacklist: {exc}")

    def is_blacklisted(self, skill_id: str) -> bool:
        return skill_id in self._blacklist

    def blacklist_skill(self, skill_id: str, reason: str = "manual") -> None:
        """Add the specified skill to the permanent blacklist (called manually or during automatic scanning).

        Once blacklisted, `route` will never return this skill; it is also removed from the in-memory registry
        to avoid the SkillPromoter `_update_skill` branch being triggered again.
        """
        if skill_id in self._blacklist:
            return
        skill = self.skills.get(skill_id)
        stats = {}
        if skill is not None:
            down_total = skill.downstream_success + skill.downstream_failure
            stats = {
                "downstream_success": skill.downstream_success,
                "downstream_failure": skill.downstream_failure,
                "downstream_success_rate": skill.downstream_success_rate,
                "usage_count": skill.usage_count,
                "baseline_at_ban": self.baseline_accuracy,
            }
            stats["downstream_total"] = down_total
        self._blacklist[skill_id] = {
            "reason": reason,
            "banned_at": datetime.now().isoformat(),
            "stats": stats,
        }
        # Also remove the skill from the in-memory registry to prevent `mark_injected` / `record_outcome`
        # from continuing to accumulate downstream metrics, and to block subsequent `maybe_promote` updates.
        self.skills.pop(skill_id, None)
        self._persist_blacklist()
        logger.info(
            f"[SkillRouter] Blacklisted skill={skill_id} reason={reason} stats={stats}"
        )

    def unblacklist_skill(self, skill_id: str) -> bool:
        """Remove a skill from the blacklist (called manually)."""
        if skill_id not in self._blacklist:
            return False
        self._blacklist.pop(skill_id, None)
        self._persist_blacklist()
        logger.info(f"[SkillRouter] Removed skill from blacklist skill={skill_id}")
        return True

    def list_blacklist(self) -> Dict[str, Dict]:
        return dict(self._blacklist)

    def _auto_blacklist_verified_bad(self) -> None:
        """Scan all registered skills and add to the blacklist those that reach the "verified negative effect" threshold."""
        if not self.skills:
            return
        to_ban: List[str] = []
        for skill_id, skill in self.skills.items():
            reason = self._auto_ban_reason(skill)
            if reason:
                to_ban.append((skill_id, reason))
        for skill_id, reason in to_ban:
            self.blacklist_skill(skill_id, reason=reason)

    def _auto_ban_reason(self, skill: Skill) -> Optional[str]:
        """Determine whether a skill meets the automatic blacklisting condition, returning a reason string or None.

        Two trigger conditions (blacklisted if either is satisfied):
        1. Sufficient-samples variant: down_total >= min_downstream_samples
           and downstream_success_rate < baseline - margin
           (reuses the semantics of the original _is_verified_bad)
        2. Early-stop variant: down_total >= hard_fail_min_samples
           and downstream_success == 0
           (a pure 0 win rate is necessarily worse than the baseline even with few samples)
        """
        down_total = skill.downstream_success + skill.downstream_failure
        # 2) Early stop: 3+ injections all failed
        if (
            down_total >= self.hard_fail_min_samples
            and skill.downstream_success == 0
        ):
            return (
                f"hard_fail: {skill.downstream_success}/{down_total} wins, "
                f"baseline={self.baseline_accuracy:.1%}"
            )
        # 1) Sufficient samples: below the baseline margin
        if (
            down_total >= self.min_downstream_samples
            and self.baseline_accuracy > 0
            and skill.downstream_success_rate
            < self.baseline_accuracy - self.disable_below_baseline_margin
        ):
            return (
                f"below_baseline: rate={skill.downstream_success_rate:.1%} "
                f"< baseline {self.baseline_accuracy:.1%} - {self.disable_below_baseline_margin:.1%} "
                f"(n={down_total})"
            )
        return None

    def register_skill(self, skill: Skill):
        """Register a skill. Blacklisted skills are discarded directly and never enter the routing candidate set."""
        if skill.skill_id in self._blacklist:
            logger.info(
                f"[SkillRouter] Skipping registration of blacklisted skill: {skill.skill_id} "
                f"(reason={self._blacklist[skill.skill_id].get('reason')})"
            )
            return
        if not self._sanitize_skill_instructions(skill):
            logger.info(
                f"[SkillRouter] Skipping registration of skill with no valid instructions: {skill.skill_id}"
            )
            return
        self.skills[skill.skill_id] = skill

    @staticmethod
    def _extract_learning_type_from_skill_id(skill_id: str) -> str:
        """Reverse-parse learning_type from SKILL-<learning_type>__<question_type>-<date>."""
        if not skill_id.startswith("SKILL-"):
            return ""
        body = skill_id[len("SKILL-"):]
        parts = body.rsplit("-", 1)
        if len(parts) == 2 and parts[-1].isdigit():
            body = parts[0]
        if "__" in body:
            return body.split("__", 1)[0]
        return body

    def _sanitize_skill_instructions(self, skill: Skill) -> bool:
        """Clean special_instructions before registration to prevent historical truncated/vague instructions from entering the injection pipeline.

        Strategy change: template instructions (instructions_source == "template") are no longer accepted for registration,
        because template instructions overlap heavily with the base prompt and only increase token cost without adding information after injection.
        Only LLM-generated instructions (which have passed the quality check) may enter the routing.
        """
        try:
            from .skill_promoter import SkillPromoter

            # Reject registration of template skills directly
            if getattr(skill, "instructions_source", "template") == "template":
                logger.info(
                    f"[SkillRouter] Skipping registration of template skill (duplicate of base prompt): {skill.skill_id}"
                )
                return False

            learning_type = self._extract_learning_type_from_skill_id(skill.skill_id)
            if SkillPromoter.validate_special_instructions(
                skill.special_instructions,
                learning_type,
                self.visual_layer_enabled,
            ):
                return True

            # When the LLM instructions are unqualified, no longer fall back to a template; reject registration directly
            old_preview = (skill.special_instructions or "").replace("\n", " ")[:100]
            logger.warning(
                f"[SkillRouter] Skill LLM instructions unqualified, rejecting registration (no fallback to template): "
                f"{skill.skill_id} instructions={old_preview!r}"
            )
            return False
        except Exception as exc:
            logger.warning(
                f"[SkillRouter] Failed to validate skill instructions, skipping registration: "
                f"{skill.skill_id} ({exc})"
            )
            return False

    def _persist_instruction_fields(self, skill: Skill) -> None:
        """Write the special_instructions repaired during the registration stage back to the JSON sidecar."""
        if self.skills_dir is None:
            return
        sidecar_path = self.skills_dir / f"{skill.skill_id}.json"
        if not sidecar_path.exists():
            return
        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            data["special_instructions"] = skill.special_instructions
            data["instructions_source"] = getattr(skill, "instructions_source", "template")
            sidecar_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                f"[SkillRouter] Failed to write back repaired skill instructions: "
                f"{skill.skill_id} ({exc})"
            )

    def mark_injected(self, skill_id: str):
        """Called when a Skill is injected into the Agent; only increments usage_count."""
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        skill.usage_count += 1

    def record_outcome(self, skill_id: str, is_correct: bool):
        """Record the downstream win/loss of an injected Skill.

        This is the "truly meaningful success rate": it reflects whether the agent answered correctly after the Skill was injected.
        """
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        if is_correct:
            skill.downstream_success += 1
        else:
            skill.downstream_failure += 1
        total = skill.downstream_success + skill.downstream_failure
        skill.downstream_success_rate = (
            skill.downstream_success / total if total > 0 else 0.0
        )

        # Persist the latest downstream metrics back to the sidecar (as fault-tolerant as possible, without affecting the main flow)
        if self.skills_dir is not None:
            try:
                sidecar_path = self.skills_dir / f"{skill_id}.json"
                if sidecar_path.exists():
                    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    data["usage_count"] = skill.usage_count
                    data["downstream_success"] = skill.downstream_success
                    data["downstream_failure"] = skill.downstream_failure
                    data["downstream_success_rate"] = skill.downstream_success_rate
                    sidecar_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as exc:
                logger.warning(
                    f"[SkillRouter] Failed to write back downstream metrics: {skill_id} ({exc})"
                )

        # P0: after every downstream metric update, immediately check whether it should be blacklisted
        reason = self._auto_ban_reason(skill)
        if reason:
            self.blacklist_skill(skill_id, reason=reason)

    def load_prior_skills(self) -> List[Skill]:
        """Scan JSON sidecars from skills_dir, deserialize historical skills, and register them.

        Returns the list of successfully loaded Skills for the orchestrator to sync to SkillPromoter.
        Fault tolerance: a single sidecar parse failure does not affect other files.
        """
        loaded: List[Skill] = []
        if not self.skills_dir or not self.skills_dir.exists():
            logger.info(f"[SkillRouter] skills_dir does not exist, skipping historical skill loading: {self.skills_dir}")
            return loaded

        sidecar_files = sorted(self.skills_dir.glob("*.json"))
        if not sidecar_files:
            logger.info(f"[SkillRouter] No historical skill sidecar found in skills_dir: {self.skills_dir}")
            return loaded

        for sidecar_path in sidecar_files:
            # Skip the blacklist persistence file itself
            if sidecar_path.name == _BLACKLIST_FILENAME:
                continue
            try:
                data = json.loads(sidecar_path.read_text(encoding="utf-8"))
                # Skip blacklisted skills directly: neither register nor add to the loaded list
                sid = data.get("skill_id", "")
                if sid and sid in self._blacklist:
                    logger.info(
                        f"[SkillRouter] Skipping loading of blacklisted skill: {sid}"
                    )
                    continue
                # P2 compatibility: the question_type of a historical skill may be a comma-joined combined
                # key (e.g. "cross_modal_reasoning,multi_detail_reasoning"),
                # but the new clustering logic only recognizes single tags. Here we downgrade according to
                # the primary-tag priority of infer_question_type, ensuring old skills can still hit a single-tag qt under the new routing.
                raw_qt = data.get("question_type", "general") or "general"
                if "," in raw_qt:
                    try:
                        from .learning_capture import _pick_primary_tag
                        parts = [p.strip() for p in raw_qt.split(",") if p.strip()]
                        normalized_qt = _pick_primary_tag(parts) if parts else "general"
                    except Exception:
                        normalized_qt = raw_qt.split(",", 1)[0].strip() or "general"
                    data["question_type"] = normalized_qt
                # Negative-sample clustering (calibration / search_fail) no longer treats query-prefix
                # statistics as a dispatch suggestion: even if video_drilldown / event_first remain in the historical JSON,
                # they are normalized to an empty string during deserialization, consistent with the new constraint of _create_skill.
                skill_id_val = data.get("skill_id", "")
                negative_prefixes = ("SKILL-calibration__", "SKILL-search_fail__")
                raw_strategy = data.get("recommended_search_strategy", "event_first")
                if skill_id_val.startswith(negative_prefixes):
                    norm_strategy = ""
                else:
                    norm_strategy = raw_strategy
                # Compatible with old sidecars: historical versions wrote 5 into the file as the default display value,
                # which was not the actually learned recommended number of rounds. After the new default became 10, migrate the old default value to 10.
                raw_max_rounds = data.get("recommended_max_rounds", 10)
                norm_max_rounds = 10 if raw_max_rounds == 5 else raw_max_rounds

                skill = Skill(
                    skill_id=data["skill_id"],
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    trigger_conditions=list(data.get("trigger_conditions", [])),
                    question_type=data.get("question_type", "general"),
                    recommended_search_strategy=norm_strategy,
                    recommended_prompt_template=data.get("recommended_prompt_template"),
                    recommended_max_rounds=norm_max_rounds,
                    special_instructions=data.get("special_instructions", ""),
                    instructions_source=data.get("instructions_source", "template"),
                    version=data.get("version", 1),
                    created_from=list(data.get("created_from", [])),
                    success_count=data.get("success_count", 0),
                    failure_count=data.get("failure_count", 0),
                    success_rate=data.get("success_rate", 0.0),
                    usage_count=data.get("usage_count", 0),
                    downstream_success=data.get("downstream_success", 0),
                    downstream_failure=data.get("downstream_failure", 0),
                    downstream_success_rate=data.get("downstream_success_rate", 0.0),
                    last_updated=data.get("last_updated", ""),
                    examples=list(data.get("examples", [])),
                    instruction_slots=[
                        InstructionSlot(
                            slot_id=s.get("slot_id", ""),
                            pattern_keywords=list(s.get("pattern_keywords", [])),
                            context_keywords=list(s.get("context_keywords", [])),
                            instructions=s.get("instructions", ""),
                            hit_count=s.get("hit_count", 0),
                            success_count=s.get("success_count", 0),
                        )
                        for s in data.get("instruction_slots", [])
                    ],
                )
                self.register_skill(skill)
                if skill.skill_id in self.skills:
                    loaded.append(skill)
            except Exception as exc:
                logger.warning(
                    f"[SkillRouter] Failed to parse historical skill sidecar: {sidecar_path.name} ({exc})"
                )

        logger.info(
            f"[SkillRouter] Loaded {len(loaded)} historical skills from {self.skills_dir}"
        )
        return loaded

    def route(self, question: TemporalQuestion) -> Optional[Skill]:
        """Match the most suitable skill for the given question.

        Scoring dimensions:
          1. question_type exact match (weight 0.3)
          2. trigger_conditions / keyword rule match (weight 0.4): the main signal
          3. effective_rate (weight 0.3): based on downstream win rate + prior,
             the fewer the samples the closer to the baseline, preventing a just-promoted skill from dominating with success_rate=1.0.

        Hard constraint (to avoid a "template-only skill" hitting indiscriminately):
          - At least one of the following must be satisfied to qualify:
              a) question_type exact hit (skill_qt != "general" and == inferred_qt)
              b) rule_score > 0 (hits at least one non-question_type trigger word)
          The approach of relying on "a general skill always defaults to 0.3" is blocked by this constraint.

        Cold-start protection:
          - If a skill's downstream sample count >= min_downstream_samples
            and its downstream win rate < baseline - disable_below_baseline_margin,
            it is automatically removed from the candidates (verified negative effect).
        """
        if not self.skills:
            return None

        self.last_slot_route = None
        candidates = []
        q_text = question.question or ""
        q_lower = q_text.lower()

        # Infer question_type from the question (consistent with LearningCapture's inference)
        inferred_qt = self._infer_question_type(question)

        for skill_id, skill in self.skills.items():
            # Cold-start protection: skip verified-negative-effect skills directly
            if skill_id in self._blacklist or self._is_verified_bad(skill):
                continue

            skill_qt = getattr(skill, "question_type", "") or "general"

            # 1) question_type exact match
            qt_exact_hit = skill_qt != "general" and skill_qt == inferred_qt
            if qt_exact_hit:
                qt_score = 1.0
            elif skill_qt == "general":
                qt_score = 0.3  # a general skill neither gains nor loses points
            else:
                qt_score = 0.0

            # 2) rule matching
            rule_score = self._rule_match(q_lower, skill)

            # Hard constraint: must have an exact qt hit or rule_score > 0, otherwise skip directly,
            # to avoid a template-only skill sneaking into routing via the default score
            if not qt_exact_hit and rule_score <= 0:
                continue

            # 3) effective_rate: downstream takes priority; use baseline as the prior during cold start
            effective_rate = self._effective_rate(skill)

            # Combined score (rule weight 0.3 -> 0.4, qt 0.5 -> 0.3, eff 0.2 -> 0.3)
            combined_score = qt_score * 0.3 + rule_score * 0.4 + effective_rate * 0.3

            if combined_score > self.route_threshold:
                candidates.append((skill, combined_score, effective_rate))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_skill, best_score, best_rate = candidates[0]
        logger.debug(
            f"Skill routing: {best_skill.name} (score={best_score:.2f}, "
            f"effective={best_rate:.2f}, qt={inferred_qt})"
        )
        return best_skill
    def _effective_rate(self, skill: Skill) -> float:
        """Effective win-rate estimate with a prior.

        - When samples are insufficient: smooth toward baseline_accuracy (typical pseudo-count k = min_downstream_samples)
        - When samples are sufficient: use downstream_success_rate directly
        - When neither is available: fall back to success_rate (the source correctness rate)
        """
        down_total = skill.downstream_success + skill.downstream_failure
        if down_total == 0:
            # Fully cold start: lean toward the baseline to prevent a new skill from seizing routing with 1.0
            return self.baseline_accuracy if self.baseline_accuracy > 0 else 0.3
        k = max(1, self.min_downstream_samples)
        prior = self.baseline_accuracy if self.baseline_accuracy > 0 else 0.3
        # Bayesian smoothing: (wins + k*prior) / (total + k)
        return (skill.downstream_success + k * prior) / (down_total + k)

    def _is_verified_bad(self, skill: Skill) -> bool:
        """Whether there is enough evidence that injecting this skill drags down performance."""
        down_total = skill.downstream_success + skill.downstream_failure
        if down_total < self.min_downstream_samples:
            return False
        if self.baseline_accuracy <= 0:
            return False
        return (
            skill.downstream_success_rate
            < self.baseline_accuracy - self.disable_below_baseline_margin
        )

    @staticmethod
    def _infer_question_type(question: TemporalQuestion) -> str:
        """Reuse the inference logic of learning_capture.infer_question_type (imported lazily here to avoid circular references)."""
        try:
            from .learning_capture import infer_question_type
            return infer_question_type(question)
        except Exception:
            return "general"

    def _rule_match(self, question_lower: str, skill: Skill) -> float:
        """Keyword-based rule matching.

        Supports four trigger condition formats:
          1. `keyword:<cognitive_pattern_name>`: cognitive pattern name match (e.g. spatial_location)
          2. `keyword:<word_or_phrase>`: word-boundary match (supports bigram phrases)
          3. `question_type=<qt>`: for display only, does not participate in keyword scoring (qt is handled by the upper layer)
          4. Natural sentence (legacy format): counts as a match if any token hits after tokenization
        """
        import re as _re

        if not skill.trigger_conditions:
            return 0.0

        # Lazily import cognitive pattern matching (to avoid circular references)
        try:
            from .cognitive_patterns import (
                match_cognitive_patterns,
                COGNITIVE_PATTERN_NAMES,
                is_low_value_keyword,
                normalize_cognitive_keyword,
            )
            question_patterns = match_cognitive_patterns(question_lower)
        except Exception:
            question_patterns = []
            COGNITIVE_PATTERN_NAMES = {
                "counting", "temporal_order", "causal", "person_identity",
                "person_relationship", "spatial_location", "comparison",
                "intent_plan", "yes_no_verification", "manner_method",
                "object_identification",
            }
            is_low_value_keyword = lambda _: False
            normalize_cognitive_keyword = lambda x: x

        matched, total = 0, 0
        for cond in skill.trigger_conditions:
            cond_l = cond.strip().lower()
            # Skip the question_type meta-condition (handled by the outer route layer)
            if cond_l.startswith("question_type="):
                continue
            total += 1
            if cond_l.startswith("keyword:"):
                kw = cond_l.split(":", 1)[1].strip()
                if not kw:
                    continue
                kw = normalize_cognitive_keyword(kw)
                if is_low_value_keyword(kw):
                    logger.debug(
                        f"[SkillRouter] Ignoring low-value trigger keyword: {cond_l!r} "
                        f"for skill {skill.skill_id}"
                    )
                    continue

                # First check whether it is a cognitive pattern name (e.g. spatial_location, person_identity, counting)
                if kw in COGNITIVE_PATTERN_NAMES:
                    # Cognitive pattern name match: check whether the question belongs to that cognitive pattern
                    if kw in question_patterns:
                        matched += 1
                else:
                    # Word-boundary match: avoid "color" matching "colorado"
                    # also matches bigrams (containing spaces) correctly
                    pattern = r'\b' + _re.escape(kw) + r'\b'
                    if _re.search(pattern, question_lower):
                        matched += 1
            else:
                # Legacy-format fallback: counts as a match if any token hits
                keywords = [w.strip().lower() for w in cond_l.split() if len(w.strip()) > 2]
                if any(kw in question_lower for kw in keywords):
                    matched += 1
        if total == 0:
            return 0.0
        return matched / total

    def route_instruction_slot(self, question: TemporalQuestion,
                               skill: Skill) -> Optional[str]:
        """Within an already-matched Skill, route to a specific instruction slot by cognitive pattern.

        If the skill has instruction_slots, try to match the question's cognitive pattern to the corresponding slot,
        and return only that slot's instructions. If no slot matches, return None (the caller falls back
        to skill.special_instructions).

        Args:
            question: the current question
            skill: the skill already matched by route()

        Returns:
            The instructions string of the matched slot, or None
        """
        if not skill.instruction_slots:
            self.last_slot_route = {
                "skill_id": skill.skill_id,
                "matched": False,
                "reason": "no_slots",
            }
            return None

        q_text = (question.question or "").lower()

        # Match using the cognitive pattern dictionary
        from .cognitive_patterns import (
            match_cognitive_patterns,
            is_low_value_keyword,
            normalize_cognitive_keyword,
        )
        matched_patterns = match_cognitive_patterns(q_text)

        if not matched_patterns:
            self.last_slot_route = {
                "skill_id": skill.skill_id,
                "matched": False,
                "reason": "no_question_pattern",
            }
            return None

        # Find the matching slot among the skill's slots
        import re as _re
        best_slot = None
        best_score = 0.0

        for slot in skill.instruction_slots:
            slot_score = 0.0

            # 1) pattern_keywords match (high weight)
            if slot.pattern_keywords:
                pattern_matched = 0
                for pk in slot.pattern_keywords:
                    pk_norm = normalize_cognitive_keyword(pk)
                    if not pk_norm or is_low_value_keyword(pk_norm):
                        continue
                    # pk may be a pattern name such as "counting", "spatial_location"
                    if pk_norm in matched_patterns:
                        pattern_matched += 1
                    else:
                        # it may also be a specific indicator phrase
                        pattern = r'\b' + _re.escape(pk_norm) + r'\b'
                        if _re.search(pattern, q_text):
                            pattern_matched += 1
                if slot.pattern_keywords:
                    slot_score += 0.6 * (pattern_matched / len(slot.pattern_keywords))

            # 2) context_keywords match (low weight)
            if slot.context_keywords:
                ctx_matched = 0
                for ck in slot.context_keywords:
                    ck_norm = normalize_cognitive_keyword(ck)
                    if not ck_norm or is_low_value_keyword(ck_norm):
                        continue
                    pattern = r'\b' + _re.escape(ck_norm) + r'\b'
                    if _re.search(pattern, q_text):
                        ctx_matched += 1
                slot_score += 0.4 * (ctx_matched / len(slot.context_keywords))

            if slot_score > best_score:
                best_score = slot_score
                best_slot = slot

        # A match score of at least 0.3 is required to be considered valid
        if best_slot and best_score >= 0.3 and best_slot.instructions:
            best_slot.hit_count += 1
            self.last_slot_route = {
                "skill_id": skill.skill_id,
                "matched": True,
                "slot_id": best_slot.slot_id,
                "score": best_score,
                "matched_patterns": list(matched_patterns),
            }
            logger.debug(
                f"[SkillRouter] Sub-skill slot match: {best_slot.slot_id} "
                f"(score={best_score:.2f}) for question: {q_text[:50]}..."
            )
            return best_slot.instructions

        self.last_slot_route = {
            "skill_id": skill.skill_id,
            "matched": False,
            "reason": "score_below_threshold",
            "score": best_score,
            "matched_patterns": list(matched_patterns),
        }
        return None
