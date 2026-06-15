"""
SkillPromoter: automatically distills reusable Skills from accumulated Learnings.
"""
import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from .learning_types import Learning, LearningType
from .skill_types import Skill
from .cognitive_patterns import (
    match_cognitive_patterns,
    extract_pattern_keywords,
    extract_context_keywords,
    generate_hybrid_keywords,
    TRANSFERABLE_CONTEXT_PATTERNS,
    COGNITIVE_PATTERN_NAMES,
    FUNCTION_WORDS,
    is_low_value_keyword,
    normalize_cognitive_keyword,
)

logger = logging.getLogger(__name__)

# ---- json-repair: soft dependency, used to repair truncated / trailing-comma / unclosed JSON from LLM output ----
try:
    from json_repair import repair_json as _repair_json  # type: ignore
    _HAS_JSON_REPAIR = True
except Exception:  # pragma: no cover - fall back to standard JSON parsing when not installed
    _repair_json = None
    _HAS_JSON_REPAIR = False


class SkillPromoter:
    """Automatically distills reusable Skills from accumulated Learnings.

    Inspired by OpenClaw's Promote mechanism:
    when enough learnings of the same kind accumulate, automatically promote them into a structured Skill.

    Args:
        skills_dir: directory where skill files are stored
        promote_threshold: trigger promotion when learnings of the same kind reach N items
        use_llm_instructions: whether to call the LLM to generate special_instructions (P1 module)
        instructions_llm_model: name of the LLM model used to generate instructions
    """

    def __init__(self, skills_dir: str, promote_threshold: int = 3,
                 use_llm_instructions: bool = False,
                 instructions_llm_model: str = "gemini-2.5-flash",
                 visual_layer_enabled: bool = False,
                 agent_mode: str = "multi_round_search"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.promote_threshold = promote_threshold
        self.use_llm_instructions = use_llm_instructions
        self.instructions_llm_model = instructions_llm_model
        self.visual_layer_enabled = bool(visual_layer_enabled)
        self.agent_mode = agent_mode
        # The control_api_harness mode does not support the VIDEO:/NEIGHBOR:/KEYFRAME: prefixes
        self._is_control_mode = (agent_mode == "control_api_harness")
        self._existing_skills: Dict[str, Skill] = {}
        # Plan C: temporary storage for LLM-generated trigger_keywords (written by _call_instruction_llm)
        self._last_llm_keywords: List[str] = []
        # P0: blacklist query callback (injected by the orchestrator as SkillRouter.is_blacklisted)
        # If not set, defaults to always returning False, preserving the old behavior
        self._is_blacklisted = lambda sid: False

    def set_blacklist_checker(self, checker):
        """Inject a blacklist query callback from the orchestrator (usually SkillRouter.is_blacklisted)."""
        if callable(checker):
            self._is_blacklisted = checker

    def load_existing_skills(self, skills: List[Skill]):
        """Register already-loaded historical skills into the internal map to avoid recreating skills for the same cluster.

        The key construction rule is consistent with `_cluster_key_for`:
            cluster_key = f"{learning_type}__{question_type}"
        If a historical skill has no question_type (older version), it defaults to 'general'.
        """
        for s in skills:
            lt = self._extract_learning_type_from_skill_id(s.skill_id)
            qt = getattr(s, "question_type", "") or "general"
            if lt:
                cluster_key = f"{lt}__{qt}"
                self._existing_skills[cluster_key] = s

    @staticmethod
    def _extract_learning_type_from_skill_id(skill_id: str) -> str:
        """Recover learning_type from SKILL-<learning_type>__<question_type>-<date>.

        Compatible with the old format SKILL-<learning_type>-<date> (in which case question_type is treated as general).
        """
        if not skill_id.startswith("SKILL-"):
            return ""
        body = skill_id[len("SKILL-"):]
        # Remove the trailing date (YYYYMMDD)
        parts = body.rsplit("-", 1)
        if len(parts) == 2 and parts[-1].isdigit():
            body = parts[0]
        # Compatible with the __ separator
        if "__" in body:
            return body.split("__", 1)[0]
        return body

    def maybe_promote(self, learnings: List[Learning]) -> List[Skill]:
        """Check whether any learnings can be promoted into skills."""
        # Cluster by error type
        clusters = self._cluster_learnings(learnings)

        new_skills = []
        for cluster_key, cluster_learnings in clusters.items():
            if len(cluster_learnings) < self.promote_threshold:
                continue

            # Check whether a corresponding Skill already exists
            existing = self._existing_skills.get(cluster_key)
            # P0: if an existing skill is blacklisted, neither recreate nor update it, just skip
            if existing is not None and self._is_blacklisted(existing.skill_id):
                logger.info(
                    f"[SkillPromoter] Skipping promote for blacklisted skill: "
                    f"{existing.skill_id}"
                )
                # Immediately mark this batch of learnings as promoted to avoid reclustering them next time
                for l in cluster_learnings:
                    l.promoted = True
                continue

            if existing:
                updated = self._update_skill(existing, cluster_learnings)
                if updated is not None:
                    new_skills.append(updated)
                else:
                    logger.info(
                        f"[SkillPromoter] Skill update failed (invalid LLM generation), skipping: {cluster_key}"
                    )
            else:
                skill = self._create_skill(cluster_key, cluster_learnings)
                if skill is None:
                    # LLM generation failed, abandon this promote and retry on the next batch
                    logger.info(
                        f"[SkillPromoter] Skill creation failed (invalid LLM generation), skipping: {cluster_key}"
                    )
                elif self._is_blacklisted(skill.skill_id):
                    logger.info(
                        f"[SkillPromoter] Newly created skill is already blacklisted, discarding: "
                        f"{skill.skill_id}"
                    )
                    # Remove from _existing_skills so it will not be revived in the update branch next time
                    self._existing_skills.pop(cluster_key, None)
                else:
                    new_skills.append(skill)

            # Mark as promoted
            for l in cluster_learnings:
                l.promoted = True

        return new_skills

    # Planner/Ledger-specific learnings are used directly by WisdomDistiller.generate_decompose_guidance,
    # and are not promoted into ordinary retrieval skills, to avoid mistakenly injecting "decomposition decision experience" as a search strategy.
    _PLANNER_ONLY_LEARNING_TYPES = {
        "subtask_stall",
        "decompose_win",
        "decompose_fail",
        "decompose_miss",
    }

    def _cluster_learnings(self, learnings: List[Learning]) -> Dict[str, List[Learning]]:
        """Cluster by (learning_type, question_type_tag).

        P2 improvement: if a Learning has multiple question_type_tags, it falls into
        the bucket corresponding to each single tag. This way a question of type "Cross-Modal Reasoning, Multi-Detail
        Reasoning" contributes to both the calibration__cross_modal_reasoning and
        calibration__multi_detail_reasoning buckets, no longer requiring a hard-coded
        "rare first" priority, and avoiding sparse buckets caused by combinatorial explosion.

        Note: the same Learning may appear in multiple clusters, but the `promoted` flag
        is set uniformly only in `maybe_promote` and does not affect the clustering stage.
        """
        clusters: Dict[str, List[Learning]] = {}
        for l in learnings:
            if l.promoted:
                continue
            if l.learning_type.value in self._PLANNER_ONLY_LEARNING_TYPES:
                continue
            keys = self._cluster_keys_for(l)
            for key in keys:
                if key not in clusters:
                    clusters[key] = []
                clusters[key].append(l)
        return clusters

    @staticmethod
    def _cluster_keys_for(l: Learning) -> List[str]:
        """Return all cluster keys that a Learning should fall into.

        If question_type_tags is non-empty, each tag produces one key;
        otherwise fall back to question_type (the primary tag) producing a single key.
        """
        tags = getattr(l, "question_type_tags", None) or []
        if not tags:
            qt = getattr(l, "question_type", "") or "general"
            return [f"{l.learning_type.value}__{qt}"]
        return [f"{l.learning_type.value}__{tag}" for tag in tags]

    @staticmethod
    def _cluster_key_for(l: Learning) -> str:
        """Backward compatible: return the cluster key of the primary tag."""
        qt = getattr(l, "question_type", "") or "general"
        return f"{l.learning_type.value}__{qt}"

    # Error-type learning_type: the query-prefix distribution of these clusters represents the "failure path",
    # so their reference value is negative. Copying them only copies erroneous behavior. For such skills,
    # we fix recommended_search_strategy to an empty string. Later rendering/injection
    # will skip this field, avoiding presenting a failure strategy as a recommendation.
    _NEGATIVE_LEARNING_TYPES = {"calibration", "search_fail"}

    def _create_skill(self, cluster_key: str,
                      learnings: List[Learning]) -> Skill:
        """Create a new skill from a group of learnings."""
        # Recover learning_type / question_type from cluster_key
        if "__" in cluster_key:
            lt_val, qt_val = cluster_key.split("__", 1)
        else:
            lt_val, qt_val = cluster_key, "general"

        # Count the most frequently used strategy (only meaningful for positive clusters; disabled for error clusters)
        if lt_val in self._NEGATIVE_LEARNING_TYPES:
            best_strategy = ""  # Disabled: avoid treating the failure path as a recommendation
        else:
            strategy_counter = Counter(
                l.search_strategy_used for l in learnings if l.search_strategy_used
            )
            best_strategy = (
                strategy_counter.most_common(1)[0][0]
                if strategy_counter else "event_first"
            )

        # Count the success rate
        successes = sum(1 for l in learnings if l.is_correct)
        total = len(learnings)

        # Build more precise trigger conditions: include question_type + high-frequency keywords extracted from learnings (rule-based fallback)
        trigger_conditions = [f"question_type={qt_val}"]
        rule_keywords = self._extract_trigger_keywords(learnings)
        trigger_conditions.extend(rule_keywords)

        skill = Skill(
            skill_id=f"SKILL-{lt_val}__{qt_val}-{datetime.now().strftime('%Y%m%d')}",
            name=f"{qt_val} x {lt_val} handling skill",
            description=f"Skill distilled from {total} learnings of type [{lt_val}] + question_type={qt_val}",
            trigger_conditions=trigger_conditions,
            question_type=qt_val,
            recommended_search_strategy=best_strategy,
            version=1,
            created_from=[l.learning_id for l in learnings],
            success_count=successes,
            failure_count=total - successes,
            success_rate=successes / total if total > 0 else 0.0,
            last_updated=datetime.now().isoformat(),
        )

        # Add cases as few-shot examples:
        # - Positive clusters (hard_win/search_win/...) take success cases
        # - Negative clusters (calibration/search_fail/error) take failure cases, giving the agent truly referable "what went wrong" evidence
        if lt_val in self._NEGATIVE_LEARNING_TYPES or lt_val == "error":
            for l in learnings:
                if not l.is_correct and len(skill.examples) < 3:
                    skill.examples.append({
                        "question": l.question,
                        "search_queries": l.search_queries_used,
                        "answer": l.agent_answer,
                        "ground_truth": getattr(l, "ground_truth", ""),                        "failure_note": (
                            f"agent answered with confidence={round(l.confidence, 2)} "
                            f"after {l.num_rounds} rounds, but it was wrong."
                        ),
                    })
        else:
            for l in learnings:
                if l.is_correct and len(skill.examples) < 3:
                    skill.examples.append({
                        "question": l.question,
                        "search_queries": l.search_queries_used,
                        "answer": l.agent_answer,
                    })

        # P1: generate special_instructions with the LLM on demand
        if self.use_llm_instructions:
            try:
                skill.special_instructions = self._generate_special_instructions(
                    lt_val, qt_val, learnings
                )
                skill.special_instructions = self._normalize_special_instructions(
                    skill.special_instructions, lt_val, self.visual_layer_enabled
                )
                skill.instructions_source = "llm"
                # Plan C: prefer LLM-generated trigger_keywords (after posterior validation)
                llm_keywords = self._validate_llm_keywords(
                    self._last_llm_keywords, learnings
                )
                if llm_keywords:
                    # Replace the rule-extracted keywords, keeping the question_type condition
                    skill.trigger_conditions = [f"question_type={qt_val}"]
                    skill.trigger_conditions.extend(
                        [f"keyword:{kw}" for kw in llm_keywords]
                    )
            except Exception as exc:
                logger.warning(
                    f"[SkillPromoter] LLM failed to generate special_instructions, abandoning this promote: {exc}"
                )
                # Do not degrade to a template; just abandon this promote
                # and retry after the next batch accumulates more learnings
                return None
        else:
            # When use_llm_instructions=False, do not create a template skill either
            # Template instructions heavily overlap with the base prompt; injecting them only increases token cost without adding information
            logger.info(
                f"[SkillPromoter] use_llm_instructions=False, skipping skill creation: {cluster_key}"
            )
            return None

        self._existing_skills[cluster_key] = skill

        # Generate instruction_slots: group learnings by cognitive pattern and generate dedicated instructions for each group
        self._generate_instruction_slots(skill, learnings)

        self._write_skill_markdown(skill)
        return skill

    def _generate_instruction_slots(self, skill: Skill,
                                    learnings: List[Learning]) -> None:
        """Group learnings by cognitive pattern and generate a dedicated instruction slot for each group.

        Slots are only generated when multiple different cognitive patterns exist among the learnings.
        If all learnings belong to the same pattern, no slots are generated (the skill-level generic instructions are used).

        This implements the requirement of "routing to different instructions by keyword within the same question_type".
        """
        from .skill_types import InstructionSlot

        # Group by cognitive pattern
        pattern_groups: Dict[str, List[Learning]] = {}
        for l in learnings:
            q = l.question or ""
            patterns = match_cognitive_patterns(q)
            if patterns:
                # Take the first matching pattern as the primary pattern
                primary_pattern = patterns[0]
            else:
                primary_pattern = "_other"
            if primary_pattern not in pattern_groups:
                pattern_groups[primary_pattern] = []
            pattern_groups[primary_pattern].append(l)

        # Only generate slots when 2+ different patterns exist and each group has at least 2 learnings
        valid_groups = {
            k: v for k, v in pattern_groups.items()
            if len(v) >= 2 and k != "_other"
        }

        if len(valid_groups) < 2:
            # Not diverse enough, do not generate slots
            return

        slots = []
        for pattern_name, group_learnings in valid_groups.items():
            # Extract keywords for each pattern group
            group_questions = [l.question or "" for l in group_learnings]
            pattern_kws = extract_pattern_keywords(group_questions, topk=2)
            context_kws = extract_context_keywords(group_questions, topk=2)

            # Extract instructions related to this pattern from the skill's special_instructions
            # or generate an instruction summary dedicated to this pattern
            slot_instructions = self._extract_slot_instructions(
                skill.special_instructions, pattern_name, group_learnings
            )

            pattern_keywords = [pattern_name]
            context_keywords = [
                kw for kw in extract_context_keywords(group_questions, topk=2)
                if not is_low_value_keyword(kw)
            ]

            if slot_instructions:
                slot = InstructionSlot(
                    slot_id=pattern_name,
                    pattern_keywords=pattern_keywords,
                    context_keywords=context_keywords,
                    instructions=slot_instructions,
                )
                slots.append(slot)

        if len(slots) >= 2:
            skill.instruction_slots = slots
            logger.info(
                f"[SkillPromoter] Generated {len(slots)} instruction slots for {skill.skill_id}: "
                f"{[s.slot_id for s in slots]}"
            )

    def _extract_slot_instructions(self, base_instructions: str,
                                   pattern_name: str,
                                   group_learnings: List[Learning]) -> str:
        """Extract/generate dedicated instructions for a specific cognitive pattern group.

        Strategy:
          1. If base_instructions contains multiple instructions (separated by "- "),
             try to select the most relevant instruction based on the semantics of pattern_name
          2. If they cannot be separated, generate a short pattern-specific hint based on the characteristics of this group of learnings
        """
        if not base_instructions:
            return ""

        # Try to split instructions by "- "
        lines = [l.strip() for l in base_instructions.split("\n") if l.strip().startswith("- ")]

        if len(lines) <= 1:
            # Only one instruction, cannot split, return empty to fall back to the skill level
            return ""

        # Select the most relevant instruction based on the semantic keywords of the cognitive pattern
        pattern_relevance_keywords = {
            "counting": ["count", "number", "times", "frequency", "repetitive", "cycle", "occurrence"],
            "temporal_order": ["sequence", "order", "before", "after", "first", "temporal", "chronological"],
            "causal": ["cause", "reason", "because", "result", "consequence", "lead"],
            "person_identity": ["identity", "character", "person", "who", "name", "distinguish"],
            "person_relationship": ["relationship", "attitude", "care", "treat", "feel", "opinion"],
            "spatial_location": ["location", "place", "position", "where", "put", "section", "layer"],
            "comparison": ["compare", "more", "better", "difference", "versus", "familiar"],
            "intent_plan": ["should", "plan", "intend", "goal", "purpose", "supposed"],
            "manner_method": ["method", "how", "technique", "approach", "step", "procedure"],
            "yes_no_verification": ["verify", "confirm", "check", "true", "false", "did"],
            "object_identification": ["identify", "what", "which", "type", "kind", "object"],
        }

        relevance_words = pattern_relevance_keywords.get(pattern_name, [])
        if not relevance_words:
            return ""

        # Compute the relevance of each instruction to this pattern
        scored_lines = []
        for line in lines:
            line_lower = line.lower()
            score = sum(1 for w in relevance_words if w in line_lower)
            scored_lines.append((line, score))

        # Select the highest-scoring instructions (must have at least 1 point)
        scored_lines.sort(key=lambda x: -x[1])
        relevant_lines = [line for line, score in scored_lines if score > 0]

        if relevant_lines:
            return "\n".join(relevant_lines)

        # If nothing is clearly relevant, return empty (fall back to the skill level)
        return ""

    @staticmethod
    def _extract_trigger_keywords(learnings: List[Learning], topk: int = 5) -> List[str]:
        """Extract highly discriminative trigger keywords from the question text of learnings.

        Uses a two-layer strategy:
          Layer 1 (cognitive pattern): match against a predefined cognitive-pattern dictionary, describing "what reasoning the question requires"
          Layer 2 (transferable context): extract scene-level vocabulary from the question text that is transferable across videos

        No longer uses a pure statistical TF-IDF method, avoiding generating entity words or overly broad words.
        """
        questions = [(l.question or "") for l in learnings]

        # Generate hybrid keywords using the cognitive-pattern dictionary
        hybrid_keywords = generate_hybrid_keywords(
            questions,
            max_pattern=min(3, topk),
            max_context=min(2, topk - min(3, topk)),
        )

        if hybrid_keywords:
            return hybrid_keywords[:topk]

        # If the cognitive-pattern dictionary has no hits at all (very rare), fall back to transferable context extraction
        context_kws = extract_context_keywords(questions, topk=topk)
        context_kws = [kw for kw in context_kws if not is_low_value_keyword(kw)]
        if context_kws:
            return [f"keyword:{kw}" for kw in context_kws]

        # Final fallback: empty list (let the LLM-generated keywords take over)
        return []
    def _update_skill(self, skill: Skill,
                      new_learnings: List[Learning]) -> Skill:
        """Update an existing skill with new learnings."""
        skill.version += 1
        skill.created_from.extend([l.learning_id for l in new_learnings])

        new_successes = sum(1 for l in new_learnings if l.is_correct)
        skill.success_count += new_successes
        skill.failure_count += len(new_learnings) - new_successes
        total = skill.success_count + skill.failure_count
        skill.success_rate = skill.success_count / total if total > 0 else 0.0

        # Update few-shot examples: negative clusters only append failure cases; positive clusters only append success cases.
        lt_val_existing = self._extract_learning_type_from_skill_id(skill.skill_id)
        is_negative_cluster = (
            lt_val_existing in self._NEGATIVE_LEARNING_TYPES
            or lt_val_existing == "error"
        )
        for l in new_learnings:
            if is_negative_cluster and not l.is_correct:
                skill.examples.append({
                    "question": l.question,
                    "search_queries": l.search_queries_used,
                    "answer": l.agent_answer,
                    "ground_truth": getattr(l, "ground_truth", ""),
                    "failure_note": (
                        f"agent answered with confidence={round(l.confidence, 2)} "
                        f"after {l.num_rounds} rounds, but it was wrong."
                    ),
                })
                if len(skill.examples) > 5:
                    skill.examples.pop(0)
            elif (not is_negative_cluster) and l.is_correct:
                skill.examples.append({
                    "question": l.question,
                    "search_queries": l.search_queries_used,
                    "answer": l.agent_answer,
                })
                if len(skill.examples) > 5:
                    skill.examples.pop(0)

        # P1: when there are new learnings, let the LLM decide whether the instructions need to be refined.
        # - No valid old instructions: generate from scratch.
        # - Valid old instructions exist: first decide whether the new signals reveal complementary knowledge not covered by the old instructions;
        #   if not, allow the LLM to return keep_current=true, keeping the current instructions and trigger conditions unchanged.
        if self.use_llm_instructions:
            previous_instructions = skill.special_instructions or ""
            previous_source = getattr(skill, "instructions_source", "template")
            attempted_refine = False
            self._last_refine_kept_current = False
            try:
                lt_val = self._extract_learning_type_from_skill_id(skill.skill_id)
                qt_val = getattr(skill, "question_type", "general") or "general"
                old_instructions = skill.special_instructions or ""
                is_template_or_empty = (
                    not old_instructions
                    or getattr(skill, "instructions_source", "template") == "template"
                    or old_instructions.startswith("[template]")
                    or not self.validate_special_instructions(
                        old_instructions, lt_val, self.visual_layer_enabled
                    )
                )
                if is_template_or_empty:
                    # Generate from scratch, avoiding refining on top of historical half-finished/invalid instructions.
                    skill.special_instructions = self._generate_special_instructions(
                        lt_val, qt_val, new_learnings
                    )
                else:
                    # Iteratively optimize based on old instructions + new cases. The LLM may also decide no rewrite is needed.
                    attempted_refine = True
                    skill.special_instructions = self._refine_special_instructions(
                        lt_val, qt_val, new_learnings, old_instructions
                    )
                skill.special_instructions = self._normalize_special_instructions(
                    skill.special_instructions, lt_val, self.visual_layer_enabled
                )
                skill.instructions_source = "llm"

                if getattr(self, "_last_refine_kept_current", False):
                    logger.info(
                        f"[SkillPromoter] Refine decided no skill instruction update is needed, keeping the current special_instructions: {skill.skill_id}"
                    )
                else:
                    # Only update trigger conditions with LLM keywords when generating from scratch or actually refining.
                    # When keep_current=true, keep the old trigger_conditions to avoid local noise in the new batch breaking the routing.
                    llm_keywords = self._validate_llm_keywords(
                        self._last_llm_keywords, new_learnings
                    )
                    if llm_keywords:
                        skill.trigger_conditions = [f"question_type={qt_val}"]
                        skill.trigger_conditions.extend(
                            [f"keyword:{kw}" for kw in llm_keywords]
                        )
            except Exception as exc:
                if attempted_refine and previous_instructions:
                    skill.special_instructions = previous_instructions
                    skill.instructions_source = previous_source
                    logger.warning(
                        f"[SkillPromoter] LLM instruction refine failed while updating the skill, rolled back to the old instructions: {exc}"
                    )
                else:
                    # Generating from scratch also failed; keep the old instructions (if any), do not degrade to a template
                    if previous_instructions and previous_source == "llm":
                        skill.special_instructions = previous_instructions
                        skill.instructions_source = previous_source
                        logger.warning(
                            f"[SkillPromoter] LLM instruction generation failed while updating the skill, keeping the old LLM instructions: {exc}"
                        )
                    else:
                        # The old instructions are also a template or empty; abandon the update and return None to indicate this update is invalid
                        logger.warning(
                            f"[SkillPromoter] LLM instruction generation failed while updating the skill and there are no valid old instructions, skipping the update: {exc}"
                        )
                        return None

        skill.last_updated = datetime.now().isoformat()
        self._write_skill_markdown(skill)
        return skill

    def _write_skill_markdown(self, skill: Skill):
        """Write the skill to a Markdown file."""
        filepath = self.skills_dir / f"{skill.skill_id}.md"

        learning_type = self._extract_learning_type_from_skill_id(skill.skill_id)
        is_negative_cluster = (
            learning_type in self._NEGATIVE_LEARNING_TYPES
            or learning_type == "error"
        )

        examples_text = ""
        for i, ex in enumerate(skill.examples):
            examples_text += f"\n#### Example {i+1}\n"
            examples_text += f"- **Question**: {ex.get('question', '')[:100]}\n"
            examples_text += f"- **Search queries**: {ex.get('search_queries', [])}\n"
            if is_negative_cluster:
                examples_text += f"- **Agent wrong answer**: {ex.get('answer', '')[:100]}\n"
                examples_text += f"- **Correct answer**: {ex.get('ground_truth', '')[:100]}\n"
                if ex.get("failure_note"):
                    examples_text += f"- **Failure note**: {ex.get('failure_note', '')[:160]}\n"
            else:
                examples_text += f"- **Correct answer**: {ex.get('answer', '')[:100]}\n"
        examples_header = "Failure cases" if is_negative_cluster else "Success cases"

        # Downstream metrics display
        down_total = skill.downstream_success + skill.downstream_failure
        if down_total > 0:
            downstream_line = (
                f"**Downstream win rate** (after use): "
                f"{skill.downstream_success_rate:.1%} "
                f"({skill.downstream_success}/{down_total}) | "
                f"**Injection count**: {skill.usage_count}"
            )
        else:
            downstream_line = (
                f"**Downstream win rate** (after use): not yet injected | "
                f"**Injection count**: {skill.usage_count}"
            )

        # Negative clusters (calibration / search_fail) do not display a concrete strategy. These strategies are
        # posterior statistics of the failure path and should not be issued as recommendations. Other cases display normally.
        if skill.recommended_search_strategy:
            strategy_display = f"`{skill.recommended_search_strategy}`"
        else:
            strategy_display = (
                "_N/A (guided by the if-then rules in special_instructions)_"
            )

        content = f"""# {skill.name}

> {skill.description}

**Version**: v{skill.version} | **Source signal quality** (cluster correctness rate): {skill.success_rate:.1%} ({skill.success_count}/{skill.success_count + skill.failure_count}) | **Last updated**: {skill.last_updated}

{downstream_line}

**Question Type**: `{getattr(skill, 'question_type', 'general')}`

**Instructions Source**: `{getattr(skill, 'instructions_source', 'template')}`

## Trigger conditions

{chr(10).join(f'- {c}' for c in skill.trigger_conditions)}

## Recommended strategy

- **Search strategy**: {strategy_display}
- **Max reasoning rounds**: {skill.recommended_max_rounds}

## Special instructions

```
{skill.special_instructions or '(none)'}
```

## {examples_header}
{examples_text}

## Evolution history

Derived from learnings: {', '.join(skill.created_from[-10:])}{'...' if len(skill.created_from) > 10 else ''}
"""
        filepath.write_text(content, encoding="utf-8")

        # Also write a JSON sidecar for precise deserialization across runs
        sidecar_path = filepath.with_suffix(".json")
        sidecar_data = {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "trigger_conditions": list(skill.trigger_conditions),
            "question_type": getattr(skill, "question_type", "general"),
            "recommended_search_strategy": skill.recommended_search_strategy,
            "recommended_prompt_template": skill.recommended_prompt_template,
            "recommended_max_rounds": skill.recommended_max_rounds,
            "special_instructions": skill.special_instructions,
            "instructions_source": getattr(
                skill, "instructions_source", "template"
            ),
            "version": skill.version,
            "created_from": list(skill.created_from),
            "success_count": skill.success_count,
            "failure_count": skill.failure_count,
            "success_rate": skill.success_rate,
            "usage_count": skill.usage_count,
            "downstream_success": skill.downstream_success,
            "downstream_failure": skill.downstream_failure,
            "downstream_success_rate": skill.downstream_success_rate,
            "last_updated": skill.last_updated,
            "examples": list(skill.examples),
            "instruction_slots": [
                {
                    "slot_id": s.slot_id,
                    "pattern_keywords": list(s.pattern_keywords),
                    "context_keywords": list(s.context_keywords),
                    "instructions": s.instructions,
                    "hit_count": s.hit_count,
                    "success_count": s.success_count,
                }
                for s in getattr(skill, "instruction_slots", [])
            ],
        }
        sidecar_path.write_text(
            json.dumps(sidecar_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Skill saved: {filepath} (+ {sidecar_path.name})")

    # ------------------------------------------------------------------
    # P1: special_instructions generation
    # ------------------------------------------------------------------
    @staticmethod
    def _template_instructions(learning_type: str, question_type: str,
                               visual_layer_enabled: bool = False,
                               is_control_mode: bool = False) -> str:
        """Templated fallback instructions based on learning_type / question_type.

        Serves as a safe fallback when LLM generation fails, the output is invalid, or use_llm_instructions=False.
        The template must keep 2-4 Markdown bullets, and each must be an executable
        guardrail/action, avoiding continuing to inject low-quality LLM summaries into the Agent.

        When is_control_mode=True, generate a template suitable for the control_api_harness mode,
        without the VIDEO:/NEIGHBOR:/KEYFRAME: prefixes (this mode only supports plain text queries).
        """
        if is_control_mode:
            return SkillPromoter._template_instructions_control(
                learning_type, question_type
            )

        visual_action = "`KEYFRAME:<query>` or `VIDEO:<event_id>:<query>`" if visual_layer_enabled else "`VIDEO:<event_id>:<query>`"
        lt_templates = {
            "calibration": (
                "- Treat this as a calibration guardrail: do not answer from indirect evidence; lower confidence unless retrieved text directly names the requested entity, count, attribute, or location.\n"
                "- Before finalizing, run a targeted `VIDEO:<event_id>:<query>` or `NEIGHBOR:<prev|next>:<query>` check when the evidence is scene-level or ambiguous.\n"
                "- If verification still lacks direct support, state uncertainty rather than inferring from the closest matching event."
            ),
            "search_fail": (
                "- Treat repeated empty or low-detail retrievals as a search-failure guardrail; verify with a different retrieval mode instead of paraphrasing the same `event_first` query.\n"
                "- Switch to `VIDEO:<event_id>:<query>` for the most relevant event when summaries lack object, action, count, or spatial details.\n"
                "- Use `NEIGHBOR:<prev|next>:<query>` once when the answer depends on a preceding setup or following outcome."
            ),
            "error": (
                "- Before finalizing, verify the candidate answer against retrieved text and reject answers supported only by loose semantic similarity.\n"
                "- Prefer literal evidence from `event_first`, `VIDEO:<event_id>:<query>`, or `NEIGHBOR:<prev|next>:<query>` over inferred details."
            ),
            "search_win": (
                "- Start with a focused `event_first` keyword query using the main entity and requested attribute.\n"
                "- If the first result directly states the answer, answer without extra search; otherwise drill down with `VIDEO:<event_id>:<query>`."
            ),
            "hard_win": (
                "- Plan two complementary searches before answering: one broad `event_first` query and one targeted `VIDEO:<event_id>:<query>` drilldown.\n"
                "- Use `NEIGHBOR:<prev|next>:<query>` when temporal order, cause, or missing context cannot be resolved inside the focus event."
            ),
        }
        qt_hint = {
            "counting": "\n- For counting questions, enumerate items explicitly and recount before answering.",
            "temporal_order": "\n- For temporal-order questions, verify ordering with at least two `event_first` or `NEIGHBOR:<prev|next>:<query>` evidence points before answering.",
            "ocr_text": "\n- For OCR/text questions, quote literal retrieved text from `VIDEO:<event_id>:<query>` or `event_first` and avoid paraphrasing.",
            "person_identification": "\n- For person-identification questions, cross-check face_id references whenever available.",
            "location": "\n- For location questions, inspect scene/background cues with `VIDEO:<event_id>:<query>` or adjacent events when the focus event is ambiguous.",
            "cross_modal_reasoning": f"\n- For visible objects, spatial relations, counts, colors, text, or fine actions, verify with {visual_action} instead of relying only on event summaries.",
        }
        base = lt_templates.get(
            learning_type,
            "- Start with the retrieval mode recommended by this skill, then verify the answer against direct retrieved evidence.\n"
            "- If evidence is incomplete, run a targeted `VIDEO:<event_id>:<query>` or `NEIGHBOR:<prev|next>:<query>` before finalizing.",
        )
        return base + qt_hint.get(question_type, "")

    @staticmethod
    def _template_instructions_control(learning_type: str, question_type: str) -> str:
        """Template instructions dedicated to the control_api_harness mode.

        This mode only supports plain text queries and does not support the VIDEO:/NEIGHBOR:/KEYFRAME: prefixes.
        The instructions focus on: query strategy diversification, character ID mapping, evidence verification, and uncertainty expression.
        """
        lt_templates = {
            "calibration": (
                "- Treat this as a calibration guardrail: do not answer from indirect evidence; lower confidence unless retrieved text directly names the requested entity, count, attribute, or location.\n"
                "- When evidence is scene-level or ambiguous, search with more specific keywords targeting the exact entity or attribute before finalizing.\n"
                "- If verification still lacks direct support, state uncertainty rather than inferring from the closest matching event."
            ),
            "search_fail": (
                "- Treat repeated empty or low-detail retrievals as a search-failure guardrail; rephrase the query with completely different keywords instead of paraphrasing.\n"
                "- Try searching with character IDs (resolve via 'What is the character id of <name>') when name-based queries return empty.\n"
                "- Use shorter, more specific keyword queries focusing on the core entity and action."
            ),
            "error": (
                "- Before finalizing, verify the candidate answer against retrieved text and reject answers supported only by loose semantic similarity.\n"
                "- Prefer literal evidence from retrieved memories over inferred details; search with different keywords if evidence is ambiguous."
            ),
            "search_win": (
                "- Start with a focused keyword query using the main entity and requested attribute.\n"
                "- If the first result directly states the answer, answer without extra search; otherwise try a more specific query targeting the missing detail."
            ),
            "hard_win": (
                "- Plan two complementary searches before answering: one broad query and one targeted query with specific entity + attribute keywords.\n"
                "- When temporal order or cause is unclear, search for events before and after the target event separately."
            ),
        }
        qt_hint = {
            "counting": "\n- For counting questions, enumerate items explicitly and recount before answering.",
            "temporal_order": "\n- For temporal-order questions, search for multiple events and verify their ordering before answering.",
            "ocr_text": "\n- For OCR/text questions, quote literal retrieved text and avoid paraphrasing.",
            "person_identification": "\n- For person-identification questions, resolve character IDs to names and cross-check references.",
            "location": "\n- For location questions, search for scene/background cues with specific location keywords when the initial result is ambiguous.",
            "cross_modal_reasoning": "\n- For visible objects, spatial relations, counts, colors, or fine actions, search with specific descriptive keywords rather than abstract queries.",
        }
        base = lt_templates.get(
            learning_type,
            "- Start with a focused keyword query, then verify the answer against direct retrieved evidence.\n"
            "- If evidence is incomplete, try a completely different query with alternative keywords before finalizing.",
        )
        return base + qt_hint.get(question_type, "")

    @staticmethod
    def _query_mode_label(query: str) -> str:
        """Normalize a retrieval query into a retrieval mode that the LLM can learn from."""
        q = (query or "").strip().lower()
        if q.startswith("video:"):
            return "VIDEO"
        if q.startswith("neighbor:"):
            return "NEIGHBOR"
        if q.startswith("keyframe:"):
            return "KEYFRAME"
        return "event_first"

    @classmethod
    def _build_instruction_signal(cls, learning_type: str,
                                  question_type: str,
                                  learnings: List[Learning]) -> dict:
        """Build a mechanized signal for the LLM, preventing it from producing vague summaries of raw cases.

        Core idea: the LLM does not directly "write an experience summary", but instead generates
        executable Agent guardrails/actions from the structured signal. This reduces half-finished prompts and empty talk.

        P2 enhancement: attach failure_mechanism / success_mechanism to each case,
        and compute the dominant_mechanism at the top level, so the LLM writes dedicated instructions targeting the main mechanism
        rather than extracting the greatest common factor across mixed cases.
        """
        case_items = cls._build_case_items(learnings, max_cases=6)
        mode_counter: Counter = Counter()
        mechanism_counter: Counter = Counter()
        for item in case_items:
            mode_counter.update(item.get("retrieval_modes", []))
            mech = item.get("mechanism", "unknown")
            if mech and mech != "unknown":
                mechanism_counter[mech] += 1

        is_negative = learning_type in cls._NEGATIVE_LEARNING_TYPES
        objective = (
            "Convert repeated failure evidence into guardrails that prevent the same wrong answer path."
            if is_negative else
            "Convert repeated successful evidence into a reusable retrieval-and-verification action."
        )
        required_shape = (
            "Use if/when triggers, name one retrieval mode, and include a stop rule such as lower confidence, reject, or state uncertainty."
            if is_negative else
            "Use if/when triggers, name one retrieval mode, and include how to verify direct evidence before answering."
        )

        # Dominant mechanism: the LLM should write instructions targeting this mechanism
        dominant = mechanism_counter.most_common(1)[0][0] if mechanism_counter else "mixed"
        mechanism_dist = dict(mechanism_counter.most_common())

        instruction_target = (
            f"Generate a guardrail specifically targeting '{dominant}' failures in {question_type} questions."
            if is_negative else
            f"Generate a reusable action pattern based on the '{dominant}' success mechanism in {question_type} questions."
        )

        # Common-pattern induction: help the LLM discover the deeper commonality among failure cases
        cross_case_pattern = cls._infer_cross_case_pattern(learnings, is_negative)

        return {
            "learning_type": learning_type,
            "question_type": question_type,
            "cluster_polarity": "negative_guardrail" if is_negative else "positive_action",
            "objective": objective,
            "required_instruction_shape": required_shape,
            "instruction_target": instruction_target,
            "dominant_mechanism": dominant,
            "mechanism_distribution": mechanism_dist,
            "common_retrieval_modes": [m for m, _ in mode_counter.most_common()],
            "cross_case_pattern_analysis": cross_case_pattern,
            "cases": case_items,
        }

    @classmethod
    def _infer_cross_case_pattern(cls, learnings: List[Learning],
                                  is_negative: bool) -> dict:
        """Induce a common pattern from multiple cases to help the LLM discover deeper regularities.

        Does not rely on the LLM; pure rule-based analysis. Outputs a structured commonality signal so that the LLM can write
        specific instructions targeting a concrete pattern rather than generic retrieval advice.
        """
        questions = [(l.question or "") for l in learnings]
        answers_wrong = [(l.agent_answer or "") for l in learnings if not l.is_correct]
        answers_correct = [(l.ground_truth or "") for l in learnings if not l.is_correct]
        confidences = [l.confidence for l in learnings if not l.is_correct]

        pattern = {
            "num_cases": len(learnings),
            "num_failures": sum(1 for l in learnings if not l.is_correct),
        }

        # 1. Question commonality: detect common question words
        question_lower = [q.lower() for q in questions]
        question_patterns = []
        pattern_keywords = {
            "placement/location": ["where should", "where to place", "where to put", "placed back", "be placed"],
            "temporal/when": ["when is", "when did", "when does", "what time", "what date"],
            "counting": ["how many", "how much", "count", "number of"],
            "identity/who": ["who is", "who are", "whose", "which person"],
            "reason/why": ["why did", "why does", "why is", "what reason", "what cause"],
            "color/appearance": ["what color", "what does.*look", "wearing", "appearance"],
            "action/activity": ["what is.*doing", "what did.*do", "what activity", "what action"],
        }
        for pattern_name, keywords in pattern_keywords.items():
            count = sum(1 for q in question_lower if any(kw in q for kw in keywords))
            if count >= max(2, len(questions) * 0.3):
                question_patterns.append(pattern_name)
        pattern["question_commonality"] = question_patterns or ["mixed/no dominant pattern"]

        # 2. Wrong-answer commonality
        if answers_wrong:
            wrong_lower = [a.lower() for a in answers_wrong]
            wrong_traits = []
            # Detect whether they are all location/scene descriptions
            location_words = ["table", "shelf", "cabinet", "desk", "floor", "bed", "chair",
                              "wall", "window", "door", "room", "kitchen", "living"]
            loc_count = sum(1 for a in wrong_lower if any(w in a for w in location_words))
            if loc_count >= len(wrong_lower) * 0.5:
                wrong_traits.append("wrong answers are mostly visually plausible locations/scenes")
            # Detect whether they are all high-confidence
            if confidences and sum(1 for c in confidences if c >= 0.9) >= len(confidences) * 0.7:
                wrong_traits.append("agent was overconfident (>=0.9) in most wrong answers")
            pattern["wrong_answer_traits"] = wrong_traits or ["no clear pattern in wrong answers"]

        # 3. Correct-answer commonality
        if answers_correct:
            correct_lower = [a.lower() for a in answers_correct]
            correct_traits = []
            # Detect whether correct answers come from dialogue/instructions
            dialogue_words = ["said", "told", "mentioned", "asked", "suggested",
                              "instruction", "according to", "behind", "inside"]
            dial_count = sum(1 for a in correct_lower if any(w in a for w in dialogue_words))
            if dial_count >= len(correct_lower) * 0.3:
                correct_traits.append("correct answers often reference dialogue/instructions/hidden locations")
            pattern["correct_answer_traits"] = correct_traits or ["no clear pattern in correct answers"]

        # 4. Inductive hypotheses
        if is_negative:
            hypotheses = []
            if "placement/location" in pattern.get("question_commonality", []):
                hypotheses.append(
                    "The agent may be answering based on where objects ARE (visual scene) "
                    "rather than where they SHOULD BE (from dialogue/instructions)."
                )
            if "agent was overconfident" in pattern.get("wrong_answer_traits", []):
                hypotheses.append(
                    "The agent finalizes too quickly without seeking corroborating evidence. "
                    "It should search for at least 2 independent sources before answering."
                )
            if not hypotheses:
                hypotheses.append(
                    "Analyze the cases to find what evidence type the agent missed "
                    "and what alternative search angle would have found the correct answer."
                )
            pattern["failure_hypothesis"] = hypotheses

        return pattern

    @staticmethod
    def _build_case_items(learnings: List[Learning], max_cases: int = 8) -> list:
        """Build the list of cases fed into the LLM from the learnings.

        P2 enhancement: attach mechanism and observed_gap to each case, so the LLM can see
        the structured failure/success mechanism rather than only the raw QA pairs.
        """
        sample = learnings[:max_cases]
        case_items = []
        for l in sample:
            queries = (l.search_queries_used or [])[:6]
            retrieval_modes = []
            for q in queries:
                mode = SkillPromoter._query_mode_label(q)
                if mode not in retrieval_modes:
                    retrieval_modes.append(mode)
            agent_answer = (l.agent_answer or "")[:120]
            ground_truth = (l.ground_truth or "")[:120]
            mechanism = SkillPromoter._infer_mechanism(l)
            item = {
                "question": (l.question or "")[:200],
                "outcome": "success" if l.is_correct else "failure",
                "mechanism": mechanism,
                "observed_gap": SkillPromoter._mechanism_explanation(l, mechanism),
                "agent_answer": agent_answer,
                "ground_truth": ground_truth,
                "confidence": round(l.confidence, 2),
                "num_rounds": l.num_rounds,
                "retrieval_modes": retrieval_modes,
                "search_queries": queries,
                "search_strategy_used": l.search_strategy_used or "",
            }
            case_items.append(item)
        return case_items

    @staticmethod
    def _infer_mechanism(l) -> str:
        """Infer the failure/success mechanism from the question, answer, GT, and trajectory.

        Pure rules, no LLM needed. Used to provide structured labels to _build_instruction_signal
        so the LLM can write dedicated instructions targeting the main mechanism.
        """
        q = (l.question or "").lower()
        agent = (l.agent_answer or "").lower().strip()
        gt = (l.ground_truth or "").lower().strip()
        queries = l.search_queries_used or []

        if l.is_correct:
            # Success mechanism
            has_keyframe = any(
                isinstance(qr, str) and qr.upper().startswith("KEYFRAME:")
                for qr in queries
            )
            has_video = any(
                isinstance(qr, str) and qr.upper().startswith("VIDEO:")
                for qr in queries
            )
            if has_keyframe:
                return "visual_verification_success"
            if has_video:
                return "detail_drilldown_success"
            if l.num_rounds <= 1:
                return "direct_hit_success"
            return "multi_step_convergence"

        # Failure mechanism
        # 1. Unanswerable but the agent gave an answer
        unanswerable_signals = (
            "cannot", "not provide", "insufficient", "does not mention",
            "no information", "not shown", "not visible", "not available",
            "unable to determine",
        )
        if any(s in gt for s in unanswerable_signals):
            if not any(s in agent for s in unanswerable_signals):
                return "should_reject_but_answered"

        # 2. Yes/no judgment reversal
        agent_yn = agent[:4].rstrip(",. ")
        gt_yn = gt[:4].rstrip(",. ")
        if gt_yn in ("yes", "no") and agent_yn in ("yes", "no") and gt_yn != agent_yn:
            return "yes_no_polarity_error"

        # 3. Entity confusion
        gt_upper = set(
            w for w in (l.ground_truth or "").split()
            if len(w) > 1 and w[0].isupper() and not w.isupper()
        )
        agent_upper = set(
            w for w in (l.agent_answer or "").split()
            if len(w) > 1 and w[0].isupper() and not w.isupper()
        )
        if gt_upper and agent_upper and not gt_upper.intersection(agent_upper):
            return "entity_confusion"

        # 4. Insufficient retrieval depth
        modes = set()
        for qr in queries:
            if not isinstance(qr, str):
                continue
            if qr.upper().startswith("VIDEO:"):
                modes.add("VIDEO")
            elif qr.upper().startswith("KEYFRAME:"):
                modes.add("KEYFRAME")
            elif qr.upper().startswith("NEIGHBOR:"):
                modes.add("NEIGHBOR")
        if not modes:
            return "insufficient_retrieval_depth"

        # 5. High-confidence error
        if l.confidence >= 0.8:
            return "overconfident_wrong_answer"

        # 6. Numeric/counting error
        if any(w in q for w in ("how many", "how much", "count", "number of")):
            return "counting_error"

        return "answer_mismatch"

    @staticmethod
    def _mechanism_explanation(l, mechanism: str) -> str:
        """Generate a short structured explanation for a mechanism label, replacing the previous failure_gap."""
        templates = {
            "should_reject_but_answered": (
                "Ground truth indicates the information is unavailable or not shown, "
                "but agent fabricated an answer instead of stating uncertainty."
            ),
            "yes_no_polarity_error": (
                "Agent gave the opposite yes/no answer. "
                "The retrieval likely returned ambiguous evidence that was misinterpreted as confirmation."
            ),
            "entity_confusion": (
                "Agent attributed the correct action/property to the wrong entity. "
                "Multiple entities were present in the retrieved context."
            ),
            "insufficient_retrieval_depth": (
                "Agent only used event_first (coarse summary) without drilling down via VIDEO/KEYFRAME. "
                "The answer required fine-grained detail."
            ),
            "overconfident_wrong_answer": (
                f"Agent answered with confidence={l.confidence:.0%} but was wrong. "
                "High confidence suggests seemingly relevant but misleading evidence was found."
            ),
            "counting_error": (
                "Agent gave an incorrect count. Counting questions require explicit enumeration "
                "and re-verification before answering."
            ),
            "visual_verification_success": (
                "Agent used KEYFRAME visual inspection to verify the answer with direct visual evidence."
            ),
            "detail_drilldown_success": (
                "Agent drilled down into a specific event via VIDEO to find precise supporting evidence."
            ),
            "direct_hit_success": (
                "Agent found the answer in a single retrieval round without needing multi-step search."
            ),
            "multi_step_convergence": (
                "Agent converged on the correct answer through multiple retrieval steps, "
                "combining evidence from different sources."
            ),
        }
        explanation = templates.get(mechanism, "")
        if not explanation:
            if l.is_correct:
                explanation = f"Agent correctly answered after {l.num_rounds} rounds of retrieval."
            else:
                explanation = (
                    f"Agent answer '{(l.agent_answer or '')[:60]}' differs from "
                    f"ground truth '{(l.ground_truth or '')[:60]}'."
                )
        return explanation

    @staticmethod
    def _clean_code_block(raw: str) -> str:
        """Remove the ``` code block markers that may wrap the LLM response.

        Note: do not use str.strip("`"). It would greedily strip all leading/trailing backticks,
        also eating inline code such as `VIDEO:...` inside the bullet content, causing truncation.

        Improved strategy:
          1. First try to extract the content inside the code fence (even if there is explanatory text before the fence)
          2. If there is no code fence, just return the stripped original text
        """
        import re
        t = raw.strip()

        # Strategy 1: extract the content inside the ```...``` fence (supports prefix text before the fence)
        # Loose matching: there may or may not be a newline after the fence start, and there may or may not be a newline before the fence end
        fence_match = re.search(
            r'```[a-zA-Z]*\s*\n?(.*?)\n?\s*```',
            t,
            re.DOTALL,
        )
        if fence_match:
            content = fence_match.group(1).strip()
            if content:
                return content

        # Strategy 2: remove leading/trailing ``` lines (compatible with the old logic)
        t = re.sub(r'^```[a-zA-Z]*\s*\n?', '', t)
        t = re.sub(r'\n?```\s*$', '', t)
        return t.strip()

    @classmethod
    def validate_special_instructions(cls, instructions: str,
                                      learning_type: str = "",
                                      visual_layer_enabled: bool = False) -> bool:
        """Decide whether special_instructions are safe enough to enter the injection pipeline."""
        return cls._instruction_quality_issue(
            instructions, learning_type, visual_layer_enabled
        ) is None

    @classmethod
    def _normalize_special_instructions(cls, raw: str,
                                        learning_type: str = "",
                                        visual_layer_enabled: bool = False) -> str:
        """Clean and validate the LLM-generated special_instructions.

        This is the key quality gate of P1: the LLM output must be 2-4 complete, executable
        Markdown bullets. Any half-finished sentence, placeholder, pure slogan, or content lacking a retrieval/verification action
        will be rejected, and the caller then degrades to the template guardrail.
        """
        cleaned = cls._clean_code_block(raw)
        json_items = cls._extract_json_instruction_items(cleaned)
        bullets = []
        if json_items is not None:
            for item in json_items:
                text = " ".join(str(item).strip().split())
                if text.startswith(("- ", "* ", "• ")):
                    text = text[2:].strip()
                bullets.append("- " + text)
        else:
            # Fallback: try JSON extraction once more directly from the raw text (without _clean_code_block)
            # This handles cases where _clean_code_block accidentally truncates or the JSON contains special characters
            json_items_raw = cls._extract_json_instruction_items(raw.strip())
            if json_items_raw is not None:
                for item in json_items_raw:
                    text = " ".join(str(item).strip().split())
                    if text.startswith(("- ", "* ", "• ")):
                        text = text[2:].strip()
                    bullets.append("- " + text)
            else:
                for line in cleaned.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(("* ", "• ")):
                        stripped = "- " + stripped[2:].strip()
                    if stripped[:3].isdigit() and stripped[3:5] in {". ", ") "}:
                        stripped = "- " + stripped[5:].strip()
                    if stripped.startswith("- "):
                        bullets.append("- " + stripped[2:].strip())
                    else:
                        # Non-bullet text is usually the LLM's explanatory prefix/suffix, so directly mark it as invalid.
                        raise RuntimeError(
                            f"invalid special_instructions format: non-bullet line {stripped[:60]!r}"
                        )

        normalized = "\n".join(bullets)
        issue = cls._instruction_quality_issue(
            normalized, learning_type, visual_layer_enabled
        )
        if issue:
            raise RuntimeError(f"invalid special_instructions: {issue}")
        return normalized

    @staticmethod
    def _extract_json_instruction_items(text: str) -> Optional[List[str]]:
        """Extract {"instructions": [...]} or [...] from the LLM response.

        Fault-tolerant strategy:
          1. Try to parse the entire text directly
          2. Try to extract a closed { ... } or [ ... ] substring
          3. Try the bare fragment from the first {/[ to the end
          4. When standard JSON fails, use json-repair to fix truncated/trailing-comma/unclosed structures
        """
        import re as _re

        def _items_from_data(data) -> Optional[List[str]]:
            if isinstance(data, dict):
                items = data.get("instructions")
            else:
                items = data
            if (
                isinstance(items, list)
                and all(isinstance(x, str) for x in items)
                and items
            ):
                return items
            return None

        candidates = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        fence_match = _re.search(
            r'```(?:json)?\s*\n?([\s\S]*)', stripped, _re.IGNORECASE
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            if inner.endswith("```"):
                inner = inner[:-3].strip()
            if inner:
                candidates.append(inner)

        if "{" in text and "}" in text:
            candidates.append(text[text.find("{"):text.rfind("}") + 1].strip())
        if "[" in text and "]" in text:
            candidates.append(text[text.find("["):text.rfind("]") + 1].strip())
        first_brace = text.find("{")
        if first_brace >= 0:
            candidates.append(text[first_brace:].strip())
        first_bracket = text.find("[")
        if first_bracket >= 0:
            candidates.append(text[first_bracket:].strip())

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            # First attempt: parse directly
            try:
                items = _items_from_data(json.loads(candidate))
                if items is not None:
                    return items
            except Exception:
                pass

            # Second attempt: clean control characters that may cause JSON parsing to fail
            try:
                sanitized = candidate.replace('\r\n', '\\n').replace('\r', '\\n')
                sanitized = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
                items = _items_from_data(json.loads(sanitized))
                if items is not None:
                    return items
            except Exception:
                pass

            # Third attempt: use json-repair to fix LLM truncated/trailing-comma/unclosed JSON.
            if _HAS_JSON_REPAIR and _repair_json is not None:
                try:
                    repaired = _repair_json(candidate, return_objects=True)
                    items = _items_from_data(repaired)
                    if items is not None:
                        logger.info("[SkillPromoter] Repaired special_instructions JSON via json-repair")
                        return items
                except Exception as exc:
                    logger.debug(f"[SkillPromoter] json-repair exception while repairing special_instructions: {exc}")

        return None

    @staticmethod
    def _extract_json_instruction_obj(text: str) -> Optional[Dict]:
        """Best-effort extraction of a JSON object from the LLM response, reused for trigger_keywords."""
        import re as _re

        stripped = (text or "").strip()
        candidates = []
        if stripped:
            candidates.append(stripped)

        fence_match = _re.search(
            r'```(?:json)?\s*\n?([\s\S]*)', stripped, _re.IGNORECASE
        )
        if fence_match:
            inner = fence_match.group(1).strip()
            if inner.endswith("```"):
                inner = inner[:-3].strip()
            if inner:
                candidates.append(inner)

        if "{" in text and "}" in text:
            candidates.append(text[text.find("{"):text.rfind("}") + 1].strip())
        first_brace = text.find("{")
        if first_brace >= 0:
            candidates.append(text[first_brace:].strip())

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            try:
                sanitized = candidate.replace('\r\n', '\\n').replace('\r', '\\n')
                sanitized = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
                data = json.loads(sanitized)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            if _HAS_JSON_REPAIR and _repair_json is not None:
                try:
                    repaired = _repair_json(candidate, return_objects=True)
                    if isinstance(repaired, dict):
                        logger.info("[SkillPromoter] Repaired trigger_keywords JSON via json-repair")
                        return repaired
                except Exception as exc:
                    logger.debug(f"[SkillPromoter] json-repair exception while repairing trigger_keywords: {exc}")
        return None

    @classmethod
    def _instruction_quality_issue(cls, instructions: str,
                                   learning_type: str = "",
                                   visual_layer_enabled: bool = False) -> Optional[str]:
        cleaned = cls._clean_code_block(instructions or "")
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) < 1 or len(lines) > 5:
            return f"expected 1-5 bullet lines, got {len(lines)}"

        # Relaxed action terms: include domain-specific verbs, no longer requiring a retrieval action
        action_terms = (
            "event_first", "verify", "verification", "double-check",
            "cross-check", "inspect", "enumerate", "recount", "drill down",
            "search", "retrieve", "check", "compare", "confirm", "support",
            "do not answer", "state uncertainty", "lower confidence",
            "reject", "defer", "quote", "literal evidence",
            # Newly added: domain-specific verbs, allowing instructions to describe evidence types and search angles
            "prioritize", "look for", "focus on", "avoid", "distrust",
            "corroborate", "distinguish", "identify", "target",
            "instead of", "rather than", "not just", "also check",
            "dialogue", "conversation", "instruction", "verbal",
            "confidence", "finalize", "conclude", "answer",
        )
        retrieval_action_pattern = (
            r"(?<![a-z0-9_])(?:video|neighbor)(?:\s*[:<`])"
        )
        retrieval_action_name_pattern = (
            r"(?<![A-Za-z0-9_])(?:VIDEO|NEIGHBOR)(?![A-Za-z0-9_])"
        )
        if visual_layer_enabled:
            retrieval_action_pattern = (
                r"(?<![a-z0-9_])(?:video|neighbor|keyframe)(?:\s*[:<`])"
            )
            retrieval_action_name_pattern = (
                r"(?<![A-Za-z0-9_])(?:VIDEO|NEIGHBOR|KEYFRAME)(?![A-Za-z0-9_])"
            )
        # Relaxed negative guardrail terms: accept more vocabulary that expresses confidence calibration
        negative_guardrail_terms = (
            "do not answer", "lower confidence", "state uncertainty", "reject",
            "guardrail", "verify", "verification", "direct support",
            "direct evidence", "distrust", "insufficient", "corroborate",
            "not finalize", "do not finalize", "avoid", "wrong",
            "trap", "misleading", "overconfident", "caution",
            "confidence", "calibrat",
        )
        incomplete_suffixes = (
            ",", ";", ":", "-", "—", " and", " or", " after",
            " before", " when", " while", " if", " with", " to", " a", " an",
            " the", " of", " in", " on", " for", " by", " as", " only provides a",
            " only provides an", " only provides the", " only shows a",
            " only shows an", " only shows the",
        )
        generic_phrases = (
            "be careful", "analyze carefully", "ensure accuracy",
            "consider the context", "use context", "broader context",
            "broader contextual information", "relevant events",
            "after identifying relevant events", "gather more information",
            "pay attention", "look for clues", "make sure",
        )
        body_text = "\n".join(lines).lower()
        if not visual_layer_enabled and "keyframe:" in body_text:
            return "KEYFRAME action unavailable when visual layer is disabled"

        for line in lines:
            if not line.startswith("- "):
                return f"line is not a Markdown bullet: {line[:60]!r}"
            body = line[2:].strip()
            body_lower = body.lower()
            if not body or "<instruction" in body_lower or "optional" in body_lower:
                return "contains placeholder text"
            if any(body_lower.endswith(suffix) for suffix in incomplete_suffixes):
                return f"incomplete bullet ending: {body[:80]!r}"
            if not re.search(r'[.!?]$|`$', body):
                return f"bullet lacks sentence-final punctuation: {body[:80]!r}"
            if re.search(r'\b(a|an|the|of|in|on|for|with|to|by|as)$', body_lower):
                return f"incomplete bullet ending in function word: {body[:80]!r}"
            if len(body.split()) < 5:
                return f"bullet too short: {body[:80]!r}"
            if len(body.split()) > 80:
                return f"bullet too long: {body[:80]!r}"
            if any(phrase in body_lower for phrase in generic_phrases):
                return f"bullet is too generic: {body[:80]!r}"
            has_action_term = any(term in body_lower for term in action_terms)
            has_retrieval_action = (
                re.search(retrieval_action_pattern, body_lower) is not None
                or re.search(retrieval_action_name_pattern, body) is not None
            )
            if not (has_action_term or has_retrieval_action):
                return f"bullet lacks executable action: {body[:80]!r}"

        if learning_type in cls._NEGATIVE_LEARNING_TYPES:
            if not any(term in body_text for term in negative_guardrail_terms):
                return "negative skill lacks calibration/search-failure guardrail"
            # Relaxed: no longer force a negative skill to contain a retrieval action or verification
            # because domain-specific confidence calibration instructions (e.g., "distrust visually obvious answers")
            # are equally valid and do not need to explicitly name a retrieval action
        return None

    def _call_instruction_llm(self, prompt: str, learning_type: str,
                              purpose: str,
                              keep_current_instructions: str = "") -> str:
        """Call the LLM and perform one structured repair when the output is invalid.

        This does not relax validation; instead it feeds the validation error back to the LLM, letting it re-output
        complete short sentences according to the JSON schema, reducing the chance of directly falling back to a template.

        The refine scenario allows the LLM to return keep_current=true, indicating that the new signals do not bring
        complementary knowledge beyond the original prompt and the current skill, in which case the old instructions are kept unchanged.

        Side effect: if the LLM returns a trigger_keywords field, it is stored in
        self._last_llm_keywords for the caller to use.
        """
        from .llm_utils import call_llm

        # Reset the previous keywords / refine decision
        self._last_llm_keywords: List[str] = []
        self._last_refine_kept_current = False

        response = call_llm(
            model=self.instructions_llm_model,
            prompt=prompt,
            timeout=90,
            max_retries=3,
            json_mode=True,
        )
        if not response or not response.strip():
            raise RuntimeError(f"LLM returned empty response for {purpose}")

        if purpose.startswith("refine") and self._should_keep_current_instructions(response):
            if not keep_current_instructions:
                raise RuntimeError("LLM requested keep_current but no current instructions were provided")
            self._last_refine_kept_current = True
            self._last_llm_keywords = []
            return keep_current_instructions

        # Try to extract trigger_keywords (before normalize, because normalize may discard non-instructions fields)
        self._last_llm_keywords = self._extract_trigger_keywords_from_llm_response(response)

        try:
            return self._normalize_special_instructions(
                response, learning_type, self.visual_layer_enabled
            )
        except RuntimeError as first_exc:
            repair_prompt = f"""Your previous output failed validation for {purpose}: {first_exc}

Output ONLY raw JSON parseable by json.loads(). Start your response with {{ and end with }}.

Schema for generation/refinement:
{{"keep_current":false,"instructions":["complete actionable sentence.","complete actionable sentence."],"trigger_keywords":["pattern_word_1","pattern_phrase_2"]}}

Schema for refine only, when no update is needed:
{{"keep_current":true,"reason":"current instructions are already complementary to the base prompt and the new batch adds no new reusable pattern"}}

Hard constraints:
- 1 to 5 instruction strings when keep_current is false. No markdown, no code fences, no commentary.
- Every string must be a complete sentence, not a dangling clause.
- Every string must provide DOMAIN-SPECIFIC knowledge: what evidence type to look for, what search angle works, what confidence trap to avoid. Do NOT merely name retrieval modes (e.g., "use VIDEO to verify") without specifying WHAT to search for.
- Every string must include an actionable verb such as search, prioritize, avoid, distrust, verify, focus on, look for, compare, reject, corroborate, distinguish, or target.
- For negative clusters, at least one string must address confidence calibration or error prevention (e.g., distrust, avoid, do not finalize, overconfident, insufficient, corroborate, wrong, trap, misleading).
- Avoid generic phrases such as "be careful", "broader context", "relevant events", or "make sure".
- trigger_keywords: 3-5 cognitive-pattern-level keywords describing the REASONING TYPE needed (e.g. "spatial_location", "object_identification", "manner_method", "counting", "yes_no_verification"). Use cognitive pattern names when possible. NOT entity names, NOT function-word phrases, NOT single common verbs, NOT generic video-QA words.

Previous invalid output:
{response[:1500]}
"""
            repaired = call_llm(
                model=self.instructions_llm_model,
                prompt=repair_prompt,
                timeout=60,
                max_retries=1,
                json_mode=True,
            )
            if not repaired or not repaired.strip():
                raise first_exc

            if purpose.startswith("refine") and self._should_keep_current_instructions(repaired):
                if not keep_current_instructions:
                    raise first_exc
                self._last_refine_kept_current = True
                self._last_llm_keywords = []
                return keep_current_instructions

            # Try to extract keywords again from the repaired response
            repaired_kws = self._extract_trigger_keywords_from_llm_response(repaired)
            if repaired_kws:
                self._last_llm_keywords = repaired_kws

            return self._normalize_special_instructions(
                repaired, learning_type, self.visual_layer_enabled
            )

    @classmethod
    def _should_keep_current_instructions(cls, response: str) -> bool:
        """Parse the keep_current decision from the refine output."""
        data = cls._extract_json_instruction_obj(response or "")
        if not isinstance(data, dict):
            return False
        value = data.get("keep_current", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "keep", "keep_current"}
        return False

    @classmethod
    def _extract_trigger_keywords_from_llm_response(cls, response: str) -> List[str]:
        """Extract the trigger_keywords field from the LLM's JSON response.

        Fault tolerance: if JSON parsing fails or the field does not exist, return an empty list.
        """
        data = cls._extract_json_instruction_obj(response or "")
        if not isinstance(data, dict):
            return []

        kws = data.get("trigger_keywords", [])
        if not (isinstance(kws, list) and all(isinstance(k, str) for k in kws)):
            return []

        # Clean: strip whitespace, lowercase, remove the "keyword:" prefix (if the LLM mistakenly added it)
        cleaned = []
        for k in kws:
            k = k.strip().lower()
            if k.startswith("keyword:"):
                k = k[8:].strip()
            if k and len(k) >= 3:
                cleaned.append(k)
        return cleaned

    @staticmethod
    def _validate_llm_keywords(llm_keywords: List[str],
                               learnings: List[Learning],
                               min_df: int = 2,
                               max_keywords: int = 5) -> List[str]:
        """Posterior validation of the LLM-generated trigger_keywords.

        Validation rules (layered):
          1. Cognitive pattern first: if a keyword matches an indicator in the cognitive-pattern dictionary, pass directly
          2. Transferability check: non-cognitive-pattern words must pass the following checks:
             a) Must not be a proper noun (an entity that appears only in a specific video)
             b) Must word-boundary match in the question of at least min_df learnings
             c) Must not be a generic video-QA domain word
          3. Keep at most max_keywords

        Return the list of keywords that pass validation. If all are filtered out, return an empty list.
        """
        import re as _re
        from .cognitive_patterns import (
            COGNITIVE_PATTERNS,
            COGNITIVE_PATTERN_NAMES,
            TRANSFERABLE_CONTEXT_PATTERNS,
            FUNCTION_WORDS,
            is_low_value_keyword,
            normalize_cognitive_keyword,
        )

        if not llm_keywords:
            return []

        # Build the set of cognitive-pattern indicators (for fast lookup)
        pattern_indicators: set = set()
        for pat in COGNITIVE_PATTERNS:
            for ind in pat["indicators"]:
                pattern_indicators.add(ind.lower())

        # Blacklist of generic video-QA domain words
        domain_stop = {
            "video", "clip", "scene", "moment", "part", "segment", "frame",
            "shown", "seen", "appear", "appears", "appeared", "show", "shows",
            "during", "after", "before", "while", "first", "last", "next",
            "second", "third", "end", "beginning", "start", "time",
            "person", "people", "man", "woman", "one", "two", "three",
            "something", "thing", "things", "way", "much", "many", "more",
            "most", "also", "still", "already", "get", "got", "gets",
            "make", "made", "makes", "take", "took", "takes",
            "say", "said", "says", "tell", "told", "tells",
            "come", "came", "comes", "give", "gave", "gives",
            "see", "saw", "look", "looks", "looked", "watch", "watching",
            "use", "used", "uses", "using", "go", "went", "goes", "going",
            "know", "known", "think", "thought",
            "new", "old", "good", "bad", "big", "small", "long", "short",
            "right", "left", "back", "front", "top", "bottom",
            "other", "others", "another", "same", "different",
            "like", "want", "need", "try", "keep", "let", "put",
            "called", "name", "named", "type", "kind",
            "happen", "happened", "happens", "event", "events",
            "actually", "really", "likely", "probably", "originally",
            "able", "unable", "possible", "impossible",
            "entire", "whole", "total", "full", "complete",
            "main", "major", "important", "specific", "particular",
            "involved", "related", "based", "according",
            "visual", "visually", "visible", "raw", "material", "materials",
            "local", "construct", "construction", "build", "building",
            "evidence", "information", "detail", "details", "object", "objects",
            "action", "actions", "activity", "activities", "context", "cue", "cues",
        }

        # Collect all question text (lowercased)
        questions_lower = [
            (l.question or "").lower() for l in learnings
        ]
        video_names = [getattr(l, "video_name", "") or "" for l in learnings]
        unique_video_names = {v for v in video_names if v}

        # Collect proper-noun candidates (for the transferability check)
        proper_noun_candidates: set = set()
        for l in learnings:
            q_raw = l.question or ""
            for idx, tok_raw in enumerate(q_raw.split()):
                clean = "".join(ch for ch in tok_raw if ch.isalnum())
                if not clean:
                    continue
                if idx > 0 and clean[0].isupper():
                    proper_noun_candidates.add(clean.lower())

        # Relax min_df for small samples
        effective_min_df = min_df if len(learnings) >= 4 else 1

        validated = []
        seen = set()
        for kw in llm_keywords:
            raw_kw = kw.strip().lower()
            kw = normalize_cognitive_keyword(raw_kw)
            if not kw or len(kw) < 3:
                continue
            if is_low_value_keyword(kw):
                continue

            # Layer 1: cognitive-pattern name/indicator passes after being normalized to a pattern name
            if kw in COGNITIVE_PATTERN_NAMES:
                if kw not in seen:
                    validated.append(kw)
                    seen.add(kw)
                continue

            # Layer 2: transferable-context whitelist passes directly, but function-word phrases are still rejected
            if kw in TRANSFERABLE_CONTEXT_PATTERNS:
                if kw not in seen:
                    validated.append(kw)
                    seen.add(kw)
                continue

            # Layer 3: general validation
            kw_words = kw.split()

            # 3a) Check whether it is a domain generic word or a function-word phrase
            if all(w in domain_stop or w in FUNCTION_WORDS for w in kw_words):
                continue
            if any(w in FUNCTION_WORDS for w in (kw_words[:1] + kw_words[-1:])):
                continue

            # 3b) Check whether it is a proper noun (not transferable)
            if any(w in proper_noun_candidates for w in kw_words):
                # If the keyword contains a proper noun, skip
                continue

            # 3c) Word-boundary match: check how many questions it appears in
            pattern = r'\b' + _re.escape(kw) + r'\b'
            matched_indices = [
                i for i, q in enumerate(questions_lower)
                if _re.search(pattern, q)
            ]
            df = len(matched_indices)
            if df < effective_min_df:
                continue

            # 3d) True cross-video transferability check: if the samples span multiple videos,
            # the keyword must not appear only in the questions of a single video.
            if len(unique_video_names) >= 2:
                matched_videos = {
                    video_names[i] for i in matched_indices
                    if i < len(video_names) and video_names[i]
                }
                if len(matched_videos) < 2:
                    continue

            # 3e) Additional transferability check: words must not be too short (words <=3 chars are usually generic)
            if len(kw_words) == 1 and len(kw) <= 3:
                continue

            if kw not in seen:
                validated.append(kw)
                seen.add(kw)

        return validated[:max_keywords]

    def _generate_special_instructions(self, learning_type: str,
                                       question_type: str,
                                       learnings: List[Learning]) -> str:
        """Call the LLM to generate executable instructions for this cluster from scratch.

        Input: a few concrete cases of this cluster + a statistical summary
        Output: 2-4 short, Agent-behavior-oriented instructions (in English, to be concatenated into the system prompt)
        """
        total = len(learnings)
        correct = sum(1 for l in learnings if l.is_correct)
        signal = self._build_instruction_signal(
            learning_type, question_type, learnings
        )
        negative_rule = (
            "- Because cluster_polarity is negative_guardrail, at least one instruction must explicitly prevent the observed failure path with do not answer, lower confidence, reject, or state uncertainty."
            if learning_type in self._NEGATIVE_LEARNING_TYPES else
            "- Because cluster_polarity is positive_action, focus on the retrieval-and-verification action that made successful cases repeatable."
        )

        prompt = f"""You are converting clustered video-QA learning signals into reusable Agent instructions.

{self._available_retrieval_actions_text()}

{self._base_prompt_capabilities_text()}

SOURCE_SIGNAL_JSON:
{json.dumps(signal, ensure_ascii=False, indent=2)}

Stats: total={total}, correct={correct}, accuracy={(correct / total) if total else 0:.1%}

Deduplication constraint (CRITICAL — read before generating):
- The base system prompt ALREADY teaches the agent all generic retrieval strategies listed above.
- Your instructions MUST NOT repeat base capabilities. Instead, provide DOMAIN-SPECIFIC knowledge that the base prompt cannot cover:
  * What TYPE of evidence most likely contains the answer for this question pattern (e.g., "answers to placement questions come from dialogue/instructions, not visual scenes")
  * What SPECIFIC query reformulation works when the obvious query fails (e.g., "search for verbal instructions mentioning the object rather than the object's current location")
  * What CONFIDENCE CALIBRATION rule applies (e.g., "do not finalize when only one source confirms a location — search for corroborating dialogue")
  * What COMMON WRONG-ANSWER TRAP to avoid (e.g., "the visually obvious location is often wrong for 'should' questions")

Specificity requirement:
- Before writing instructions, identify the COMMON FAILURE PATTERN across all cases in SOURCE_SIGNAL_JSON:
  * What do the questions have in common? (e.g., all ask "where should X be placed")
  * What do the wrong answers have in common? (e.g., all are visually plausible but not from dialogue)
  * What do the correct answers have in common? (e.g., all come from verbal instructions or memory)
  * Why did the agent fail? (e.g., it searched for the object's current location instead of dialogue about where it SHOULD go)
- Your instructions must address THIS specific pattern, not generic retrieval advice.

BANNED instruction patterns (these add ZERO value over the base prompt):
- "Use VIDEO/NEIGHBOR/KEYFRAME/search to verify/compare/inspect..." (already in base prompt)
- "Search with different keywords when stuck" (already in base prompt)
- "State uncertainty if evidence is insufficient" (already in base prompt)
- Any instruction that merely names a retrieval mode without domain-specific knowledge about WHAT to search for or WHY the obvious approach fails

Generation protocol:
- Infer one reusable mechanism from SOURCE_SIGNAL_JSON; do not summarize cases or restate stats.
- Each instruction must follow trigger -> action -> verification/stop-rule.
- Name concrete retrieval actions exactly as {self._retrieval_action_names_text()} when retrieval is needed{self._retrieval_example_text()}.
- Each instruction must include an executable verb such as use, verify, inspect, compare, retrieve, reject, quote, or state uncertainty.
- Each instruction must contain domain-specific knowledge (what evidence type, what search angle, what trap to avoid) that goes BEYOND the base prompt.
{negative_rule}
- Do not write generic advice such as "be careful", "use broader context", "look at relevant events", or "make sure".
- Do not write dangling clauses such as "After identifying relevant events,".

trigger_keywords protocol:
- Extract 3-5 keywords that describe the COGNITIVE REASONING PATTERN needed to answer these questions.
- Prefer canonical cognitive pattern names: "counting", "spatial_location", "temporal_order", "causal", "person_identity", "person_relationship", "comparison", "intent_plan", "manner_method", "yes_no_verification", "object_identification".
- Keywords MUST fall into one of two categories:
  Category A (Cognitive Pattern): Words/phrases that describe WHAT TYPE OF REASONING is needed.
    Examples: "how many" (counting), "spatial location" (where-questions), "temporal order" (sequence), "causal reasoning" (why-questions), "person relationship" (attitude/feeling), "comparison" (more/better), "intent" (should/plan to), "manner method" (how did).
  Category B (Transferable Context): Scene-level words that are REUSABLE across different videos.
    Examples: "cooking", "robot get", "homework", "birthday party", "cleaning", "care about", "familiar with", "occupation".
- STRICTLY FORBIDDEN keywords:
  * Specific entity names (Nancy, Bob, Tom, William, Emma)
  * Specific object names that only appear in one video (soup, broccoli, chessboard, balloons)
  * Generic video-QA words (video, person, after, during, scene, shown, appear)
  * Function-word phrases (in the, on the, of the, to the, with the)
  * Single common verbs (taken, should, put, get, make)
- Prefer bigram/trigram phrases over single words. Single words are acceptable ONLY if they are clearly a cognitive pattern indicator (e.g., "counting", "comparison").
- Each keyword must appear (as a pattern) in at least 2 of the sample questions.

Output ONLY raw JSON parseable by json.loads(). Start your response with {{ and end with }}. No markdown, no code fences, no commentary.

Schema:
{{"instructions":["When <trigger>, use <action> and verify <condition>.","If <condition>, reject <answer> or state uncertainty."],"trigger_keywords":["pattern_word_1","pattern phrase 2","pattern_word_3"]}}
"""
        return self._call_instruction_llm(
            prompt, learning_type, "special_instructions"
        )

    def _refine_special_instructions(self, learning_type: str,
                                     question_type: str,
                                     new_learnings: List[Learning],
                                     old_instructions: str) -> str:
        """Based on the old instructions + new cases, let the LLM iteratively optimize the skill instructions.

        Differences from _generate_special_instructions:
        - Pass the old instructions in as context so the LLM refines on top of them
        - New cases may reveal failure modes not covered by the old instructions, or confirm the effectiveness of the old instructions
        """
        total_new = len(new_learnings)
        correct_new = sum(1 for l in new_learnings if l.is_correct)
        signal = self._build_instruction_signal(
            learning_type, question_type, new_learnings
        )
        negative_rule = (
            "- For negative clusters, keep or add at least one guardrail that blocks the repeated wrong-answer path with do not answer, lower confidence, reject, or state uncertainty."
            if learning_type in self._NEGATIVE_LEARNING_TYPES else
            "- For positive clusters, keep only actions that are supported by the new successful evidence."
        )

        prompt = f"""You are refining reusable Agent instructions for video-QA.

{self._available_retrieval_actions_text()}

{self._base_prompt_capabilities_text()}

Current instructions:
{old_instructions}

NEW_SOURCE_SIGNAL_JSON:
{json.dumps(signal, ensure_ascii=False, indent=2)}

New batch stats: total={total_new}, correct={correct_new}, accuracy={(correct_new / total_new) if total_new else 0:.1%}

Refine decision protocol (CRITICAL):
- First decide whether the current instructions should change.
- Return keep_current=true if the current instructions are already complementary to the base prompt AND NEW_SOURCE_SIGNAL_JSON does not reveal a new reusable failure mode, evidence type, query reformulation, or confidence trap.
- Return keep_current=true if changing would only paraphrase existing instructions or repeat base prompt capabilities.
- Return keep_current=false only when the new batch provides a concrete, reusable, domain-specific improvement.
- Do not refine just because new examples arrived.

Deduplication constraint (CRITICAL):
- The base system prompt ALREADY teaches the agent all generic retrieval strategies.
- Remove any old instruction that merely repeats base capabilities (e.g., "use VIDEO to verify", "search with different keywords").
- Keep or add ONLY instructions that provide domain-specific knowledge: what evidence type to look for, what search angle works, what confidence trap to avoid.

Refinement protocol when keep_current=false:
- Preserve an old instruction only if it contains domain-specific knowledge NOT covered by the base prompt AND is supported by NEW_SOURCE_SIGNAL_JSON.
- Replace generic advice with trigger -> action -> verification/stop-rule instructions that include WHAT to search for (evidence type) and WHY the obvious approach fails.
- Name concrete retrieval actions exactly as {self._retrieval_action_names_text()} when retrieval is needed.
{negative_rule}
- Do not write generic advice such as "be careful", "use broader context", "look at relevant events", or "make sure".
- Do not write dangling clauses such as "After identifying relevant events,".

BANNED instruction patterns (add ZERO value):
- "Use VIDEO/NEIGHBOR/search to verify/compare/inspect..." without specifying WHAT evidence type to look for
- "Search with different keywords when stuck" (already in base prompt)
- "State uncertainty if evidence is insufficient" (already in base prompt)

trigger_keywords protocol when keep_current=false:
- Extract 3-5 discriminative pattern-level keywords/phrases from the sample questions.
- Prefer canonical cognitive pattern names: "counting", "spatial_location", "temporal_order", "causal", "person_identity", "person_relationship", "comparison", "intent_plan", "manner_method", "yes_no_verification", "object_identification".
- Keywords must reflect the QUESTION PATTERN (e.g. "spatial_location", "person_relationship", "manner_method", "yes_no_verification") — NOT specific entities (e.g. "Nancy", "Bob"), NOT function-word phrases (e.g. "in the", "on the"), and NOT generic video-QA words (e.g. "video", "person", "after", "during", "scene", "shown", "appear").
- Prefer bigram phrases (e.g. "cooking technique") over single words when they add specificity.
- Each keyword must appear in at least 2 of the sample questions (word-boundary match).

Output ONLY raw JSON parseable by json.loads(). Start your response with {{ and end with }}. No markdown, no code fences, no commentary.

Schema when no update is needed:
{{"keep_current":true,"reason":"Current instructions already cover the new reusable pattern without repeating the base prompt."}}

Schema when update is needed:
{{"keep_current":false,"instructions":["When <trigger>, use <action> and verify <condition>.","If <condition>, reject <answer> or state uncertainty."],"trigger_keywords":["pattern_word_1","pattern phrase 2","pattern_word_3"]}}
"""
        return self._call_instruction_llm(
            prompt,
            learning_type,
            "refine_special_instructions",
            keep_current_instructions=old_instructions,
        )

    def _available_retrieval_actions_text(self) -> str:
        if self._is_control_mode:
            # The control_api_harness mode only supports plain text queries
            return (
                "- Plain text query: semantic search over the memory bank. "
                "The query is encoded into embeddings for vector similarity search.\n"
                "- Character ID query: use 'What is the name of <character_i>' or "
                "'What is the character id of <name>' to resolve character mappings."
            )
        lines = [
            "- `event_first`: coarse EventGraph keyword search.",
            "- `VIDEO:<event_id>:<query>`: fine-grained drilldown inside one event.",
            "- `NEIGHBOR:<prev|next>:<query>`: adjacent-event context lookup.",
        ]
        if self.visual_layer_enabled:
            lines.append(
                "- `KEYFRAME:<query>`: visual keyframe inspection for visible objects, attributes, counts, text, spatial relations, and fine actions."
            )
        return "\n".join(lines)

    def _retrieval_action_names_text(self) -> str:
        if self._is_control_mode:
            return "`plain text query`, `character ID query`"
        actions = [
            "`event_first`",
            "`VIDEO:<event_id>:<query>`",
            "`NEIGHBOR:<prev|next>:<query>`",
        ]
        if self.visual_layer_enabled:
            actions.append("`KEYFRAME:<query>`")
        return ", ".join(actions)

    def _retrieval_example_text(self) -> str:
        """Generate the retrieval action example text in the LLM prompt."""
        if self._is_control_mode:
            return (
                "; write queries as plain text, e.g. "
                "\"What is the name of <character_0>\", "
                "\"robot placing object on table\""
            )
        examples = "VIDEO:<event_id>:<query>, NEIGHBOR:<prev|next>:<query>"
        if self.visual_layer_enabled:
            examples += ", or KEYFRAME:<query>"
        return f"; write retrieval actions with the colon form, e.g. {examples}"

    def _base_prompt_capabilities_text(self) -> str:
        """Generate a summary of the capabilities already present in the original system prompt, telling the LLM what does not need to be repeated.

        This is the key to solving the problem of self-evolved instructions overlapping with the original prompt: let the LLM know
        what the baseline prompt has already taught the model, so it only generates incremental domain-specific knowledge.
        """
        if self._is_control_mode:
            return """BASE CAPABILITIES ALREADY IN THE AGENT'S SYSTEM PROMPT (DO NOT REPEAT these — they add zero value):
- The agent already knows to use semantic vector search over the memory bank.
- The agent already knows to write queries different from previous ones.
- The agent already knows to resolve character IDs via "What is the name of <character_i>".
- The agent already knows to use character ID instead of name for searching after mapping.
- The agent already knows to guess when information is insufficient (last round).
- The agent already knows the Action: [Answer] or [Search] / Content: format.

YOUR INSTRUCTIONS MUST PROVIDE KNOWLEDGE THAT THE BASE PROMPT CANNOT COVER:
- What TYPE of evidence most likely contains the answer for this question pattern
- What SEARCH ANGLE works when the obvious query fails for this pattern
- What CONFIDENCE CALIBRATION rule applies (when to distrust a seemingly good answer)
- What COMMON WRONG-ANSWER TRAP to avoid for this question pattern"""

        base = """BASE CAPABILITIES ALREADY IN THE AGENT'S SYSTEM PROMPT (DO NOT REPEAT these — they add zero value):
- The agent already knows HOW to use all retrieval modes (plain query, VIDEO, NEIGHBOR"""
        if self.visual_layer_enabled:
            base += ", KEYFRAME"
        base += """).
- The agent already knows to write declarative queries with subject + predicate + context.
- The agent already knows to avoid duplicate queries and switch modes when stuck.
- The agent already knows to resolve character IDs and use placeholders in queries.
- The agent already knows to compare options before answering (MCQ mode).
- The agent already knows when to use VIDEO (fine-grained detail needed) vs NEIGHBOR (wrong focus event).
- The agent already knows not to retry near-duplicate queries.

YOUR INSTRUCTIONS MUST PROVIDE KNOWLEDGE THAT THE BASE PROMPT CANNOT COVER:
- What TYPE of evidence most likely contains the answer for this question pattern (e.g., "dialogue" vs "visual scene" vs "character action")
- What SEARCH ANGLE works when the obvious query fails for this pattern (e.g., "search for verbal instructions mentioning the object" instead of "search for the object's location")
- What CONFIDENCE CALIBRATION rule applies (e.g., "do not finalize with confidence=1.0 when only one clip confirms a location")
- What COMMON WRONG-ANSWER TRAP to avoid (e.g., "the visually obvious answer is often wrong for 'should' questions")
- What ALTERNATIVE INTERPRETATION of the question leads to the correct evidence"""
        return base
