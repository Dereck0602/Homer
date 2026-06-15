"""
Cognitive Pattern Vocabulary

Defines the cognitive pattern taxonomy of questions, used to replace pure statistical
methods for generating trigger_keywords. Keywords should describe "what type of reasoning
the question requires", rather than the surface words that appear in the question.

Two layers:
  1. pattern_keywords (cognitive pattern): matched from a predefined vocabulary, describing the reasoning type
  2. context_keywords (task context): extracted from the question text, but must be transferable across videos
"""
import re
from typing import List, Dict, Tuple, Set
from collections import Counter


# ============================================================
# Layer 1: cognitive pattern vocabulary
# Each pattern contains:
#   - name: pattern name (used as keyword prefix)
#   - indicators: words/phrases that, when appearing in a question, indicate this pattern
#   - description: semantic description of the pattern
# ============================================================

COGNITIVE_PATTERNS: List[Dict] = [
    {
        "name": "counting",
        "indicators": [
            "how many", "how much", "count", "number of", "total",
            "how often", "frequency", "times",
        ],
        "description": "reasoning that requires counting or quantification",
    },
    {
        "name": "temporal_order",
        "indicators": [
            "before", "after", "first", "then", "order", "sequence",
            "earlier", "later", "next", "previous", "following",
            "when did", "at what point", "which came first",
        ],
        "description": "questions that require temporal-order reasoning",
    },
    {
        "name": "causal",
        "indicators": [
            "why", "because", "reason", "cause", "result", "lead to",
            "consequence", "due to", "so that", "in order to",
        ],
        "description": "questions that require causal reasoning",
    },
    {
        "name": "person_identity",
        "indicators": [
            "who", "whose", "which person", "which character",
            "who is", "who was", "who did",
        ],
        "description": "questions that require identifying a person's identity",
    },
    {
        "name": "person_relationship",
        "indicators": [
            "relationship", "care about", "treat", "attitude",
            "feel about", "opinion of", "think of", "towards",
            "friendly", "close to", "familiar with",
        ],
        "description": "questions that require understanding interpersonal relationships or attitudes",
    },
    {
        "name": "spatial_location",
        "indicators": [
            "where", "location", "located", "placed", "put", "position",
            "which section", "which part", "which layer", "which shelf",
            "which table", "which room", "which cabinet", "which drawer",
            "which container", "which rack", "which place", "which location",
            "beside", "next to", "near",
        ],
        "description": "questions that require spatial localization",
    },
    {
        "name": "comparison",
        "indicators": [
            "more", "better", "most", "compared", "difference",
            "prefer", "rather", "versus", "which is",
            "more familiar", "more experienced",
            "or more", "better than", "worse than",
        ],
        "description": "questions that require comparative judgment",
    },
    {
        "name": "intent_plan",
        "indicators": [
            "should", "want to", "plan to", "intend", "going to",
            "supposed to", "need to", "goal", "purpose",
        ],
        "description": "questions that require understanding intent or plans",
    },
    {
        "name": "yes_no_verification",
        "indicators": [
            "did", "does", "is", "was", "are", "were",
            "has", "have", "can", "could",
        ],
        "description": "yes/no verification questions (require verifying facts)",
        # Note: these indicators are too broad, so they only count when matched at the sentence start
        "match_mode": "sentence_start",
    },
    {
        "name": "manner_method",
        "indicators": [
            "how did", "how does", "how to", "method", "way",
            "technique", "approach", "steps", "procedure",
            "by themselves", "follow the recipe",
        ],
        "description": "questions that require understanding manner or method",
    },
    {
        "name": "object_identification",
        "indicators": [
            "what", "which", "what kind", "what type",
            "what color", "what brand", "what material",
        ],
        "description": "questions that require identifying an object or attribute",
        "match_mode": "sentence_start",
    },
]

COGNITIVE_PATTERN_NAMES: Set[str] = {pat["name"] for pat in COGNITIVE_PATTERNS}

# Function words / low-value phrases cannot be used as routing keywords. They may appear
# in natural-language questions but do not express a stable cognitive pattern.
FUNCTION_WORDS: Set[str] = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with",
    "from", "by", "as", "about", "into", "onto", "over", "under",
    "and", "or", "but", "if", "when", "while", "after", "before",
    "during", "than", "then", "this", "that", "these", "those", "it",
}

LOW_VALUE_KEYWORD_PHRASES: Set[str] = {
    "in the", "on the", "at the", "of the", "to the", "for the",
    "with the", "from the", "by the", "as the", "in a", "on a",
    "at a", "of a", "to a", "for a", "with a", "from a",
}

_PATTERN_ALIAS_TO_NAME: Dict[str, str] = {}
for _pat in COGNITIVE_PATTERNS:
    _name = _pat["name"].lower()
    _PATTERN_ALIAS_TO_NAME[_name] = _name
    _PATTERN_ALIAS_TO_NAME[_name.replace("_", " ")] = _name
    for _ind in _pat["indicators"]:
        _PATTERN_ALIAS_TO_NAME[_ind.lower()] = _name


def normalize_cognitive_keyword(keyword: str) -> str:
    """Normalize a cognitive keyword produced by the LLM/rules into a pattern name."""
    kw = " ".join((keyword or "").strip().lower().replace("-", " ").split())
    if not kw:
        return ""
    return _PATTERN_ALIAS_TO_NAME.get(kw, kw.replace(" ", "_") if kw in COGNITIVE_PATTERN_NAMES else kw)


def is_low_value_keyword(keyword: str) -> bool:
    """Determine whether a keyword is merely a function word or a non-discriminative surface phrase."""
    kw = " ".join((keyword or "").strip().lower().split())
    if not kw:
        return True
    if kw in LOW_VALUE_KEYWORD_PHRASES:
        return True
    words = kw.split()
    if all(w in FUNCTION_WORDS for w in words):
        return True
    if len(words) > 1 and (words[0] in FUNCTION_WORDS or words[-1] in FUNCTION_WORDS):
        return True
    return False

# Pre-compile indicators into regex patterns (improves matching efficiency)
_COMPILED_PATTERNS: List[Tuple[Dict, List[re.Pattern]]] = []
for _pat in COGNITIVE_PATTERNS:
    _compiled = []
    for _ind in _pat["indicators"]:
        if _pat.get("match_mode") == "sentence_start":
            # match only at the sentence start
            _compiled.append(re.compile(r'^\s*' + re.escape(_ind) + r'\b', re.IGNORECASE))
        else:
            # word-boundary matching
            _compiled.append(re.compile(r'\b' + re.escape(_ind) + r'\b', re.IGNORECASE))
    _COMPILED_PATTERNS.append((_pat, _compiled))


def match_cognitive_patterns(question: str) -> List[str]:
    """Match cognitive patterns from the question text and return the matched pattern names.

    Example return value: ["counting", "spatial_location"]
    """
    matched = []
    for pat_def, compiled_list in _COMPILED_PATTERNS:
        for regex in compiled_list:
            if regex.search(question):
                matched.append(pat_def["name"])
                break  # a pattern only needs to match once
    return matched


def extract_pattern_keywords(questions: List[str], topk: int = 5) -> List[str]:
    """Extract cognitive-pattern-level keywords from a set of questions.

    Strategy:
      1. Match cognitive patterns for each question
      2. Count the occurrence frequency of each pattern
      3. Select the topk most frequent patterns as pattern keywords
      4. For each selected pattern, find the most specific indicator phrase that actually appears in the questions

    Return format: ["counting:how many", "spatial_location:where should"]
    """
    # Count the number of hits per pattern and the specific indicator hits
    pattern_counts: Counter = Counter()
    indicator_hits: Dict[str, Counter] = {}  # pattern_name -> {indicator: count}

    for q in questions:
        q_lower = q.lower().strip()
        for pat_def, compiled_list in _COMPILED_PATTERNS:
            pname = pat_def["name"]
            for i, regex in enumerate(compiled_list):
                if regex.search(q_lower):
                    pattern_counts[pname] += 1
                    if pname not in indicator_hits:
                        indicator_hits[pname] = Counter()
                    indicator_hits[pname][pat_def["indicators"][i]] += 1
                    break

    if not pattern_counts:
        return []

    # Select the most frequent patterns
    top_patterns = pattern_counts.most_common(topk)

    result = []
    for pname, count in top_patterns:
        # Must appear in at least 2 questions (relaxed to 1 when there are too few samples)
        min_count = 2 if len(questions) >= 4 else 1
        if count < min_count:
            continue
        # Find the most frequently hit indicator for this pattern
        if pname in indicator_hits:
            best_indicator = indicator_hits[pname].most_common(1)[0][0]
            result.append(f"{pname}:{best_indicator}")
        else:
            result.append(pname)

    return result[:topk]


# ============================================================
# Layer 2: transferable context-word extraction
# ============================================================

# Characteristics of non-transferable entity words (used for filtering)
_ENTITY_INDICATORS = {
    # Person names are usually capitalized and not in the common vocabulary
    # Specific entity words such as foods and objects usually appear only in a particular video
}

# Whitelist of transferable context words (scene-level words, reusable across videos)
TRANSFERABLE_CONTEXT_PATTERNS: Set[str] = {
    # Scenes
    "cooking", "cleaning", "homework", "birthday", "party", "shopping",
    "exercise", "meeting", "dinner", "breakfast", "lunch",
    # Actions (high-level abstraction)
    "robot get", "robot put", "robot wash", "robot pick",
    "should the robot", "robot place",
    # Relationships
    "care about", "familiar with", "good at", "busy at",
    # Attributes
    "favorite", "habit", "personality", "occupation", "profession",
}


def extract_context_keywords(questions: List[str],
                             all_questions_corpus: List[str] = None,
                             topk: int = 3) -> List[str]:
    """Extract transferable context-level keywords from questions.

    Transferability checks:
      1. Cannot be a proper noun (a capitalized word that is not at the sentence start)
      2. If all_questions_corpus is provided, the keyword must appear in questions from multiple different videos
      3. Prefer words in the TRANSFERABLE_CONTEXT_PATTERNS whitelist

    Args:
        questions: list of questions in the current skill cluster
        all_questions_corpus: global question corpus (used for cross-video verification), optional
        topk: maximum number of keywords to return
    """
    # Basic stop words
    stop = {
        "what", "who", "when", "where", "why", "how", "the", "is", "are",
        "was", "were", "does", "do", "did", "a", "an", "of", "in", "on",
        "at", "and", "or", "to", "for", "with", "this", "that", "these",
        "those", "it", "be", "by", "as", "from", "has", "have", "had",
        "about", "into", "than", "then", "his", "her", "him", "she", "he",
        "they", "them", "their", "its", "not", "no",
    }

    # Extract bigram candidates
    bigram_counter: Counter = Counter()
    for q in questions:
        tokens = re.findall(r'[a-z]+', q.lower())
        tokens = [t for t in tokens if t not in stop and len(t) > 2]
        for i in range(len(tokens) - 1):
            bi = f"{tokens[i]} {tokens[i+1]}"
            bigram_counter[bi] += 1

    # Prefer bigrams in the whitelist
    whitelist_hits = []
    for bi, count in bigram_counter.most_common():
        if count < 2:
            break
        if bi in TRANSFERABLE_CONTEXT_PATTERNS:
            whitelist_hits.append(bi)

    # If there are enough whitelist hits, return directly
    if len(whitelist_hits) >= topk:
        return whitelist_hits[:topk]

    # Otherwise select transferable ones from the high-frequency bigrams
    result = list(whitelist_hits)
    for bi, count in bigram_counter.most_common():
        if len(result) >= topk:
            break
        if count < 2 or bi in result:
            continue
        # Transferability check: must not contain a proper noun
        words = bi.split()
        # Check whether these words always start with an uppercase letter in the original questions (proper noun)
        is_entity = False
        for q in questions:
            q_words = q.split()
            for i, w in enumerate(q_words):
                if w.lower() in words and i > 0 and w[0].isupper():
                    is_entity = True
                    break
            if is_entity:
                break
        if not is_entity:
            result.append(bi)

    return result[:topk]


def generate_hybrid_keywords(questions: List[str],
                             all_questions_corpus: List[str] = None,
                             max_pattern: int = 3,
                             max_context: int = 2) -> List[str]:
    """Generate hybrid keywords: pattern-level + context-level.

    Return format: ["keyword:counting:how many", "keyword:robot get"]
    where pattern keywords carry a pattern prefix and context keywords do not.
    """
    # Layer 1: cognitive patterns
    pattern_kws = extract_pattern_keywords(questions, topk=max_pattern)

    # Layer 2: transferable context
    context_kws = extract_context_keywords(
        questions, all_questions_corpus, topk=max_context
    )

    # Combine: pattern keywords use the original indicator as the actual matching word,
    # but filter out indicators that are too short or too broad.
    # Blacklist of overly broad single-word keywords: although these words are indicators
    # of cognitive patterns, they have too little discriminative power on their own.
    _OVERLY_BROAD_SINGLE_WORDS = {
        "who", "what", "where", "when", "why", "how",
        "did", "does", "is", "was", "are", "were",
        "has", "have", "can", "could", "should",
        "more", "most", "better", "which",
    }

    result = []
    seen = set()
    for pk in pattern_kws:
        # pk format: "counting:how many". Routing keeps only the cognitive pattern name,
        # to avoid surface words like `keyword:in the` / `keyword:which` polluting trigger conditions.
        pattern_name = pk.split(":", 1)[0] if ":" in pk else pk
        pattern_name = normalize_cognitive_keyword(pattern_name)
        if pattern_name and pattern_name in COGNITIVE_PATTERN_NAMES and pattern_name not in seen:
            result.append(f"keyword:{pattern_name}")
            seen.add(pattern_name)

    for ck in context_kws:
        if is_low_value_keyword(ck):
            continue
        if ck not in seen:
            result.append(f"keyword:{ck}")
            seen.add(ck)

    return result
