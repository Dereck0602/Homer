# -*- coding: utf-8 -*-
"""
Prompt definitions dedicated to Plan D (following the prompt_generate_full_memory style).

Core changes:
  - Remove all voice_id / voice feature related content
  - Let the API model (e.g. Gemini) perform ASR via the video + audio channels and attribute speech to face_id on its own
  - Provide only face features as person references
  - Remove Equivalence recognition (since there is no independent voice_id)
  - Keep the detailed categorization structure of the original prompt_generate_full_memory
"""

prompt_generate_memory_simplified = """You are given a video (with its original audio track) along with a set of face features. Each face feature is represented by a cropped face image or a video frame with a bounding box, identified by a unique ID enclosed in angle brackets (e.g. <face_0>, <face_1>). Some face features may belong to the same character across different clips.

**Important**: You must listen carefully to the audio track of the video. Perform Automatic Speech Recognition (ASR) on the audio, and attribute each spoken utterance to the corresponding <face_X> based on visual-audio cues such as:
  - Lip movement alignment
  - Camera focus and framing
  - Spatial position of speakers
  - Turn-taking patterns in conversation

If you cannot confidently determine which face is speaking, use a descriptive phrase (e.g., "an off-screen speaker") instead of guessing.

Your Tasks (produce both in the same response):

1. **Episodic Memory** (the ordered list of atomic captions)
   Using the provided face IDs, generate a detailed and cohesive description of the current video clip. The description should capture the complete set of observable and inferable events in the clip. Your output should incorporate the following categories (but is not limited to them):
    (a) Characters' Appearance: Describe the characters' appearance, such as their clothing, facial features, or any distinguishing characteristics.
    (b) Characters' Actions & Movements: Describe specific gesture, movement, or interaction performed by the characters.
    (c) Characters' Spoken Dialogue: Quote—or, if necessary, summarize—what is spoken by the characters. Attribute each utterance to the corresponding <face_X>.
    (d) Characters' Contextual Behavior: Describe the characters' roles in the scene or their interaction with other characters, focusing on their behavior, emotional state, or relationships.

2. **Semantic Memory** (the ordered list of high-level thinking conclusions)
   Produce concise, high-level reasoning-based conclusions across four categories:
    (a) Character-level Attributes – Infer abstract attributes for each character, such as: Name (if explicitly stated), Personality (e.g., confident, nervous), Role/profession (e.g., host, newcomer), Interests or background (when inferable), Distinctive behaviors or traits (e.g., speaks formally, fidgets). Avoid restating visual facts—focus on identity construction.
    (b) Interpersonal Relationships & Dynamics – Describe the relationships and interactions between characters: Roles (e.g., host-guest, leader-subordinate), Emotions or tone (e.g., respect, tension), Power dynamics (e.g., who leads), Evidence of cooperation, exclusion, conflict, etc.
    (c) Video-level Plot Understanding – Summarize the scene-level narrative, such as: Main event or theme, Narrative arc or sequence (e.g., intro → discussion → reaction), Overall tone (e.g., formal, tense), Cause-effect or group dynamics.
    (d) Contextual & General Knowledge – Include general knowledge that can be learned from the video, such as: Likely setting or genre (e.g., corporate meeting, game show), Cultural/procedural norms, Real-world knowledge (e.g., "Alice market is pet-friendly"), Common-sense or format conventions.

Strict Requirements (apply to both sections unless noted):

1. If a character has a provided face ID, refer to that character only with the ID (e.g. <face_0>, <face_1>).
2. If no ID exists, use a short descriptive phrase (e.g. "a man in a blue shirt").
3. Do not use "he," "she," "they," pronouns, or invented names to refer to characters.
4. Keep face IDs consistent throughout.
5. Describe only what is grounded in the video or obviously inferable.
6. Include natural time and location cues and setting hints when inferable.
7. Each Episodic Memory line must express one event/detail; split sentences if needed.
8. Output English only.

Additional Rules for Episodic Memory:
1. Do not mix unrelated aspects in one memory sentence.
2. Focus on appearance, actions/movements, spoken dialogue (quote or summary), contextual behavior.

Additional Rules for Semantic Memory:
1. Do not repeat simple surface observations already in the episodic memory.
2. Provide only final conclusions, not reasoning steps.

Expected Output Format:

Return the result as a single JSON object containing exactly two keys:

{
  "episodic_memory": [
    "In the bright conference room, <face_0> enters confidently, giving a professional appearance as <face_0> approaches <face_1> to shake hands.",
    "<face_0> wears a black suit with a white shirt and tie, has short black hair and wears glasses.",
    "<face_1> is dressed in a striking red dress with long brown hair.",
    "<face_1> smiles warmly and greets <face_0>, then sits down at the table, glancing at a phone briefly while occasionally looking up.",
    "<face_0> says to the group: 'Good afternoon, everyone. Let us begin the meeting.'",
    "<face_1> listens attentively to <face_0>'s words, nodding in agreement. The atmosphere is professional, with the participants settling into their roles for the meeting.",
    "<face_0> adjusts a tie and begins discussing the agenda, engaging the participants in a productive conversation."
  ],
  "semantic_memory": [
    "<face_0>'s name is David.",
    "<face_0> holds a position of authority, likely as the meeting organizer or a senior executive.",
    "<face_1> shows social awareness and diplomacy, possibly experienced in client-facing roles.",
    "<face_0> demonstrates control and composure, suggesting a high level of professionalism and confidence under pressure.",
    "The interaction between <face_0> and <face_1> suggests a working relationship built on mutual respect.",
    "The overall tone of the meeting is structured and goal-oriented, indicating it is part of a larger organizational workflow."
  ]
}

Please only return the valid JSON object (which starts with "{" and ends with "}") containing two string lists in "episodic_memory" and "semantic_memory", without any additional explanation or formatting."""


prompt_generate_memory_simplified_no_face = """You are given a video (with its original audio track). No face features are available for this clip.

**Important**: You must listen carefully to the audio track of the video. Perform Automatic Speech Recognition (ASR) on the audio, and describe all speakers using short descriptive phrases (e.g., "a man in a blue shirt", "the narrator").

Your Tasks (produce both in the same response):

1. **Episodic Memory** (the ordered list of atomic captions)
   Generate a detailed and cohesive description of the current video clip. The description should capture the complete set of observable and inferable events in the clip. Your output should incorporate the following categories (but is not limited to them):
    (a) Characters' Appearance: Describe the characters' appearance, such as their clothing, facial features, or any distinguishing characteristics.
    (b) Characters' Actions & Movements: Describe specific gesture, movement, or interaction performed by the characters.
    (c) Characters' Spoken Dialogue: Quote—or, if necessary, summarize—what is spoken by the characters.
    (d) Characters' Contextual Behavior: Describe the characters' roles in the scene or their interaction with other characters, focusing on their behavior, emotional state, or relationships.

2. **Semantic Memory** (the ordered list of high-level thinking conclusions)
   Produce concise, high-level reasoning-based conclusions across four categories:
    (a) Character-level Attributes – Name (if stated), personality, role/profession, interests, distinctive behaviors.
    (b) Interpersonal Relationships & Dynamics – Roles, emotions, power dynamics, cooperation/conflict.
    (c) Video-level Plot Understanding – Main event, narrative arc, overall tone, cause-effect.
    (d) Contextual & General Knowledge – Setting, cultural norms, real-world knowledge.

Strict Requirements:
1. Use short descriptive phrases to refer to characters.
2. Do not use "he," "she," "they," pronouns alone without context.
3. Describe only what is grounded in the video or obviously inferable.
4. Include natural time and location cues when inferable.
5. Each Episodic Memory line must express one atomic event/detail.
6. Do not repeat surface observations in Semantic Memory.
7. Output English only.

Expected Output Format:

Return the result as a single JSON object containing exactly two keys:

{
  "episodic_memory": [
    "..."
  ],
  "semantic_memory": [
    "..."
  ]
}

Please only return the valid JSON object (which starts with "{" and ends with "}") containing two string lists in "episodic_memory" and "semantic_memory", without any additional explanation or formatting."""
