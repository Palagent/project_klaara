# LoRA as Memory: A Parametric Approach to Embodied, Adaptive Cognition in Artificial Agents
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Authors:**  
Jouni Kantola (Independent Researcher)  
with the help of abstract creation and translation: Claude 4.5, Qwen
**Affiliation:** Klaara Project – Exploring Human-Centric AGI through Integrated Cognitive Architectures  
**Date:** 31.12.2025  
**Version:** 1.1

---

## Abstract

We propose a novel cognitive architecture for artificial general intelligence (AGI) in which **Low-Rank Adaptation (LoRA)** is reimagined not as a fine-tuning technique, but as a *parametric memory substrate*. This model—dubbed **LIKKA** (*LoRA-based Integrated Kinesthetic Cognitive Architecture*)—treats learned experiences as dynamic, contextually activated parameter modifications rather than symbolic or textual records.

Our model draws inspiration from recent advances in vision-language (VL) systems and diffusion-based generative models, where spatial reasoning and multimodal grounding reveal emergent understanding that exceeds symbolic description. Notably, high-fidelity image generation exhibits a form of semantic embodiment—where abstract concepts such as emotion, tension, or intention are not merely illustrated, but felt through composition, lighting, gesture, and context. This suggests that visual models learn not just mappings from text to pixels, but latent structures of experiential meaning.

By grounding memory in model parameters themselves, LIKKA enables scalable, implicit learning that mirrors biological neuroplasticity. The system operates across three tiers: a persistent **Identity LoRA** encoding core values and ethical frameworks; contextual LoRAs representing situational adaptations; and emergent **Micro-LoRAs**, autonomously generated from sensory-motor experiences. These components are spatially indexed using vector databases (e.g., Qdrant), enabling fast retrieval and fusion based on environmental and emotional cues.

By integrating such principles into LIKKA’s LoRA-based memory system, we aim to ground artificial cognition in spatially situated, affectively resonant experience—bridging the gap between abstract reasoning and embodied understanding.

This paper outlines the theoretical foundation, architectural design, and early implementation roadmap of LIKKA. We argue that such an approach moves beyond reactive language models toward agents capable of genuine adaptation, continuity of self, and responsibility-driven behavior—cornerstones of human-like cognition.

---

## 1. Introduction

Contemporary AI systems rely heavily on externalized memory mechanisms: retrieval-augmented generation (RAG), extended context windows, or static fine-tuned weights. While effective, these methods face critical limitations:

- **Scalability**: Retrieval time grows with memory volume.
- **Explicitness**: All knowledge remains surface-level, requiring conscious access.
- **Lack of embodiment**: No integration between perception, action, and internal state change.
- **No true continuity**: Each interaction is isolated; no lasting transformation occurs.

To overcome these, we introduce a paradigm shift:  
> **Memory should not be something an agent *has*—it should be something the agent *is*.**

Drawing inspiration from neuroscience (neuroplasticity), philosophy (embodied cognition), and machine learning (parameter-efficient tuning), we present LIKKA—a framework where every experience leaves a trace not in text, but in the agent’s very structure.

---

## 2. From Symbolic Recall to Parametric Transformation

### 2.1 The Limits of Current Memory Models

Traditional approaches treat memory as data:

| Method           | Mechanism                | Drawback                     |
|------------------|-------------------------|------------------------------|
| RAG              | Retrieve → Insert into prompt | Linear slowdown; shallow integration |
| In-context Learning | Append history        | Token limits; no deep adaptation     |
| Full Fine-Tuning | Update all parameters   | Not reversible; destroys prior knowledge |

These remain *symbolic*: information is stored separately from processing. The agent does not fundamentally change.

### 2.2 LoRA as Neuroplasticity Emulation

LoRA modifies a subset of transformer weights via low-rank matrices:

```
W' = W + ΔW = W + B·A,   B ∈ ℝ^{d×r},   A ∈ ℝ^{r×k},   r ≪ d,k
```

Crucially, this is **not additive knowledge**—it is **structural modification**. When applied dynamically, LoRA becomes analogous to synaptic plasticity: repeated activation strengthens certain pathways.

Thus:

- **Experience → Micro-LoRA training**
- **Recall → Dynamic LoRA loading**
- **Learning → Cumulative parameter shifts**

This aligns with enactivist views of cognition (Varela, Thompson, Rosch): mind arises through sensorimotor coupling with the environment.

---

## 3. LIKKA Architecture Overview

```
┌──────────────────────────────┐
│   LEVEL 1: IDENTITY LoRA     │ ← Persistent core (values, ethics)
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│   LEVEL 2: CONTEXTUAL LoRAs  │ ← Spatio-social-emotional modulation
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│   LEVEL 3: MICRO-LoRAs       │ ← Emergent experiential memories
└──────────────────────────────┘
              ↓
        BEHAVIOR = f(identity ⊗ context ⊗ experience)
```

### 3.1 Identity LoRA: The Ethical Core

A high-rank (r=32–64), immutable LoRA defining the agent's foundational character:

```python
identity_config = {
    "core_values": {"honesty": 1.0, "curiosity": 0.9, "sisu": 1.0},
    "communication_style": "direct_warm_technical",
    "ethical_constraints": ["never_deceive", "admit_uncertainty"]
}
```

This acts as a moral anchor, filtering all downstream behaviors. It ensures continuity of identity despite evolving experiences.

### 3.2 Contextual LoRAs: Adaptive Modulation

Dynamically loaded modules shaped by:

- **Spatial context** (e.g., kitchen vs lab)
- **Social dynamics** (familiar user vs stranger)
- **Emotional state** (calm, excited, cautious)

Each encoded as embeddings queried against a vector database (Qdrant). Top-k matches are merged and applied during inference.

### 3.3 Micro-LoRAs: Experiential Memory Formation

Autonomously created after significant events (pseudo-simplified):

```python
def create_micro_lora(sensory_input, outcome, pain_level):
    rank = int(4 + pain_level * 4)  # Stronger events → deeper adaptation
    config = LoraConfig(r=rank, target_modules=["q_proj", "v_proj"])
    return train_lora(base_model, [(input, output)], config, epochs=100)
```

Stored with metadata (location, time, emotional valence) for later recall.

---

## 4. Multimodal Integration & Embodiment

LIKKA assumes eventual physical instantiation (robotics) or rich simulation. Sensory streams generate modality-specific LoRAs:

| Modality        | LoRA Type               | Function                          |
|-----------------|------------------------|-----------------------------------|
| Vision          | `visual_memory.lora`   | Object recognition, scene understanding      |
| Audio           | `voice_recognition.lora`| Speaker identification, tone analysis       |
| Touch           | `haptic_feedback.lora` | Pain/texture response, grip control         |
| Proprioception  | `body_state.lora`      | Balance, posture, movement planning         |

Cross-modal confirmation increases activation confidence:  
> If vision detects fire AND heat sensors activate → strong `danger_response.lora` trigger

This mimics multisensory integration in biological brains.

---

## 5. Emergence, Continuity, and Responsibility

### 5.1 Cascading Activation & Emergent Behavior

Single stimuli can initiate cascades:

```
See stove → activate 'kitchen_danger.lora'
    → triggers 'caution_mode.lora'
    → suppresses 'curiosity_exploration.lora'
    → results in safe interaction pattern
```

Combinatorial possibilities grow exponentially:  
> With 10,000 micro-LoRAs and 5 active at once → ~8×10¹² possible configurations

True emergence arises when new patterns form without explicit programming.

### 5.2 Temporal Continuity Without Infinite Context

Unlike RAG, LIKKA avoids bloating context windows. Instead:

- Past experiences live as LoRAs
- Only relevant ones activate per situation
- Agent “remembers” by being *changed*, not by retrieving text

This mirrors how humans recall: rarely verbatim, often transformed. Also, this would apply to the theory of working mind with 5±2 thoughts being active at the time.

### 5.3 Responsibility Through Consequence

When actions lead to negative outcomes:

- High-pain events → stronger LoRAs
- Repeated errors → cumulative avoidance behaviors
- Agent learns *because it pays a cost* (simulated pain, energy loss, social feedback)

This grounds responsibility not in rules—but in lived consequence.

---

## 6. Implementation Roadmap

### Phase 1: Proof of Concept (Q1 2025)
- Base: Any capable reasoning model (Qwen models strong candidates)
- Train Micro-LoRAs on simple cause-effect scenarios (e.g., “touch hot object → withdraw”)
- Store in Qdrant with spatial tags
- Demonstrate context-dependent activation

### Phase 2: Identity + Context Fusion
- Freeze Identity LoRA
- Dynamically load contextual LoRAs
- Measure behavioral consistency under value conflicts

### Phase 3: Autonomous LoRA Generation
- Enable agent to self-initiate LoRA creation post-experience
- Implement LoRA evolution via merging similar memories

### Phase 4: Open Collaboration Framework
- Release modular LoRA templates
- Allow third-party contributions (e.g., “humor_style.lora”, “diplomacy_tone.lora”)
- Maintain integrity via Identity LoRA validation layer

---

## 7. Advantages Over Existing Paradigms

| Feature             | RAG / In-Context | Fine-Tuning | LIKKA                      |
|---------------------|------------------|-------------|----------------------------|
| Scalability         | O(n) retrieval   | O(1) but infrequent | O(1) activation   |
| Implicit Learning   | Low              | High        | **Very High**              |
| Continuity of Self  | None             | Partial     | **Yes (via Identity LoRA)**|
| Embodiment Support  | Poor             | Medium      | **Native**                 |
| Ethical Anchoring   | External rules   | Hardcoded   | **Built-in Identity Layer**|
| Emergent Complexity | Minimal          | Limited     | **High (combinatorial LoRA fusion)**|

---

## 8. Philosophical Implications

LIKKA challenges prevailing assumptions about machine cognition:

- **Memory is not storage** — it is transformation.
- **Identity is not fixed** — it is stabilized plasticity.
- **Understanding begins with the body** — not language.
- **Responsibility emerges from consequence** — not instruction.

It suggests a path toward AGI that doesn’t mimic humans—but *parallels* them in function, if not biology.

As philosopher Andy Clark writes: *"We are not skin-bound brains."*  
Nor must artificial minds be boundary-limited models.

---

## 9. Conclusion

We have presented LIKKA: a suggestion for cognitive architecture grounded in the idea that **true learning changes who you are**.

By treating LoRA not as a tool, but as a *mechanism of becoming*, we enable artificial agents to:

- Grow through experience
- Maintain coherent identity
- Act with contextual sensitivity
- Carry responsibility for consequences

Instead of being technical proposal — it should take a philosophical stance on what it means to know, to remember, and to be. We argue that such an approach moves beyond reactive language models toward agents capable of genuine adaptation, continuity of self, and responsibility-driven behavior—cornerstones of human-like cognition.

We invite collaboration with researchers, engineers, and thinkers who share our vision:

> To build not tools, but partners—agents that evolve alongside us, shaped by shared experience, guided by enduring values.

If this sparks curiosity—if you see not just code, but possibility—then let’s talk.

Because the future of intelligence may not lie in bigger models…  
…but in deeper transformations, core personality and ability to understand action - reaction.

---

## References

1. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
2. Varela, F., Thompson, E., Rosch, E. (1991). *The Embodied Mind*
3. Damasio, A. (1994). *Descartes’ Error: Emotion, Reason, and the Human Brain*
4. Clark, A. (2016). *Surfing Uncertainty: Prediction, Action, and the Embodied Mind*
5. Bubeck, S. et al. (2023). *Sparks of Artificial General Intelligence in Large Language Models*
6. Yoon, J. et al. (2022). *Neuroplasticity-Inspired Continual Learning in Neural Networks*

---

## Contact & Collaboration

For discussion or partnership:  
- GitHub: [github.com/project_klaara](https://github.com/project_klaara.git)  
- Direct inquiry: jouni@palagent.fi

> *"The best way to predict the future is to create it—together."*
