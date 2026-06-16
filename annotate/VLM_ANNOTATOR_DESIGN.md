# VLM Annotator — Design

Goal: turn the pruned, level-clearing Contra traces into **think+act VLA
samples** of the form

```
[ 8 image frames ]  →  [ text plan for the next move ]  →  [ 8 actions ]
```

The 8 actions are ground truth (from the trace). The 8 frames are ground
truth (replayed). The only thing we have to *manufacture* is the **text plan**.
This doc specifies how `annotate/vlm_annotator.py` produces that plan with the
VLM service in `annotate/serve_vlm.sh`, and how the result feeds the dataset
builder.

---

## 1. The key idea: hindsight annotation

A plan is a description of *what the player is about to do and why*. At training
time the model only sees the **input frames** and must emit the plan before it
has seen the consequences. But to *write a good label* we are allowed to cheat:
we are a teacher with hindsight.

So to label the plan for a chunk of 8 actions we show the VLM the **frames of
that chunk as it actually unfolds** (the player moving, jumping, the enemy
dying) and ask it to *narrate the movement plan that produced this*. The VLM is
not guessing the future — it is describing footage it can see, which makes the
label accurate and grounded. The trained policy never gets this hindsight; it
learns to *predict* the plan from the preceding frames alone.

**Decision (revised after a level-1 A/B): hybrid grounding.** We first tried
frames-only. On the 2B model the plans were fluent but (a) invented shooting in
~50% of chunks where no fire was pressed and (b) occasionally flipped left/right,
even though Contra only advances rightward. So we now feed a **factual one-line
button summary** (`--action_hint`) into the *Move* sentence only — direction,
jump count, and fire-steps decoded from the real inputs. Terrain and Threats
stay pure-vision (game-agnostic); only the action-intent clause is disciplined by
ground truth, which is legitimate hindsight since we *know* the inputs.

A/B on the first 14 chunks of `win_level1_202603301145`: phantom-shoot chunks
fell 7→2; residual left-slips (3) are now confined to the Terrain clause, not
the Move. Decoded per-step action tokens are still written to the output JSON
regardless, for inspection/validation.

---

## 2. Indexing and sample boundaries

A trace is `actions[0..N-1]`, shape `(N, 9)` uint8, plus an `initial_state`.

Replaying through the emulator (IMAGE obs, `SKIP=3`) yields one logical frame
per action step. Define `F[i]` = the screen at step `i`, i.e. the state the
player observes *before* committing `actions[i]`. Replay also gives the final
frame `F[N]`.

We cut the trace into **non-overlapping chunks of `CHUNK = 8` actions**. Chunk
`k` covers steps `[8k, 8k+8)`.

```
        chunk k-1                chunk k
   ┌───────────────────┐   ┌───────────────────┐
F: … F[8k-8] … F[8k-1]  F[8k] … F[8k+7]  F[8k+8] …
        observation         actions+plan label
```

For one **training instance** the dataset builder pairs:

| field            | source                                              |
|------------------|-----------------------------------------------------|
| input frames (8) | `F[8k-8 .. 8k)` — the previous chunk's frames        |
| (optional) prev plan | `plan_{k-1}`                                     |
| **plan label**   | `plan_k`  ← produced by this annotator               |
| action labels (8)| `actions[8k .. 8k+8)`                                |

So one instance spans 16 action steps. The annotator's *only* job is to emit
`plan_k` for every chunk `k`. The input/output pairing and frame downsampling
(192px etc.) stay in the dataset builder (`vla/datasets/`), exactly as today.

**Annotation window for `plan_k`:** show the VLM `F[8k .. 8k+8]` — the 8 step
frames plus the post-chunk frame `F[8k+8]` so the result of the last action is
visible (9 images, well under the server's 12-image limit). Frames only; no
action tokens in the prompt (see §1).

---

## 3. Pipeline

```
clear_trace/*.npz ──replay(IMAGE)──► per-step frames F[0..N]
                                        │
                for each 8-action chunk │  frames F[8k..8k+8]  +  decoded actions[8k..8k+8)
                                        ▼
                            VLM service (Qwen3-VL @ :8000)
                                        │  one short plan sentence
                                        ▼
                        annotate/clear_trace_plans/<trace>.json
```

Reuse `contra.replay`: one IMAGE env per process (stable_retro allows only one),
`rewind_state` to the trace's `initial_state`, then step `SKIP` times per action
capturing the pre-action screen — same loop as `replay_actions(want_video=True)`.
Collect all frames first, then iterate chunks and call the VLM. Mirror
`ram_annotator`'s `--trace` / `--trace_dir` reuse of a single env.

---

## 5. VLM prompt design

A first try ("one short sentence, what's on screen + the plan") gave low-variance,
under-descriptive output — the 2B model fell back to stock phrasing like "clear
the soldiers". Asking openly for more detail then produced run-on paragraphs that
degenerated into repeated lines. The structure below fixes both.

**System.** Three fixed sentences, one per line, each capped, so the output is
descriptive but bounded:

> You are an expert Contra (NES) player narrating your tactical read of the
> battlefield … Reply with EXACTLY three short sentences, one per line:
> 1. **Terrain** — the ground ahead and any pit/gap/water/cliff edge to jump over
>    or avoid falling into.
> 2. **Threats** — which enemies/turrets are where, and any incoming bullets to
>    dodge (say "no enemies on screen" if none).
> 3. **Move** — where to run or jump and what to shoot to get through safely.
> Be concrete about positions; describe only what is visible; do not invent
> enemies, bullets or shooting. Each sentence < 20 words.

**User.** The 9 frames as base64 PNG `image_url` parts in order + the instruction.
With `--action_hint` (the chosen default, see §1), a factual button summary is
appended and the model is told to make the **Move** sentence consistent with it
(rightward-only, fire/no-fire). Terrain/Threats remain pure-vision. Zero-shot —
no RAM-derived few-shots.

**Decoding.** `temperature ≈ 0.2`, `max_tokens ≈ 110`, `frequency_penalty ≈ 0.4`
(the penalty kills the repetition loops the open-ended prompt produced). The
three lines are joined into one space-separated `plan` string in the output.

Output style target (Terrain / Threats / Move):

> "Terrain: a cliff edge drops into water with a gap to jump. Threats: a red
> turret on the right, a projectile falling from above. Move: jump the gap to the
> right and keep advancing."

---

## 6. VLM service

`annotate/serve_vlm.sh` already serves an OpenAI-compatible API:

- endpoint `http://localhost:8000/v1`, model name **`annotator`**
- default checkpoint `tmp/models/Qwen3-VL-2B-Instruct-FP8`
- `--limit-mm-per-prompt '{"image": 12}'` → our 9 frames/call fit
- `--max-model-len 8192`

Client: the `openai` Python SDK (v2 installed), `base_url` + a dummy api key.
Frames encoded as PNG → base64 `data:` URLs. Add `--base_url` / `--model` CLI
flags so a remote/alternate server can be swapped in. The GPU is exclusive while
serving — annotation and training must not run at the same time.

Throughput note: ~19k chunks at chunk=10 today → ~24k chunks at chunk=8. One
synchronous request per chunk is slow; batch with a bounded thread pool
(`--concurrency`, default ~8) and write each trace's JSON when done so the job is
resumable (skip traces whose output already exists).

---

## 7. Output format

Mirror `ram_annotator` so the two are interchangeable downstream. One JSON per
trace in `annotate/clear_trace_plans/`:

```json
{
  "trace": "win_level1_202603301145.npz",
  "chunk": 8,
  "model": "annotator",
  "annotations": [
    {
      "t": 0,
      "plan": "No threats yet — push right and fire to scout the path ahead.",
      "actions": ["R+F", "R+F", "R+_", "R+J", "R+J", "R+F", "R+F", "R+F"]
    }
  ]
}
```

Keep the `t` (chunk start step) and a readable `actions` list for inspection.
We drop `ram_annotator`'s `facts` block (it is RAM-derived); validation instead
compares against the RAM annotations file for the same trace.