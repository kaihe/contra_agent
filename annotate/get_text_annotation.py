"""
Text annotation pipeline: Gemini video analysis + Gemma text embeddings.

Public API
----------
    narrative, macros = annotate_video(video_path, api_key, fps)
    embed_text_annotations(va, proto_path)
"""

import json
import os
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from annotate.proto.video_annotation_pb2 import VideoAnnotation  # type: ignore

MODEL          = "gemini-3-flash-preview"
GEMMA_ST_MODEL = "google/embeddinggemma-300m"  # SentenceTransformer producing [1, 768] embeddings
_gemma_st: SentenceTransformer | None = None

_PROMPT_FILE       = os.path.join(os.path.dirname(__file__), "annotation_prompt.txt")
_ANNOTATION_PROMPT: str | None = None


@dataclass
class TextAnnotation:
    frame_idx:   int
    instruction: str
    duration:    float
    provider:    str
    version:     str


def _get_prompt() -> str:
    global _ANNOTATION_PROMPT
    if _ANNOTATION_PROMPT is None:
        with open(_PROMPT_FILE) as f:
            _ANNOTATION_PROMPT = f.read()
    return _ANNOTATION_PROMPT


def _get_gemma_st() -> SentenceTransformer:
    global _gemma_st
    if _gemma_st is None:
        _gemma_st = SentenceTransformer(GEMMA_ST_MODEL)
    return _gemma_st


def _mmss_to_seconds(ts: str) -> float | None:
    if ts is None:
        return None
    m = re.match(r"^(\d+):(\d{2})$", ts.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def annotate_video(video_path: str, api_key: str, fps: int) -> tuple[str, list[dict]]:
    """Upload video to Gemini and parse macro instructions + narrative.

    Returns:
        narrative: str
        macros: list of {instruction, start_frame, end_frame}
    """
    client = genai.Client(api_key=api_key)

    print(f"  Uploading {os.path.basename(video_path)}...", flush=True)
    video_file = client.files.upload(
        file=video_path,
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"File upload failed: {video_file.state.name}")

    print(f"  Querying Gemini ({MODEL})...", flush=True)
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(parts=[
            types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type="video/mp4")),
            types.Part(text=_get_prompt()),
        ])],
    )
    client.files.delete(name=video_file.name)

    raw = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
    raw = re.sub(r"\s*```$", "", raw)
    parsed = json.loads(raw)

    macros = []
    for entry in parsed.get("macro_instructions", []):
        start_s = _mmss_to_seconds(entry.get("start"))
        end_s   = _mmss_to_seconds(entry.get("end"))
        macros.append({
            "instruction": entry["instruction"],
            "start_frame": round(start_s * fps) if start_s is not None else 0,
            "end_frame":   round(end_s   * fps) if end_s   is not None else None,
        })

    narrative = parsed.get("narrative", "")
    print(f"  Got {len(macros)} macro instruction(s).")
    return narrative, macros


def embed_text_annotations(va: VideoAnnotation, proto_path: str) -> None:
    """Encode each FrameTextAnnotation instruction with Gemma SentenceTransformer
    and write the embeddings back into the proto file at proto_path."""
    model = _get_gemma_st()
    print("  Embedding text annotations with Gemma...", flush=True)

    for fa in va.frame_annotations:
        for fta in fa.frame_text_annotation:
            vec = model.encode(fta.instruction, convert_to_numpy=True)  # (768,)
            embedding = vec.reshape(1, -1)  # (1, 768)

            tokenizer_map = fta.text_embedding_dict[MODEL]
            tensor = tokenizer_map.text_embeddings["gemma"]
            del tensor.shape[:]
            del tensor.values[:]
            tensor.shape.extend(list(embedding.shape))
            tensor.values.extend(embedding.flatten().tolist())

    with open(proto_path, "wb") as f:
        f.write(va.SerializeToString())
