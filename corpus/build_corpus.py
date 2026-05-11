# corpus/build_corpus.py
"""Build ALT entity corpus embeddings from action labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Optional

import torch
import torch.nn.functional as F
from pathlib import Path


def _install_package(pip_spec: str) -> None:
    """Install a missing package into the current Python environment."""
    cmd = [sys.executable, "-m", "pip", "install", pip_spec]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install dependency: {pip_spec}") from exc


def _ensure_package(import_name: str, pip_spec: str) -> None:
    """Ensure a Python package can be imported, installing if missing."""
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing dependency: {pip_spec}")
        _install_package(pip_spec)


def _ensure_spacy_model(model_name: str = "en_core_web_sm") -> None:
    """Ensure a spaCy language model is available."""
    import spacy
    try:
        spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        print(f"Downloading spaCy model: {model_name}")
        download(model_name)
        spacy.load(model_name)


def _ensure_wordnet_data() -> None:
    """Ensure WordNet corpora are installed and loadable."""
    import nltk
    from nltk.corpus import wordnet as wn

    download_dir = Path.home() / "nltk_data"
    download_dir.mkdir(parents=True, exist_ok=True)
    if str(download_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(download_dir))
    for pkg in ("wordnet", "omw-1.4"):
        print(f"Ensuring NLTK resource: {pkg}")
        ok = nltk.download(pkg, download_dir=str(download_dir), quiet=False)
        if not ok:
            cmd = [sys.executable, "-m", "nltk.downloader", "-d", str(download_dir), pkg]
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Failed to download NLTK resource: {pkg}") from exc

    # Validate by actually loading from the corpus API.
    try:
        _ = wn.synsets("person")
    except LookupError as exc:
        raise RuntimeError(
            f"WordNet is not loadable from NLTK paths: {nltk.data.path}"
        ) from exc


def normalize_label(label: str) -> str:
    """Split concatenated and camelCase labels into spaced lowercase text."""
    _ensure_package("wordninja", "wordninja")
    import wordninja

    normalized = label.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", normalized)

    pieces: list[str] = []
    for token in normalized.split():
        if any(char.isalpha() for char in token):
            pieces.extend(wordninja.split(token.lower()))
        else:
            pieces.append(token.lower())
    return " ".join(pieces).strip()

def get_entity_noun(label: str, nlp: Any) -> str:
    """Extract main noun phrase from action label."""
    doc = nlp(normalize_label(label))
    nouns = [t.text for t in doc if t.pos_ in ('NOUN', 'PROPN')]
    if nouns:
        return nouns[-1]   # last noun = object of the action
    for t in doc:
        if t.dep_ == 'ROOT':
            return t.text
    return label.split()[-1]

def get_wordnet_def(word: str, context: Optional[str] = None) -> str:
    """Return WordNet definition, optionally disambiguated via Lesk."""
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk

    if context:
        synset = lesk(context.split(), word, pos=wn.NOUN) or lesk(context.split(), word)
        if synset is not None:
            return synset.definition()

    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(word)
    if synsets:
        return synsets[0].definition()
    return word

def is_action_related(texts, tokenizer, model, device: str) -> list[bool]:
    """Binary filter using T5 text2text prompt for action relevance."""
    prompts = [f"question: Is this related to a human action? context: {text} answer:" for text in texts]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**encoded, max_new_tokens=2)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.strip().lower().startswith("yes") for d in decoded]


def _load_labels(label_file: str) -> list[str]:
    """Load labels from either {'labels': [...]} or flat list JSON."""
    with open(label_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        labels = payload.get("labels")
    elif isinstance(payload, list):
        labels = payload
    else:
        labels = None
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"Unsupported label file format: {label_file}")
    return labels


def _load_body_parts(body_parts_file: str) -> list[str]:
    """Load body part list from a JSON file."""
    if not body_parts_file:
        return []
    with open(body_parts_file, "r", encoding="utf-8") as handle:
        parts = json.load(handle)
    if not isinstance(parts, list) or not all(isinstance(x, str) for x in parts):
        raise ValueError(f"Unsupported body parts file format: {body_parts_file}")
    return [p.strip() for p in parts if p.strip()]


def _load_expansion_cache(cache_path: str) -> dict[str, list[str]]:
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return {
        str(k): [str(v).strip() for v in vals if str(v).strip()]
        for k, vals in payload.items()
        if isinstance(vals, list)
    }


def _save_expansion_cache(cache_path: str, cache: dict[str, list[str]]) -> None:
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)


def _parse_llm_list(text: str) -> list[str]:
    cleaned = text.replace("\n", ",").replace(";", ",")
    cleaned = re.sub(r"\b\d+\.", ",", cleaned)
    pieces = [p.strip(" -\t\r\n") for p in cleaned.split(",")]
    return [p for p in pieces if p]


def _is_valid_expansion(candidate: str, noun: str) -> bool:
    lowered = candidate.strip().lower()
    if not lowered or lowered == noun.lower():
        return False
    if len(lowered) < 2 or len(lowered) > 40:
        return False
    if not any(ch.isalpha() for ch in lowered):
        return False
    banned = ("list of", "noun", "phrase", "related", "comma", "return")
    if any(bad in lowered for bad in banned):
        return False
    return True


def _post_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> dict[str, Any]:
    _ensure_package("requests", "requests")
    import requests

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code < 400:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504):
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_for = float(retry_after)
                except ValueError:
                    sleep_for = base_sleep * (2 ** attempt)
            else:
                sleep_for = base_sleep * (2 ** attempt)
            time.sleep(sleep_for)
            continue
        response.raise_for_status()
    response.raise_for_status()
    return {}


def _expand_nouns_with_transformers(
    nouns: list[str],
    per_noun: int,
    model_name: str,
    cache_path: str,
) -> dict[str, list[str]]:
    """Expand nouns with a local transformers model."""
    _ensure_package("transformers", "transformers")
    _ensure_package("sentencepiece", "sentencepiece")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    raw_cache = _load_expansion_cache(cache_path)
    cache: dict[str, list[str]] = {}
    for noun, items in raw_cache.items():
        cleaned = [item for item in items if _is_valid_expansion(item, noun)]
        if cleaned:
            cache[noun] = cleaned
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    for noun in nouns:
        if noun in cache and len(cache[noun]) >= per_noun:
            continue
        prompt = (
            f"Generate {per_noun} distinct concrete nouns related to '{noun}'. "
            "Return only a comma-separated list of nouns or short noun phrases."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            num_beams=4,
        )
        expansions_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        expansions = [
            item for item in _parse_llm_list(expansions_text)
            if _is_valid_expansion(item, noun)
        ]
        existing = cache.get(noun, [])
        merged = []
        seen = {e.lower() for e in existing}
        for item in expansions:
            key = item.lower()
            if key and key not in seen and key != noun.lower():
                merged.append(item)
                seen.add(key)
        cache[noun] = existing + merged

    _save_expansion_cache(cache_path, cache)
    return cache


def _expand_nouns_with_openai(
    nouns: list[str],
    per_noun: int,
    model_name: str,
    cache_path: str,
    request_sleep: float = 0.2,
) -> dict[str, list[str]]:
    """Expand nouns with OpenAI Chat Completions API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAI LLM expansion.")

    raw_cache = _load_expansion_cache(cache_path)
    cache: dict[str, list[str]] = {}
    for noun, items in raw_cache.items():
        cleaned = [item for item in items if _is_valid_expansion(item, noun)]
        if cleaned:
            cache[noun] = cleaned

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://api.openai.com/v1/chat/completions"

    for noun in nouns:
        if noun in cache and len(cache[noun]) >= per_noun:
            continue
        prompt = (
            f"Generate {per_noun} distinct concrete nouns related to '{noun}'. "
            "Return only a comma-separated list of nouns or short noun phrases."
        )
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        data = _post_with_retries(url, headers, payload)
        content = data["choices"][0]["message"]["content"]
        expansions = [
            item for item in _parse_llm_list(content)
            if _is_valid_expansion(item, noun)
        ]
        existing = cache.get(noun, [])
        merged = []
        seen = {e.lower() for e in existing}
        for item in expansions:
            key = item.lower()
            if key and key not in seen and key != noun.lower():
                merged.append(item)
                seen.add(key)
        cache[noun] = existing + merged
        if request_sleep:
            time.sleep(request_sleep)

    _save_expansion_cache(cache_path, cache)
    return cache


def _expand_nouns_with_groq(
    nouns: list[str],
    per_noun: int,
    model_name: str,
    cache_path: str,
    request_sleep: float = 0.2,
) -> dict[str, list[str]]:
    """Expand nouns with Groq's OpenAI-compatible API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set; cannot use Groq LLM expansion.")

    raw_cache = _load_expansion_cache(cache_path)
    cache: dict[str, list[str]] = {}
    for noun, items in raw_cache.items():
        cleaned = [item for item in items if _is_valid_expansion(item, noun)]
        if cleaned:
            cache[noun] = cleaned

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://api.groq.com/openai/v1/chat/completions"

    for noun in nouns:
        if noun in cache and len(cache[noun]) >= per_noun:
            continue
        prompt = (
            f"Generate {per_noun} distinct concrete nouns related to '{noun}'. "
            "Return only a comma-separated list of nouns or short noun phrases."
        )
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        data = _post_with_retries(url, headers, payload)
        content = data["choices"][0]["message"]["content"]
        expansions = [
            item for item in _parse_llm_list(content)
            if _is_valid_expansion(item, noun)
        ]
        existing = cache.get(noun, [])
        merged = []
        seen = {e.lower() for e in existing}
        for item in expansions:
            key = item.lower()
            if key and key not in seen and key != noun.lower():
                merged.append(item)
                seen.add(key)
        cache[noun] = existing + merged
        if request_sleep:
            time.sleep(request_sleep)

    _save_expansion_cache(cache_path, cache)
    return cache


def expand_nouns_with_llm(
    nouns: list[str],
    per_noun: int,
    model_name: str,
    cache_path: str,
    provider: str = "transformers",
) -> dict[str, list[str]]:
    """Expand nouns with an LLM provider and return a map noun -> expansions."""
    if provider == "openai":
        return _expand_nouns_with_openai(nouns, per_noun, model_name, cache_path)
    if provider == "groq":
        return _expand_nouns_with_groq(nouns, per_noun, model_name, cache_path)
    if provider == "transformers":
        return _expand_nouns_with_transformers(nouns, per_noun, model_name, cache_path)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_corpus(
    label_file: str,
    output_dir: str,
    output_path: Optional[str] = None,
    device: str = "cpu",
    use_lesk: bool = True,
    use_t5_filter: bool = False,
    t5_model_name: str = "t5-small",
    expand_corpus: bool = False,
    llm_model_name: str = os.getenv("LLM_MODEL", "google/flan-t5-small"),
    llm_provider: str = os.getenv("LLM_PROVIDER", "transformers"),
    expansions_per_noun: int = 3,
    min_corpus_size: int = 500,
    body_parts_file: str = "corpus/body_parts.json",
    expansion_cache: str = "corpus/llm_expansions.json",
) -> None:
    _ensure_package("clip", "git+https://github.com/openai/CLIP.git")
    _ensure_package("spacy", "spacy")
    _ensure_package("nltk", "nltk")
    if use_t5_filter:
        _ensure_package("transformers", "transformers")
        _ensure_package("sentencepiece", "sentencepiece")

    import clip
    import spacy

    labels = _load_labels(label_file)
    _ensure_spacy_model("en_core_web_sm")
    _ensure_wordnet_data()
    nlp = spacy.load("en_core_web_sm")

    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    t5_tokenizer = t5_model = None
    if use_t5_filter:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)
        t5_model.eval()

    embeddings = []
    entities_log = []
    body_parts = _load_body_parts(body_parts_file) if expand_corpus else []

    normalized_labels = []
    nouns_by_label = []
    for label in labels:
        normalized_label = normalize_label(label)
        normalized_labels.append(normalized_label)
        nouns_by_label.append(get_entity_noun(normalized_label, nlp))

    unique_nouns = sorted({n for n in nouns_by_label if n})
    llm_expansions: dict[str, list[str]] = {}
    if expand_corpus and unique_nouns:
        needed = max(0, min_corpus_size - (len(labels) + len(body_parts)))
        per_noun = max(expansions_per_noun, math.ceil(needed / max(1, len(unique_nouns))))
        llm_expansions = expand_nouns_with_llm(
            unique_nouns,
            per_noun=per_noun,
            model_name=llm_model_name,
            cache_path=expansion_cache,
            provider=llm_provider,
        )

    def add_entry(entry: dict[str, str], seen: set[str]) -> None:
        key = entry["context"].strip().lower()
        if not key or key in seen:
            return
        entities_log.append(entry)
        seen.add(key)

    with torch.no_grad():
        seen_contexts: set[str] = set()
        for label, normalized_label, noun in zip(labels, normalized_labels, nouns_by_label):
            defn = get_wordnet_def(noun, context=normalized_label if use_lesk else None)
            # Use label + entity context
            entity_text = f"{noun}: {defn}" if defn != noun else noun
            if use_t5_filter and t5_model is not None and t5_tokenizer is not None:
                keep = is_action_related([entity_text], t5_tokenizer, t5_model, device=device)[0]
                if not keep:
                    entity_text = noun
            add_entry(
                {"label": label, "entity": noun, "context": entity_text, "source": "label"},
                seen_contexts,
            )

        for part in body_parts:
            add_entry(
                {"label": "body_part", "entity": part, "context": part, "source": "body_part"},
                seen_contexts,
            )

        for noun, expansions in llm_expansions.items():
            for expansion in expansions:
                add_entry(
                    {
                        "label": "llm_expansion",
                        "entity": noun,
                        "context": expansion,
                        "source": "llm_expansion",
                    },
                    seen_contexts,
                )

        for entry in entities_log:
            tokens = clip.tokenize([entry["context"]], truncate=True).to(device)
            emb = clip_model.encode_text(tokens)          # (1, 512)
            emb = F.normalize(emb.float(), dim=-1)
            embeddings.append(emb.cpu())

    S = torch.cat(embeddings, dim=0).half()               # (K, 512) fp16
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(output_path) if output_path else (out_dir / "corpus_embeddings.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(S, save_path)
    with open(out_dir / "corpus_entities.json", "w", encoding="utf-8") as handle:
        json.dump(entities_log, handle, indent=2)
    meta = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "label_file": str(label_file),
        "output_path": str(save_path),
        "use_lesk": bool(use_lesk),
        "use_t5_filter": bool(use_t5_filter),
        "t5_model_name": str(t5_model_name),
        "expand_corpus": bool(expand_corpus),
        "llm_model_name": str(llm_model_name),
        "llm_provider": str(llm_provider),
        "expansions_per_noun": int(expansions_per_noun),
        "min_corpus_size": int(min_corpus_size),
        "body_parts_file": str(body_parts_file),
        "expansion_cache": str(expansion_cache),
    }
    with open(out_dir / "corpus_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Corpus saved: {S.shape} → {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", default="corpus/action_labels.json")
    parser.add_argument("--output_dir", default="corpus/")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--use_lesk", action="store_true", default=False)
    parser.add_argument("--use_t5_filter", action="store_true", default=False)
    parser.add_argument("--t5_model_name", default="t5-small")
    parser.add_argument("--expand_corpus", action="store_true", default=False)
    parser.add_argument("--llm_model_name", default=os.getenv("LLM_MODEL", "google/flan-t5-small"))
    parser.add_argument(
        "--llm_provider",
        default=os.getenv("LLM_PROVIDER", "transformers"),
        choices=["transformers", "openai", "groq"],
    )
    parser.add_argument("--expansions_per_noun", type=int, default=3)
    parser.add_argument("--min_corpus_size", type=int, default=500)
    parser.add_argument("--body_parts_file", default="corpus/body_parts.json")
    parser.add_argument("--expansion_cache", default="corpus/llm_expansions.json")
    args = parser.parse_args()
    build_corpus(
        args.label_file,
        args.output_dir,
        args.output_path,
        args.device,
        use_lesk=args.use_lesk,
        use_t5_filter=args.use_t5_filter,
        t5_model_name=args.t5_model_name,
        expand_corpus=args.expand_corpus,
        llm_model_name=args.llm_model_name,
        llm_provider=args.llm_provider,
        expansions_per_noun=args.expansions_per_noun,
        min_corpus_size=args.min_corpus_size,
        body_parts_file=args.body_parts_file,
        expansion_cache=args.expansion_cache,
    )
