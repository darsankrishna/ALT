# corpus/build_corpus.py
"""Build ALT entity corpus embeddings from action labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
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

def get_entity_noun(label: str, nlp: Any) -> str:
    """Extract main noun phrase from action label."""
    doc = nlp(label)
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


def build_corpus(
    label_file: str,
    output_dir: str,
    output_path: Optional[str] = None,
    device: str = "cpu",
    use_lesk: bool = True,
    use_t5_filter: bool = False,
    t5_model_name: str = "t5-small",
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

    with torch.no_grad():
        for label in labels:
            noun = get_entity_noun(label, nlp)
            defn = get_wordnet_def(noun, context=label if use_lesk else None)
            # Use label + entity context
            entity_text = f"{noun}: {defn}" if defn != noun else noun
            if use_t5_filter and t5_model is not None and t5_tokenizer is not None:
                keep = is_action_related([entity_text], t5_tokenizer, t5_model, device=device)[0]
                if not keep:
                    entity_text = noun
            entities_log.append({"label": label, "entity": noun, "context": entity_text})

            tokens = clip.tokenize([entity_text], truncate=True).to(device)
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
    args = parser.parse_args()
    build_corpus(
        args.label_file,
        args.output_dir,
        args.output_path,
        args.device,
        use_lesk=args.use_lesk,
        use_t5_filter=args.use_t5_filter,
        t5_model_name=args.t5_model_name,
    )
