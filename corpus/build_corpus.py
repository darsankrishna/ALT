# corpus/build_corpus.py
import json, torch, argparse
import torch.nn.functional as F
from pathlib import Path

def get_entity_noun(label, nlp):
    """Extract main noun phrase from action label."""
    doc = nlp(label)
    nouns = [t.text for t in doc if t.pos_ in ('NOUN', 'PROPN')]
    if nouns:
        return nouns[-1]   # last noun = object of the action
    for t in doc:
        if t.dep_ == 'ROOT':
            return t.text
    return label.split()[-1]

def get_wordnet_def(word):
    from nltk.corpus import wordnet as wn
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(word)
    if synsets:
        return synsets[0].definition()
    return word

def build_corpus(label_file, output_dir, device="cpu"):
    import clip
    import spacy

    labels = json.load(open(label_file))["labels"]
    nlp = spacy.load("en_core_web_sm")

    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    embeddings = []
    entities_log = []

    with torch.no_grad():
        for label in labels:
            noun = get_entity_noun(label, nlp)
            defn = get_wordnet_def(noun)
            # Use label + entity context
            entity_text = f"{noun}: {defn}" if defn != noun else noun
            entities_log.append({"label": label, "entity": noun, "context": entity_text})

            tokens = clip.tokenize([entity_text], truncate=True).to(device)
            emb = clip_model.encode_text(tokens)          # (1, 512)
            emb = F.normalize(emb.float(), dim=-1)
            embeddings.append(emb.cpu())

    S = torch.cat(embeddings, dim=0).half()               # (K, 512) fp16
    out_dir = Path(output_dir)
    torch.save(S, out_dir / "corpus_embeddings.pt")
    json.dump(entities_log, open(out_dir / "corpus_entities.json", "w"), indent=2)
    print(f"Corpus saved: {S.shape} → {out_dir / 'corpus_embeddings.pt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file", default="corpus/action_labels.json")
    parser.add_argument("--output_dir", default="corpus/")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()
    build_corpus(args.label_file, args.output_dir, args.device)