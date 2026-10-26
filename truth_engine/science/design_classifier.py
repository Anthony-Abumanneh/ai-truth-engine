"""Study-design classifier from title + abstract text (step 8.5).

Labels come free from PubMed: every indexed paper carries expert-assigned
publication types and MeSH terms ("Randomized Controlled Trial", "Animals").
We pull ~1,000 abstracts per design with those tags (distant supervision) and
train TF-IDF + logistic regression: small, fast on CPU, and easy to inspect.

It only runs where PubMed's own tags are missing (OpenAlex papers without study
designs, web pages, local documents) and after the high-precision regex rules
have answered "unknown" (scripts/eval_study_design.py: rules are ~100% precise
but find only 5-48% of most designs from text alone).

    python -m truth_engine.science.design_classifier        # fetch data, train, report, save
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from ..config import DATA_DIR

MODEL_PATH = DATA_DIR / "models" / "study_design_clf.joblib"
QUERIES = {   # our design label -> PubMed query (tags = labels)
    "meta_analysis": "meta-analysis[pt]",
    "systematic_review": "systematic review[pt] NOT meta-analysis[pt]",
    "rct": "randomized controlled trial[pt]",
    "cohort": "(observational study[pt] OR cohort studies[mh] OR case-control studies[mh] OR cross-sectional "
              "studies[mh]) NOT randomized controlled trial[pt] NOT review[pt] AND humans[mh]",
    "case_report": "case reports[pt]",
    "animal": "animals[mh] NOT humans[mh] NOT review[pt] NOT clinical trial[pt]",
    "in_vitro": "cell line[mh] NOT animals[mh] NOT clinical trial[pt] NOT review[pt] NOT case reports[pt]",
    "narrative_review": "review[pt] NOT systematic review[pt] NOT meta-analysis[pt]",
}
FILTER = " AND hasabstract AND 2015:2024[dp]"
MIN_PROB = 0.6          # below this the classifier abstains and the design stays "unknown"


def fetch_dataset(per_class: int = 1000) -> list[dict]:
    from .pubmed import PubMedClient
    pm = PubMedClient(offline=False)
    rows, seen = [], set()
    for label, q in QUERIES.items():
        ids = [i for i in pm.search(q + FILTER, retmax=per_class) if i not in seen]
        for k in range(0, len(ids), 200):
            for r in pm.fetch(ids[k:k + 200]):
                pmid = r.pmid or ""
                if pmid in seen or not r.abstract:
                    continue
                seen.add(pmid)
                rows.append({"pmid": pmid, "label": label, "text": f"{r.title}. {r.abstract}"})
        print(f"  {label:18} {sum(x['label'] == label for x in rows)}", flush=True)
    return rows


def is_test(pmid: str) -> bool:
    return int(hashlib.sha1(pmid.encode()).hexdigest(), 16) % 5 == 0       # 20% held out, by PMID


def train(rows: list[dict]) -> dict:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.pipeline import make_pipeline
    tr = [r for r in rows if not is_test(r["pmid"])]
    te = [r for r in rows if is_test(r["pmid"])]
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
                        LogisticRegression(C=4.0, max_iter=2000))
    clf.fit([r["text"] for r in tr], [r["label"] for r in tr])
    pred = clf.predict([r["text"] for r in te])
    report = classification_report([r["label"] for r in te], pred, output_dict=True, zero_division=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return {"train": len(tr), "test": len(te), "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "per_class": {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score")}
                          for k, v in report.items() if k in QUERIES}}


@lru_cache(maxsize=1)
def _model():
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def predict(title: str, abstract: str) -> tuple[str, float] | None:
    """(design, probability) when confident, else None."""
    clf = _model()
    text = f"{title}. {abstract}".strip(". ")
    if clf is None or len(text) < 80:            # too little text to judge (titles alone, snippets)
        return None
    probs = clf.predict_proba([text])[0]
    i = int(probs.argmax())
    return (clf.classes_[i], round(float(probs[i]), 3)) if probs[i] >= MIN_PROB else None


if __name__ == "__main__":
    path = DATA_DIR / "finetune" / "study_design.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in open(path)]
    else:
        rows = fetch_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows))
    print(json.dumps(train(rows), indent=1))
