"""How accurate is the text-only study-design detector? Gold = PubMed's own
publication-type tags. Only matters for sources without those tags (OpenAlex
papers lacking study designs, web pages, local documents).

    python scripts/eval_study_design.py [per_class]
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truth_engine.science.pubmed import PubMedClient  # noqa: E402
from truth_engine.science.study_design import classify  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
QUERIES = {   # PubMed tag query -> design label our detector should produce
    "meta_analysis": "meta-analysis[pt] AND hasabstract",
    "systematic_review": "systematic review[pt] NOT meta-analysis[pt] AND hasabstract",
    "rct": "randomized controlled trial[pt] AND hasabstract",
    "observational": "(observational study[pt] OR cohort studies[mh]) NOT randomized controlled trial[pt] "
                     "NOT review[pt] AND humans[mh] AND hasabstract",
    "case_report": "case reports[pt] AND hasabstract",
    "animal": "animals[mh] NOT humans[mh] NOT review[pt] AND hasabstract",
}
pm = PubMedClient(offline=False)
gold, pred = [], []
for label, q in QUERIES.items():
    recs = pm.search_records(q + " AND 2018:2024[dp]", retmax=n)
    for r in recs:
        prof = classify(r.title, r.abstract)            # text only: no PubMed tags, no MeSH
        gold.append(label)
        pred.append(prof.design)
    print(f"  {label:18} {len(recs)} records", flush=True)
labels = list(QUERIES)
print(f"\ntext-only detector vs PubMed tags, n={len(gold)}: accuracy {sum(g == p for g, p in zip(gold, pred)) / len(gold):.3f}")
for lab in labels:
    rows = [p for g, p in zip(gold, pred) if g == lab]
    tp = sum(p == lab for p in rows)
    prec = tp / max(1, pred.count(lab))
    print(f"  {lab:18} recall {tp / max(1, len(rows)):.2f}  precision {prec:.2f}  top predictions {Counter(rows).most_common(3)}")
