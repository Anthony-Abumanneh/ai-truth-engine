"""Investigation tree (spec §12): the evidence trail behind every verdict,
shaped for the UI.

    input text / question                         (root)
      claim  [status, confidence]                 (step)
        supports:    lineage 1 (strongest source first, its copies listed)
        contradicts: lineage 2 ...
        red flags                                  (rejected = reasons to doubt)
"""
from __future__ import annotations

from ..schemas import Claim, ClaimVerdict, EvidenceItem, TreeNode
from ..textutils import short


def _lineage_nodes(items: list[EvidenceItem], relation: str) -> list[TreeNode]:
    groups: dict[int, list[EvidenceItem]] = {}
    for i in items:
        groups.setdefault(i.lineage if i.lineage is not None else -id(i), []).append(i)
    out = []
    for members in sorted(groups.values(), key=lambda g: -max(i.weight for i in g)):
        best = max(members, key=lambda i: i.weight)
        label = (best.source.metadata or {}).get("study_label") or best.source.source_type
        copies = sorted({i.source.title for i in members if i.source.id != best.source.id})
        node = TreeNode(label=best.source.title, detail=f"{label}; \"{short(best.passage.text, 200)}\"",
                        value=best.source.url, relation=relation, weight=round(best.weight, 3))
        node.children = [TreeNode(label=t, detail="restates / derived from the same study", relation="correlates")
                         for t in copies]
        out.append(node)
    return out


def claim_node(claim: Claim, v: ClaimVerdict) -> TreeNode:
    conf = v.confidence_detail
    node = TreeNode(label=claim.text, status=v.status, weight=v.confidence, relation="step",
                    detail=(f"{conf.label} confidence ({conf.calibrated:.2f}); " if conf else "") + short(v.rationale, 300))
    node.children = _lineage_nodes(v.supporting, "supports") + _lineage_nodes(v.contradicting, "contradicts")
    node.children += [TreeNode(label=f.kind.replace("_", " "), detail=f.message, relation="rejected", weight=f.severity)
                      for f in v.red_flags]
    return node


def build_tree(text: str, claims: list[Claim], verdicts: list[ClaimVerdict]) -> TreeNode:
    by_id = {v.claim_id: v for v in verdicts}
    root = TreeNode(label=short(text, 200), relation="root")
    root.children = [claim_node(c, by_id[c.id]) for c in claims if c.id in by_id]
    return root
