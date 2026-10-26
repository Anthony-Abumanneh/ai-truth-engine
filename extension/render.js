// Renders an audit result with DOM APIs only (no innerHTML): source text is untrusted.
var TE_LABEL = {supported: "✓ Supported", partially_supported: "◐ Partly supported", contradicted: "✗ Contradicted",
  unverified: "? Unverified", inferred: "→ Inferred", speculative: "~ Speculative"};
var TE_COLOR = {supported: "#1f8a4c", inferred: "#1f8a4c", partially_supported: "#b07a00", contradicted: "#c4362f",
  unverified: "#6b6b70", speculative: "#7a4bc4"};
var TE_CSS = `
  .te { font: 13.5px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1d1d1f; }
  .te * { box-sizing: border-box; }
  .te .claim { border: 1px solid #e4e4e0; border-radius: 10px; padding: 10px 12px; margin: 10px 0; background: #fff; }
  .te .badge { display: inline-block; color: #fff; font-size: 11.5px; font-weight: 700; padding: 1px 8px; border-radius: 999px; }
  .te .text { font-weight: 600; margin: 6px 0 4px; }
  .te .meta { color: #6b6b70; font-size: 12px; }
  .te .why { color: #4a4a4f; font-size: 12.5px; margin: 6px 0; }
  .te .flag { display: inline-block; font-size: 11.5px; background: #fdeee2; color: #a3420e; border-radius: 5px; padding: 1px 6px; margin: 2px 4px 2px 0; }
  .te .ev { font-size: 12.5px; border-left: 3px solid #e4e4e0; padding-left: 8px; margin: 6px 0; }
  .te .ev.for { border-color: #1f8a4c; } .te .ev.against { border-color: #c4362f; }
  .te a { color: #2f5bd3; }
  .te .note { color: #6b6b70; font-size: 11.5px; margin-top: 8px; }
`;

function teEl(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function teRenderResult(root, r, apiUrl) {
  root.replaceChildren();
  if (!r.claims.length) root.append(teEl("p", "meta", "No check-worthy science or health claims found in the selection."));
  for (const c of r.claims) {
    const box = teEl("div", "claim");
    const badge = teEl("span", "badge", TE_LABEL[c.status] || c.status);
    badge.style.background = TE_COLOR[c.status] || "#6b6b70";
    box.append(badge, teEl("div", "text", c.text));
    const lean = c.leaning ? ` · leans ${(TE_LABEL[c.leaning] || c.leaning).replace(/^[^ ]+ /, "").toLowerCase()}` : "";
    box.append(teEl("div", "meta", `Confidence ${c.confidence_label || ""} (${Math.round(c.confidence * 100)}%)${lean}`));
    box.append(teEl("div", "why", c.rationale.length > 280 ? c.rationale.slice(0, 280) + "…" : c.rationale));
    for (const f of c.red_flags) { const s = teEl("span", "flag", "⚑ " + f.kind.replaceAll("_", " ")); s.title = f.message; box.append(s); }
    for (const [list, cls] of [[c.supporting, "for"], [c.contradicting, "against"]]) {
      for (const e of list.slice(0, 2)) {
        const ev = teEl("div", "ev " + cls);
        const url = /^https?:\/\//i.test(e.url || "") ? e.url : null;
        const title = url ? teEl("a", null, e.title) : teEl("b", null, e.title);
        if (url) { title.href = url; title.target = "_blank"; title.rel = "noopener"; }
        ev.append(cls === "for" ? "For: " : "Against: ", title, teEl("div", "meta", e.study || ""), teEl("div", null, "“" + e.text + "”"));
        box.append(ev);
      }
    }
    root.append(box);
  }
  const note = teEl("div", "note", "Not medical advice. ");
  if (apiUrl && r.query_id) {
    const a = teEl("a", null, "Open full report");
    a.href = apiUrl.replace(/\/+$/, "") + "/"; a.target = "_blank"; a.rel = "noopener";
    note.append(a);
  }
  root.append(note);
}
