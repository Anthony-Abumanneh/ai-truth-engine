const out = document.getElementById("out");
const style = document.createElement("style");
style.textContent = TE_CSS;
document.head.append(style);

document.getElementById("sel").onclick = async () => {
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  try {
    const [{result}] = await chrome.scripting.executeScript({target: {tabId: tab.id}, func: () => getSelection().toString()});
    document.getElementById("text").value = result || "";
  } catch { out.replaceChildren(teEl("p", "why", "Can't read the selection on this page.")); }
};

document.getElementById("go").onclick = async () => {
  const text = document.getElementById("text").value.trim();
  if (!text) return;
  out.replaceChildren(teEl("p", "meta", "Checking claims against the literature…"));
  try {
    const {result, apiUrl} = await teAudit(text);
    teRenderResult(out, result, apiUrl);
  } catch (e) {
    out.replaceChildren(teEl("p", "why", `Couldn't reach the Truth Engine (${e.message}). Start it with: python -m truth_engine serve`));
  }
};
