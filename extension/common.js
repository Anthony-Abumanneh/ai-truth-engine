// Shared by the background worker and the popup.
const TE_DEFAULTS = {apiUrl: "http://localhost:8000", live: false};

async function teSettings() {
  return {...TE_DEFAULTS, ...(await chrome.storage.sync.get(["apiUrl", "live"]))};
}

async function teAudit(text) {
  const {apiUrl, live} = await teSettings();
  const res = await fetch(apiUrl.replace(/\/+$/, "") + "/api/audit", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: text.slice(0, 6000), live}),
  });
  if (!res.ok) throw new Error(`server answered ${res.status}`);
  return {result: await res.json(), apiUrl};
}
