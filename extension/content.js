// Injected only after the user picks "Check with Truth Engine"; shows a side panel.
(() => {
  if (window.__teInjected) return;
  window.__teInjected = true;
  let host, body;

  function panel() {
    if (host?.isConnected) return body;
    host = document.createElement("div");
    host.style.cssText = "all: initial; position: fixed; top: 16px; right: 16px; z-index: 2147483647;";
    const shadow = host.attachShadow({mode: "closed"});
    const style = document.createElement("style");
    style.textContent = TE_CSS + `
      .panel { width: 380px; max-height: calc(100vh - 32px); overflow: auto; background: #f7f7f5; border: 1px solid #ddd;
        border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.18); padding: 12px 14px; }
      .head { display: flex; justify-content: space-between; align-items: center; font-weight: 700; }
      .x { border: 0; background: none; font-size: 18px; cursor: pointer; color: #6b6b70; }`;
    const wrap = document.createElement("div");
    wrap.className = "te panel";
    const head = document.createElement("div");
    head.className = "head";
    head.append("AI Truth Engine");
    const x = document.createElement("button");
    x.className = "x"; x.textContent = "×"; x.title = "Close";
    x.onclick = () => host.remove();
    head.append(x);
    body = document.createElement("div");
    wrap.append(head, body);
    shadow.append(style, wrap);
    document.documentElement.append(host);
    return body;
  }

  chrome.runtime.onMessage.addListener(msg => {
    const b = panel();
    if (msg.type === "te-loading") {
      b.replaceChildren(teEl("p", "meta", "Checking claims against the literature…"), teEl("p", "why", "“" + msg.text.slice(0, 300) + "”"));
    } else if (msg.type === "te-result") {
      teRenderResult(b, msg.result, msg.apiUrl);
    } else if (msg.type === "te-error") {
      b.replaceChildren(teEl("p", "why", msg.message));
    }
  });
})();
