importScripts("common.js");

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({id: "te-check", title: "Check with Truth Engine", contexts: ["selection"]});
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "te-check" || !tab?.id || !info.selectionText) return;
  // activeTab: injecting the panel is allowed only because the user just clicked our menu item
  await chrome.scripting.executeScript({target: {tabId: tab.id}, files: ["render.js", "content.js"]});
  const send = msg => chrome.tabs.sendMessage(tab.id, msg).catch(() => {});
  send({type: "te-loading", text: info.selectionText});
  try {
    const {result, apiUrl} = await teAudit(info.selectionText);
    send({type: "te-result", result, apiUrl});
  } catch (e) {
    send({type: "te-error", message: `Couldn't reach the Truth Engine (${e.message}). Is the server running? ` +
      "Start it with: python -m truth_engine serve"});
  }
});
