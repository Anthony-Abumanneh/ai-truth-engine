(async () => {
  const s = await teSettings();
  document.getElementById("api").value = s.apiUrl;
  document.getElementById("live").checked = s.live;
})();
document.getElementById("save").onclick = async () => {
  const apiUrl = document.getElementById("api").value.trim().replace(/\/+$/, "");
  const msg = document.getElementById("msg");
  let origin;
  try { origin = new URL(apiUrl).origin; } catch { msg.textContent = "Not a valid URL."; return; }
  const local = /^http:\/\/(localhost|127\.0\.0\.1):8000$/.test(origin);
  if (!local && !(await chrome.permissions.request({origins: [origin + "/*"]}))) {
    msg.textContent = "Permission to reach that server was not granted."; return;
  }
  await chrome.storage.sync.set({apiUrl, live: document.getElementById("live").checked});
  msg.textContent = "Saved.";
};
