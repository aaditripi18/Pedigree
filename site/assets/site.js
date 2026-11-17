(() => {
  const tip = document.createElement("div"); tip.className = "tip"; tip.setAttribute("role", "tooltip"); document.body.append(tip);
  let cur = null;
  function show(el) {
    cur = el; tip.textContent = "";
    el.dataset.tip.split("|").forEach((line, i) => { const d = document.createElement("div"); d.textContent = line; if (!i) d.className = "tip-h"; tip.append(d); });
    tip.style.display = "block";
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let x = r.left + r.width / 2 - w / 2, y = r.top - h - 10;
    if (y < 64) y = r.bottom + 10;
    x = Math.max(8, Math.min(window.innerWidth - w - 8, x));
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hide() { cur = null; tip.style.display = "none"; }
  document.querySelectorAll("[data-tip]").forEach(el => {
    el.addEventListener("mouseenter", () => show(el)); el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", () => show(el)); el.addEventListener("blur", hide);
    el.addEventListener("click", ev => { ev.stopPropagation(); cur === el ? hide() : show(el); });
  });
  document.addEventListener("click", hide); window.addEventListener("scroll", hide, { passive: true });
  document.querySelectorAll("[data-copy]").forEach(b => b.addEventListener("click", async () => {
    const src = document.querySelector(b.dataset.copy);
    const text = src.innerText.split("\n").filter(l => !l.startsWith("#")).map(l => l.replace(/^\$ /, "")).join("\n");
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy", 1500); } catch (e) {}
  }));
})();
