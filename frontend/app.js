// ---------- Night sky / Milky Way background ----------
(function starfield() {
  const canvas = document.getElementById("sky");
  const ctx = canvas.getContext("2d");
  let width, height, dpr;
  let stars = [];
  let shootingStars = [];

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedStars();
  }

  function seedStars() {
    const count = Math.floor((width * height) / 5500);
    stars = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      r: Math.random() * 1.3 + 0.3,
      baseAlpha: Math.random() * 0.6 + 0.3,
      twinkleSpeed: Math.random() * 0.015 + 0.004,
      phase: Math.random() * Math.PI * 2,
      hue: Math.random() < 0.15 ? "200, 220, 255" : "255, 255, 255",
    }));
  }

  function maybeSpawnShootingStar() {
    if (shootingStars.length < 2 && Math.random() < 0.006) {
      const startX = Math.random() * width * 0.8;
      const startY = Math.random() * height * 0.35;
      const angle = (Math.PI / 5) + Math.random() * (Math.PI / 10);
      const speed = 6 + Math.random() * 5;
      shootingStars.push({
        x: startX,
        y: startY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        maxLife: 45 + Math.random() * 20,
      });
    }
  }

  function draw(time) {
    ctx.clearRect(0, 0, width, height);

    for (const s of stars) {
      const twinkle = Math.sin(time * s.twinkleSpeed + s.phase) * 0.35 + 0.65;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${s.hue}, ${(s.baseAlpha * twinkle).toFixed(3)})`;
      ctx.fill();
    }

    maybeSpawnShootingStar();
    shootingStars = shootingStars.filter((star) => star.life < star.maxLife);
    for (const star of shootingStars) {
      const progress = star.life / star.maxLife;
      const alpha = progress < 0.15 ? progress / 0.15 : 1 - (progress - 0.15) / 0.85;
      const tailX = star.x - star.vx * 6;
      const tailY = star.y - star.vy * 6;

      const gradient = ctx.createLinearGradient(star.x, star.y, tailX, tailY);
      gradient.addColorStop(0, `rgba(210, 225, 255, ${alpha})`);
      gradient.addColorStop(1, "rgba(210, 225, 255, 0)");

      ctx.strokeStyle = gradient;
      ctx.lineWidth = 1.6;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(star.x, star.y);
      ctx.lineTo(tailX, tailY);
      ctx.stroke();

      star.x += star.vx;
      star.y += star.vy;
      star.life += 1;
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  requestAnimationFrame(draw);
})();

// ---------- Chat ----------
(function chat() {
  const API_ASK_URL = "/ask";
  const SESSION_KEY = "rag_session_id";

  const messagesEl = document.getElementById("messages");
  const composer = document.getElementById("composer");
  const input = document.getElementById("question-input");
  const sendBtn = document.getElementById("send-btn");
  const resetBtn = document.getElementById("reset-btn");

  let sessionId = localStorage.getItem(SESSION_KEY) || null;
  let busy = false;

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMessage(role, text, sources) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.textContent = text;

    if (sources && sources.length) {
      const chips = document.createElement("div");
      chips.className = "sources";
      sources.forEach((s) => {
        const chip = document.createElement("span");
        chip.className = "source-chip";
        chip.textContent = s;
        chips.appendChild(chip);
      });
      el.appendChild(chips);
    }

    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
  }

  function showTyping() {
    const el = document.createElement("div");
    el.className = "typing";
    el.id = "typing-indicator";
    el.innerHTML = "<span></span><span></span><span></span>";
    messagesEl.appendChild(el);
    scrollToBottom();
  }

  function hideTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function errorMessageFor(status) {
    if (status === 503) return "Hazırda çox məşğulam, bir azdan yenə cəhd et 🙏";
    if (status === 504) return "Cavab gözlədiyimdən çox çəkdi, bir də sınayaq?";
    if (status === 502) return "Bu sualı emal edərkən nəsə səhv getdi, üzr istəyirəm.";
    if (status === 422) return "Sualını görə bilmədim, bir də yaz görək?";
    return "Naməlum bir xəta oldu, bir azdan yenidən cəhd et.";
  }

  async function sendMessage(text) {
    busy = true;
    sendBtn.disabled = true;
    addMessage("user", text);
    showTyping();

    try {
      const response = await fetch(API_ASK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, top_k: 4, session_id: sessionId }),
      });

      const data = await response.json().catch(() => ({}));
      hideTyping();

      if (!response.ok) {
        addMessage("assistant error", data.detail || errorMessageFor(response.status));
        return;
      }

      sessionId = data.session_id;
      localStorage.setItem(SESSION_KEY, sessionId);
      addMessage("assistant", data.answer, data.sources);
    } catch (err) {
      hideTyping();
      addMessage("assistant error", "Serverlə əlaqə qurula bilmədi. İnternetini yoxla, zəhmət olmasa.");
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || busy) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(text);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composer.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  resetBtn.addEventListener("click", () => {
    localStorage.removeItem(SESSION_KEY);
    sessionId = null;
    messagesEl.innerHTML = "";
    showWelcome();
  });

  function showWelcome() {
    addMessage(
      "assistant",
      "Salam! Mən Dory-yəm ✦ Universitet, fakültə, qəbul şərtləri — nə bilmək istəsən, mənbələrə əsaslanaraq sənə kömək etməyə çalışacam. Nədən başlayaq?"
    );
  }

  showWelcome();
})();
