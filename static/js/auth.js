/* =====================================================
   VoxEmotion – Auth Page JavaScript
   Client-side validation, particle bg, cookie banner
   ===================================================== */
'use strict';

// ── Animated background (same as main app) ────────────────────────────────────
(function () {
  const c = document.getElementById('bg-canvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  let W, H, particles = [];
  function resize() { W = c.width = window.innerWidth; H = c.height = window.innerHeight; }
  function init() {
    particles = [];
    const n = Math.floor((W * H) / 14000);
    for (let i = 0; i < n; i++) {
      particles.push({ x: Math.random()*W, y: Math.random()*H,
        r: Math.random()*1.4+0.3, vx: (Math.random()-0.5)*0.15,
        vy: (Math.random()-0.5)*0.15, alpha: Math.random()*0.45+0.1,
        hue: Math.random()>0.5?260:190 });
    }
  }
  function draw() {
    ctx.clearRect(0,0,W,H);
    particles.forEach(p => {
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0)p.x=W; if(p.x>W)p.x=0;
      if(p.y<0)p.y=H; if(p.y>H)p.y=0;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle=`hsla(${p.hue},80%,70%,${p.alpha})`; ctx.fill();
    });
    for(let i=0;i<particles.length;i++)
      for(let j=i+1;j<particles.length;j++){
        const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<85){ ctx.beginPath();
          ctx.strokeStyle=`hsla(260,70%,65%,${0.05*(1-d/85)})`;
          ctx.lineWidth=0.5;
          ctx.moveTo(particles[i].x,particles[i].y);
          ctx.lineTo(particles[j].x,particles[j].y); ctx.stroke(); }
      }
    requestAnimationFrame(draw);
  }
  resize(); init(); draw();
  window.addEventListener('resize', () => { resize(); init(); });
})();

// ── Panel switcher ─────────────────────────────────────────────────────────────
function showPanel(id) {
  document.querySelectorAll('.form-panel').forEach(p => p.classList.add('hidden'));
  const target = document.getElementById(id);
  if (target) { target.classList.remove('hidden'); target.querySelector('input')?.focus(); }
}

// ── Password visibility toggle ────────────────────────────────────────────────
document.querySelectorAll('.toggle-pw').forEach(btn => {
  btn.addEventListener('click', () => {
    const inp = document.getElementById(btn.dataset.target);
    if (!inp) return;
    inp.type = inp.type === 'password' ? 'text' : 'password';
    btn.textContent = inp.type === 'password' ? '👁' : '🙈';
  });
});

// ── Password strength meter ───────────────────────────────────────────────────
function getStrength(pw) {
  let score = 0;
  if (pw.length >= 8)  score++;
  if (pw.length >= 12) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  return score;
}

function updateStrength(inputId, fillId, labelId) {
  const inp   = document.getElementById(inputId);
  const fill  = document.getElementById(fillId);
  const label = document.getElementById(labelId);
  if (!inp || !fill || !label) return;
  inp.addEventListener('input', () => {
    const score = getStrength(inp.value);
    const pct   = (score / 5) * 100;
    const colors = ['#ff5252','#ff5252','#ffd740','#ffd740','#00e676','#00e676'];
    const labels = ['Very Weak','Weak','Fair','Good','Strong','Very Strong'];
    fill.style.width      = pct + '%';
    fill.style.background = colors[score] || '#00e676';
    label.textContent     = inp.value ? labels[score] || 'Very Strong' : 'Enter password';
    label.style.color     = colors[score] || '#00e676';
  });
}

updateStrength('reg-password',  'pw-fill', 'pw-label');
updateStrength('new-password',  'pw-fill', 'pw-label');

// ── DOB formatter ─────────────────────────────────────────────────────────────
const dobInput = document.getElementById('reg-dob');
if (dobInput) {
  dobInput.addEventListener('input', (e) => {
    let v = e.target.value.replace(/\D/g, '');
    if (v.length > 2)  v = v.slice(0,2) + '/' + v.slice(2);
    if (v.length > 5)  v = v.slice(0,5) + '/' + v.slice(5);
    e.target.value = v.slice(0, 10);
  });
}

// ── Full name: allow only letters ─────────────────────────────────────────────
const nameInput = document.getElementById('reg-name');
if (nameInput) {
  nameInput.addEventListener('input', () => {
    nameInput.value = nameInput.value
      .replace(/[^A-Za-z '\-]/g, '')
      .replace(/\s{2,}/g, ' ')
      .trimStart();
  });
}

// ── Client-side validation helpers ───────────────────────────────────────────
function setError(fieldId, msg) {
  const el = document.getElementById(fieldId);
  if (el) el.textContent = msg;
}
function clearErrors(...ids) {
  ids.forEach(id => { const el = document.getElementById(id); if(el) el.textContent=''; });
}
function markInput(inputId, hasError) {
  const el = document.getElementById(inputId);
  if (el) el.classList.toggle('input--error', hasError);
}

function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email);
}
function validatePassword(pw) {
  return pw.length >= 8 && /[A-Z]/.test(pw) && /[0-9]/.test(pw);
}
function validateDOB(dob) {
  if (!/^\d{2}\/\d{2}\/\d{4}$/.test(dob)) return false;
  const [dd, mm, yyyy] = dob.split('/').map(Number);
  const d = new Date(yyyy, mm-1, dd);
  if (d.getFullYear() !== yyyy || d.getMonth() !== mm-1 || d.getDate() !== dd) return false;
  const age = (new Date() - d) / (365.25 * 24 * 3600 * 1000);
  return age >= 5 && age <= 120;
}

function validateFullname(name) {
  return /^[A-Za-z]+(?:[ '-][A-Za-z]+)*$/.test(name);
}

// ── Login form validation ─────────────────────────────────────────────────────
const loginForm = document.getElementById('login-form');
if (loginForm) {
  loginForm.addEventListener('submit', function(e) {
    clearErrors('err-login-email','err-login-pw');
    let ok = true;
    const email = document.getElementById('login-email').value.trim();
    const pw    = document.getElementById('login-password').value;
    if (!validateEmail(email)) {
      setError('err-login-email', 'Please enter a valid email address.');
      markInput('login-email', true); ok = false;
    }
    if (!pw) {
      setError('err-login-pw', 'Password is required.');
      markInput('login-password', true); ok = false;
    }
    if (!ok) { e.preventDefault(); return; }
    setLoading('login-btn', true);
  });
}

// ── Register form validation ──────────────────────────────────────────────────
const registerForm = document.getElementById('register-form');
if (registerForm) {
  registerForm.addEventListener('submit', function(e) {
    clearErrors('err-reg-name','err-reg-dob','err-reg-email','err-reg-pw','err-reg-confirm');
    let ok = true;

    const name    = document.getElementById('reg-name').value.trim();
    const dob     = document.getElementById('reg-dob').value.trim();
    const email   = document.getElementById('reg-email').value.trim();
    const pw      = document.getElementById('reg-password').value;
    const confirm = document.getElementById('reg-confirm').value;
    const agreed  = document.getElementById('agree-terms').checked;

    if (!name || !validateFullname(name)) {
      setError('err-reg-name', "Name can include letters, spaces, hyphens, and apostrophes only.");
      markInput('reg-name', true); ok = false;
    }
    if (!validateDOB(dob)) {
      setError('err-reg-dob', 'Enter a valid date in DD/MM/YYYY format.');
      markInput('reg-dob', true); ok = false;
    }
    if (!validateEmail(email)) {
      setError('err-reg-email', 'Please enter a valid email address.');
      markInput('reg-email', true); ok = false;
    }
    if (!validatePassword(pw)) {
      setError('err-reg-pw', 'Password must be at least 8 characters with 1 uppercase letter and 1 number.');
      markInput('reg-password', true); ok = false;
    }
    if (pw !== confirm) {
      setError('err-reg-confirm', 'Passwords do not match.');
      markInput('reg-confirm', true); ok = false;
    }
    if (!agreed) {
      alert('Please accept the Terms & Conditions to continue.'); ok = false;
    }
    if (!ok) { e.preventDefault(); return; }
    setLoading('register-btn', true);
  });
}

// ── Forgot password form ──────────────────────────────────────────────────────
const forgotForm = document.getElementById('forgot-form');
if (forgotForm) {
  forgotForm.addEventListener('submit', function(e) {
    clearErrors('err-forgot-email');
    const email = document.getElementById('forgot-email').value.trim();
    if (!validateEmail(email)) {
      setError('err-forgot-email', 'Please enter a valid email address.');
      markInput('forgot-email', true);
      e.preventDefault(); return;
    }
    setLoading('forgot-btn', true);
  });
}

// ── Reset password form ───────────────────────────────────────────────────────
const resetForm = document.getElementById('reset-form') ||
                  document.querySelector('form[action*="reset-password"]');
if (resetForm) {
  const newPw  = document.getElementById('new-password');
  const confPw = document.getElementById('confirm-password');
  resetForm.addEventListener('submit', function(e) {
    let ok = true;
    if (newPw && !validatePassword(newPw.value)) {
      setError('err-new-pw', 'Password must be at least 8 chars with 1 uppercase and 1 number.');
      ok = false;
    }
    if (newPw && confPw && newPw.value !== confPw.value) {
      setError('err-confirm-pw', 'Passwords do not match.');
      ok = false;
    }
    if (!ok) e.preventDefault();
  });
}

// ── Loading state ─────────────────────────────────────────────────────────────
function setLoading(btnId, on) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  const text   = btn.querySelector('.btn-text');
  const loader = btn.querySelector('.btn-loader');
  btn.disabled = on;
  if (text)   text.hidden   = on;
  if (loader) loader.hidden = !on;
}

// ── Cookie banner ─────────────────────────────────────────────────────────────
(function () {
  const banner = document.getElementById('cookie-banner');
  if (!banner) return;
  const consent = localStorage.getItem('vox_cookie_consent');
  if (!consent) {
    setTimeout(() => banner.classList.add('show'), 800);
  }
})();

function acceptCookies(level) {
  localStorage.setItem('vox_cookie_consent', level);
  localStorage.setItem('vox_cookie_date', new Date().toISOString());
  const banner = document.getElementById('cookie-banner');
  if (banner) {
    banner.classList.remove('show');
    setTimeout(() => banner.style.display = 'none', 400);
  }
}
