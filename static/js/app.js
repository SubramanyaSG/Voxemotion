/* ===================================================================
   VoxEmotion – Frontend JavaScript
   =================================================================== */

'use strict';

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  textEmotion : 'neutral',
  fileEmotion : 'neutral',
  selectedFile: null,
  audioFile   : null,
  audioText   : '',
  audioCtx    : null,
  audioBuffer : null,
  words       : [],
  wordTimings : [],
  isPlaying   : false,
  startTime   : 0,
  pauseOffset : 0,
  rafId       : null,
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

const elTextInput    = $('text-input');
const elCharCount    = $('char-count');
const elResetText    = $('reset-text');
const elSubmitText   = $('submit-text');
const elDropzone     = $('dropzone');
const elFileInput    = $('file-input');
const elBrowseBtn    = $('browse-btn');
const elDropContent  = $('drop-content');
const elFileSelected = $('file-selected');
const elFileName     = $('selected-file-name');
const elFileSize     = $('selected-file-size');
const elRemoveFile   = $('remove-file');
const elResetFile    = $('reset-file');
const elSubmitFile   = $('submit-file');
const elOutputPanel  = $('output-panel');
const elMetaEmotion  = $('meta-emotion');
const elMetaDuration = $('meta-duration');
const elWaveformCanvas = $('waveform-canvas');
const elWaveformProg = $('waveform-progress');
const elBtnPlay      = $('btn-play');
const elBtnPause     = $('btn-pause');
const elBtnStop      = $('btn-stop');
const elBtnDownload  = $('btn-download');
const elTimeCurrent  = $('time-current');
const elTimeTotal    = $('time-total');
const elSeekBar      = $('seek-bar');
const elSeekFill     = $('seek-fill');
const elSeekThumb    = $('seek-thumb');
const elTranscript   = $('transcript-text');
const elToast        = $('toast');
const elAudio        = $('audio-player');
const elBgCanvas     = $('bg-canvas');

// ─── Animated Background ─────────────────────────────────────────────────────
(function initBackground() {
  const ctx = elBgCanvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = elBgCanvas.width  = window.innerWidth;
    H = elBgCanvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = [];
    const count = Math.floor((W * H) / 12000);
    for (let i = 0; i < count; i++) {
      particles.push({
        x    : Math.random() * W,
        y    : Math.random() * H,
        r    : Math.random() * 1.5 + 0.3,
        vx   : (Math.random() - 0.5) * 0.18,
        vy   : (Math.random() - 0.5) * 0.18,
        alpha: Math.random() * 0.5 + 0.15,
        hue  : Math.random() > 0.5 ? 260 : 190,
      });
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue},80%,70%,${p.alpha})`;
      ctx.fill();
    });

    // Connect nearby particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 90) {
          ctx.beginPath();
          ctx.strokeStyle = `hsla(260,70%,65%,${0.06 * (1 - d / 90)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  resize();
  createParticles();
  draw();
  window.addEventListener('resize', () => { resize(); createParticles(); });
})();

// ─── Emotion Chip Selection ───────────────────────────────────────────────────
$$('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const group   = chip.dataset.group;
    const emotion = chip.dataset.emotion;
    $$(`[data-group="${group}"]`).forEach(c => c.classList.remove('chip--active'));
    chip.classList.add('chip--active');
    if (group === 'text') state.textEmotion = emotion;
    else                  state.fileEmotion = emotion;
  });
});

// ─── Textarea character counter ───────────────────────────────────────────────
elTextInput.addEventListener('input', () => {
  elCharCount.textContent = elTextInput.value.length.toLocaleString();
});

// ─── Reset Text ───────────────────────────────────────────────────────────────
elResetText.addEventListener('click', () => {
  elTextInput.value = '';
  elCharCount.textContent = '0';
  $$('[data-group="text"]').forEach(c => {
    c.classList.toggle('chip--active', c.dataset.emotion === 'neutral');
  });
  state.textEmotion = 'neutral';
});

// ─── Reset File ───────────────────────────────────────────────────────────────
function clearFileSelection() {
  state.selectedFile = null;
  elFileInput.value  = '';
  elDropContent.hidden  = false;
  elFileSelected.hidden = true;
  $$('[data-group="file"]').forEach(c => {
    c.classList.toggle('chip--active', c.dataset.emotion === 'neutral');
  });
  state.fileEmotion = 'neutral';
}

elResetFile.addEventListener('click', clearFileSelection);
elRemoveFile.addEventListener('click', clearFileSelection);

// ─── Dropzone ─────────────────────────────────────────────────────────────────
elBrowseBtn.addEventListener('click', () => elFileInput.click());
elDropzone.addEventListener('click', e => {
  if (e.target === elDropzone || e.target.classList.contains('dropzone')) {
    elFileInput.click();
  }
});

elDropzone.addEventListener('dragover', e => {
  e.preventDefault();
  elDropzone.classList.add('drag-over');
});
elDropzone.addEventListener('dragleave', () => elDropzone.classList.remove('drag-over'));
elDropzone.addEventListener('drop', e => {
  e.preventDefault();
  elDropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});

elFileInput.addEventListener('change', () => {
  if (elFileInput.files[0]) handleFileSelect(elFileInput.files[0]);
});

const MAX_FILE_BYTES = 50 * 1024 * 1024;

function handleFileSelect(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['txt', 'pdf', 'docx'].includes(ext)) {
    showToast(`Unsupported file type: .${ext}`, 'error');
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showToast('File exceeds 50 MB limit', 'error');
    return;
  }
  state.selectedFile   = file;
  elFileName.textContent = file.name;
  elFileSize.textContent = formatBytes(file.size);
  elDropContent.hidden   = true;
  elFileSelected.hidden  = false;
}

function formatBytes(bytes) {
  if (bytes < 1024)       return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ─── Submit Text ──────────────────────────────────────────────────────────────
elSubmitText.addEventListener('click', async () => {
  const text = elTextInput.value.trim();
  if (!text) { showToast('Please enter some text first.', 'error'); return; }
  setLoading(elSubmitText, true);
  try {
    const res = await fetch('/synthesize', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ text, emotion: state.textEmotion }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');
    await handleAudioResult(data);
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    setLoading(elSubmitText, false);
  }
});

// ─── Submit File ──────────────────────────────────────────────────────────────
elSubmitFile.addEventListener('click', async () => {
  if (!state.selectedFile) { showToast('Please select a file first.', 'error'); return; }
  const formData = new FormData();
  formData.append('file',    state.selectedFile);
  formData.append('emotion', state.fileEmotion);
  setLoading(elSubmitFile, true);
  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');
    await handleAudioResult(data);
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    setLoading(elSubmitFile, false);
  }
});

// ─── Handle audio result from server ─────────────────────────────────────────
async function handleAudioResult(data) {
  stopAudio();
  state.audioFile = data.file;
  state.audioText = data.text || '';
  state.words     = tokenizeWords(state.audioText);

  // Show output panel immediately
  elOutputPanel.hidden = false;
  elOutputPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Emotion badge
  const EMOTION_LABELS = {
    angry: '😠 Angry', happy: '😊 Happy', neutral: '😐 Neutral',
    sad: '😢 Sad', surprise: '😲 Surprise'
  };
  elMetaEmotion.textContent = EMOTION_LABELS[data.emotion] || data.emotion;

  // Set audio source with cache-busting to force fresh fetch
  const audioSrc = `/audio/${data.file}?t=${Date.now()}`;
  elAudio.src    = '';          // reset first
  elAudio.src    = audioSrc;
  elBtnDownload.href     = `/audio/${data.file}`;
  elBtnDownload.download = data.file;

  // Show duration from server immediately (before decode)
  if (data.duration) {
    elTimeTotal.textContent    = formatTime(data.duration);
    elMetaDuration.textContent = `${data.duration.toFixed(1)} s`;
  }

  // Build transcript right away
  buildTranscript(state.words);

  // Decode audio buffer for waveform (async, don't block UI)
  try {
    // Create fresh AudioContext each time to avoid stale state
    if (state.audioCtx) {
      try { await state.audioCtx.close(); } catch(_) {}
    }
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    const resp = await fetch(`/audio/${data.file}?t=${Date.now()}`);
    if (!resp.ok) throw new Error(`Fetch failed: ${resp.status}`);
    const arrBuf = await resp.arrayBuffer();
    if (arrBuf.byteLength === 0) throw new Error('Empty audio buffer');

    state.audioBuffer = await state.audioCtx.decodeAudioData(arrBuf);
    const dur = state.audioBuffer.duration;
    elTimeTotal.textContent    = formatTime(dur);
    elMetaDuration.textContent = `${dur.toFixed(1)} s`;
    buildWordTimings(state.words, dur);
    drawWaveform(state.audioBuffer, 0);
  } catch (e) {
    console.warn('Waveform decode failed:', e);
    // Draw a placeholder flat waveform so panel doesn't look broken
    drawFlatWaveform();
    if (data.duration) buildWordTimings(state.words, data.duration);
  }

  showToast('✅ Audio ready — press Play!', 'success');
}

// ─── Waveform drawing ─────────────────────────────────────────────────────────
function drawWaveform(audioBuffer, progress = 0) {
  const canvas = elWaveformCanvas;
  const ctx    = canvas.getContext('2d');
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;

  const data     = audioBuffer.getChannelData(0);
  const step     = Math.ceil(data.length / W);
  const mid      = H / 2;
  const splitPx  = Math.floor(W * progress);

  ctx.clearRect(0, 0, W, H);

  for (let x = 0; x < W; x++) {
    let min = 0, max = 0;
    for (let j = 0; j < step; j++) {
      const s = data[x * step + j] || 0;
      if (s < min) min = s;
      if (s > max) max = s;
    }
    const top  = mid - max * mid * 0.85;
    const bot  = mid - min * mid * 0.85;
    const h    = Math.max(1, bot - top);

    if (x < splitPx) {
      ctx.fillStyle = `rgba(110,86,255,0.9)`;
    } else {
      ctx.fillStyle = `rgba(100,100,180,0.35)`;
    }
    ctx.fillRect(x, top, 1, h);
  }
}

// Fallback: draw a simple animated placeholder waveform
function drawFlatWaveform() {
  const canvas = elWaveformCanvas;
  const ctx    = canvas.getContext('2d');
  const W = canvas.width  = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  ctx.clearRect(0, 0, W, H);
  const mid = H / 2;
  // Draw random-height bars as placeholder
  for (let x = 0; x < W; x += 3) {
    const h = (Math.sin(x * 0.05) * 0.4 + Math.random() * 0.3 + 0.1) * mid * 0.8;
    ctx.fillStyle = 'rgba(100,100,180,0.35)';
    ctx.fillRect(x, mid - h, 2, h * 2);
  }
}
function tokenizeWords(text) {
  return text.match(/\S+/g) || [];
}

function buildWordTimings(words, duration) {
  if (!words.length) return;
  // Evenly distribute word timings (approximation without forced alignment)
  const avgWPM  = 150;
  const estDur  = Math.max(duration, (words.length / avgWPM) * 60);
  const wordDur = estDur / words.length;
  state.wordTimings = words.map((_, i) => ({
    start: i * wordDur,
    end  : (i + 1) * wordDur,
  }));
}

// ─── Transcript builder ───────────────────────────────────────────────────────
function buildTranscript(words) {
  elTranscript.innerHTML = '';
  words.forEach((word, i) => {
    const span = document.createElement('span');
    span.className    = 'word';
    span.dataset.index = i;
    span.textContent  = word + ' ';
    elTranscript.appendChild(span);
  });
}

// ─── Audio controls ───────────────────────────────────────────────────────────
elAudio.addEventListener('loadedmetadata', () => {
  elTimeTotal.textContent = formatTime(elAudio.duration);
});

elAudio.addEventListener('timeupdate', () => {
  const pct = elAudio.duration ? elAudio.currentTime / elAudio.duration : 0;
  updateSeek(pct);
  elTimeCurrent.textContent = formatTime(elAudio.currentTime);
  highlightWord(elAudio.currentTime);
  elWaveformProg.style.width = (pct * 100) + '%';
  if (state.audioBuffer) drawWaveform(state.audioBuffer, pct);
});

elAudio.addEventListener('ended', () => {
  elBtnPlay.hidden  = false;
  elBtnPause.hidden = true;
  state.isPlaying   = false;
  updateSeek(1);
  unhighlightAll();
});

elBtnPlay.addEventListener('click', async () => {
  // Resume AudioContext if suspended (browser autoplay policy)
  if (state.audioCtx && state.audioCtx.state === 'suspended') {
    await state.audioCtx.resume();
  }
  // Ensure audio is loaded before playing
  if (elAudio.readyState < 2) {
    await new Promise(resolve => {
      elAudio.addEventListener('canplay', resolve, { once: true });
    });
  }
  try {
    await elAudio.play();
    elBtnPlay.hidden  = true;
    elBtnPause.hidden = false;
    state.isPlaying   = true;
  } catch (e) {
    showToast('Playback failed. Try clicking play again.', 'error');
    console.error('Play error:', e);
  }
});

elBtnPause.addEventListener('click', () => {
  elAudio.pause();
  elBtnPlay.hidden  = false;
  elBtnPause.hidden = true;
  state.isPlaying   = false;
});

elBtnStop.addEventListener('click', stopAudio);

function stopAudio() {
  elAudio.pause();
  elAudio.currentTime = 0;
  elBtnPlay.hidden    = false;
  elBtnPause.hidden   = true;
  state.isPlaying     = false;
  updateSeek(0);
  elTimeCurrent.textContent = '0:00';
  unhighlightAll();
  elWaveformProg.style.width = '0%';
}

// ─── Seek bar ─────────────────────────────────────────────────────────────────
function updateSeek(pct) {
  const clamped = Math.max(0, Math.min(1, pct));
  elSeekFill.style.width  = (clamped * 100) + '%';
  elSeekThumb.style.left  = (clamped * 100) + '%';
}

elSeekBar.addEventListener('click', e => {
  const rect = elSeekBar.getBoundingClientRect();
  const pct  = (e.clientX - rect.left) / rect.width;
  if (elAudio.duration) {
    elAudio.currentTime = pct * elAudio.duration;
  }
});

// Waveform click to seek
elWaveformCanvas.parentElement.addEventListener('click', e => {
  const rect = elWaveformCanvas.parentElement.getBoundingClientRect();
  const pct  = (e.clientX - rect.left) / rect.width;
  if (elAudio.duration) elAudio.currentTime = pct * elAudio.duration;
});

// ─── Word highlighting ────────────────────────────────────────────────────────
function highlightWord(currentTime) {
  const timings = state.wordTimings;
  if (!timings.length) return;
  const wordSpans = elTranscript.querySelectorAll('.word');
  timings.forEach((t, i) => {
    if (!wordSpans[i]) return;
    if (currentTime >= t.start && currentTime < t.end) {
      wordSpans[i].classList.add('highlight');
      wordSpans[i].classList.remove('done');
      // Scroll into view if needed
      wordSpans[i].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    } else if (currentTime >= t.end) {
      wordSpans[i].classList.remove('highlight');
      wordSpans[i].classList.add('done');
    } else {
      wordSpans[i].classList.remove('highlight', 'done');
    }
  });
}

function unhighlightAll() {
  elTranscript.querySelectorAll('.word').forEach(s => s.classList.remove('highlight', 'done'));
}

// ─── Toast ────────────────────────────────────────────────────────────────────
let _toastTimer;
function showToast(msg, type = 'info') {
  elToast.textContent  = msg;
  elToast.className    = `toast toast--${type} show`;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { elToast.classList.remove('show'); }, 3500);
}

// ─── Loading state toggle ─────────────────────────────────────────────────────
function setLoading(btn, loading) {
  const inner  = btn.querySelector('.btn-inner');
  const loader = btn.querySelector('.btn-loader');
  btn.disabled    = loading;
  if (inner)  inner.hidden  = loading;
  if (loader) loader.hidden = !loading;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatTime(sec) {
  if (!isFinite(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
