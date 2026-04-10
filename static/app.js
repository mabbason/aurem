let ws = null;
let activeSessionId = null;
let isRecording = false;
let timerInterval = null;
let startTime = null;
let viewingSessionId = null;
let availableDevices = [];

// --- Toast ---

function showToast(message, duration = 3000) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

// --- Collapsible sections ---

function toggleSection(name) {
    const body = document.getElementById(`section-${name}`);
    const chevron = document.getElementById(`chevron-${name}`);
    body.classList.toggle('collapsed');
    chevron.classList.toggle('collapsed');
    localStorage.setItem(`section-${name}-collapsed`, body.classList.contains('collapsed'));
}

function restoreSectionStates() {
    for (const name of ['devices', 'sessions']) {
        const collapsed = localStorage.getItem(`section-${name}-collapsed`) === 'true';
        if (collapsed) {
            document.getElementById(`section-${name}`).classList.add('collapsed');
            document.getElementById(`chevron-${name}`).classList.add('collapsed');
        }
    }
}

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => console.log('WebSocket connected');

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'segment' && isRecording) {
            appendSegment(msg.data);
        } else if (msg.type === 'session_state') {
            if (msg.data.active) {
                msg.data.segments.forEach(seg => appendSegment(seg));
            }
        } else if (msg.type === 'diarization_complete') {
            // Re-render transcript with updated speaker labels
            const transcript = document.getElementById('transcript');
            transcript.innerHTML = '';
            msg.data.segments.forEach(seg => appendSegment(seg));
            showToast('Speaker identification complete');
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 2000);
    };
}

// --- Device selection ---

function getIgnoredDevices() {
    const saved = localStorage.getItem('ignoredDevices');
    return saved ? new Set(JSON.parse(saved)) : new Set();
}

function saveIgnoredDevices(ignored) {
    localStorage.setItem('ignoredDevices', JSON.stringify([...ignored]));
}

function ignoreDevice(index) {
    const ignored = getIgnoredDevices();
    ignored.add(index);
    saveIgnoredDevices(ignored);

    // Also uncheck it
    const selected = getSelectedDeviceIndices().filter(i => i !== index);
    localStorage.setItem('selectedDevices', JSON.stringify(selected));

    renderDeviceList(new Set(selected));
}

function resetIgnoredDevices() {
    localStorage.removeItem('ignoredDevices');
    const selected = new Set(getSelectedDeviceIndices());
    renderDeviceList(selected);
}

async function loadDevices() {
    const resp = await fetch('/api/devices');
    availableDevices = await resp.json();

    // Load saved selection from localStorage, fall back to defaults
    const saved = localStorage.getItem('selectedDevices');
    let selectedIndices;
    if (saved) {
        selectedIndices = new Set(JSON.parse(saved));
        const validIndices = new Set(availableDevices.map(d => d.index));
        selectedIndices = new Set([...selectedIndices].filter(i => validIndices.has(i)));
        if (selectedIndices.size === 0) {
            selectedIndices = new Set(availableDevices.filter(d => d.default).map(d => d.index));
        }
    } else {
        selectedIndices = new Set(availableDevices.filter(d => d.default).map(d => d.index));
    }

    renderDeviceList(selectedIndices);
}

function renderDeviceList(selectedIndices) {
    const list = document.getElementById('device-list');

    if (availableDevices.length === 0) {
        list.innerHTML = '<p class="empty">No devices found</p>';
        return;
    }

    const ignored = getIgnoredDevices();
    const visible = availableDevices.filter(d => !ignored.has(d.index));

    if (visible.length === 0) {
        list.innerHTML = '<p class="empty">All sources hidden</p>' +
            '<p class="device-reset" onclick="resetIgnoredDevices()">Reset audio sources</p>';
        return;
    }

    const items = visible.map(d => {
        const checked = selectedIndices.has(d.index) ? 'checked' : '';
        const icon = d.type === 'loopback' ? '\u{1F50A}' : '\u{1F3A4}';
        const shortName = d.name.replace(/\[Loopback\]/i, '').replace(/\(.*?\)/g, '').trim();
        const disabledAttr = isRecording ? 'disabled' : '';
        return `
            <div class="device-item" title="${d.name}">
                <label class="device-label">
                    <input type="checkbox" value="${d.index}" ${checked}
                           onchange="onDeviceToggle()" ${disabledAttr}>
                    <span class="device-icon">${icon}</span>
                    <span class="device-name">${shortName}</span>
                </label>
                <button class="btn-ignore" onclick="ignoreDevice(${d.index})" title="Ignore">&times;</button>
            </div>
        `;
    }).join('');

    const resetLink = ignored.size > 0
        ? '<p class="device-reset" onclick="resetIgnoredDevices()">Reset audio sources</p>'
        : '';

    list.innerHTML = items + resetLink;
}

function getSelectedDeviceIndices() {
    const checkboxes = document.querySelectorAll('#device-list input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => parseInt(cb.value));
}

function onDeviceToggle() {
    const selected = getSelectedDeviceIndices();
    localStorage.setItem('selectedDevices', JSON.stringify(selected));
}

// --- Session management ---

async function startSession() {
    const devices = getSelectedDeviceIndices();
    if (devices.length === 0) {
        alert('Select at least one audio source');
        return;
    }

    const resp = await fetch('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ devices }),
    });
    const data = await resp.json();

    activeSessionId = data.id;
    viewingSessionId = data.id;
    isRecording = true;

    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('status').className = 'status recording';
    document.getElementById('status').textContent = 'Recording';
    document.getElementById('export-bar').style.display = 'none';

    // Disable device checkboxes during recording
    document.querySelectorAll('#device-list input[type="checkbox"]').forEach(cb => cb.disabled = true);

    const transcript = document.getElementById('transcript');
    transcript.innerHTML = '';

    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);

    loadSessions();
}

async function stopSession() {
    const resp = await fetch('/api/session/stop', { method: 'POST' });
    const data = await resp.json();

    isRecording = false;
    activeSessionId = null;

    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('status').className = 'status idle';
    document.getElementById('status').textContent = 'Idle';

    // Re-enable device checkboxes
    document.querySelectorAll('#device-list input[type="checkbox"]').forEach(cb => cb.disabled = false);

    clearInterval(timerInterval);

    if (data.segment_count === 0) {
        showToast('No audio segments detected');
        viewingSessionId = null;
        document.getElementById('transcript').innerHTML = '<p class="placeholder">Start a recording or select a past session to view the transcript.</p>';
        document.getElementById('export-bar').style.display = 'none';
    } else if (viewingSessionId) {
        document.getElementById('export-bar').style.display = 'flex';
    }

    loadSessions();
}

function updateTimer() {
    if (!startTime) return;
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    document.getElementById('timer').textContent = `${h}:${m}:${s}`;
}

function appendSegment(seg) {
    const transcript = document.getElementById('transcript');
    const placeholder = transcript.querySelector('.placeholder');
    if (placeholder) placeholder.remove();

    const div = document.createElement('div');
    div.className = 'segment';
    div.dataset.speaker = seg.speaker;

    const ts = formatTimestamp(seg.start);
    div.innerHTML = `
        <div class="meta">
            <span class="timestamp">[${ts}]</span>
            <span class="speaker">${seg.speaker}</span>
        </div>
        <div class="text">${escapeHtml(seg.text)}</div>
    `;

    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
}

function formatTimestamp(seconds) {
    const h = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
    const s = String(Math.floor(seconds % 60)).padStart(2, '0');
    const ms = Math.floor((seconds % 1) * 10);
    return `${h}:${m}:${s}.${ms}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDuration(seconds) {
    if (!seconds || seconds <= 0) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

async function loadSessions() {
    const resp = await fetch('/api/sessions');
    const sessions = await resp.json();
    const list = document.getElementById('session-list');

    if (sessions.length === 0) {
        list.innerHTML = '<p class="empty">No sessions yet</p>';
        return;
    }

    list.innerHTML = sessions.map(s => {
        const date = new Date(s.started_at);
        const dateStr = date.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric',
        }) + ', ' + date.toLocaleTimeString('en-US', {
            hour: 'numeric', minute: '2-digit',
        });
        const duration = formatDuration(s.duration);
        const title = escapeHtml(s.title || s.id);
        const active = s.id === viewingSessionId ? ' active' : '';
        return `
            <div class="session-item${active}" data-id="${s.id}">
                <div class="session-row" onclick="sessionClick('${s.id}')" ondblclick="sessionDblClick(event, '${s.id}')">
                    <div class="session-title">${title}</div>
                    <div class="session-meta">
                        <span class="session-date">${dateStr}</span>
                        <span class="session-duration">${duration}</span>
                    </div>
                </div>
                <button class="btn-delete" onclick="event.stopPropagation(); deleteSession('${s.id}')" title="Delete session">&times;</button>
            </div>
        `;
    }).join('');
}

let clickTimer = null;

function sessionClick(sessionId) {
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; return; }
    clickTimer = setTimeout(() => {
        clickTimer = null;
        viewSession(sessionId);
    }, 250);
}

function sessionDblClick(event, sessionId) {
    event.stopPropagation();
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
    const item = event.currentTarget.closest('.session-item');
    const titleEl = item.querySelector('.session-title');
    startRename(sessionId, titleEl);
}

function startRename(sessionId, el) {
    const current = el.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'rename-input';
    input.value = current;

    let finished = false;
    const finish = async () => {
        if (finished) return;
        finished = true;
        const newTitle = input.value.trim();
        if (newTitle && newTitle !== current) {
            await fetch(`/api/sessions/${sessionId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle }),
            });
        }
        loadSessions();
    };

    input.addEventListener('blur', finish);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') input.blur();
        if (e.key === 'Escape') { input.value = current; input.blur(); }
    });

    el.replaceWith(input);
    input.focus();
    input.select();
}

async function viewSession(sessionId) {
    if (isRecording && sessionId !== activeSessionId) {
        return;
    }

    const resp = await fetch(`/api/sessions/${sessionId}`);
    const data = await resp.json();

    viewingSessionId = sessionId;
    const transcript = document.getElementById('transcript');
    transcript.innerHTML = '';

    data.segments.forEach(seg => appendSegment(seg));

    document.getElementById('export-bar').style.display = 'flex';
    loadSpeakerMapping(sessionId);
    loadSessions();
}

async function deleteSession(sessionId) {
    if (!confirm('Delete this recording?')) return;
    await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    if (viewingSessionId === sessionId) {
        viewingSessionId = null;
        document.getElementById('transcript').innerHTML = '<p class="placeholder">Start a recording or select a past session to view the transcript.</p>';
        document.getElementById('export-bar').style.display = 'none';
        document.getElementById('speaker-mapping').style.display = 'none';
    }
    loadSessions();
}

function exportSession(fmt) {
    if (!viewingSessionId) return;
    window.open(`/api/sessions/${viewingSessionId}/export/${fmt}`, '_blank');
}

async function pollStatus() {
    try {
        const resp = await fetch('/api/session/status');
        const data = await resp.json();

        const audioEl = document.getElementById('audio-status');
        if (data.active && data.audio_capturing) {
            audioEl.innerHTML = '<span class="connected">Capturing</span>';
        } else if (data.active) {
            audioEl.innerHTML = '<span class="disconnected">Starting...</span>';
        } else {
            audioEl.innerHTML = 'Idle';
        }
    } catch (e) {
        // Server not reachable
    }
}

// --- AI Config ---

let aiConfigured = false;
let diarizationAvailable = false;

const API_MODELS = {
    anthropic: [
        { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4' },
        { id: 'claude-haiku-4-5-20251001', name: 'Claude Haiku 4.5' },
    ],
    openai: [
        { id: 'gpt-4o', name: 'GPT-4o' },
        { id: 'gpt-4o-mini', name: 'GPT-4o Mini' },
    ],
};

async function loadAiConfig() {
    try {
        const resp = await fetch('/api/ai-config');
        const cfg = await resp.json();
        aiConfigured = !!cfg.provider;
        updateAiExportButtons();
        return cfg;
    } catch (e) {
        return {};
    }
}

function updateAiExportButtons() {
    const btnSummary = document.getElementById('btn-summary');
    const btnLessons = document.getElementById('btn-lessons');

    if (aiConfigured) {
        btnSummary.disabled = false;
        btnSummary.title = 'Generate AI summary';
    } else {
        btnSummary.disabled = true;
        btnSummary.title = 'Configure AI to unlock';
    }

    if (aiConfigured && diarizationAvailable) {
        btnLessons.disabled = false;
        btnLessons.title = 'Generate lesson notes from transcript';
    } else if (aiConfigured && !diarizationAvailable) {
        // Allow lessons even without diarization — speaker labels will just be generic
        btnLessons.disabled = false;
        btnLessons.title = 'Generate lesson notes (speaker labels unavailable without diarization)';
    } else {
        btnLessons.disabled = true;
        btnLessons.title = 'Configure AI to unlock';
    }
}

async function loadDiarizationStatus() {
    try {
        const resp = await fetch('/api/diarization/status');
        const data = await resp.json();
        diarizationAvailable = data.available;
        updateAiExportButtons();

        const el = document.getElementById('diarization-status');
        if (data.available) {
            el.innerHTML = 'Speakers: <span class="active">On</span>';
        } else if (data.hf_token_set) {
            el.innerHTML = 'Speakers: <span class="inactive">Loading failed</span>';
        } else {
            el.innerHTML = 'Speakers: <span class="inactive">Off</span>';
        }
    } catch (e) {
        // ignore
    }
}

async function openConfigModal() {
    const modal = document.getElementById('config-modal');
    modal.style.display = 'flex';

    const cfg = await loadAiConfig();
    document.getElementById('test-result').textContent = '';

    // Reset toggle buttons
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('ollama-settings').style.display = 'none';
    document.getElementById('api-settings').style.display = 'none';

    if (cfg.provider === 'ollama') {
        selectProvider('ollama', false);
        document.getElementById('ollama-url').value = cfg.ollama_url || 'http://localhost:11434';
        // Fetch models then select saved one
        await fetchOllamaModels();
        if (cfg.ollama_model) {
            document.getElementById('ollama-model').value = cfg.ollama_model;
        }
    } else if (cfg.provider === 'api') {
        selectProvider('api', false);
        document.getElementById('api-provider').value = cfg.api_provider || 'anthropic';
        document.getElementById('api-key').value = '';
        document.getElementById('api-key-preview').textContent = cfg.api_key_preview
            ? `Current key: ${cfg.api_key_preview}`
            : '';
        updateApiModels();
        if (cfg.api_model) {
            document.getElementById('api-model').value = cfg.api_model;
        }
    }
}

function closeConfigModal() {
    document.getElementById('config-modal').style.display = 'none';
}

function selectProvider(provider, clearTest = true) {
    document.querySelectorAll('.toggle-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.provider === provider);
    });

    document.getElementById('ollama-settings').style.display = provider === 'ollama' ? 'block' : 'none';
    document.getElementById('api-settings').style.display = provider === 'api' ? 'block' : 'none';

    if (clearTest) {
        document.getElementById('test-result').textContent = '';
    }

    if (provider === 'ollama') {
        fetchOllamaModels();
    }
}

function getSelectedProvider() {
    const active = document.querySelector('.toggle-btn.active');
    return active ? active.dataset.provider : '';
}

async function fetchOllamaModels() {
    const select = document.getElementById('ollama-model');
    const url = document.getElementById('ollama-url').value.trim();

    select.innerHTML = '<option value="">Loading...</option>';

    try {
        const resp = await fetch(`${url}/api/tags`);
        const data = await resp.json();
        const models = data.models || [];

        if (models.length === 0) {
            select.innerHTML = '<option value="">No models found</option>';
            return;
        }

        select.innerHTML = models.map(m =>
            `<option value="${m.name}">${m.name}</option>`
        ).join('');
    } catch (e) {
        select.innerHTML = '<option value="">Cannot reach Ollama</option>';
    }
}

function updateApiModels() {
    const provider = document.getElementById('api-provider').value;
    const select = document.getElementById('api-model');
    const models = API_MODELS[provider] || [];

    select.innerHTML = models.map(m =>
        `<option value="${m.id}">${m.name}</option>`
    ).join('');
}

async function testConnection() {
    const resultEl = document.getElementById('test-result');
    const btn = document.getElementById('btn-test');
    resultEl.textContent = 'Testing...';
    resultEl.className = 'test-result';
    btn.disabled = true;

    // Save config first so the backend has the latest
    await saveConfig(true);

    try {
        const resp = await fetch('/api/ai-config/test', { method: 'POST' });
        const data = await resp.json();

        if (data.ok) {
            resultEl.textContent = data.message || 'Connected!';
            resultEl.className = 'test-result success';
        } else {
            resultEl.textContent = data.error || 'Connection failed';
            resultEl.className = 'test-result error';
        }
    } catch (e) {
        resultEl.textContent = 'Request failed';
        resultEl.className = 'test-result error';
    }

    btn.disabled = false;
}

async function saveConfig(silent = false) {
    const provider = getSelectedProvider();
    const body = { provider };

    if (provider === 'ollama') {
        body.ollama_url = document.getElementById('ollama-url').value.trim();
        body.ollama_model = document.getElementById('ollama-model').value;
    } else if (provider === 'api') {
        body.api_provider = document.getElementById('api-provider').value;
        body.api_model = document.getElementById('api-model').value;
        const keyInput = document.getElementById('api-key').value.trim();
        if (keyInput) body.api_key = keyInput;
    }

    await fetch('/api/ai-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    await loadAiConfig();

    if (!silent) {
        closeConfigModal();
        showToast('AI configuration saved');
    }
}

// --- Summary Export ---

async function exportSummary() {
    if (!viewingSessionId || !aiConfigured) return;

    const modal = document.getElementById('summary-modal');
    const content = document.getElementById('summary-content');
    modal.style.display = 'flex';
    content.innerHTML = '<div class="summary-loading"><div class="spinner"></div><p>Generating summary...</p></div>';

    try {
        const resp = await fetch(`/api/sessions/${viewingSessionId}/export/summary`, {
            method: 'POST',
        });
        const data = await resp.json();

        if (data.error) {
            content.innerHTML = `<p class="test-result error">${escapeHtml(data.error)}</p>`;
        } else {
            content.innerHTML = renderMarkdown(data.summary);
        }
    } catch (e) {
        content.innerHTML = '<p class="test-result error">Failed to generate summary</p>';
    }
}

async function exportLessons() {
    if (!viewingSessionId || !aiConfigured) return;

    const modal = document.getElementById('summary-modal');
    const content = document.getElementById('summary-content');
    document.querySelector('#summary-modal .modal-header h2').textContent = 'Lesson Notes';
    modal.style.display = 'flex';
    content.innerHTML = '<div class="summary-loading"><div class="spinner"></div><p>Generating lesson notes...</p></div>';

    try {
        const resp = await fetch(`/api/sessions/${viewingSessionId}/export/lessons`, {
            method: 'POST',
        });
        const data = await resp.json();

        if (data.error) {
            content.innerHTML = `<p class="test-result error">${escapeHtml(data.error)}</p>`;
        } else {
            content.innerHTML = renderMarkdown(data.lessons);
        }
    } catch (e) {
        content.innerHTML = '<p class="test-result error">Failed to generate lesson notes</p>';
    }
}

function closeSummaryModal() {
    document.getElementById('summary-modal').style.display = 'none';
    document.querySelector('#summary-modal .modal-header h2').textContent = 'Meeting Summary';
}

function copySummary() {
    const content = document.getElementById('summary-content');
    navigator.clipboard.writeText(content.innerText);
    showToast('Summary copied to clipboard');
}

function renderMarkdown(text) {
    // Simple markdown rendering for summary output
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/^- \[x\] (.+)$/gm, '<li style="list-style:none">&#9745; $1</li>')
        .replace(/^- \[ \] (.+)$/gm, '<li style="list-style:none">&#9744; $1</li>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
        .replace(/<\/ul>\s*<ul>/g, '')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

// --- Tab switching ---

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
    document.querySelectorAll('.tab-content').forEach(c => {
        c.style.display = c.id === `tab-${tab}` ? '' : 'none';
        c.classList.toggle('active', c.id === `tab-${tab}`);
    });
}

// --- File Transcription ---

let selectedFile = null;
let currentFileJobId = null;
let filePollingInterval = null;

function onFileSelected(input) {
    if (input.files.length > 0) {
        selectedFile = input.files[0];
        document.getElementById('file-drop-text').textContent = selectedFile.name;
        document.getElementById('file-drop-zone').classList.add('has-file');
        document.getElementById('btn-transcribe').disabled = false;
    }
}

// Drag and drop
document.addEventListener('DOMContentLoaded', () => {
    const zone = document.getElementById('file-drop-zone');
    if (!zone) return;

    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            document.getElementById('file-input').files = e.dataTransfer.files;
            onFileSelected(document.getElementById('file-input'));
        }
    });
});

async function startFileTranscription() {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    const numSpeakers = document.getElementById('file-num-speakers').value;
    if (numSpeakers) formData.append('num_speakers', numSpeakers);

    const labelSpeakers = document.getElementById('file-label-speakers').checked;
    formData.append('label_speakers', labelSpeakers);

    document.getElementById('btn-transcribe').disabled = true;
    document.getElementById('file-progress').style.display = 'flex';
    document.getElementById('file-progress-text').textContent = 'Uploading...';
    document.getElementById('file-export-bar').style.display = 'none';
    document.getElementById('file-speakers').style.display = 'none';
    document.getElementById('file-transcript').innerHTML = '<p class="placeholder">Processing...</p>';

    try {
        const resp = await fetch('/api/transcribe-file', { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.error) {
            document.getElementById('file-progress-text').textContent = data.error;
            document.getElementById('btn-transcribe').disabled = false;
            return;
        }

        currentFileJobId = data.job_id;
        filePollingInterval = setInterval(pollFileJob, 2000);
    } catch (e) {
        document.getElementById('file-progress-text').textContent = 'Upload failed';
        document.getElementById('btn-transcribe').disabled = false;
    }
}

async function pollFileJob() {
    if (!currentFileJobId) return;

    try {
        const resp = await fetch(`/api/transcribe-file/${currentFileJobId}/status`);
        const data = await resp.json();

        document.getElementById('file-progress-text').textContent = data.progress;

        if (data.status === 'completed') {
            clearInterval(filePollingInterval);
            filePollingInterval = null;
            document.getElementById('file-progress').style.display = 'none';
            document.getElementById('btn-transcribe').disabled = false;

            // Show speaker summary
            if (data.speakers && Object.keys(data.speakers).length > 0) {
                const speakersEl = document.getElementById('file-speakers');
                speakersEl.style.display = 'block';
                speakersEl.innerHTML = '<h3>Speakers</h3>' +
                    Object.values(data.speakers).map(s =>
                        `<div class="speaker-stat"><span class="speaker-label">${escapeHtml(s.label)}</span><span class="speaker-time">${formatDuration(s.total_speaking_time)}</span></div>`
                    ).join('');
            }

            loadFileResult();
        } else if (data.status === 'failed') {
            clearInterval(filePollingInterval);
            filePollingInterval = null;
            document.getElementById('file-progress-text').textContent = data.progress;
            document.getElementById('btn-transcribe').disabled = false;
        }
    } catch (e) {
        // keep polling
    }
}

async function loadFileResult() {
    if (!currentFileJobId) return;

    const resp = await fetch(`/api/transcribe-file/${currentFileJobId}/result`);
    const data = await resp.json();

    const transcript = document.getElementById('file-transcript');
    transcript.innerHTML = '';

    for (const seg of data.segments) {
        const div = document.createElement('div');
        div.className = 'segment';
        div.dataset.speaker = seg.speaker;
        const ts = formatTimestamp(seg.start);
        div.innerHTML = `
            <div class="meta">
                <span class="timestamp">[${ts}]</span>
                <span class="speaker">${escapeHtml(seg.speaker)}</span>
            </div>
            <div class="text">${escapeHtml(seg.text)}</div>
        `;
        transcript.appendChild(div);
    }

    document.getElementById('file-export-bar').style.display = 'flex';
}

function downloadFileResult(format) {
    if (!currentFileJobId) return;
    window.open(`/api/transcribe-file/${currentFileJobId}/result?format=${format}`, '_blank');
}

// --- Speaker mapping on session detail ---

let peopleCache = [];

async function loadSpeakerMapping(sessionId) {
    const panel = document.getElementById('speaker-mapping');
    const list = document.getElementById('speaker-mapping-list');
    const statusEl = document.getElementById('extraction-status');
    panel.style.display = 'block';
    list.innerHTML = '<p class="empty">Loading...</p>';
    statusEl.textContent = '';
    statusEl.className = 'extraction-status';

    try {
        const [speakersResp, peopleResp] = await Promise.all([
            fetch(`/api/sessions/${sessionId}/speakers`),
            fetch('/api/people'),
        ]);
        const speakersData = await speakersResp.json();
        peopleCache = await peopleResp.json();

        if (speakersData.error) {
            panel.style.display = 'none';
            return;
        }

        if (speakersData.extraction) {
            const e = speakersData.extraction;
            statusEl.textContent = `Facts: ${e.status}` + (e.error ? ` (${e.error.slice(0, 60)})` : '');
            statusEl.className = `extraction-status ${e.status}`;
        } else {
            statusEl.textContent = 'Facts: not extracted yet';
        }

        if (!speakersData.speakers.length) {
            list.innerHTML = '<p class="empty">No speakers detected</p>';
            return;
        }

        list.innerHTML = speakersData.speakers.map(s => {
            const options = ['<option value="">— Unassigned —</option>']
                .concat(peopleCache.map(p => {
                    const sel = p.id === s.person_id ? 'selected' : '';
                    return `<option value="${p.id}" ${sel}>${escapeHtml(p.name)}</option>`;
                }))
                .concat(['<option value="__new__">+ Add new person...</option>'])
                .join('');
            const factText = s.fact_count > 0 ? `${s.fact_count} fact${s.fact_count === 1 ? '' : 's'}` : 'no facts';
            return `
                <div class="speaker-mapping-row" data-speaker="${escapeHtml(s.label)}">
                    <span class="label">${escapeHtml(s.label)}</span>
                    <select onchange="onSpeakerMappingChange('${sessionId}', '${escapeHtml(s.label)}', this)">
                        ${options}
                    </select>
                    <span class="fact-count">${factText}</span>
                </div>
            `;
        }).join('');
    } catch (e) {
        list.innerHTML = '<p class="empty">Failed to load speakers</p>';
    }
}

async function onSpeakerMappingChange(sessionId, label, select) {
    const value = select.value;

    if (value === '__new__') {
        const name = prompt(`New person for ${label}:`);
        if (!name || !name.trim()) {
            select.value = '';
            return;
        }
        try {
            const resp = await fetch('/api/people', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name.trim() }),
            });
            const person = await resp.json();
            await saveSpeakerMapping(sessionId, label, person.id);
            loadSpeakerMapping(sessionId);
            showToast(`Created ${person.name}`);
        } catch (e) {
            showToast('Failed to create person');
            select.value = '';
        }
        return;
    }

    const personId = value === '' ? null : parseInt(value, 10);
    await saveSpeakerMapping(sessionId, label, personId);
    // Refresh to update fact counts
    loadSpeakerMapping(sessionId);
}

async function saveSpeakerMapping(sessionId, label, personId) {
    await fetch(`/api/sessions/${sessionId}/speakers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mappings: { [label]: personId } }),
    });
}

// --- People tab ---

let viewingPersonId = null;

async function loadPeople() {
    const list = document.getElementById('people-list');
    list.innerHTML = '<p class="empty">Loading...</p>';

    try {
        const resp = await fetch('/api/people');
        const people = await resp.json();
        peopleCache = people;

        if (people.length === 0) {
            list.innerHTML = '<p class="empty">No people yet. Add someone to start building the graph.</p>';
            return;
        }

        list.innerHTML = people.map(p => {
            const active = p.id === viewingPersonId ? ' active' : '';
            const factBit = p.fact_count > 0 ? `${p.fact_count} facts` : 'no facts';
            const last = p.last_seen
                ? formatRelativeDate(p.last_seen)
                : 'never';
            return `
                <div class="person-item${active}" onclick="viewPerson(${p.id})">
                    <div class="person-name">${escapeHtml(p.name)}</div>
                    <div class="person-meta">
                        <span>${factBit}</span>
                        <span>&middot;</span>
                        <span>${last}</span>
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) {
        list.innerHTML = '<p class="empty">Failed to load people</p>';
    }
}

function formatRelativeDate(isoString) {
    if (!isoString) return '';
    // SQLite stores UTC without 'Z' suffix; treat as UTC
    const s = isoString.endsWith('Z') || isoString.includes('+') ? isoString : isoString + 'Z';
    const d = new Date(s);
    if (isNaN(d.getTime())) return '';
    const now = new Date();
    const diffMs = now - d;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    if (diffDays === 0) return 'today';
    if (diffDays === 1) return 'yesterday';
    if (diffDays < 7) return `${diffDays}d ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`;
    return `${Math.floor(diffDays / 365)}y ago`;
}

const CATEGORY_ORDER = ['professional', 'personal', 'commitment', 'preference', 'opinion', 'context'];
const CATEGORY_LABELS = {
    professional: 'Professional',
    personal: 'Personal',
    commitment: 'Commitments',
    preference: 'Preferences',
    opinion: 'Opinions',
    context: 'Context',
};

async function viewPerson(personId) {
    viewingPersonId = personId;
    const detail = document.getElementById('person-detail');
    detail.innerHTML = '<p class="placeholder">Loading...</p>';

    try {
        const resp = await fetch(`/api/people/${personId}`);
        const data = await resp.json();
        if (data.error) {
            detail.innerHTML = `<p class="empty">${escapeHtml(data.error)}</p>`;
            return;
        }

        const { person, facts_by_category, meeting_history, total_facts } = data;
        const categoriesHtml = CATEGORY_ORDER
            .filter(cat => facts_by_category[cat] && facts_by_category[cat].length > 0)
            .map(cat => {
                const facts = facts_by_category[cat];
                const items = facts.map(f => {
                    const manualBadge = f.source === 'manual'
                        ? '<span class="fact-source-manual">manual</span>'
                        : '';
                    return `
                        <div class="fact-item">
                            <div class="fact-text">
                                <div>${escapeHtml(f.text)}</div>
                                <div class="fact-meta">
                                    ${manualBadge}
                                    ${f.confidence ? `conf ${Number(f.confidence).toFixed(2)}` : ''}
                                    ${f.session_id ? `&middot; <a href="#" onclick="openSessionFromPerson('${f.session_id}'); return false">session</a>` : ''}
                                </div>
                            </div>
                            <button class="btn-delete-fact" onclick="deleteFact(${f.id}, ${personId})" title="Delete">&times;</button>
                        </div>
                    `;
                }).join('');
                return `
                    <div class="fact-category" data-category="${cat}">
                        <div class="fact-category-header">${CATEGORY_LABELS[cat]} &middot; ${facts.length}</div>
                        ${items}
                    </div>
                `;
            }).join('');

        const meetingsHtml = meeting_history.length
            ? `<div class="meetings-list">${meeting_history.map(m => {
                const date = m.started_at ? formatRelativeDate(m.started_at) : '';
                const dur = formatDuration(m.duration);
                return `
                    <div class="meeting-item" onclick="openSessionFromPerson('${m.session_id}')">
                        <div class="meeting-title">${escapeHtml(m.title)}</div>
                        <div class="meeting-meta">${date} &middot; ${dur}</div>
                    </div>
                `;
            }).join('')}</div>`
            : '<p class="empty-state">No meeting history yet.</p>';

        detail.innerHTML = `
            <div class="person-detail-header">
                <h1>${escapeHtml(person.name)}</h1>
                <div class="person-actions">
                    <button class="btn btn-sm" onclick="openAddFactModal(${personId})">+ Fact</button>
                    <button class="btn btn-sm" onclick="deletePerson(${personId})">Delete</button>
                </div>
            </div>
            <div class="person-notes">
                <label>Notes</label>
                <textarea id="person-notes-${personId}"
                    onblur="savePersonNotes(${personId})"
                    placeholder="Free-form notes about ${escapeHtml(person.name)}...">${escapeHtml(person.notes || '')}</textarea>
            </div>
            <div class="detail-section">
                <h2>Facts (${total_facts})</h2>
                ${categoriesHtml || '<p class="empty-state">No facts yet. Add one manually or extract from a session.</p>'}
            </div>
            <div class="detail-section">
                <h2>Meeting History</h2>
                ${meetingsHtml}
            </div>
        `;
        loadPeople();
    } catch (e) {
        detail.innerHTML = '<p class="empty">Failed to load person</p>';
    }
}

async function savePersonNotes(personId) {
    const el = document.getElementById(`person-notes-${personId}`);
    if (!el) return;
    const notes = el.value;
    await fetch(`/api/people/${personId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes }),
    });
}

async function deletePerson(personId) {
    if (!confirm('Delete this person? Their facts will be unlinked but not deleted.')) return;
    await fetch(`/api/people/${personId}`, { method: 'DELETE' });
    viewingPersonId = null;
    document.getElementById('person-detail').innerHTML =
        '<p class="placeholder">Select a person to see their profile, or add someone new.</p>';
    loadPeople();
}

async function deleteFact(factId, personId) {
    if (!confirm('Delete this fact?')) return;
    await fetch(`/api/facts/${factId}`, { method: 'DELETE' });
    viewPerson(personId);
}

function openSessionFromPerson(sessionId) {
    switchTab('record');
    viewSession(sessionId);
}

// --- Add Person Modal ---

function openAddPersonModal() {
    document.getElementById('new-person-name').value = '';
    document.getElementById('new-person-notes').value = '';
    document.getElementById('add-person-modal').style.display = 'flex';
    setTimeout(() => document.getElementById('new-person-name').focus(), 50);
}

function closeAddPersonModal() {
    document.getElementById('add-person-modal').style.display = 'none';
}

async function createPerson() {
    const name = document.getElementById('new-person-name').value.trim();
    const notes = document.getElementById('new-person-notes').value.trim();
    if (!name) {
        showToast('Name required');
        return;
    }
    const resp = await fetch('/api/people', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, notes }),
    });
    const person = await resp.json();
    closeAddPersonModal();
    await loadPeople();
    viewPerson(person.id);
}

// --- Add Fact Modal ---

let addFactPersonId = null;

function openAddFactModal(personId) {
    addFactPersonId = personId;
    document.getElementById('new-fact-text').value = '';
    document.getElementById('new-fact-category').value = 'context';
    document.getElementById('add-fact-modal').style.display = 'flex';
    setTimeout(() => document.getElementById('new-fact-text').focus(), 50);
}

function closeAddFactModal() {
    document.getElementById('add-fact-modal').style.display = 'none';
    addFactPersonId = null;
}

async function createFact() {
    if (!addFactPersonId) return;
    const text = document.getElementById('new-fact-text').value.trim();
    const category = document.getElementById('new-fact-category').value;
    if (!text) {
        showToast('Fact text required');
        return;
    }
    await fetch(`/api/people/${addFactPersonId}/facts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, category }),
    });
    const personId = addFactPersonId;
    closeAddFactModal();
    viewPerson(personId);
}

// Initialize
restoreSectionStates();
connectWebSocket();
loadDevices();
loadSessions();
loadAiConfig();
loadDiarizationStatus();
setInterval(pollStatus, 3000);
pollStatus();
