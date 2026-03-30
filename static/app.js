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
        const levelClass = d.type === 'microphone' ? (d.peak > 0.005 ? 'active' : 'silent') : '';
        const levelDot = d.type === 'microphone' ? `<span class="device-level ${levelClass}" title="peak: ${d.peak}"></span>` : '';
        const shortName = d.name.replace(/\[Loopback\]/i, '').replace(/\(.*?\)/g, '').trim();
        const disabledAttr = isRecording ? 'disabled' : '';
        return `
            <div class="device-item" title="${d.name}">
                <label class="device-label">
                    <input type="checkbox" value="${d.index}" ${checked}
                           onchange="onDeviceToggle()" ${disabledAttr}>
                    <span class="device-icon">${icon}</span>
                    <span class="device-name">${shortName}</span>
                    ${levelDot}
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
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        const active = s.id === viewingSessionId ? ' active' : '';
        return `
            <div class="session-item${active}">
                <div class="session-row" onclick="viewSession('${s.id}')">
                    <div class="date">${dateStr}</div>
                    <div class="meta">${s.segment_count} segments</div>
                </div>
                <button class="btn-delete" onclick="event.stopPropagation(); deleteSession('${s.id}')" title="Delete session">&times;</button>
            </div>
        `;
    }).join('');
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
    loadSessions();
}

async function deleteSession(sessionId) {
    if (!confirm('Delete this recording?')) return;
    await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
    if (viewingSessionId === sessionId) {
        viewingSessionId = null;
        document.getElementById('transcript').innerHTML = '<p class="placeholder">Start a recording or select a past session to view the transcript.</p>';
        document.getElementById('export-bar').style.display = 'none';
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

// Initialize
restoreSectionStates();
connectWebSocket();
loadDevices();
loadSessions();
setInterval(pollStatus, 3000);
pollStatus();
