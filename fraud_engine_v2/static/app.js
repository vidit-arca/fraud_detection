let currentResults = null;
let currentView = 'overlay'; // 'overlay' or 'original'

document.addEventListener('DOMContentLoaded', () => {
    initDropZone();
    checkHealth();
});

function initDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            const data = await res.json();
            const statusEl = document.getElementById('engineStatusText');
            if (data.model_loaded) {
                statusEl.innerText = `Engine Active (${data.device} • Custom Weights)`;
            } else {
                statusEl.innerText = `Engine Active (${data.device})`;
            }
        }
    } catch (e) {
        console.warn('Health check failed:', e);
    }
}

async function handleFileUpload(file) {
    // Show Loading
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `Server returned error ${response.status}`);
        }

        const data = await response.json();
        currentResults = data;
        renderResults(data);
    } catch (err) {
        alert(`Analysis Error: ${err.message}`);
        resetApp();
    }
}

function renderResults(data) {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'flex';

    // 1. Verdict Banner
    const banner = document.getElementById('verdictBanner');
    const badge = document.getElementById('verdictBadge');
    const title = document.getElementById('verdictTitle');
    const reason = document.getElementById('verdictReason');
    const confVal = document.getElementById('confidenceVal');

    const isTampered = data.status === 'TAMPERED';
    banner.className = `verdict-banner glass-card ${isTampered ? 'tampered' : 'genuine'}`;
    badge.innerText = data.status;
    badge.className = `verdict-badge ${isTampered ? 'tampered' : 'genuine'}`;
    
    title.innerText = isTampered ? 'Document Manipulation Detected' : 'Document Verified Authentic';
    reason.innerText = data.tamper_reason;
    confVal.innerText = `${data.confidence}%`;

    // 2. Viewport Image
    currentView = 'overlay';
    updateViewportImage();

    // 3. Info Badges
    const count = data.tamper_regions_count || 0;
    document.getElementById('regionCountBadge').innerText = isTampered ? `⚠️ ${count} Tampered Region(s) Localized` : `✅ 0 Tampered Regions`;
    document.getElementById('procTimeBadge').innerText = `⏱️ ${data.processing_time}s`;

    // 4. Localized Regions List
    const regionsList = document.getElementById('regionsList');
    regionsList.innerHTML = '';
    
    if (data.tamper_boxes && data.tamper_boxes.length > 0) {
        data.tamper_boxes.forEach((box, i) => {
            const card = document.createElement('div');
            card.className = 'region-card';
            card.innerHTML = `
                <div class="region-card-header">
                    <span class="region-type">#${i + 1} ${box.type || 'Tampering Detected'}</span>
                    <span class="region-conf">${Math.round(box.confidence * 100)}%</span>
                </div>
                <div class="region-desc">${box.reason || 'Anomalous signal noise & texture'}</div>
            `;
            regionsList.appendChild(card);
        });
    } else {
        regionsList.innerHTML = `<div class="empty-regions">No localized tampering detected in this document.</div>`;
    }

    // 5. Metrics Breakdown
    const b = data.breakdown || {};
    setMetricBar('barSeg', 'valSeg', b.pixel_segmentation_score || 0);
    setMetricBar('barSignal', 'valSignal', b.signal_forensics_score || 0);
    setMetricBar('barTypo', 'valTypo', b.typography_score || 0);
    setMetricBar('barDct', 'valDct', b.dct_periodicity || 0);
}

function setMetricBar(barId, valId, score) {
    const bar = document.getElementById(barId);
    const val = document.getElementById(valId);
    const pct = Math.min(100, Math.max(0, score * 100));
    bar.style.width = `${pct}%`;
    val.innerText = score.toFixed(2);
}

function switchView(view) {
    currentView = view;
    document.getElementById('btnShowOverlay').classList.toggle('active', view === 'overlay');
    document.getElementById('btnShowOriginal').classList.toggle('active', view === 'original');
    updateViewportImage();
}

function updateViewportImage() {
    if (!currentResults) return;
    const img = document.getElementById('displayImage');
    if (currentView === 'overlay' && currentResults.overlay_url) {
        img.src = currentResults.overlay_url;
    } else {
        img.src = currentResults.original_url;
    }
}

function resetApp() {
    currentResults = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
}
