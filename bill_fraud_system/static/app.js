// ============================================
// Bill Fraud Detector — Frontend Logic
// ============================================

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const analysisSection = document.getElementById('analysisSection');
const previewImg = document.getElementById('previewImg');
const imageMeta = document.getElementById('imageMeta');
const loadingState = document.getElementById('loadingState');
const resultState = document.getElementById('resultState');
const newScanBtn = document.getElementById('newScanBtn');

// Document type selector
let selectedDocType = 'bill';
const typeBtns = document.querySelectorAll('.type-btn');

typeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        typeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedDocType = btn.dataset.type;
    });
});

// ---- Drag & Drop ----

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

newScanBtn.addEventListener('click', resetUI);

// ---- Handle File Upload ----

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a JPG, PNG, or PDF file.');
        return;
    }

    // Show preview (PDFs can't render in <img>, show placeholder)
    if (file.type === 'application/pdf') {
        previewImg.src = '';
        previewImg.alt = '';
        previewImg.style.display = 'none';
        // Insert a PDF placeholder
        const placeholder = document.createElement('div');
        placeholder.id = 'pdfPlaceholder';
        placeholder.innerHTML = `
            <div style="display:flex;flex-direction:column;align-items:center;gap:12px;padding:40px;color:#94a3b8;">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                <span style="font-size:0.9rem;font-weight:500;">PDF Document</span>
                <span style="font-size:0.78rem;color:#64748b;">Preview will appear after analysis</span>
            </div>`;
        const previewContainer = previewImg.parentElement;
        previewContainer.appendChild(placeholder);
    } else {
        previewImg.style.display = '';
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    // Switch to analysis view
    uploadSection.classList.add('hidden');
    analysisSection.classList.remove('hidden');
    analysisSection.classList.add('fade-in');

    // Show loading
    loadingState.classList.remove('hidden');
    resultState.classList.add('hidden');

    // Animate loading steps
    animateLoadingSteps();

    // Display file meta
    imageMeta.innerHTML = `
        <span>📄 ${file.name}</span>
        <span>📏 ${formatFileSize(file.size)}</span>
        <span>🏷️ ${file.type.split('/')[1].toUpperCase()}</span>
    `;

    // Upload & analyze
    analyzeImage(file);
}

// ---- Loading Steps Animation ----

function animateLoadingSteps() {
    const steps = [
        document.getElementById('step1'),
        document.getElementById('step2'),
        document.getElementById('step3'),
        document.getElementById('step4'),
    ];

    // Reset all
    steps.forEach(s => {
        s.classList.remove('active', 'done');
    });
    steps[0].classList.add('active');

    let current = 0;
    const interval = setInterval(() => {
        if (current < steps.length) {
            steps[current].classList.remove('active');
            steps[current].classList.add('done');
        }
        current++;
        if (current < steps.length) {
            steps[current].classList.add('active');
        } else {
            clearInterval(interval);
        }
    }, 1500);

    // Store interval for cleanup
    window._loadingInterval = interval;
}

// ---- API Call ----

async function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', selectedDocType);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Analysis failed');
        }

        const result = await response.json();

        // Stop loading animation
        if (window._loadingInterval) clearInterval(window._loadingInterval);

        // Mark all steps done
        document.querySelectorAll('.loading-steps .step').forEach(s => {
            s.classList.remove('active');
            s.classList.add('done');
        });

        // Short delay for visual effect
        setTimeout(() => {
            showResult(result);
        }, 600);

    } catch (error) {
        if (window._loadingInterval) clearInterval(window._loadingInterval);
        alert(`Error: ${error.message}`);
        resetUI();
    }
}

// ---- Display Result ----

function showResult(data) {
    loadingState.classList.add('hidden');
    resultState.classList.remove('hidden');
    resultState.classList.add('slide-up');

    const isTampered = data.status === 'TAMPERED';

    // Verdict icon
    const verdictIcon = document.getElementById('verdictIcon');
    verdictIcon.className = 'verdict-icon ' + (isTampered ? 'tampered' : 'genuine');
    verdictIcon.textContent = isTampered ? '⚠️' : '✅';

    // Verdict label
    const verdictLabel = document.getElementById('verdictLabel');
    verdictLabel.className = isTampered ? 'tampered' : 'genuine';
    verdictLabel.textContent = isTampered ? 'Tampered Document' : 'Genuine Document';

    // Verdict sub
    // Verdict sub
    console.log("Analysis Data:", data);

    let subText = isTampered
        ? 'Forensic analysis flagged this document'
        : 'No signs of tampering detected';

    // If specific reason provided, show it
    if (isTampered && data.tamper_reason && data.tamper_reason !== "None") {
        subText = `Analysis detected: ${data.tamper_reason}`;
    }

    document.getElementById('verdictSub').textContent = subText;

    // Confidence ring
    const confidence = data.confidence;
    const ringFill = document.getElementById('ringFill');
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (confidence / 100) * circumference;

    ringFill.className = 'ring-fill ' + (isTampered ? 'tampered' : '');
    // Trigger animation
    requestAnimationFrame(() => {
        ringFill.style.strokeDashoffset = offset;
    });

    // Animate confidence number
    animateNumber('confidenceValue', 0, confidence, 1200);

    // Score breakdown
    document.getElementById('deepScoreVal').textContent = data.deep_score.toFixed(4);
    document.getElementById('forensicScoreVal').textContent = data.forensic_score.toFixed(4);
    document.getElementById('combinedScoreVal').textContent = data.combined_score.toFixed(4);
    document.getElementById('thresholdVal').textContent = data.threshold.toFixed(4);

    // Animate bars (normalized to a visual max)
    setTimeout(() => {
        const deepNorm = Math.min(100, Math.max(5, normalizeScore(data.deep_score, 0, 5)));
        document.getElementById('deepBar').style.width = deepNorm + '%';

        const forensicNorm = Math.min(100, Math.max(5, isTampered ? 95 : normalizeScore(data.forensic_score, 0, 3)));
        document.getElementById('forensicBar').style.width = forensicNorm + '%';
    }, 300);

    // Processing time + model info
    const modelLabel = data.model_name || selectedDocType;
    document.getElementById('processingTime').textContent =
        `Analyzed in ${data.processing_time}s · Model: ${modelLabel}`;

    // Update image meta with dimensions
    if (data.image_info) {
        const info = data.image_info;
        imageMeta.innerHTML = `
            <span>📄 ${info.filename}</span>
            <span>📐 ${info.width} × ${info.height}</span>
            <span>🏷️ ${info.format}</span>
        `;
    }

    // For PDFs: swap placeholder with the server-converted image
    if (data.upload_url) {
        const placeholder = document.getElementById('pdfPlaceholder');
        if (placeholder) {
            placeholder.remove();
        }
        previewImg.src = data.upload_url;
        previewImg.style.display = '';
        previewImg.alt = 'Converted document preview';
    }
}

// ---- Helpers ----

function animateNumber(elementId, start, end, duration) {
    const el = document.getElementById(elementId);
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * eased;
        el.textContent = Math.round(current * 10) / 10;
        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

function normalizeScore(value, min, max) {
    return ((value - min) / (max - min)) * 100;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function resetUI() {
    uploadSection.classList.remove('hidden');
    analysisSection.classList.add('hidden');
    resultState.classList.add('hidden');
    loadingState.classList.remove('hidden');
    fileInput.value = '';

    // Reset ring
    document.getElementById('ringFill').style.strokeDashoffset = 326.73;
    document.getElementById('confidenceValue').textContent = '0';

    // Reset bars
    document.getElementById('deepBar').style.width = '0%';
    document.getElementById('forensicBar').style.width = '0%';

    // Remove PDF placeholder if present
    const placeholder = document.getElementById('pdfPlaceholder');
    if (placeholder) placeholder.remove();
    previewImg.style.display = '';
    previewImg.alt = 'Uploaded document preview';

    // Reset steps
    document.querySelectorAll('.loading-steps .step').forEach(s => {
        s.classList.remove('active', 'done');
    });

    if (window._loadingInterval) clearInterval(window._loadingInterval);
}
