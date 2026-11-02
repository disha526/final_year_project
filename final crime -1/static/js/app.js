/**
 * SENTINEL-AI SURVEILLANCE DASHBOARD CONTROLLER
 * Real-time 11-Crime Classification & Incident Tracker
 */

document.addEventListener('DOMContentLoaded', () => {
    // Common DOM Elements
    const videoElement = document.getElementById('videoElement');
    const threatBadge = document.getElementById('threatLevelBadge');
    const threatText = document.getElementById('threatText');
    const threatOverlay = document.getElementById('threatOverlay');
    const fpsVal = document.getElementById('fpsVal');
    const activeSourceBadge = document.getElementById('activeSourceBadge');

    // Classification Meter Elements
    const dominantBox = document.getElementById('dominantBox');
    const dominantVal = document.getElementById('dominantVal');
    const dominantConf = document.getElementById('dominantConf');

    // Tab Navigation (Upload page)
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Sample Video Cards
    const sampleCards = document.querySelectorAll('.sample-card');

    // Upload Elements
    const dropZone = document.getElementById('dropZone');
    const videoFileInput = document.getElementById('videoFileInput');
    const uploadProgressContainer = document.getElementById('uploadProgressContainer');
    const uploadProgressBar = document.getElementById('uploadProgressBar');
    const uploadStatusLabel = document.getElementById('uploadStatusLabel');

    // Webcam Elements
    const camIndexSelect = document.getElementById('camIndexSelect');
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopStreamBtn = document.getElementById('stopStreamBtn');

    // URL Elements
    const streamUrlInput = document.getElementById('streamUrlInput');
    const connectUrlBtn = document.getElementById('connectUrlBtn');
    const urlPresets = document.querySelectorAll('.url-preset');

    // Incidents
    const incidentFeed = document.getElementById('incidentFeed');
    const incidentCountBadge = document.getElementById('incidentCountBadge');
    const clearIncidentsBtn = document.getElementById('clearIncidentsBtn');
    const emptyIncidentMsg = document.getElementById('emptyIncidentMsg');

    // Audio Alarm
    const audioToggleBtn = document.getElementById('audioToggleBtn');
    const audioIcon = document.getElementById('audioIcon');
    let audioAlertEnabled = true;
    let audioCtx = null;
    let lastAudioBeep = 0;

    // Snapshot Modal
    const snapshotModal = document.getElementById('snapshotModal');
    const modalBackdrop = document.getElementById('modalBackdrop');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalTitle');
    const modalMeta = document.getElementById('modalMeta');

    // Fullscreen & Snap Utilities
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    const snapBtn = document.getElementById('snapBtn');
    const videoWrapper = document.getElementById('videoWrapper');

    // =========================================================================
    // Tab Navigation
    // =========================================================================
    if (tabBtns.length > 0) {
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.dataset.tab;
                tabBtns.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                const targetEl = document.getElementById(target);
                if (targetEl) targetEl.classList.add('active');
            });
        });
    }

    // =========================================================================
    // Video Source API
    // =========================================================================
    async function setSource(payload) {
        try {
            const resp = await fetch('/api/source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (data.success && videoElement) {
                videoElement.src = `/video_feed?t=${Date.now()}`;
            } else if (!data.success) {
                alert(`Error: ${data.error || 'Failed to switch source'}`);
            }
        } catch (err) {
            console.error('Error switching source:', err);
        }
    }

    // Sample Selection
    if (sampleCards.length > 0) {
        sampleCards.forEach(card => {
            card.addEventListener('click', () => {
                sampleCards.forEach(c => c.classList.remove('active-sample'));
                card.classList.add('active-sample');
                const sampleName = card.dataset.sample;
                if (activeSourceBadge) activeSourceBadge.textContent = `Source: Sample (${sampleName})`;
                setSource({ source_type: 'sample', sample_name: sampleName });
            });
        });
    }

    // Live Webcam Start / Stop
    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', () => {
            const camIdx = camIndexSelect ? camIndexSelect.value : 0;
            if (activeSourceBadge) activeSourceBadge.textContent = `Source: Live Camera ${camIdx}`;
            setSource({ source_type: 'webcam', camera_index: parseInt(camIdx) });
        });
    }

    if (stopStreamBtn) {
        stopStreamBtn.addEventListener('click', () => {
            if (activeSourceBadge) activeSourceBadge.textContent = `Source: Offline / Stopped`;
            setSource({ source_type: 'stop' });
        });
    }

    // Stream URL
    if (connectUrlBtn && streamUrlInput) {
        connectUrlBtn.addEventListener('click', () => {
            const url = streamUrlInput.value.trim();
            if (url) {
                if (activeSourceBadge) activeSourceBadge.textContent = `Source: Remote Stream`;
                setSource({ source_type: 'url', url: url });
            }
        });
    }

    if (urlPresets.length > 0) {
        urlPresets.forEach(preset => {
            preset.addEventListener('click', (e) => {
                e.preventDefault();
                const url = preset.dataset.url;
                if (streamUrlInput) streamUrlInput.value = url;
                if (activeSourceBadge) activeSourceBadge.textContent = `Source: Online Benchmark`;
                setSource({ source_type: 'url', url: url });
            });
        });
    }

    // =========================================================================
    // Upload File & Predict Button Controller
    // =========================================================================
    const browseFileBtn = document.getElementById('browseFileBtn');
    const selectedFilePill = document.getElementById('selectedFilePill');
    const selectedFileName = document.getElementById('selectedFileName');
    const predictBtn = document.getElementById('predictBtn');
    let chosenFile = null;

    if (browseFileBtn && videoFileInput) {
        browseFileBtn.addEventListener('click', () => {
            videoFileInput.click();
        });
    }

    if (videoFileInput) {
        videoFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                chosenFile = e.target.files[0];
                if (selectedFileName) selectedFileName.textContent = chosenFile.name;
                if (selectedFilePill) selectedFilePill.classList.add('has-file');
            }
        });
    }

    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
            if (!chosenFile) {
                alert('Please choose a video file first using the "Choose File" button.');
                return;
            }
            handleFileUpload(chosenFile);
        });
    }

    const resetUploadBtn = document.getElementById('resetUploadBtn');
    if (resetUploadBtn) {
        resetUploadBtn.addEventListener('click', async () => {
            await resetToStandby();
        });
    }

    async function resetToStandby() {
        try {
            await fetch('/api/reset', { method: 'POST' });
        } catch (e) {
            console.error('Reset error:', e);
        }

        // Reset file selection
        chosenFile = null;
        if (videoFileInput) videoFileInput.value = '';
        if (selectedFileName) selectedFileName.textContent = 'No video selected';
        if (selectedFilePill) selectedFilePill.classList.remove('has-file');
        if (activeSourceBadge) activeSourceBadge.textContent = 'Source: Awaiting Upload';
        if (uploadProgressContainer) uploadProgressContainer.style.display = 'none';

        // Refresh MJPEG stream to force standby image immediately
        if (videoElement) {
            videoElement.src = `/video_feed?t=${Date.now()}`;
        }

        // Reset classification & telemetry
        resetTelemetryUI();
    }

    function resetTelemetryUI() {
        if (threatBadge) {
            threatBadge.className = 'telemetry-pill threat-pill threat-safe';
            if (threatText) threatText.textContent = 'SAFE';
        }
        if (threatOverlay) threatOverlay.className = 'threat-overlay';
        if (dominantBox) dominantBox.className = 'dominant-box';
        if (dominantVal) dominantVal.textContent = 'Normal activity';
        if (dominantConf) dominantConf.textContent = '98.0% Confidence';
        if (fpsVal) fpsVal.textContent = '0.0 FPS';

        // Reset all 11 category progress bars
        document.querySelectorAll('.cat-bar-fill').forEach(fill => {
            fill.style.width = '0%';
            fill.className = 'cat-bar-fill';
        });
        document.querySelectorAll('.cat-pct').forEach(pct => {
            pct.textContent = '0.0%';
        });
        document.querySelectorAll('.category-row').forEach(row => {
            row.className = 'category-row';
        });

        const normalFill = document.getElementById('fill-Normal_activity');
        const normalPct = document.getElementById('pct-Normal_activity');
        const normalRow = document.getElementById('row-Normal_activity');
        if (normalFill) {
            normalFill.style.width = '98%';
            normalFill.className = 'cat-bar-fill fill-safe';
        }
        if (normalPct) normalPct.textContent = '98.0%';
        if (normalRow) normalRow.classList.add('active-cat');
    }

    // Ensure clicking "Upload Video" anywhere in navbar resets to a fresh upload
    const uploadLinks = document.querySelectorAll('a[href$="/upload"], a[href*="/upload"]');
    uploadLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            if (window.location.pathname.endsWith('/upload')) {
                // If already on /upload page, reset immediately without full page reload
                e.preventDefault();
                resetToStandby();
            } else {
                // If navigating to /upload from another page, trigger beacon reset
                if (navigator.sendBeacon) {
                    navigator.sendBeacon('/api/reset');
                }
            }
        });
    });

    // Drag & drop support if dropZone exists
    if (dropZone && videoFileInput) {
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
                chosenFile = files[0];
                if (selectedFileName) selectedFileName.textContent = chosenFile.name;
                if (selectedFilePill) selectedFilePill.classList.add('has-file');
            }
        });
    }

    function handleFileUpload(file) {
        if (!file.type.startsWith('video/') && !file.name.match(/\.(mp4|avi|mov|mkv)$/i)) {
            alert('Please select a valid video file (MP4, AVI, MOV, MKV).');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        if (predictBtn) {
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        }

        if (uploadProgressContainer) uploadProgressContainer.style.display = 'flex';
        if (uploadProgressBar) uploadProgressBar.style.width = '30%';
        if (uploadStatusLabel) uploadStatusLabel.textContent = `Uploading ${file.name}...`;

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload', true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable && uploadProgressBar) {
                const percent = Math.round((e.loaded / e.total) * 90);
                uploadProgressBar.style.width = `${percent}%`;
            }
        };

        xhr.onload = () => {
            if (uploadProgressBar) uploadProgressBar.style.width = '100%';
            if (predictBtn) {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict';
            }
            if (xhr.status === 200) {
                if (uploadStatusLabel) uploadStatusLabel.textContent = `Predicting: ${file.name}`;
                if (activeSourceBadge) activeSourceBadge.textContent = `Source: Upload (${file.name})`;
                if (videoElement) videoElement.src = `/video_feed?t=${Date.now()}`;
                setTimeout(() => {
                    if (uploadProgressContainer) uploadProgressContainer.style.display = 'none';
                }, 1800);
            } else {
                if (uploadStatusLabel) uploadStatusLabel.textContent = 'Upload failed.';
                alert('Upload failed: ' + xhr.responseText);
            }
        };

        xhr.onerror = () => {
            if (predictBtn) {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict';
            }
            if (uploadStatusLabel) uploadStatusLabel.textContent = 'Upload error.';
            alert('Network error while uploading video.');
        };

        xhr.send(formData);
    }

    // =========================================================================
    // Tactical Web Audio Siren
    // =========================================================================
    function playTacticalAlarm() {
        if (!audioAlertEnabled) return;
        const now = Date.now();
        if (now - lastAudioBeep < 2500) return;
        lastAudioBeep = now;

        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }

            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.35);
            osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.40);
            osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.75);

            gain.gain.setValueAtTime(0.18, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.8);

            osc.connect(gain);
            gain.connect(audioCtx.destination);

            osc.start();
            osc.stop(audioCtx.currentTime + 0.85);
        } catch (e) {
            console.warn('Audio alarm unavailable:', e);
        }
    }

    if (audioToggleBtn && audioIcon) {
        audioToggleBtn.addEventListener('click', () => {
            audioAlertEnabled = !audioAlertEnabled;
            if (audioAlertEnabled) {
                audioIcon.className = 'fa-solid fa-volume-high';
                audioToggleBtn.title = 'Audio Alarm Enabled';
                audioToggleBtn.classList.remove('active');
                playTacticalAlarm();
            } else {
                audioIcon.className = 'fa-solid fa-volume-xmark';
                audioToggleBtn.title = 'Audio Alarm Muted';
                audioToggleBtn.classList.add('active');
            }
        });
    }

    // =========================================================================
    // Telemetry & 11-Crime Classification Meter Poller
    // =========================================================================
    const criticalCrimes = new Set(['Fighting', 'Assault', 'Robbery', 'Shooting', 'Explosion', 'Arson']);

    async function updateTelemetry() {
        try {
            const resp = await fetch('/api/status');
            if (!resp.ok) return;
            const data = await resp.json();

            // 1. FPS
            if (fpsVal) fpsVal.textContent = `${data.fps} FPS`;

            // 2. Threat Level Badge & Glow Overlay
            const lvl = data.threat_level;
            if (threatText) threatText.textContent = lvl;
            if (threatBadge) {
                threatBadge.className = 'telemetry-pill threat-pill';
                if (lvl === 'CRITICAL') threatBadge.classList.add('threat-critical');
                else if (lvl === 'ELEVATED') threatBadge.classList.add('threat-elevated');
                else threatBadge.classList.add('threat-safe');
            }

            if (threatOverlay) {
                threatOverlay.className = 'threat-overlay';
                if (lvl === 'CRITICAL') threatOverlay.classList.add('active-critical');
                else if (lvl === 'ELEVATED') threatOverlay.classList.add('active-elevated');
            }

            if (lvl === 'CRITICAL') {
                playTacticalAlarm();
            }

            // 3. Dominant Crime Activity Box
            if (dominantVal && data.top_crime) {
                dominantVal.textContent = data.top_crime;
                if (dominantConf) dominantConf.textContent = `${data.top_conf}% Confidence`;

                if (dominantBox) {
                    dominantBox.className = 'dominant-box';
                    if (data.top_crime !== 'Normal activity') {
                        if (criticalCrimes.has(data.top_crime)) {
                            dominantBox.classList.add('dominant-critical');
                        } else {
                            dominantBox.classList.add('dominant-warning');
                        }
                    }
                }
            }

            // 4. Update All 11 Crime Category Progress Bars
            if (data.crime_probabilities) {
                for (const [catName, pct] of Object.entries(data.crime_probabilities)) {
                    const slug = catName.replace(/ /g, '_');
                    const pctEl = document.getElementById(`pct-${slug}`);
                    const fillEl = document.getElementById(`fill-${slug}`);
                    const rowEl = document.getElementById(`row-${slug}`);

                    if (pctEl) pctEl.textContent = `${pct}%`;
                    if (fillEl) {
                        fillEl.style.width = `${Math.min(100, Math.max(0, pct))}%`;
                        fillEl.className = 'cat-bar-fill';

                        if (catName === 'Normal activity') {
                            fillEl.classList.add('fill-safe');
                        } else if (criticalCrimes.has(catName)) {
                            fillEl.classList.add('fill-critical');
                        } else {
                            fillEl.classList.add('fill-warning');
                        }
                    }

                    if (rowEl) {
                        rowEl.className = 'category-row';
                        if (catName === data.top_crime && pct > 20.0) {
                            if (catName === 'Normal activity') {
                                rowEl.classList.add('active-cat');
                            } else if (criticalCrimes.has(catName)) {
                                rowEl.classList.add('active-cat-critical');
                            } else {
                                rowEl.classList.add('active-cat-warning');
                            }
                        }
                    }
                }
            }

        } catch (err) {
            // Transient fetch errors silently ignored
        }
    }

    // =========================================================================
    // Incident Feed Poller
    // =========================================================================
    let renderedIncidentIds = new Set();

    async function updateIncidents() {
        if (!incidentFeed) return;
        try {
            const resp = await fetch('/api/incidents');
            if (!resp.ok) return;
            const data = await resp.json();

            if (incidentCountBadge) incidentCountBadge.textContent = data.total;

            if (data.incidents.length === 0) {
                if (emptyIncidentMsg) emptyIncidentMsg.style.display = 'flex';
                incidentFeed.innerHTML = '';
                incidentFeed.appendChild(emptyIncidentMsg);
                renderedIncidentIds.clear();
                return;
            }

            if (emptyIncidentMsg) emptyIncidentMsg.style.display = 'none';

            const currentIds = new Set(data.incidents.map(i => i.id));
            let hasNew = false;
            for (let id of currentIds) {
                if (!renderedIncidentIds.has(id)) {
                    hasNew = true;
                    break;
                }
            }

            if (!hasNew && data.incidents.length === renderedIncidentIds.size) {
                return;
            }

            renderedIncidentIds = currentIds;
            incidentFeed.innerHTML = '';

            data.incidents.forEach(item => {
                const el = document.createElement('div');
                el.className = 'incident-item';

                const isCrit = item.severity === 'CRITICAL';
                const labelClass = isCrit ? 'crit' : 'warn';
                const thumbSrc = item.snapshot ? `/alerts/${item.snapshot}` : '';

                el.innerHTML = `
                    ${thumbSrc ? `<img src="${thumbSrc}" class="incident-thumb" alt="Thumbnail">` : `<div class="incident-thumb"></div>`}
                    <div class="incident-details">
                        <div class="incident-badge-row">
                            <span class="incident-label ${labelClass}">${item.label}</span>
                            <span class="incident-time">${item.timestamp.split(' ')[1] || item.timestamp}</span>
                        </div>
                        <div class="incident-subtext">
                            <span>Conf: <strong>${item.confidence}%</strong></span> • 
                            <span>Severity: <strong style="color:${isCrit ? '#ff3355' : '#ff8800'}">${item.severity}</strong></span>
                        </div>
                    </div>
                `;

                el.addEventListener('click', () => {
                    openSnapshotModal(item);
                });

                incidentFeed.appendChild(el);
            });

        } catch (err) {
            console.error('Error fetching incidents:', err);
        }
    }

    if (clearIncidentsBtn) {
        clearIncidentsBtn.addEventListener('click', async () => {
            if (confirm('Clear all recorded incidents?')) {
                await fetch('/api/clear_incidents', { method: 'POST' });
                renderedIncidentIds.clear();
                updateIncidents();
            }
        });
    }

    // Modal
    function openSnapshotModal(item) {
        if (!snapshotModal) return;
        if (modalTitle) modalTitle.textContent = `Evidence: ${item.label} (#${item.id})`;
        if (modalImage) modalImage.src = item.snapshot ? `/alerts/${item.snapshot}` : '';
        if (modalMeta) {
            modalMeta.innerHTML = `
                <strong>Timestamp:</strong> ${item.timestamp} | 
                <strong>Confidence:</strong> ${item.confidence}% | 
                <strong>Severity:</strong> <span style="color:${item.severity === 'CRITICAL' ? '#ff3355' : '#ff8800'}">${item.severity}</span>
            `;
        }
        snapshotModal.classList.add('open');
    }

    function closeModal() {
        if (snapshotModal) snapshotModal.classList.remove('open');
    }

    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
    if (modalBackdrop) modalBackdrop.addEventListener('click', closeModal);

    // Fullscreen & Snap Utilities
    if (fullscreenBtn && videoWrapper) {
        fullscreenBtn.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                videoWrapper.requestFullscreen().catch(e => alert(`Fullscreen error: ${e.message}`));
            } else {
                document.exitFullscreen();
            }
        });
    }

    if (snapBtn && videoElement) {
        snapBtn.addEventListener('click', () => {
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.naturalWidth || 960;
            canvas.height = videoElement.naturalHeight || 540;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            const a = document.createElement('a');
            a.href = canvas.toDataURL('image/jpeg');
            a.download = `cctv_evidence_${Date.now()}.jpg`;
            a.click();
        });
    }

    // Polling Intervals
    setInterval(updateTelemetry, 700);
    setInterval(updateIncidents, 1200);
    updateTelemetry();
    updateIncidents();
});
