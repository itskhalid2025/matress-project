document.addEventListener('DOMContentLoaded', () => {
    // Clock
    setInterval(() => {
        const now = new Date();
        document.getElementById('sys-clock').textContent = now.toLocaleTimeString();
    }, 1000);

    const btnProcess = document.getElementById('btn-process');
    const spinner = document.getElementById('processing-spinner');
    const searchInput = document.getElementById('history-search');

    let allRecords = [];

    // Camera Function Dropdowns & Real-time Live Stream Re-binding
    const functionSelects = document.querySelectorAll('.feed-function-select');

    async function syncFeedConfig() {
        try {
            const res = await fetch('/api/config/camera_functions');
            const feedConfig = await res.json();
            for (const [feedId, func] of Object.entries(feedConfig)) {
                const selectEl = document.querySelector(`.feed-function-select[data-feed="${feedId}"]`);
                if (selectEl) {
                    selectEl.value = func;
                }
            }
        } catch (err) {
            console.error('Failed to sync feed config:', err);
        }
    }
    syncFeedConfig();

    functionSelects.forEach(select => {
        select.addEventListener('change', async (e) => {
            const feedId = e.target.getAttribute('data-feed');
            const selectedFunction = e.target.value;

            try {
                const res = await fetch('/api/config/camera_functions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ feed_id: feedId, function: selectedFunction })
                });
                const data = await res.json();
                if (data.success) {
                    const img = document.getElementById(`img-${feedId.replace('_', '-')}`);
                    if (img) {
                        const baseUrl = img.src.split('?')[0];
                        img.src = `${baseUrl}?t=${Date.now()}`;
                    }
                }
            } catch (err) {
                console.error('Failed to update feed function:', err);
            }
        });
    });

    // Check 3 Camera Connection Statuses
    async function checkCameraStatus() {
        try {
            const res = await fetch('/api/camera_status');
            const data = await res.json();

            const badgeQr = document.getElementById('badge-qr');
            const badgeBill = document.getElementById('badge-bill');
            const badgeTop = document.getElementById('badge-top');

            badgeQr.querySelector('.dot').className = `dot ${data.camera_qr ? 'green' : 'red'}`;
            badgeBill.querySelector('.dot').className = `dot ${data.camera_bill ? 'green' : 'red'}`;
            badgeTop.querySelector('.dot').className = `dot ${data.camera_top ? 'green' : 'red'}`;
        } catch (e) {
            console.error('Camera status check failed:', e);
        }
    }
    setInterval(checkCameraStatus, 5000);
    checkCameraStatus();

    // Trigger 1-Click Process
    btnProcess.addEventListener('click', async () => {
        btnProcess.disabled = true;
        spinner.classList.remove('hidden');

        try {
            const res = await fetch('/api/process', { method: 'POST' });
            const data = await res.json();

            if (data.success && data.record) {
                updateResultsPanel(data.record);
                loadHistory();
            } else {
                alert('Inspection process failed.');
            }
        } catch (e) {
            alert('Server error during inspection process.');
            console.error(e);
        } finally {
            btnProcess.disabled = false;
            spinner.classList.add('hidden');
        }
    });

    // Update Result UI Cards & Banners
    function updateResultsPanel(rec) {
        const banner = document.getElementById('status-banner');
        const icon = document.getElementById('status-icon');
        const title = document.getElementById('status-title');
        const desc = document.getElementById('status-desc');

        const overallStatus = rec.overall_status || 'FAIL';
        banner.className = `status-banner ${overallStatus.toLowerCase()}`;
        icon.textContent = overallStatus;
        title.textContent = `INSPECTION RESULT: ${overallStatus}`;
        desc.textContent = `ID: ${rec.inspection_id} | Completed at ${new Date(rec.timestamp).toLocaleTimeString()}`;

        // Dimensions Card
        const topData = rec.top_camera_data || {};
        const dims = topData.measured_dimensions || {};
        const expDims = topData.expected_dimensions || {};
        const dimStatus = rec.dimension_status || 'FAIL';

        document.getElementById('res-dim-l').textContent = `${dims.length_cm ?? '--'} cm (Exp: ${expDims.length_cm ?? '--'} cm)`;
        document.getElementById('res-dim-w').textContent = `${dims.width_cm ?? '--'} cm (Exp: ${expDims.width_cm ?? '--'} cm)`;
        document.getElementById('res-dim-h').textContent = `${dims.thickness_cm ?? '--'} cm (Exp: ${expDims.thickness_cm ?? '--'} cm)`;

        const badgeDim = document.getElementById('res-dim-status');
        badgeDim.className = `tag-badge ${dimStatus.toLowerCase()}`;
        badgeDim.textContent = dimStatus;

        // 1. QR Card
        document.getElementById('res-qr-product').textContent = rec.qr_code_data?.product_name || 'Not Detected';
        document.getElementById('res-qr-batch').textContent = rec.qr_code_data?.batch_no || 'N/A';
        document.getElementById('res-qr-itemid').textContent = rec.qr_code_data?.inventory_item_id || 'N/A';

        // 2. Side OCR Card
        document.getElementById('res-ocr-conf').textContent = `${rec.verification_result?.side_ocr_similarity ?? 0}%`;
        document.getElementById('res-ocr-text').textContent = rec.side_ocr_data?.full_text || 'No Text Extracted';

        // 3. Texture Card
        document.getElementById('res-texture-pred').textContent = rec.texture_data?.predicted_category || 'N/A';
        document.getElementById('res-texture-conf').textContent = `${rec.texture_data?.confidence ?? 0}%`;

        // 4. Corner Label Card
        const cornerLabel = topData.corner_label || {};
        document.getElementById('res-corner-product').textContent = cornerLabel.product_name || 'Not Detected';
        document.getElementById('res-corner-size').textContent = cornerLabel.size || 'N/A';

        // Mismatch Container
        const mismatchContainer = document.getElementById('mismatch-container');
        const mismatchList = document.getElementById('mismatch-list');

        const mismatches = rec.verification_result?.mismatches || [];
        if (mismatches.length > 0) {
            mismatchContainer.classList.remove('hidden');
            mismatchList.innerHTML = mismatches.map(m => `
                <li class="mismatch-item">
                    <strong>${m.source}</strong>: Expected "<em>${m.expected}</em>", Detected "<em>${m.detected}</em>"
                </li>
            `).join('');
        } else {
            mismatchContainer.classList.add('hidden');
            mismatchList.innerHTML = '';
        }
    }

    // Load History Table
    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            allRecords = await res.json();
            renderHistoryTable(allRecords);
        } catch (e) {
            console.error('Failed to load history:', e);
        }
    }

    function renderHistoryTable(records) {
        const tbody = document.getElementById('history-table-body');
        if (!records || records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" class="text-center">No inspection history records found.</td></tr>';
            return;
        }

        tbody.innerHTML = records.map(r => {
            const dims = r.top_camera_data?.measured_dimensions || {};
            const corner = r.top_camera_data?.corner_label || {};
            return `
            <tr class="clickable-row" onclick="viewRecordDetails('${r.inspection_id}')">
                <td><strong>${r.inspection_id}</strong></td>
                <td>${new Date(r.timestamp).toLocaleString()}</td>
                <td><span class="tag-${(r.overall_status || 'fail').toLowerCase()}">${r.overall_status || 'FAIL'}</span></td>
                <td><span class="tag-${(r.identity_status || 'fail').toLowerCase()}">${r.identity_status || 'FAIL'}</span></td>
                <td><span class="tag-${(r.dimension_status || 'fail').toLowerCase()}">${r.dimension_status || 'FAIL'}</span></td>
                <td>${r.product_variety}</td>
                <td>${r.batch_number}</td>
                <td>${dims.length_cm ?? '--'} x ${dims.width_cm ?? '--'} x ${dims.thickness_cm ?? '--'} cm</td>
                <td>${corner.product_name || 'N/A'}</td>
                <td>
                    <button class="btn-view" onclick="event.stopPropagation(); viewRecordDetails('${r.inspection_id}')">View Details</button>
                </td>
            </tr>
            `;
        }).join('');
    }

    // Search Filter
    searchInput.addEventListener('input', (e) => {
        const q = e.target.value.toLowerCase();
        const filtered = allRecords.filter(r => 
            (r.inspection_id && r.inspection_id.toLowerCase().includes(q)) ||
            (r.product_variety && r.product_variety.toLowerCase().includes(q)) ||
            (r.batch_number && r.batch_number.toLowerCase().includes(q)) ||
            (r.item_id && r.item_id.toLowerCase().includes(q))
        );
        renderHistoryTable(filtered);
    });

    // Modal
    const modal = document.getElementById('image-modal');
    document.getElementById('modal-close').addEventListener('click', () => modal.classList.add('hidden'));

    window.viewRecordDetails = (inspectionId) => {
        const rec = allRecords.find(r => r.inspection_id === inspectionId);
        if (!rec) return;

        const body = document.getElementById('modal-body');
        document.getElementById('modal-title').textContent = `Inspection Record Details`;
        document.getElementById('modal-subtitle').textContent = `ID: ${rec.inspection_id} | Timestamp: ${new Date(rec.timestamp).toLocaleString()}`;

        const overallStatus = rec.overall_status || 'FAIL';
        const identityStatus = rec.identity_status || 'FAIL';
        const dimStatus = rec.dimension_status || 'FAIL';

        const topData = rec.top_camera_data || {};
        const dims = topData.measured_dimensions || {};
        const expDims = topData.expected_dimensions || {};
        const corner = topData.corner_label || {};

        const qr = rec.qr_code_data || {};
        const sideOcr = rec.side_ocr_data || {};
        const texture = rec.texture_data || {};

        const mismatches = rec.verification_result?.mismatches || [];
        const imgs = rec.image_paths || {};

        body.innerHTML = `
            <!-- Top Status Badges -->
            <div style="display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;">
                <div class="status-banner ${overallStatus.toLowerCase()}" style="flex: 1; padding: 10px 14px;">
                    <strong style="font-size: 0.9rem;">Overall Status:</strong> 
                    <span class="tag-badge ${overallStatus.toLowerCase()}">${overallStatus}</span>
                </div>
                <div class="status-banner ${identityStatus.toLowerCase()}" style="flex: 1; padding: 10px 14px;">
                    <strong style="font-size: 0.9rem;">Identity Match:</strong> 
                    <span class="tag-badge ${identityStatus.toLowerCase()}">${identityStatus}</span>
                </div>
                <div class="status-banner ${dimStatus.toLowerCase()}" style="flex: 1; padding: 10px 14px;">
                    <strong style="font-size: 0.9rem;">Dimension Check:</strong> 
                    <span class="tag-badge ${dimStatus.toLowerCase()}">${dimStatus}</span>
                </div>
            </div>

            <!-- Dimensions & 4-Way Identity Grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px;">
                <!-- Dimensions -->
                <div class="res-card highlighted-card">
                    <div class="res-card-header">
                        <span class="res-badge green">TOP DIMS</span>
                        <h3>Measured vs Expected Dimensions</h3>
                    </div>
                    <div class="res-card-body">
                        <div class="res-row"><label>Length (L):</label> <strong>${dims.length_cm ?? '--'} cm (Exp: ${expDims.length_cm ?? '--'} cm)</strong></div>
                        <div class="res-row"><label>Width (W):</label> <strong>${dims.width_cm ?? '--'} cm (Exp: ${expDims.width_cm ?? '--'} cm)</strong></div>
                        <div class="res-row"><label>Thickness (H):</label> <strong>${dims.thickness_cm ?? '--'} cm (Exp: ${expDims.thickness_cm ?? '--'} cm)</strong></div>
                    </div>
                </div>

                <!-- Product & Batch Summary -->
                <div class="res-card">
                    <div class="res-card-header">
                        <span class="res-badge blue">SUMMARY</span>
                        <h3>Product & Batch Metadata</h3>
                    </div>
                    <div class="res-card-body">
                        <div class="res-row"><label>Product Variety:</label> <strong>${rec.product_variety || 'Not Detected'}</strong></div>
                        <div class="res-row"><label>Batch Number:</label> <span>${rec.batch_number || 'N/A'}</span></div>
                        <div class="res-row"><label>Item ID:</label> <span>${rec.item_id || 'N/A'}</span></div>
                    </div>
                </div>
            </div>

            <!-- 4-Way Sensor Verification Cards -->
            <div class="results-grid-4" style="margin-bottom: 16px;">
                <!-- 1. QR Code -->
                <div class="res-card">
                    <div class="res-card-header">
                        <span class="res-badge blue">CAM 1: QR CODE</span>
                        <h3>QR Code Payload</h3>
                    </div>
                    <div class="res-card-body">
                        <div class="res-row"><label>Variety:</label> <strong>${qr.product_name || 'Not Detected'}</strong></div>
                        <div class="res-row"><label>Batch:</label> <span>${qr.batch_no || 'N/A'}</span></div>
                        <div class="res-row"><label>Item ID:</label> <span>${qr.inventory_item_id || 'N/A'}</span></div>
                        <div class="res-row"><label>Raw Text:</label> <span class="text-truncate" style="max-width: 140px;">${qr.raw_text || 'N/A'}</span></div>
                    </div>
                </div>

                <!-- 2. Side Bill OCR -->
                <div class="res-card">
                    <div class="res-card-header">
                        <span class="res-badge purple">CAM 2: SIDE OCR</span>
                        <h3>Side Bill OCR</h3>
                    </div>
                    <div class="res-card-body">
                        <div class="res-row"><label>Similarity Score:</label> <strong>${rec.verification_result?.side_ocr_similarity ?? 0}%</strong></div>
                        <div class="res-row"><label>Extracted Text:</label> <span class="text-truncate" style="max-width: 140px;">${sideOcr.full_text || 'None'}</span></div>
                        <div class="res-row"><label>Items Count:</label> <span>${(sideOcr.items || sideOcr.extracted_items || []).length} items</span></div>

                    </div>
                </div>

                <!-- 3. PyTorch Texture AI -->
                <div class="res-card">
                    <div class="res-card-header">
                        <span class="res-badge cyan">CAM 2: TEXTURE AI</span>
                        <h3>Texture Pattern AI</h3>
                    </div>
                    <div class="res-card-body">
                        <div class="res-row"><label>Predicted Variety:</label> <strong>${texture.predicted_category || 'N/A'}</strong></div>
                        <div class="res-row"><label>Confidence:</label> <span>${texture.confidence ?? 0}%</span></div>
                    </div>
                </div>

                <!-- 4. Corner Label OCR -->
                <div class="res-card">
                    <div class="res-card-header">
                        <span class="res-badge yellow">CAM 3: CORNER LABEL</span>
                        <h3>Corner Label OCR</h3>
                    </div>
                    <div class="res-card-body">
                        <div class="res-row"><label>Detected Variety:</label> <strong>${corner.product_name || 'Not Detected'}</strong></div>
                        <div class="res-row"><label>Label Size:</label> <span>${corner.size || 'N/A'}</span></div>
                        <div class="res-row"><label>Match Score:</label> <span>${corner.fuzzy_similarity ?? 0}%</span></div>
                    </div>
                </div>
            </div>

            <!-- Mismatches List if any -->
            ${mismatches.length > 0 ? `
                <div class="mismatch-card" style="margin-bottom: 16px;">
                    <div class="mismatch-header">
                        <svg width="20" height="20" viewBox="0 0 24 24"><path fill="#ff4d4f" d="M12 2L1 21h22L12 2zm1 14h-2v-2h2v2zm0-4h-2V8h2v4z"/></svg>
                        <h3>Verification Mismatches (${mismatches.length})</h3>
                    </div>
                    <ul class="mismatch-list">
                        ${mismatches.map(m => `
                            <li class="mismatch-item">
                                <strong>${m.source}</strong>: Expected "<em>${m.expected}</em>", Detected "<em>${m.detected}</em>"
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}

            <!-- 3 Annotated Camera Images & Crops -->
            <div style="margin-top: 20px;">
                <h3 style="font-size: 0.95rem; margin-bottom: 12px; color: var(--accent-blue);">Inspection Result Images & Boundary Labels</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px;">
                    <div>
                        <h4 style="font-size: 0.8rem; margin-bottom: 6px; color: var(--accent-blue);">Cam 1: QR Scan & Overlay</h4>
                        ${imgs.annotated_qr ? `<img src="/results/${imgs.annotated_qr}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">` : '<div style="color:var(--text-muted); padding:20px;">No Image</div>'}
                    </div>
                    <div>
                        <h4 style="font-size: 0.8rem; margin-bottom: 6px; color: var(--accent-purple);">Cam 2: Side Bill OCR Bounding Boxes</h4>
                        ${imgs.annotated_ocr ? `<img src="/results/${imgs.annotated_ocr}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">` : '<div style="color:var(--text-muted); padding:20px;">No Image</div>'}
                    </div>
                    <div>
                        <h4 style="font-size: 0.8rem; margin-bottom: 6px; color: var(--pass-green);">Cam 3: Top Dims & Label Region</h4>
                        ${imgs.annotated_top ? `<img src="/results/${imgs.annotated_top}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">` : '<div style="color:var(--text-muted); padding:20px;">No Image</div>'}
                    </div>
                </div>

                ${imgs.corner_label_crop ? `
                    <div style="margin-top: 16px; text-align: center;">
                        <h4 style="font-size: 0.8rem; margin-bottom: 6px; color: var(--accent-yellow);">Cropped Corner Label Region</h4>
                        <img src="/results/${imgs.corner_label_crop}" style="max-height: 180px; border-radius: 8px; border: 1px solid var(--border-color);">
                    </div>
                ` : ''}
            </div>
        `;
        modal.classList.remove('hidden');
    };

    // Backward compatibility alias for viewRecordImages
    window.viewRecordImages = window.viewRecordDetails;

    loadHistory();
});

