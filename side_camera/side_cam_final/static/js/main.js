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
            <tr>
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
                    <button class="btn-view" onclick="viewRecordImages('${r.inspection_id}')">Images</button>
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

    window.viewRecordImages = (inspectionId) => {
        const rec = allRecords.find(r => r.inspection_id === inspectionId);
        if (!rec || !rec.image_paths) return;

        const body = document.getElementById('modal-body');
        document.getElementById('modal-title').textContent = `Inspection Artifacts (${inspectionId})`;

        const imgs = rec.image_paths;
        body.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
                <div>
                    <h4 style="margin-bottom: 8px; color: var(--accent-blue);">Cam 1: QR Result</h4>
                    <img src="/results/${imgs.annotated_qr}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">
                </div>
                <div>
                    <h4 style="margin-bottom: 8px; color: var(--accent-purple);">Cam 2: Side Bill OCR</h4>
                    <img src="/results/${imgs.annotated_ocr}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">
                </div>
                <div>
                    <h4 style="margin-bottom: 8px; color: var(--pass-green);">Cam 3: Top Dims & Label</h4>
                    <img src="/results/${imgs.annotated_top}" style="width: 100%; border-radius: 8px; border: 1px solid var(--border-color);">
                </div>
            </div>
            ${imgs.corner_label_crop ? `
                <div style="margin-top: 16px; text-align: center;">
                    <h4 style="margin-bottom: 8px; color: var(--accent-yellow);">Cropped Corner Label Region</h4>
                    <img src="/results/${imgs.corner_label_crop}" style="max-height: 180px; border-radius: 8px; border: 1px solid var(--border-color);">
                </div>
            ` : ''}
        `;
        modal.classList.remove('hidden');
    };

    loadHistory();
});
