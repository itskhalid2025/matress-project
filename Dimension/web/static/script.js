// script.js — Mattress Dimension Dashboard Frontend Logic

function setActiveEdge(edgeName) {
    fetch('/api/set_active_edge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edge: edgeName })
    }).catch(err => console.error("Error setting active edge:", err));
}

function saveCalibration() {
    const topVal = parseFloat(document.getElementById('input-top').value) || 100.0;
    const rightVal = parseFloat(document.getElementById('input-right').value) || 120.0;
    const bottomVal = parseFloat(document.getElementById('input-bottom').value) || 100.0;
    const leftVal = parseFloat(document.getElementById('input-left').value) || 120.0;

    fetch('/api/set_edge_lengths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            top: topVal,
            right: rightVal,
            bottom: bottomVal,
            left: leftVal
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            document.getElementById('status-text').innerText = "STEP 2: BORDER CALIBRATION SAVED (READY FOR MATTRESS)";
            document.getElementById('system-status').querySelector('.dot').className = "dot green";
            alert("✅ 4-Edge Calibration Saved Successfully!\n\nTop: " + topVal + " cm\nRight: " + rightVal + " cm\nBottom: " + bottomVal + " cm\nLeft: " + leftVal + " cm");
        } else {
            alert("Error saving calibration: " + data.error);
        }
    })
    .catch(err => console.error("Error saving calibration:", err));
}

function processMattress() {
    document.getElementById('status-text').innerText = "STEP 3: PROCESSING MATTRESS DIMENSIONS...";
    
    fetch('/api/process_dimension', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        if (data.success && data.result.success) {
            const r = data.result;
            document.getElementById('res-breadth').innerText = r.width_cm + " cm";
            document.getElementById('res-breadth-in').innerText = r.width_in + " in";
            
            document.getElementById('res-length').innerText = r.length_cm + " cm";
            document.getElementById('res-length-in').innerText = r.length_in + " in";
            
            document.getElementById('res-area').innerText = r.area_sq_m + " sq.m";
            document.getElementById('res-px-cm').innerText = (r.pixels_per_cm || "--") + " px/cm";
            document.getElementById('res-status').innerText = "MEASURED PASS";
            document.getElementById('res-status').className = "metric-val status-pass";
            
            document.getElementById('status-text').innerText = "STEP 3: MEASUREMENT COMPLETE (WIDTH: " + r.width_cm + " cm | LENGTH: " + r.length_cm + " cm)";
        } else {
            alert("Processing Error: " + (data.result ? data.result.error : data.error));
        }
    })
    .catch(err => console.error("Error processing mattress:", err));
}

function resetSystem() {
    fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('res-breadth').innerText = "-- cm";
        document.getElementById('res-breadth-in').innerText = "-- in";
        document.getElementById('res-length').innerText = "-- cm";
        document.getElementById('res-length-in').innerText = "-- in";
        document.getElementById('res-area').innerText = "-- sq.m";
        document.getElementById('res-status').innerText = "READY";
        
        document.getElementById('status-text').innerText = "STEP 1: DETECTING BLACK BORDER";
        document.getElementById('system-status').querySelector('.dot').className = "dot yellow";
    })
    .catch(err => console.error("Error resetting system:", err));
}
