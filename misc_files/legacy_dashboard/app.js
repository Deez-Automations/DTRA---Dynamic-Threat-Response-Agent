const API_BASE = 'http://localhost:5000/api';
let currentFile = null;

// Wait for page to load
document.addEventListener('DOMContentLoaded', function () {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const activateBtn = document.getElementById('activate-btn');
    const consoleOutput = document.getElementById('console-output');

    // Click to upload
    uploadZone.onclick = function () {
        fileInput.click();
    };

    // File selected
    fileInput.onchange = function (e) {
        currentFile = e.target.files[0];
        if (currentFile) {
            uploadZone.querySelector('.upload-text').textContent = currentFile.name;
            activateBtn.disabled = false;
            addLog('info', 'File loaded: ' + currentFile.name);
        }
    };

    // Analyze button
    activateBtn.onclick = async function () {
        if (!currentFile) return;

        addLog('system', 'Starting analysis...');
        activateBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(API_BASE + '/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                addLog('error', 'ERROR: ' + data.error);
                activateBtn.disabled = false;
                return;
            }

            if (data.stats) {
                addLog('success', '=== ANALYSIS COMPLETE ===');
                addLog('info', 'Total Packets: ' + data.stats.total_packets);
                addLog('info', 'Threats Detected: ' + data.stats.threats_detected);

                if (data.stats.classification_metrics) {
                    const cm = data.stats.classification_metrics;
                    addLog('success', 'Accuracy: ' + cm.accuracy + '%');
                    addLog('success', 'Detection Rate: ' + cm.detection_rate + '%');
                    addLog('info', 'False Alarm Rate: ' + cm.false_alarm_rate + '%');
                }

                if (data.stats.confusion_matrix) {
                    const cf = data.stats.confusion_matrix;
                    addLog('info', 'TP: ' + cf.tp + ' | TN: ' + cf.tn);
                    addLog('info', 'FP: ' + cf.fp + ' | FN: ' + cf.fn);
                }

                addLog('success', 'Processing Time: ' + data.stats.processing_time + 's');
            }

        } catch (error) {
            addLog('error', 'ERROR: ' + error.message);
            activateBtn.disabled = false;
        }
    };

    function addLog(type, message) {
        const line = document.createElement('div');
        line.className = 'console-line ' + type;
        const time = new Date().toLocaleTimeString();
        line.innerHTML = `<span class="console-time">[${time}]</span> <span class="console-text">${message}</span>`;
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    addLog('system', 'Dashboard ready');
});
