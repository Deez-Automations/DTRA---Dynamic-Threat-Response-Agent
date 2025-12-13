// Simple working version - testing only file upload
const API_BASE = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const activateBtn = document.getElementById('activate-btn');
    let currentFile = null;

    // Click upload zone to trigger file input
    uploadZone.addEventListener('click', () => {
        console.log('Upload zone clicked');
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name);
            currentFile = file;
            uploadZone.querySelector('.upload-text').textContent = file.name;
            activateBtn.disabled = false;
        }
    });

    // Handle activate button
    activateBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        console.log('Starting upload...');
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log('Response:', data);

            if (data.stats) {
                alert(`SUCCESS! Analyzed ${data.stats.total_packets} packets`);
            } else if (data.error) {
                alert(`ERROR: ${data.error}`);
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`ERROR: ${error.message}`);
        }
    });

    console.log('Dashboard initialized');
});
