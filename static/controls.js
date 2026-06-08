async function postJson(url, body) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
}

function setButtonActive(button, active) {
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
}

function setIcon(button, iconName) {
    button.innerHTML = `<i data-lucide="${iconName}"></i>`;
    window.lucide?.createIcons();
}

export function initControls(renderer, hooks = {}) {
    const pauseButton = document.getElementById('pause-button');
    const cameraInput = document.getElementById('camera-input');
    const sourceButton = document.getElementById('source-button');
    const uploadInput = document.getElementById('upload-input');
    const uploadButton = document.getElementById('upload-button');
    const videoOverlay = document.getElementById('video-overlay');

    let paused = false;
    pauseButton.addEventListener('click', async () => {
        paused = !paused;
        setButtonActive(pauseButton, paused);
        setIcon(pauseButton, paused ? 'play' : 'pause');
        await postJson('/api/pause', { paused });
        hooks.onControl?.(`Capture ${paused ? 'paused' : 'running'}`);
    });

    sourceButton.addEventListener('click', async () => {
        await postJson('/api/source', { source: cameraInput.value || '0' });
        hooks.onControl?.(`Camera ${cameraInput.value || '0'}`);
    });

    uploadButton.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async () => {
        if (!uploadInput.files.length) return;
        const form = new FormData();
        form.append('file', uploadInput.files[0]);
        const response = await fetch('/api/upload', { method: 'POST', body: form });
        if (!response.ok) throw new Error(await response.text());
        hooks.onControl?.('Video uploaded');
    });

    document.querySelectorAll('[data-toggle]').forEach((button) => {
        button.addEventListener('click', () => {
            const name = button.dataset.toggle;
            const active = !button.classList.contains('active');
            setButtonActive(button, active);
            if (name === 'video') {
                videoOverlay.classList.toggle('hidden', !active);
            } else {
                renderer.setOverlay(name, active);
            }
        });
    });

    document.getElementById('reset-camera').addEventListener('click', () => {
        renderer.resetCamera();
    });
}
