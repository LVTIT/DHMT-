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

function reloadVideoOverlay(videoOverlay) {
    videoOverlay.src = `/video_feed?ts=${Date.now()}`;
}

export function initControls(renderer, hooks = {}) {
    const pauseButton = document.getElementById('pause-button');
    const cameraInput = document.getElementById('camera-input');
    const scanCamerasButton = document.getElementById('scan-cameras-button');
    const sourceButton = document.getElementById('source-button');
    const uploadInput = document.getElementById('upload-input');
    const uploadButton = document.getElementById('upload-button');
    const videoOverlay = document.getElementById('video-overlay');
    const aspectButton = document.getElementById('aspect-button');
    const fitButton = document.getElementById('fit-button');
    const rotateButton = document.getElementById('rotate-button');

    let paused = false;
    const cameraConfig = {
        aspect: '9:16',
        fit: 'contain',
        rotate: 0,
    };

    setButtonActive(aspectButton, true);

    async function applyCameraConfig() {
        await postJson('/api/camera-config', cameraConfig);
        videoOverlay.classList.toggle('fit-cover', cameraConfig.fit === 'cover');
        reloadVideoOverlay(videoOverlay);
        hooks.onControl?.(`${cameraConfig.aspect} ${cameraConfig.fit} ${cameraConfig.rotate} deg`);
    }

    pauseButton.addEventListener('click', async () => {
        paused = !paused;
        setButtonActive(pauseButton, paused);
        setIcon(pauseButton, paused ? 'play' : 'pause');
        await postJson('/api/pause', { paused });
        hooks.onControl?.(`Capture ${paused ? 'paused' : 'running'}`);
    });

    sourceButton.addEventListener('click', async () => {
        await postJson('/api/source', { source: cameraInput.value || '0' });
        reloadVideoOverlay(videoOverlay);
        hooks.onControl?.(`Source ${cameraInput.value || '0'}`);
    });

    scanCamerasButton.addEventListener('click', async () => {
        hooks.onControl?.('Scanning cameras...');
        const response = await fetch('/api/cameras?max_index=8&all_backends=true');
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        const cameras = payload.cameras || [];
        if (!cameras.length) {
            hooks.onControl?.('No readable camera found');
            return;
        }
        const camera = cameras.find(item => item.index === 1) || cameras[0];
        cameraInput.value = camera.source || `${camera.index}@${camera.backend}`;
        cameraConfig.aspect = camera.height > camera.width ? '9:16' : '16:9';
        aspectButton.textContent = cameraConfig.aspect;
        await applyCameraConfig();
        hooks.onControl?.(`Found ${cameraInput.value} mean ${camera.frame_mean}`);
    });

    uploadButton.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async () => {
        if (!uploadInput.files.length) return;
        const form = new FormData();
        form.append('file', uploadInput.files[0]);
        const response = await fetch('/api/upload', { method: 'POST', body: form });
        if (!response.ok) throw new Error(await response.text());
        reloadVideoOverlay(videoOverlay);
        hooks.onControl?.('Video uploaded');
    });

    aspectButton.addEventListener('click', async () => {
        const next = cameraConfig.aspect === '9:16' ? '16:9' : '9:16';
        cameraConfig.aspect = next;
        aspectButton.textContent = next;
        await applyCameraConfig();
    });

    fitButton.addEventListener('click', async () => {
        cameraConfig.fit = cameraConfig.fit === 'contain' ? 'cover' : 'contain';
        setButtonActive(fitButton, cameraConfig.fit === 'cover');
        await applyCameraConfig();
    });

    rotateButton.addEventListener('click', async () => {
        cameraConfig.rotate = (cameraConfig.rotate + 90) % 360;
        await applyCameraConfig();
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
