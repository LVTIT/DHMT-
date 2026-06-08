export class PoseWebSocket {
    constructor({ url, onStatus, onMessage, onFps }) {
        this.url = url;
        this.onStatus = onStatus;
        this.onMessage = onMessage;
        this.onFps = onFps;
        this.socket = null;
        this.reconnectTimer = null;
        this.pingTimer = null;
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
    }

    connect() {
        this._setStatus('connecting', 'Connecting');
        try {
            this.socket = new WebSocket(this.url);
        } catch {
            this._scheduleReconnect();
            return;
        }

        this.socket.onopen = () => {
            this._setStatus('connected', 'Connected');
            this.frameCount = 0;
            this.lastFpsTime = performance.now();
            clearInterval(this.pingTimer);
            this.pingTimer = setInterval(() => this.send('ping'), 1500);
            if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        };

        this.socket.onmessage = (event) => {
            const payload = JSON.parse(event.data);
            this.onMessage?.(payload);
            this._tickFps();
        };

        this.socket.onclose = () => {
            clearInterval(this.pingTimer);
            this._setStatus('offline', 'Disconnected');
            this._scheduleReconnect();
        };

        this.socket.onerror = () => {
            if (this.socket) this.socket.close();
        };
    }

    send(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(message);
        }
    }

    _setStatus(state, text) {
        this.onStatus?.({ state, text });
    }

    _scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, 1400);
    }

    _tickFps() {
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime < 1000) return;
        const fps = Math.round(this.frameCount * 1000 / (now - this.lastFpsTime));
        this.onFps?.(fps);
        this.frameCount = 0;
        this.lastFpsTime = now;
    }
}
