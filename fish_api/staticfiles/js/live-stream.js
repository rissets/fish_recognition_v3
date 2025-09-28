/**
 * Live Fish Recognition Stream Application
 * Dedicated for real-time camera processing with advanced segmentation overlay
 */

class LiveStreamApp {
    constructor() {
        this.ws = null;
        this.stream = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isProcessing = false;
        this.isCameraActive = false;
        this.processingInterval = null;
        this.fpsCounter = 0;
        this.lastFpsTime = Date.now();
        this.sessionStartTime = null;
        
        // Statistics
        this.stats = {
            framesProcessed: 0,
            fishDetected: 0,
            facesDetected: 0,
            totalProcessingTime: 0,
            successfulFrames: 0,
            detectionHistory: []
        };
        
        // Settings
        this.settings = {
            includeFaces: true,
            includeSegmentation: true,
            processingInterval: 500,
            qualityThreshold: 0.3,
            autoProcess: true
        };
        
        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Live Stream App');
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.checkApiHealth();
        this.startSessionTimer();
    }
    
    initializeElements() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('overlayCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas properties for better performance
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'high';
    }
    
    setupEventListeners() {
        // Camera controls
        document.getElementById('startCameraBtn').addEventListener('click', () => this.startCamera());
        document.getElementById('stopCameraBtn').addEventListener('click', () => this.stopCamera());
        document.getElementById('toggleProcessingBtn').addEventListener('click', () => this.toggleProcessing());
        
        // Settings
        document.getElementById('includeFaces').addEventListener('change', (e) => {
            this.settings.includeFaces = e.target.checked;
            this.updateWebSocketSettings();
        });
        
        document.getElementById('includeSegmentation').addEventListener('change', (e) => {
            this.settings.includeSegmentation = e.target.checked;
            this.updateWebSocketSettings();
        });
        
        document.getElementById('processingInterval').addEventListener('change', (e) => {
            this.settings.processingInterval = parseInt(e.target.value);
            this.updateProcessingInterval();
            this.updateWebSocketSettings();
        });
        
        document.getElementById('qualityThreshold').addEventListener('input', (e) => {
            this.settings.qualityThreshold = parseFloat(e.target.value);
            document.getElementById('qualityValue').textContent = e.target.value;
            this.updateWebSocketSettings();
        });
        
        // Video events
        this.video.addEventListener('loadedmetadata', () => {
            this.resizeCanvas();
        });
        
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });
    }
    
    async startCamera() {
        try {
            console.log('📹 Starting camera...');
            this.updateCameraStatus('requesting', 'Requesting access...');
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                },
                audio: false
            });
            
            this.video.srcObject = this.stream;
            this.isCameraActive = true;
            
            this.updateCameraStatus('active', 'Active');
            this.updateUI();
            
            console.log('✅ Camera started successfully');
            this.showNotification('Camera started successfully', 'success');
            
        } catch (error) {
            console.error('❌ Camera error:', error);
            this.updateCameraStatus('error', 'Error');
            this.showNotification(`Camera error: ${error.message}`, 'error');
        }
    }
    
    stopCamera() {
        console.log('🛑 Stopping camera...');
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        this.video.srcObject = null;
        this.isCameraActive = false;
        this.isProcessing = false;
        
        this.clearCanvas();
        this.updateCameraStatus('inactive', 'Inactive');
        this.updateProcessingIndicator('idle', 'Idle');
        this.updateUI();
        
        console.log('✅ Camera stopped');
        this.showNotification('Camera stopped', 'info');
    }
    
    toggleProcessing() {
        if (!this.isCameraActive) {
            this.showNotification('Please start camera first', 'warning');
            return;
        }
        
        if (this.isProcessing) {
            this.stopProcessing();
        } else {
            this.startProcessing();
        }
    }
    
    startProcessing() {
        console.log('🔄 Starting recognition processing...');
        this.isProcessing = true;
        this.updateProcessingIndicator('processing', 'Processing');
        
        this.processingInterval = setInterval(() => {
            this.captureAndSendFrame();
        }, this.settings.processingInterval);
        
        this.updateUI();
        this.showNotification('Recognition started', 'success');
    }
    
    stopProcessing() {
        console.log('⏸️ Stopping recognition processing...');
        this.isProcessing = false;
        
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        this.updateProcessingIndicator('idle', 'Idle');
        this.updateUI();
        this.showNotification('Recognition stopped', 'info');
    }
    
    captureAndSendFrame() {
        if (!this.isCameraActive || !this.video.videoWidth || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        try {
            // Create temporary canvas for frame capture
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = this.video.videoWidth;
            tempCanvas.height = this.video.videoHeight;
            
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(this.video, 0, 0);
            
            // Convert to base64
            const frameData = tempCanvas.toDataURL('image/jpeg', 0.8);
            
            // Send to WebSocket
            const payload = {
                type: 'camera_frame',
                data: {
                    frame_data: frameData,
                    frame_id: Date.now(),
                    include_faces: this.settings.includeFaces,
                    include_segmentation: this.settings.includeSegmentation,
                    quality_threshold: this.settings.qualityThreshold,
                    timestamp: new Date().toISOString()
                }
            };
            
            console.log('📤 Sending frame for processing...');
            this.ws.send(JSON.stringify(payload));
            
            // Update FPS
            this.updateFPS();
            
        } catch (error) {
            console.error('❌ Frame capture error:', error);
        }
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/recognition/`;
        
        console.log('🔌 Connecting to WebSocket:', wsUrl);
        this.updateConnectionStatus('connecting', 'Connecting...');
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.updateConnectionStatus('connected', 'Connected');
            this.updateWebSocketStatus('connected', 'Connected');
            this.updateWebSocketSettings();
        };
        
        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (error) {
                console.error('❌ WebSocket message error:', error);
            }
        };
        
        this.ws.onclose = () => {
            console.log('🔌 WebSocket disconnected');
            this.updateConnectionStatus('disconnected', 'Disconnected');
            this.updateWebSocketStatus('disconnected', 'Disconnected');
            
            // Auto-reconnect after 3 seconds
            setTimeout(() => {
                if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.updateConnectionStatus('error', 'Error');
            this.updateWebSocketStatus('error', 'Error');
        };
    }
    
    handleWebSocketMessage(message) {
        console.log('📨 WebSocket message:', message.type);
        
        switch (message.type) {
            case 'connection_established':
                console.log('🎯 Connection established:', message.data);
                break;
                
            case 'recognition_result':
                this.handleRecognitionResult(message.data);
                break;
                
            case 'session_stats':
                this.updateSessionStats(message.data);
                break;
                
            case 'quality_warning':
                console.warn('⚠️ Quality warning:', message.data);
                break;
                
            case 'frame_skipped':
                console.log('⏭️ Frame skipped:', message.data.reason);
                break;
                
            case 'error':
            case 'frame_error':
                console.error('❌ Processing error:', message.data);
                this.showNotification(message.data.message || message.data.error, 'error');
                break;
                
            default:
                console.log('❓ Unknown message type:', message.type);
        }
    }
    
    handleRecognitionResult(result) {
        console.log('🎯 Recognition result:', result);
        
        // Update statistics
        this.stats.framesProcessed++;
        if (result.fish_detections && result.fish_detections.length > 0) {
            this.stats.fishDetected += result.fish_detections.length;
            this.stats.successfulFrames++;
        }
        if (result.faces && result.faces.length > 0) {
            this.stats.facesDetected += result.faces.length;
        }
        if (result.total_processing_time) {
            this.stats.totalProcessingTime += result.total_processing_time;
        }
        
        // Draw overlay
        this.drawRecognitionOverlay(result);
        
        // Add to detection history
        this.addToDetectionHistory(result);
        
        // Update UI
        this.updateStatsDisplay();
    }
    
    drawRecognitionOverlay(result) {
        this.clearCanvas();
        
        if (!result.fish_detections && !result.faces) {
            return;
        }
        
        console.log('🎨 Drawing recognition overlay...');
        
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;
        
        // Draw fish detections with segmentation
        if (result.fish_detections) {
            result.fish_detections.forEach((fish, index) => {
                this.drawFishDetection(fish, index, scaleX, scaleY);
            });
        }
        
        // Draw face detections
        if (result.faces) {
            result.faces.forEach((face, index) => {
                this.drawFaceDetection(face, index, scaleX, scaleY);
            });
        }
    }
    
    drawFishDetection(fish, index, scaleX, scaleY) {
        const [x1, y1, x2, y2] = fish.bbox;
        
        // Draw segmentation polygon first (if available)
        if (fish.segmentation && fish.segmentation.has_segmentation && fish.segmentation.polygon_data) {
            this.drawSegmentationPolygon(fish.segmentation.polygon_data, scaleX, scaleY);
        }
        
        // Draw bounding box
        this.ctx.strokeStyle = '#10B981';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(
            x1 * scaleX, 
            y1 * scaleY, 
            (x2 - x1) * scaleX, 
            (y2 - y1) * scaleY
        );
        
        // Draw label with classification
        const classification = fish.classification && fish.classification[0];
        let label = `Fish ${index + 1}`;
        
        if (classification) {
            label = `${classification.name} (${(classification.accuracy * 100).toFixed(1)}%)`;
        }
        
        if (fish.segmentation && fish.segmentation.has_segmentation) {
            label += ' 🔍'; // Segmentation indicator
        }
        
        this.drawLabel(label, x1 * scaleX, y1 * scaleY, '#10B981');
    }
    
    drawSegmentationPolygon(polygonData, scaleX, scaleY) {
        if (!polygonData || polygonData.length < 3) return;
        
        console.log('🔍 Drawing segmentation polygon with', polygonData.length, 'points');
        
        // Draw filled polygon
        this.ctx.beginPath();
        this.ctx.moveTo(polygonData[0][0] * scaleX, polygonData[0][1] * scaleY);
        
        for (let i = 1; i < polygonData.length; i++) {
            this.ctx.lineTo(polygonData[i][0] * scaleX, polygonData[i][1] * scaleY);
        }
        
        this.ctx.closePath();
        
        // Fill with semi-transparent yellow
        this.ctx.fillStyle = 'rgba(251, 191, 36, 0.3)';
        this.ctx.fill();
        
        // Stroke with solid yellow
        this.ctx.strokeStyle = '#FBBF24';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Add glow effect
        this.ctx.shadowColor = '#FBBF24';
        this.ctx.shadowBlur = 8;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }
    
    drawFaceDetection(face, index, scaleX, scaleY) {
        const [x1, y1, x2, y2] = face.bbox;
        
        // Draw bounding box
        this.ctx.strokeStyle = '#EF4444';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(
            x1 * scaleX, 
            y1 * scaleY, 
            (x2 - x1) * scaleX, 
            (y2 - y1) * scaleY
        );
        
        // Draw label
        this.drawLabel('Face', x1 * scaleX, y1 * scaleY, '#EF4444');
    }
    
    drawLabel(text, x, y, color) {
        this.ctx.font = 'bold 14px Arial';
        const metrics = this.ctx.measureText(text);
        const labelWidth = metrics.width + 10;
        const labelHeight = 20;
        
        // Draw background
        this.ctx.fillStyle = color;
        this.ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);
        
        // Draw text
        this.ctx.fillStyle = 'white';
        this.ctx.fillText(text, x + 5, y - 6);
    }
    
    addToDetectionHistory(result) {
        const detection = {
            timestamp: new Date(),
            fishCount: result.fish_detections ? result.fish_detections.length : 0,
            faceCount: result.faces ? result.faces.length : 0,
            processingTime: result.total_processing_time || 0,
            classifications: []
        };
        
        if (result.fish_detections) {
            result.fish_detections.forEach(fish => {
                if (fish.classification && fish.classification[0]) {
                    detection.classifications.push({
                        name: fish.classification[0].name,
                        accuracy: fish.classification[0].accuracy,
                        hasSegmentation: fish.segmentation && fish.segmentation.has_segmentation
                    });
                }
            });
        }
        
        this.stats.detectionHistory.unshift(detection);
        
        // Keep only last 10 detections
        if (this.stats.detectionHistory.length > 10) {
            this.stats.detectionHistory = this.stats.detectionHistory.slice(0, 10);
        }
        
        this.updateDetectionDisplay();
    }
    
    updateDetectionDisplay() {
        const container = document.getElementById('recentDetections');
        
        if (this.stats.detectionHistory.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <svg class="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    <p>No detections yet. Start the camera to begin recognition.</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.stats.detectionHistory.map(detection => {
            const classificationsHtml = detection.classifications.map(cls => `
                <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800 mr-1 mb-1">
                    ${cls.name} (${(cls.accuracy * 100).toFixed(1)}%)
                    ${cls.hasSegmentation ? ' 🔍' : ''}
                </span>
            `).join('');
            
            return `
                <div class="bg-gray-50 rounded-lg p-3 border">
                    <div class="flex justify-between items-start mb-2">
                        <div class="text-sm font-medium text-gray-900">
                            ${detection.fishCount} Fish, ${detection.faceCount} Faces
                        </div>
                        <div class="text-xs text-gray-500">
                            ${detection.timestamp.toLocaleTimeString()}
                        </div>
                    </div>
                    <div class="text-xs text-gray-600 mb-2">
                        Processing: ${detection.processingTime.toFixed(2)}s
                    </div>
                    <div>
                        ${classificationsHtml}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    resizeCanvas() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;
        
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.canvas.style.width = this.video.offsetWidth + 'px';
        this.canvas.style.height = this.video.offsetHeight + 'px';
    }
    
    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    updateWebSocketSettings() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const settings = {
                include_faces: this.settings.includeFaces,
                include_segmentation: this.settings.includeSegmentation,
                quality_threshold: this.settings.qualityThreshold,
                min_processing_interval: this.settings.processingInterval / 1000,
                auto_process: this.settings.autoProcess
            };
            
            console.log('⚙️ Updating WebSocket settings:', settings);
            
            this.ws.send(JSON.stringify({
                type: 'settings_update',
                data: settings
            }));
        }
    }
    
    updateProcessingInterval() {
        if (this.processingInterval && this.isProcessing) {
            clearInterval(this.processingInterval);
            this.processingInterval = setInterval(() => {
                this.captureAndSendFrame();
            }, this.settings.processingInterval);
        }
    }
    
    updateFPS() {
        this.fpsCounter++;
        const now = Date.now();
        
        if (now - this.lastFpsTime >= 1000) {
            const fps = this.fpsCounter;
            document.getElementById('fpsIndicator').textContent = `${fps} FPS`;
            this.fpsCounter = 0;
            this.lastFpsTime = now;
        }
    }
    
    updateStatsDisplay() {
        document.getElementById('framesProcessed').textContent = this.stats.framesProcessed;
        document.getElementById('fishDetected').textContent = this.stats.fishDetected;
        document.getElementById('facesDetected').textContent = this.stats.facesDetected;
        
        const avgTime = this.stats.framesProcessed > 0 ? 
            (this.stats.totalProcessingTime / this.stats.framesProcessed) * 1000 : 0;
        document.getElementById('avgProcessingTime').textContent = `${avgTime.toFixed(0)}ms`;
        
        const successRate = this.stats.framesProcessed > 0 ? 
            (this.stats.successfulFrames / this.stats.framesProcessed) * 100 : 0;
        document.getElementById('successRate').textContent = `${successRate.toFixed(1)}%`;
    }
    
    startSessionTimer() {
        this.sessionStartTime = Date.now();
        
        setInterval(() => {
            if (this.sessionStartTime) {
                const elapsed = Date.now() - this.sessionStartTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                document.getElementById('sessionDuration').textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }
    
    updateUI() {
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const toggleBtn = document.getElementById('toggleProcessingBtn');
        const toggleBtnText = document.getElementById('processingBtnText');
        
        startBtn.disabled = this.isCameraActive;
        stopBtn.disabled = !this.isCameraActive;
        toggleBtn.disabled = !this.isCameraActive;
        
        if (this.isProcessing) {
            toggleBtnText.textContent = 'Stop Recognition';
            toggleBtn.className = toggleBtn.className.replace('bg-blue-600 hover:bg-blue-700', 'bg-red-600 hover:bg-red-700');
        } else {
            toggleBtnText.textContent = 'Start Recognition';
            toggleBtn.className = toggleBtn.className.replace('bg-red-600 hover:bg-red-700', 'bg-blue-600 hover:bg-blue-700');
        }
    }
    
    updateConnectionStatus(status, text) {
        const statusElement = document.getElementById('connectionStatus');
        const dot = statusElement.querySelector('div');
        const span = statusElement.querySelector('span');
        
        dot.className = 'w-3 h-3 rounded-full';
        
        switch (status) {
            case 'connected':
                dot.className += ' bg-green-500';
                break;
            case 'connecting':
                dot.className += ' bg-yellow-500 animate-pulse';
                break;
            case 'disconnected':
                dot.className += ' bg-red-500 animate-pulse';
                break;
            case 'error':
                dot.className += ' bg-red-600';
                break;
        }
        
        span.textContent = text;
    }
    
    updateWebSocketStatus(status, text) {
        const statusElement = document.getElementById('wsStatus');
        const dot = statusElement.querySelector('div');
        const span = statusElement.querySelector('span');
        
        dot.className = 'w-2 h-2 rounded-full';
        
        switch (status) {
            case 'connected':
                dot.className += ' bg-green-500';
                break;
            case 'disconnected':
                dot.className += ' bg-red-500';
                break;
            case 'error':
                dot.className += ' bg-red-600';
                break;
        }
        
        span.textContent = text;
    }
    
    updateCameraStatus(status, text) {
        const statusElement = document.getElementById('cameraStatus');
        const dot = statusElement.querySelector('div');
        const span = statusElement.querySelector('span');
        
        dot.className = 'w-2 h-2 rounded-full';
        
        switch (status) {
            case 'active':
                dot.className += ' bg-green-500';
                break;
            case 'requesting':
                dot.className += ' bg-yellow-500 animate-pulse';
                break;
            case 'inactive':
                dot.className += ' bg-gray-500';
                break;
            case 'error':
                dot.className += ' bg-red-600';
                break;
        }
        
        span.textContent = text;
    }
    
    updateProcessingIndicator(status, text) {
        const indicator = document.getElementById('processingIndicator');
        
        indicator.className = 'px-3 py-1 rounded-full text-sm';
        
        switch (status) {
            case 'processing':
                indicator.className += ' bg-green-200 text-green-800 animate-pulse';
                break;
            case 'idle':
                indicator.className += ' bg-gray-200 text-gray-600';
                break;
        }
        
        indicator.textContent = text;
    }
    
    async checkApiHealth() {
        try {
            const response = await fetch('/api/v1/health/');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateModelsStatus('loaded', 'Loaded');
                console.log('✅ API Health check passed');
            } else {
                this.updateModelsStatus('error', 'Error');
                console.error('❌ API Health check failed');
            }
        } catch (error) {
            this.updateModelsStatus('error', 'Error');
            console.error('❌ API Health check error:', error);
        }
    }
    
    updateModelsStatus(status, text) {
        const statusElement = document.getElementById('modelsStatus');
        const dot = statusElement.querySelector('div');
        const span = statusElement.querySelector('span');
        
        dot.className = 'w-2 h-2 rounded-full';
        
        switch (status) {
            case 'loaded':
                dot.className += ' bg-green-500';
                break;
            case 'loading':
                dot.className += ' bg-yellow-500 animate-pulse';
                break;
            case 'error':
                dot.className += ' bg-red-600';
                break;
        }
        
        span.textContent = text;
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `
            px-4 py-3 rounded-lg shadow-lg text-white transform transition-all duration-300 translate-x-full
        `;
        
        switch (type) {
            case 'success':
                notification.className += ' bg-green-500';
                break;
            case 'error':
                notification.className += ' bg-red-500';
                break;
            case 'warning':
                notification.className += ' bg-yellow-500';
                break;
            default:
                notification.className += ' bg-blue-500';
        }
        
        notification.innerHTML = `
            <div class="flex items-center space-x-2">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-white hover:text-gray-200">
                    ×
                </button>
            </div>
        `;
        
        document.getElementById('notifications').appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.liveStreamApp = new LiveStreamApp();
});