/**
 * Fish Recognition API Test Application
 * Comprehensive JavaScript application for testing all API features
 */

class FishRecognitionApp {
    constructor() {
        this.apiBase = '/api/v1';
        this.recognitionWsUrl = `ws://${window.location.host}/ws/recognition/`;
        this.detectionWsUrl = `ws://${window.location.host}/ws/recognition/detection/`;
        this.recognitionWs = null;
        this.detectionWs = null;
        this.videoStream = null;
        this.isRecognitionProcessing = false;
        this.isDetectionProcessing = false;
        this.detectionInterval = null;
        this.detectionShouldReconnect = false;
        this.detectionState = 'idle';
        this.detectionIndicatorConnected = false;
        this.detectionIndicatorColor = 'bg-gray-400';
        this.captureCanvas = document.createElement('canvas');
        this.batchFrames = [];
        this.pendingDetectionFrames = new Map();
        this.MIN_BATCH_FRAMES = 10;
        this.MAX_BATCH_FRAMES = 60;
        this.MAX_LWF_FILES = 60;
        this.lwfState = {
            speciesName: '',
            scientificName: '',
            files: [],
            augment: true,
            isSubmitting: false
        };
        this.stats = {
            processed: 0,
            successful: 0,
            fishDetected: 0,
            totalTime: 0
        };
        
        this.currentMode = 'image';
        this.currentCorrectionId = null; // Store current identification ID for correction
        this.settings = {
            includeFaces: true,
            includeSegmentation: true,
            includeVisualization: true,  // Enable visualization by default
            qualityThreshold: 0.3,
            processingMode: 'accuracy',
            autoProcess: true
        };
        
        // Face filter configuration
        this.faceFilterConfig = {
            enabled: true,
            iouThreshold: 0.3
        };
        
        this.init();
    }

    async connectDetectionWebSocket() {
        if (this.detectionWs && this.detectionWs.readyState === WebSocket.OPEN) {
            return;
        }

        this.detectionShouldReconnect = true;

        return new Promise((resolve, reject) => {
            let resolved = false;
            try {
                const socket = new WebSocket(this.detectionWsUrl);
                this.detectionWs = socket;

                socket.onopen = () => {
                    console.log('Detection WebSocket connected');
                    this.updateDetectionIndicator(true);
                    resolved = true;
                    resolve();
                };

                socket.onclose = () => {
                    console.log('Detection WebSocket disconnected');
                    this.updateDetectionIndicator(false);

                    if (this.detectionShouldReconnect) {
                        setTimeout(() => this.connectDetectionWebSocket(), 1500);
                    } else {
                        this.updateDetectionStatus('idle');
                    }

                    if (!resolved) {
                        resolved = true;
                        reject(new Error('Detection stream closed before ready'));
                    }
                };

                socket.onerror = (error) => {
                    console.error('Detection WebSocket error:', error);
                    this.updateDetectionIndicator(false);

                     if (!resolved) {
                        resolved = true;
                        reject(error);
                    }
                };

                socket.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.handleDetectionMessage(message);
                    } catch (error) {
                        console.error('Failed to parse detection message:', error);
                    }
                };
            } catch (error) {
                console.error('Detection WebSocket connection failed:', error);
                this.updateDetectionIndicator(false);
                this.detectionShouldReconnect = false;
                reject(error);
            }
        });
    }

    disconnectDetectionWebSocket() {
        this.detectionShouldReconnect = false;
        if (this.detectionWs) {
            try {
                this.detectionWs.close();
            } catch (error) {
                console.warn('Failed to close detection WebSocket cleanly:', error);
            }
            this.detectionWs = null;
        }
    }

    init() {
        this.setupEventListeners();
        this.syncInitialSettings();
        this.checkApiHealth();
        this.connectRecognitionWebSocket();
        this.updateUI();
        this.updateDetectionStatus('idle');
        this.refreshDetectionIndicator();
        this.updateLwfStatus();
    }
    
    syncInitialSettings() {
        // Sync settings with HTML checkbox states
        const includeFacesEl = document.getElementById('includeFaces');
        const includeSegmentationEl = document.getElementById('includeSegmentation');
        const includeVisualizationEl = document.getElementById('includeVisualization');
        const qualityThresholdEl = document.getElementById('qualityThreshold');
        
        if (includeFacesEl) this.settings.includeFaces = includeFacesEl.checked;
        if (includeSegmentationEl) this.settings.includeSegmentation = includeSegmentationEl.checked;
        if (includeVisualizationEl) this.settings.includeVisualization = includeVisualizationEl.checked;
        if (qualityThresholdEl) this.settings.qualityThreshold = parseFloat(qualityThresholdEl.value);
        
        // Sync face filter settings
        const faceFilterEnabledEl = document.getElementById('faceFilterEnabled');
        const faceFilterThresholdEl = document.getElementById('faceFilterThreshold');
        
        if (faceFilterEnabledEl) this.faceFilterConfig.enabled = faceFilterEnabledEl.checked;
        if (faceFilterThresholdEl) this.faceFilterConfig.iouThreshold = parseFloat(faceFilterThresholdEl.value);
        
        console.log('Initial settings synced:', this.settings);
        console.log('Initial face filter config synced:', this.faceFilterConfig);
        
        // Load current face filter configuration from server
        this.loadFaceFilterConfig();
    }
    
    setupEventListeners() {
        // Mode selection
        document.getElementById('imageMode').addEventListener('click', () => this.switchMode('image'));
        document.getElementById('cameraMode').addEventListener('click', () => this.switchMode('camera'));
        document.getElementById('batchMode').addEventListener('click', () => this.switchMode('batch'));
        const lwfModeBtn = document.getElementById('lwfMode');
        if (lwfModeBtn) {
            lwfModeBtn.addEventListener('click', () => this.switchMode('lwf'));
        }
        
        // Settings
        document.getElementById('includeFaces').addEventListener('change', (e) => {
            this.settings.includeFaces = e.target.checked;
            this.updateRecognitionSettings();
            this.updateDetectionSettings();
        });
        
        document.getElementById('includeSegmentation').addEventListener('change', (e) => {
            this.settings.includeSegmentation = e.target.checked;
            this.updateRecognitionSettings();
        });
        
        document.getElementById('includeVisualization').addEventListener('change', (e) => {
            this.settings.includeVisualization = e.target.checked;
            this.updateRecognitionSettings();
            this.updateDetectionSettings();
        });
        
        document.getElementById('qualityThreshold').addEventListener('input', (e) => {
            this.settings.qualityThreshold = parseFloat(e.target.value);
            document.getElementById('qualityValue').textContent = e.target.value;
            this.updateRecognitionSettings();
            this.updateDetectionSettings();
        });
        
        // Face filter settings
        document.getElementById('faceFilterEnabled').addEventListener('change', (e) => {
            this.faceFilterConfig.enabled = e.target.checked;
        });
        
        document.getElementById('faceFilterThreshold').addEventListener('input', (e) => {
            this.faceFilterConfig.iouThreshold = parseFloat(e.target.value);
            document.getElementById('faceFilterThresholdValue').textContent = e.target.value;
        });
        
        document.getElementById('applyFaceFilterBtn').addEventListener('click', this.applyFaceFilterConfig.bind(this));
        document.getElementById('resetFaceFilterBtn').addEventListener('click', this.resetFaceFilterConfig.bind(this));
        
        // Image upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('imageInput').click();
        });
        
        document.getElementById('imageInput').addEventListener('change', this.handleImageSelect.bind(this));
        document.getElementById('analyzeBtn').addEventListener('click', this.analyzeImage.bind(this));
        
        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        dropZone.addEventListener('drop', this.handleDrop.bind(this));
        dropZone.addEventListener('click', () => document.getElementById('imageInput').click());
        
        // Camera controls
        document.getElementById('startCameraBtn').addEventListener('click', this.startCamera.bind(this));
        document.getElementById('stopCameraBtn').addEventListener('click', this.stopCamera.bind(this));
        document.getElementById('captureBtn').addEventListener('click', this.captureFrame.bind(this));
        
        // Camera settings
        document.getElementById('processingMode').addEventListener('change', (e) => {
            this.settings.processingMode = e.target.value;
            this.updateRecognitionSettings();
            this.updateDetectionSettings();
            this.restartDetectionStreaming();
        });

        document.getElementById('autoProcess').addEventListener('change', (e) => {
            this.settings.autoProcess = e.target.checked;
            this.updateRecognitionSettings();
            this.updateDetectionSettings();
            this.restartDetectionStreaming();
        });
        
        // Batch processing
        document.getElementById('batchUploadBtn').addEventListener('click', () => {
            document.getElementById('batchInput').click();
        });
        
        document.getElementById('batchInput').addEventListener('change', this.handleBatchSelect.bind(this));
        document.getElementById('processBatchBtn').addEventListener('click', this.processBatch.bind(this));

        // Batch drag and drop
        const batchDropZone = document.getElementById('batchDropZone');
        batchDropZone.addEventListener('dragover', this.handleDragOver.bind(this));
        batchDropZone.addEventListener('drop', this.handleBatchDrop.bind(this));
        batchDropZone.addEventListener('click', () => document.getElementById('batchInput').click());

        // LWF adaptation
        const lwfSpeciesInput = document.getElementById('lwfSpeciesName');
        if (lwfSpeciesInput) {
            lwfSpeciesInput.addEventListener('input', (e) => {
                this.lwfState.speciesName = e.target.value;
            });
        }

        const lwfScientificInput = document.getElementById('lwfScientificName');
        if (lwfScientificInput) {
            lwfScientificInput.addEventListener('input', (e) => {
                this.lwfState.scientificName = e.target.value;
            });
        }

        const lwfAugmentInput = document.getElementById('lwfAugment');
        if (lwfAugmentInput) {
            lwfAugmentInput.addEventListener('change', (e) => {
                this.lwfState.augment = e.target.checked;
            });
        }

        const lwfImagesInput = document.getElementById('lwfImages');
        if (lwfImagesInput) {
            lwfImagesInput.addEventListener('change', this.handleLwfFileSelect.bind(this));
        }

        const lwfUploadBtn = document.getElementById('lwfUploadBtn');
        if (lwfUploadBtn) {
            lwfUploadBtn.addEventListener('click', () => {
                const input = document.getElementById('lwfImages');
                if (input) input.click();
            });
        }

        const lwfClearBtn = document.getElementById('lwfClearBtn');
        if (lwfClearBtn) {
            lwfClearBtn.addEventListener('click', () => this.clearLwfFiles('Selection cleared'));
        }

        const lwfDropZone = document.getElementById('lwfDropZone');
        if (lwfDropZone) {
            lwfDropZone.addEventListener('dragover', this.handleDragOver.bind(this));
            lwfDropZone.addEventListener('drop', this.handleLwfDrop.bind(this));
            lwfDropZone.addEventListener('click', () => {
                const input = document.getElementById('lwfImages');
                if (input) input.click();
            });
        }

        const lwfSubmitBtn = document.getElementById('runLwfBtn');
        if (lwfSubmitBtn) {
            lwfSubmitBtn.addEventListener('click', () => this.runLwfAdaptation());
        }

        // Status refresh
        document.getElementById('refreshStatusBtn').addEventListener('click', this.checkApiHealth.bind(this));
        
        // Help modal
        document.getElementById('helpBtn').addEventListener('click', this.showHelp.bind(this));
        document.getElementById('closeHelpBtn').addEventListener('click', this.hideHelp.bind(this));
        document.getElementById('helpModal').addEventListener('click', (e) => {
            if (e.target.id === 'helpModal') this.hideHelp();
        });
        
        // Correction modal
        document.getElementById('closeCorrectionBtn').addEventListener('click', this.hideCorrectionModal.bind(this));
        document.getElementById('cancelCorrectionBtn').addEventListener('click', this.hideCorrectionModal.bind(this));
        document.getElementById('correctionModal').addEventListener('click', (e) => {
            if (e.target.id === 'correctionModal') this.hideCorrectionModal();
        });
        document.getElementById('correctionForm').addEventListener('submit', this.submitCorrection.bind(this));
        
        // Test correction button
        const testCorrectionBtn = document.getElementById('testCorrectionBtn');
        if (testCorrectionBtn) {
            testCorrectionBtn.addEventListener('click', () => {
                // Open modal with test data (no real ID, just for UI testing)
                console.log('Test correction button clicked - opening modal for UI test only');
                this.showCorrectionModal(null, {
                    scientific_name: 'Katsuwonus pelamis',
                    indonesian_name: 'Ikan Cakalang',
                    english_name: 'Skipjack Tuna',
                    kelompok: 'Ikan Laut'
                });
            });
        }
    }
    
    switchMode(mode) {
        this.currentMode = mode;
        
        // Update button styles
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.remove('bg-fish-blue', 'bg-blue-600');
            btn.classList.add('bg-gray-500');
        });
        
        document.getElementById(`${mode}Mode`).classList.remove('bg-gray-500');
        document.getElementById(`${mode}Mode`).classList.add('bg-fish-blue');
        
        // Show/hide sections
        document.getElementById('imageUploadSection').classList.toggle('hidden', mode !== 'image');
        document.getElementById('cameraSection').classList.toggle('hidden', mode !== 'camera');
        document.getElementById('batchSection').classList.toggle('hidden', mode !== 'batch');
        const lwfSection = document.getElementById('lwfSection');
        if (lwfSection) {
            lwfSection.classList.toggle('hidden', mode !== 'lwf');
        }
        
        if (mode !== 'camera' && this.videoStream) {
            this.stopCamera();
        }
    }
    
    async checkApiHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health/`);
            const data = await response.json();
            
            document.getElementById('healthStatus').textContent = data.status;
            document.getElementById('healthStatus').className = `font-semibold ${data.status === 'healthy' ? 'text-green-500' : 'text-red-500'}`;
            
            document.getElementById('modelsStatus').textContent = data.models_loaded ? 'Loaded' : 'Not Loaded';
            document.getElementById('modelsStatus').className = `font-semibold ${data.models_loaded ? 'text-green-500' : 'text-red-500'}`;
            
            document.getElementById('deviceStatus').textContent = data.device || 'Unknown';
            
        } catch (error) {
            console.error('Health check failed:', error);
            document.getElementById('healthStatus').textContent = 'Error';
            document.getElementById('healthStatus').className = 'font-semibold text-red-500';
        }
    }
    
    async loadFaceFilterConfig() {
        try {
            const response = await fetch(`${this.apiBase}/config/face-filter/`);
            if (response.ok) {
                const data = await response.json();
                this.faceFilterConfig.enabled = data.enabled;
                this.faceFilterConfig.iouThreshold = data.iou_threshold;
                
                // Update UI
                const enabledEl = document.getElementById('faceFilterEnabled');
                const thresholdEl = document.getElementById('faceFilterThreshold');
                const thresholdValueEl = document.getElementById('faceFilterThresholdValue');
                
                if (enabledEl) enabledEl.checked = data.enabled;
                if (thresholdEl) thresholdEl.value = data.iou_threshold;
                if (thresholdValueEl) thresholdValueEl.textContent = data.iou_threshold;
                
                this.updateFaceFilterStatus('Loaded current configuration');
                console.log('Face filter config loaded:', data);
            } else {
                this.updateFaceFilterStatus('Failed to load configuration', 'error');
            }
        } catch (error) {
            console.error('Failed to load face filter config:', error);
            this.updateFaceFilterStatus('Error loading configuration', 'error');
        }
    }
    
    async applyFaceFilterConfig() {
        try {
            const response = await fetch(`${this.apiBase}/config/face-filter/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    enabled: this.faceFilterConfig.enabled,
                    iou_threshold: this.faceFilterConfig.iouThreshold
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateFaceFilterStatus('Configuration applied successfully', 'success');
                console.log('Face filter config applied:', data);
                
                // Update local config with server response
                if (data.config) {
                    this.faceFilterConfig.enabled = data.config.enabled;
                    this.faceFilterConfig.iouThreshold = data.config.iou_threshold;
                }
            } else {
                const errorData = await response.json();
                this.updateFaceFilterStatus(`Failed to apply: ${errorData.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            console.error('Failed to apply face filter config:', error);
            this.updateFaceFilterStatus('Error applying configuration', 'error');
        }
    }
    
    async resetFaceFilterConfig() {
        // Reset to default values
        this.faceFilterConfig.enabled = true;
        this.faceFilterConfig.iouThreshold = 0.3;
        
        // Update UI
        document.getElementById('faceFilterEnabled').checked = true;
        document.getElementById('faceFilterThreshold').value = 0.3;
        document.getElementById('faceFilterThresholdValue').textContent = '0.3';
        
        // Apply the reset configuration
        await this.applyFaceFilterConfig();
        this.updateFaceFilterStatus('Configuration reset to defaults', 'success');
    }
    
    updateFaceFilterStatus(message, type = 'info') {
        const statusEl = document.getElementById('faceFilterStatus');
        if (statusEl) {
            statusEl.textContent = message;
            statusEl.className = `mt-2 text-sm ${
                type === 'success' ? 'text-green-600' : 
                type === 'error' ? 'text-red-600' : 
                'text-gray-600'
            }`;
            
            // Clear message after 3 seconds
            setTimeout(() => {
                statusEl.textContent = '';
            }, 3000);
        }
    }
    
    connectRecognitionWebSocket() {
        try {
            this.recognitionWs = new WebSocket(this.recognitionWsUrl);
            
            this.recognitionWs.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };
            
            this.recognitionWs.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectRecognitionWebSocket(), 3000);
            };
            
            this.recognitionWs.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
            this.recognitionWs.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleRecognitionMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('statusIndicator');
        const text = document.getElementById('statusText');
        
        if (connected) {
            indicator.className = 'w-3 h-3 bg-green-500 rounded-full pulse';
            text.textContent = 'Connected';
        } else {
            indicator.className = 'w-3 h-3 bg-red-500 rounded-full';
            text.textContent = 'Disconnected';
        }
    }
    
    handleRecognitionMessage(message) {
        console.log('WebSocket message:', message);
        
        switch (message.type) {
            case 'connection_established':
                console.log('Connection established:', message.data);
                break;
                
            case 'recognition_result':
                // Extract results from the WebSocket message structure
                const resultData = {
                    ...message.data.results,  // fish_detections, faces, visualization_image, etc.
                    frame_id: message.data.frame_id,
                    processing_time: message.data.processing_time,
                    timestamp: message.data.timestamp,
                    total_processing_time: message.data.results.total_processing_time || message.data.processing_time,
                    source: message.data.source || 'stream',
                    // IMPORTANT: Include identification data for correction feature
                    identification_id: message.data.results.identification_id,
                    correction_url: message.data.results.correction_url,
                    correction_data: message.data.results.correction_data
                };
                if (message.data.batch) {
                    resultData.batch_summary = message.data.batch;
                }
                
                // DEBUG: Log the extracted data
                console.log('=== WEBSOCKET RECOGNITION RESULT ===');
                console.log('Original message.data.results:', message.data.results);
                console.log('Extracted resultData:', resultData);
                console.log('Has identification_id:', !!resultData.identification_id);
                console.log('identification_id value:', resultData.identification_id);
                console.log('Has correction_data:', !!resultData.correction_data);
                console.log('=== END WEBSOCKET DEBUG ===');
                
                this.handleRecognitionResult(resultData);
                this.updateStats(resultData);
                if (message.data.source === 'capture') {
                    if (this.videoStream && this.settings.autoProcess) {
                        this.updateDetectionStatus('searching');
                    }
                    if (!this.settings.autoProcess && this.videoStream) {
                        this.sendDetectionFrame();
                    }
                }
                break;
                
            case 'session_stats':
                this.updateSessionStats(message.data);
                break;
                
            case 'quality_warning':
                this.showQualityWarning(message.data);
                break;
                
            case 'frame_skipped':
                console.log('Frame skipped:', message.data.reason);
                break;
                
            case 'error':
            case 'frame_error':
                this.showError(message.data.message || message.data.error);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handleDetectionMessage(message) {
        switch (message.type) {
            case 'detection_ready':
                this.updateDetectionIndicator(true);
                this.updateDetectionStatus('ready', message.data);
                break;

            case 'detection_result':
                this.isDetectionProcessing = false;
                this.capturePendingBatchFrame(message.data);
                if (this.detectionState !== 'classifying') {
                    this.updateDetectionStatus(
                        message.data.has_fish ? 'fish_detected' : 'searching',
                        message.data
                    );
                } else {
                    this.updateDetectionConfidence(message.data);
                }
                this.drawDetectionOverlay(message.data);
                break;

            case 'session_stats':
                this.updateDetectionStats(message.data);
                break;

            case 'quality_warning':
                this.showQualityWarning(message.data);
                break;

            case 'frame_skipped':
                this.isDetectionProcessing = false;
                // Throttling is expected in detection stream; log only for debugging
                break;

            case 'error':
            case 'frame_error':
                this.isDetectionProcessing = false;
                this.updateDetectionStatus('error', message.data);
                this.showError(message.data.message || message.data.error);
                this.clearOverlayCanvas();
                break;

            default:
                console.log('Unknown detection message type:', message.type);
        }
    }

    refreshDetectionIndicator() {
        const indicator = document.getElementById('detectionIndicator');
        if (!indicator) return;

        const connectionClass = this.detectionIndicatorConnected ? 'pulse' : 'opacity-40';
        indicator.className = `w-3 h-3 rounded-full ${this.detectionIndicatorColor} ${connectionClass}`;
    }

    updateDetectionIndicator(connected) {
        this.detectionIndicatorConnected = connected;
        this.refreshDetectionIndicator();
    }

    updateDetectionStatus(state, data = {}) {
        this.detectionState = state;

        const statusTextEl = document.getElementById('detectionStatusText');
        const instructionEl = document.getElementById('detectionInstructionText');

        let statusText = 'Stream nonaktif';
        let instructionText = 'Aktifkan kamera untuk memulai deteksi.';

        switch (state) {
            case 'ready':
                statusText = 'Stream siap';
                instructionText = 'Arahkan kamera ke ikan untuk memulai deteksi.';
                this.detectionIndicatorColor = 'bg-blue-500';
                break;
            case 'searching':
                statusText = 'Mencari ikan...';
                instructionText = 'Gerakkan kamera hingga ikan terlihat jelas.';
                this.detectionIndicatorColor = 'bg-amber-500';
                break;
            case 'fish_detected':
                statusText = 'Ikan terdeteksi!';
                instructionText = data.guidance || 'Tahan kamera dan tekan tombol foto.';
                this.detectionIndicatorColor = 'bg-emerald-500';
                break;
            case 'classifying':
                statusText = 'Mengirim foto...';
                instructionText = 'Tunggu hasil klasifikasi dari server.';
                this.detectionIndicatorColor = 'bg-sky-500';
                break;
            case 'error':
                statusText = 'Deteksi bermasalah';
                instructionText = data.message || 'Periksa koneksi kamera atau jaringan.';
                this.detectionIndicatorColor = 'bg-red-500';
                break;
            default:
                this.detectionIndicatorColor = 'bg-gray-400';
        }

        if (statusTextEl) statusTextEl.textContent = statusText;
        if (instructionEl) instructionEl.textContent = instructionText;

        this.updateDetectionConfidence(data);
        this.refreshDetectionIndicator();

        if (state === 'idle') {
            this.updateDetectionStats({ frames_processed: 0, frames_received: 0, avg_processing_time: 0 });
        }
    }

    updateDetectionConfidence(data) {
        const confidenceEl = document.getElementById('detectionConfidenceText');
        if (!confidenceEl) return;

        if (!data || !data.detection_summary || !data.detection_summary.max_confidence) {
            confidenceEl.textContent = 'Confidence: -';
            return;
        }

        const confidence = data.detection_summary.max_confidence || 0;
        confidenceEl.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
    }

    updateDetectionStats(data) {
        const statsEl = document.getElementById('detectionStatsText');
        if (!statsEl) return;

        const processed = data.frames_processed || 0;
        const received = data.frames_received || 0;
        const avg = data.avg_processing_time ? data.avg_processing_time.toFixed(2) : '0.00';
        const bufferCount = this.batchFrames.length;

        statsEl.textContent = `Stream: ${processed}/${received} • Avg ${avg}s • Buffer ${bufferCount}/${this.MAX_BATCH_FRAMES}`;
    }

    capturePendingBatchFrame(messageData) {
        const { frame_id: frameId, has_fish: hasFish, detection_summary: summary } = messageData;
        if (frameId == null) return;

        const pending = this.pendingDetectionFrames.get(frameId);
        if (!pending) return;

        this.pendingDetectionFrames.delete(frameId);

        if (!hasFish) return;

        const frameRecord = {
            frameId,
            data: pending.data,
            timestamp: pending.timestamp,
            detectionSummary: summary || {},
        };

        const existingIndex = this.batchFrames.findIndex((item) => item.frameId === frameId);
        if (existingIndex >= 0) {
            this.batchFrames[existingIndex] = frameRecord;
        } else {
            this.batchFrames.push(frameRecord);
        }

        this.trimBatchFrames();
    }

    trimBatchFrames() {
        if (!this.batchFrames.length) return;

        this.batchFrames.sort((a, b) => {
            const confA = a.detectionSummary?.max_confidence ?? 0;
            const confB = b.detectionSummary?.max_confidence ?? 0;
            if (confA === confB) {
                return (b.timestamp || 0) - (a.timestamp || 0);
            }
            return confB - confA;
        });

        if (this.batchFrames.length > this.MAX_BATCH_FRAMES) {
            this.batchFrames = this.batchFrames.slice(0, this.MAX_BATCH_FRAMES);
        }
    }

    drawDetectionOverlay(data) {
        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('overlayCanvas');

        if (!video || !canvas) return;
        if (!video.videoWidth || !video.videoHeight) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const displayWidth = video.offsetWidth || video.videoWidth;
        const displayHeight = video.offsetHeight || video.videoHeight;

        canvas.width = displayWidth;
        canvas.height = displayHeight;
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const detections = data.detections || [];
        if (!detections.length) {
            return;
        }

        const scaleX = displayWidth / video.videoWidth;
        const scaleY = displayHeight / video.videoHeight;

        ctx.font = '14px system-ui, -apple-system, sans-serif';
        ctx.lineWidth = 2;

        detections.forEach((det) => {
            if (!det.bbox) return;
            const [x1, y1, x2, y2] = det.bbox;

            const drawX = x1 * scaleX;
            const drawY = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;

            const colorHex = data.has_fish ? '#10b981' : '#f97316';
            const fillColor = data.has_fish ? 'rgba(16, 185, 129, 0.18)' : 'rgba(249, 115, 22, 0.18)';

            ctx.strokeStyle = colorHex;
            ctx.fillStyle = fillColor;
            ctx.beginPath();
            ctx.rect(drawX, drawY, width, height);
            ctx.fill();
            ctx.stroke();

            const label = `Fish ${(det.confidence * 100).toFixed(0)}%`;
            const textWidth = ctx.measureText(label).width;
            const labelPadding = 6;
            const labelHeight = 20;
            const labelX = drawX;
            const labelY = Math.max(0, drawY - labelHeight - 2);

            ctx.fillStyle = colorHex;
            ctx.fillRect(labelX, labelY, textWidth + labelPadding * 2, labelHeight);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, labelX + labelPadding, labelY + 14);
        });
    }

    clearOverlayCanvas() {
        const canvas = document.getElementById('overlayCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.clearRect(0, 0, canvas.width || 0, canvas.height || 0);
    }

    handleLwfFileSelect(event) {
        const files = Array.from(event.target.files || []);
        event.target.value = '';
        this.addLwfFiles(files);
    }

    handleLwfDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        const files = Array.from(event.dataTransfer.files || []).filter((file) => file.type.startsWith('image/'));
        if (!files.length) {
            this.showError('Only image files are supported for LWF adaptation');
            return;
        }
        this.addLwfFiles(files);
    }

    addLwfFiles(files) {
        if (!files || !files.length) {
            return;
        }

        const existingKeys = new Set(this.lwfState.files.map((file) => `${file.name}-${file.size}-${file.lastModified}`));
        let added = 0;

        for (const file of files) {
            if (!file.type.startsWith('image/')) {
                continue;
            }

            const key = `${file.name}-${file.size}-${file.lastModified}`;
            if (existingKeys.has(key)) {
                continue;
            }

            if (this.lwfState.files.length >= this.MAX_LWF_FILES) {
                this.showError(`Maximum ${this.MAX_LWF_FILES} images reached`);
                break;
            }

            this.lwfState.files.push(file);
            existingKeys.add(key);
            added += 1;
        }

        if (!added) {
            this.updateLwfStatus('No new images added');
            return;
        }

        this.renderLwfFileList();
        this.updateLwfStatus();
    }

    renderLwfFileList() {
        const list = document.getElementById('lwfFileList');
        const preview = document.getElementById('lwfPreview');
        if (!list || !preview) return;

        list.innerHTML = '';

        if (!this.lwfState.files.length) {
            preview.classList.add('hidden');
            return;
        }

        preview.classList.remove('hidden');
        this.lwfState.files.forEach((file, index) => {
            const item = document.createElement('li');
            const sizeKb = (file.size / 1024).toFixed(1);
            item.textContent = `${index + 1}. ${file.name} (${sizeKb} KB)`;
            list.appendChild(item);
        });
    }

    clearLwfFiles(message = null) {
        this.lwfState.files = [];
        const input = document.getElementById('lwfImages');
        if (input) {
            input.value = '';
        }
        this.renderLwfFileList();
        this.updateLwfStatus(message);
    }

    updateLwfStatus(message = null) {
        const statusEl = document.getElementById('lwfStatus');
        if (!statusEl) return;
        if (message) {
            statusEl.textContent = message;
            return;
        }

        const count = this.lwfState.files.length;
        if (count > 0) {
            statusEl.textContent = `${count} image${count > 1 ? 's' : ''} ready for adaptation`;
        } else {
            statusEl.textContent = `Select up to ${this.MAX_LWF_FILES} images (any format) for adaptation`;
        }
    }

    setLwfLoading(isLoading) {
        this.lwfState.isSubmitting = isLoading;
        const button = document.getElementById('runLwfBtn');
        const spinner = document.getElementById('lwfSpinner');
        const text = document.getElementById('lwfButtonText');

        if (!button || !spinner || !text) return;

        if (isLoading) {
            button.disabled = true;
            spinner.classList.remove('hidden');
            text.classList.add('hidden');
        } else {
            button.disabled = false;
            spinner.classList.add('hidden');
            text.classList.remove('hidden');
        }
    }

    async runLwfAdaptation() {
        if (this.lwfState.isSubmitting) return;

        const speciesName = (this.lwfState.speciesName || '').trim();
        if (!speciesName) {
            this.showError('Please provide a species name for adaptation');
            return;
        }

        if (!this.lwfState.files.length) {
            this.showError('Please select at least one image for adaptation');
            return;
        }

        if (this.lwfState.files.length > this.MAX_LWF_FILES) {
            this.showError(`Maximum ${this.MAX_LWF_FILES} images allowed for a single adaptation batch`);
            return;
        }

        const formData = new FormData();
        formData.append('species_name', speciesName);
        if (this.lwfState.scientificName) {
            formData.append('scientific_name', this.lwfState.scientificName.trim());
        }
        formData.append('augment_data', this.lwfState.augment ? 'true' : 'false');
        this.lwfState.files.forEach((file) => formData.append('images', file));

        this.setLwfLoading(true);
        this.updateLwfStatus('Running adaptation...');

        try {
            const response = await fetch(`${this.apiBase}/embedding/lwf/`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Adaptation failed');
            }

            this.clearLwfFiles('Adaptation completed successfully');
            this.renderLwfResult(result);
        } catch (error) {
            console.error('LWF adaptation failed:', error);
            this.showError(error.message || 'LWF adaptation failed');
            this.updateLwfStatus('Adaptation failed');
        } finally {
            this.setLwfLoading(false);
        }
    }

    renderLwfResult(result) {
        const container = document.getElementById('resultsContainer');
        if (!container) return;

        if (container.children.length === 1 && container.children[0].textContent.includes('No results yet')) {
            container.innerHTML = '';
        }

        const card = document.createElement('div');
        card.className = 'result-card bg-white border border-emerald-300 rounded-lg p-4 shadow-sm';
        const scientificRow = result.scientific_name ? `
            <div class="text-xs text-gray-500 italic">${result.scientific_name}</div>
        ` : '';
        card.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <div>
                    <div class="text-sm font-semibold text-emerald-600">LWF Adaptation Complete</div>
                    <div class="text-xs text-gray-500">${new Date().toLocaleTimeString()}</div>
                </div>
                <div class="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded">${result.species_name}</div>
            </div>
            ${scientificRow}
            <div class="grid grid-cols-2 gap-3 text-xs text-gray-700">
                <div>
                    <div class="font-semibold">Species ID</div>
                    <div>${result.species_id}</div>
                </div>
                <div>
                    <div class="font-semibold">New Embeddings</div>
                    <div>${result.new_embeddings}</div>
                </div>
                <div>
                    <div class="font-semibold">Total Embeddings</div>
                    <div>${result.total_embeddings}</div>
                </div>
                <div>
                    <div class="font-semibold">Majority Ratio</div>
                    <div>${(result.majority_ratio * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div class="mt-3 text-xs text-gray-600">
                Centroid shift: ${result.centroid_shift.toFixed(4)}
            </div>
        `;

        container.insertBefore(card, container.firstChild || null);
    }
    
    updateRecognitionSettings() {
        if (this.recognitionWs && this.recognitionWs.readyState === WebSocket.OPEN) {
            console.log('Updating WebSocket settings...');
            
            // Get elements with null checks
            const includeFacesEl = document.getElementById('includeFaces');
            const includeSegmentationEl = document.getElementById('includeSegmentation');
            const includeVisualizationEl = document.getElementById('includeVisualization');
            const qualityThresholdEl = document.getElementById('qualityThreshold');
            
            if (!includeFacesEl || !includeSegmentationEl || !includeVisualizationEl || !qualityThresholdEl) {
                console.error('Some settings elements not found');
                return;
            }
            
            const settings = {
                include_faces: includeFacesEl.checked,
                include_segmentation: includeSegmentationEl.checked,
                include_visualization: includeVisualizationEl.checked,
                quality_threshold: parseFloat(qualityThresholdEl.value),
                auto_process: this.settings.autoProcess,
                processing_mode: 'accuracy'
            };
            
            console.log('Sending settings to WebSocket:', settings);
            
            this.recognitionWs.send(JSON.stringify({
                type: 'settings_update',
                data: settings
            }));
        }
    }

    updateDetectionSettings() {
        if (this.detectionWs && this.detectionWs.readyState === WebSocket.OPEN) {
            const includeFacesEl = document.getElementById('includeFaces');
            const qualityThresholdEl = document.getElementById('qualityThreshold');

            const detectionSettings = {
                include_faces: includeFacesEl ? includeFacesEl.checked : false,
                include_visualization: false,
                quality_threshold: qualityThresholdEl
                    ? parseFloat(qualityThresholdEl.value)
                    : this.settings.qualityThreshold,
                auto_process: this.settings.autoProcess,
                min_processing_interval: this.settings.processingMode === 'speed' ? 0.12 : 0.25
            };

            this.detectionWs.send(JSON.stringify({
                type: 'settings_update',
                data: detectionSettings
            }));
        }
    }
    
    handleImageSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.displayImagePreview(file);
        }
    }
    
    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('previewImg');
            img.src = e.target.result;
            document.getElementById('imagePreview').classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
    
    async analyzeImage() {
        const fileInput = document.getElementById('imageInput');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select an image first');
            return;
        }
        
        this.setAnalyzeLoading(true);
        
        try {
            const formData = new FormData();
            formData.append('image', file);
            formData.append('include_faces', this.settings.includeFaces ? 'true' : 'false');
            formData.append('include_segmentation', this.settings.includeSegmentation ? 'true' : 'false');
            formData.append('include_visualization', this.settings.includeVisualization ? 'true' : 'false');
            
            const response = await fetch(`${this.apiBase}/recognize/`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            // DEBUG: Log detailed response information
            console.log('=== API RESPONSE DEBUG ===');
            console.log('Response Status:', response.status);
            console.log('Response OK:', response.ok);
            console.log('Full Result:', result);
            console.log('Include Visualization Setting:', this.settings.includeVisualization);
            
            // Check if visualization is included
            if (result.visualization_image) {
                console.log('✅ Visualization image found in response');
                console.log('Visualization image length:', result.visualization_image.length);
                console.log('Visualization image starts with:', result.visualization_image.substring(0, 50));
            } else {
                console.log('❌ No visualization image in response');
            }
            
            // Check fish detections and segmentation
            if (result.fish_detections) {
                console.log('Fish detections count:', result.fish_detections.length);
                result.fish_detections.forEach((fish, index) => {
                    console.log(`Fish ${index + 1}:`, fish);
                    if (fish.segmentation) {
                        console.log(`  - Segmentation:`, fish.segmentation);
                        if (fish.segmentation.has_segmentation) {
                            console.log(`  - Has segmentation: TRUE`);
                            console.log(`  - Polygon data:`, fish.segmentation.polygon_data);
                        } else {
                            console.log(`  - Has segmentation: FALSE`);
                        }
                    } else {
                        console.log(`  - No segmentation data`);
                    }
                });
            }
            console.log('=== END DEBUG ===');
            
            if (response.ok) {
                this.handleRecognitionResult(result);
                this.updateStats(result);
            } else {
                this.showError(result.error || 'Recognition failed');
            }
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError('Network error occurred');
        } finally {
            this.setAnalyzeLoading(false);
        }
    }
    
    setAnalyzeLoading(loading) {
        const text = document.getElementById('analyzeText');
        const spinner = document.getElementById('analyzeSpinner');
        const btn = document.getElementById('analyzeBtn');
        
        if (loading) {
            text.classList.add('hidden');
            spinner.classList.remove('hidden');
            btn.disabled = true;
        } else {
            text.classList.remove('hidden');
            spinner.classList.add('hidden');
            btn.disabled = false;
        }
    }
    
    async startCamera() {
        try {
            this.videoStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });

            const video = document.getElementById('videoElement');
            video.srcObject = this.videoStream;

            this.currentMode = 'camera';
            this.clearOverlayCanvas();

            document.getElementById('startCameraBtn').classList.add('hidden');
            document.getElementById('stopCameraBtn').classList.remove('hidden');
            document.getElementById('captureBtn').classList.remove('hidden');

            this.updateDetectionStatus('searching');

            await this.connectDetectionWebSocket().catch(() => {
                this.updateDetectionStatus('error', { message: 'Gagal membuka stream deteksi.' });
            });

            this.updateDetectionSettings();

            if (this.settings.autoProcess) {
                this.startDetectionStreaming(true);
            } else {
                this.sendDetectionFrame();
            }

        } catch (error) {
            console.error('Camera access failed:', error);
            this.showError('Failed to access camera');
        }
    }
    
    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
        }

        document.getElementById('startCameraBtn').classList.remove('hidden');
        document.getElementById('stopCameraBtn').classList.add('hidden');
        document.getElementById('captureBtn').classList.add('hidden');

        this.stopDetectionStreaming();
        this.disconnectDetectionWebSocket();
        this.clearOverlayCanvas();
        this.updateDetectionStatus('idle');
        this.batchFrames = [];
        this.pendingDetectionFrames.clear();
    }
    
    startDetectionStreaming(sendImmediate = false) {
        if (!this.settings.autoProcess) return;
        if (this.detectionInterval) return;

        if (sendImmediate) {
            this.sendDetectionFrame();
        }

        const interval = this.settings.processingMode === 'speed' ? 450 : 900;

        this.detectionInterval = setInterval(() => {
            if (this.videoStream && !this.isDetectionProcessing) {
                this.sendDetectionFrame();
            }
        }, interval);
    }

    stopDetectionStreaming() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        this.isDetectionProcessing = false;
    }

    restartDetectionStreaming() {
        this.stopDetectionStreaming();
        if (this.videoStream && this.settings.autoProcess) {
            this.startDetectionStreaming(true);
        } else if (this.videoStream) {
            this.sendDetectionFrame();
        }
    }

    captureFrame() {
        this.sendRecognitionFrame();
    }

    sendDetectionFrame() {
        if (!this.detectionWs || this.detectionWs.readyState !== WebSocket.OPEN) {
            return;
        }

        const video = document.getElementById('videoElement');
        if (!video || !video.videoWidth || !video.videoHeight) {
            return;
        }

        const ctx = this.captureCanvas.getContext('2d');
        if (!ctx) return;

        const targetWidth = 480;
        const aspectRatio = video.videoWidth / video.videoHeight || 1;
        const targetHeight = Math.round(targetWidth / aspectRatio);

        this.captureCanvas.width = targetWidth;
        this.captureCanvas.height = targetHeight;

        ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

        const frameData = this.captureCanvas.toDataURL('image/jpeg', 0.6);

        const framePayload = {
            frame_data: frameData,
            frame_id: Date.now(),
            include_faces: this.settings.includeFaces,
            include_visualization: false,
            quality_threshold: this.settings.qualityThreshold,
            manual_trigger: !this.settings.autoProcess
        };

        try {
            if (this.detectionState !== 'classifying') {
                this.updateDetectionStatus('searching');
            }
            this.pendingDetectionFrames.set(framePayload.frame_id, {
                data: frameData,
                timestamp: Date.now()
            });

            if (this.pendingDetectionFrames.size > 120) {
                const oldestKey = this.pendingDetectionFrames.keys().next().value;
                this.pendingDetectionFrames.delete(oldestKey);
            }
            this.isDetectionProcessing = true;
            this.detectionWs.send(JSON.stringify({
                type: 'camera_frame',
                data: framePayload
            }));
        } catch (error) {
            console.error('Failed to send detection frame:', error);
            this.isDetectionProcessing = false;
        }
    }

    sendRecognitionFrame() {
        const video = document.getElementById('videoElement');
        if (!video || !video.videoWidth || !video.videoHeight) {
            this.showError('Kamera belum siap untuk mengambil gambar.');
            return;
        }

        if (!this.recognitionWs || this.recognitionWs.readyState !== WebSocket.OPEN) {
            this.showError('Koneksi klasifikasi belum siap.');
            return;
        }

        if (this.batchFrames.length < this.MIN_BATCH_FRAMES) {
            this.showError(`Buffer deteksi belum cukup. Diperlukan minimal ${this.MIN_BATCH_FRAMES} frame dengan ikan terdeteksi, saat ini ${this.batchFrames.length}.`);
            return;
        }

        const selectedFrames = this.batchFrames
            .slice(0, this.MAX_BATCH_FRAMES)
            .map((item) => item.data);

        const framePayload = {
            frames: selectedFrames,
            include_faces: this.settings.includeFaces,
            include_segmentation: this.settings.includeSegmentation,
            include_visualization: this.settings.includeVisualization,
            quality_threshold: this.settings.qualityThreshold
        };

        try {
            this.isRecognitionProcessing = true;
            this.updateDetectionStatus('classifying');

            this.recognitionWs.send(JSON.stringify({
                type: 'classification_batch',
                data: framePayload
            }));

            this.stopDetectionStreaming();
            this.batchFrames = [];

            if (this.settings.autoProcess) {
                setTimeout(() => {
                    if (this.videoStream) {
                        this.startDetectionStreaming(true);
                    }
                }, 2000);
            }

        } catch (error) {
            console.error('Failed to send classification batch:', error);
            this.isRecognitionProcessing = false;
            this.updateDetectionStatus('error', { message: 'Gagal mengirim batch foto ke server.' });
        }
    }
    
    handleBatchSelect(event) {
        const files = Array.from(event.target.files);
        if (files.length > 10) {
            this.showError('Maximum 10 images allowed');
            return;
        }
        
        this.displayBatchPreview(files);
    }
    
    handleBatchDrop(event) {
        event.preventDefault();
        const files = Array.from(event.dataTransfer.files).filter(file => file.type.startsWith('image/'));
        
        if (files.length > 10) {
            this.showError('Maximum 10 images allowed');
            return;
        }
        
        this.displayBatchPreview(files);
    }
    
    displayBatchPreview(files) {
        const container = document.getElementById('batchImages');
        container.innerHTML = '';
        
        files.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const div = document.createElement('div');
                div.className = 'relative';
                div.innerHTML = `
                    <img src="${e.target.result}" class="w-full h-24 object-cover rounded-lg">
                    <div class="absolute top-2 right-2 bg-fish-blue text-white text-xs px-2 py-1 rounded">
                        ${index + 1}
                    </div>
                `;
                container.appendChild(div);
            };
            reader.readAsDataURL(file);
        });
        
        document.getElementById('batchPreview').classList.remove('hidden');
        
        // Store files for processing
        this.batchFiles = files;
    }
    
    async processBatch() {
        if (!this.batchFiles || this.batchFiles.length === 0) {
            this.showError('Please select images first');
            return;
        }
        
        this.setBatchLoading(true);
        
        try {
            const formData = new FormData();
            
            this.batchFiles.forEach(file => {
                formData.append('images', file);
            });
            
            formData.append('include_faces', this.settings.includeFaces);
            formData.append('include_segmentation', this.settings.includeSegmentation);
            formData.append('include_visualization', this.settings.includeVisualization);
            
            const response = await fetch(`${this.apiBase}/recognize/batch/`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.handleBatchResults(result);
            } else {
                this.showError(result.error || 'Batch processing failed');
            }
            
        } catch (error) {
            console.error('Batch processing failed:', error);
            this.showError('Network error occurred');
        } finally {
            this.setBatchLoading(false);
        }
    }
    
    setBatchLoading(loading) {
        const text = document.getElementById('batchText');
        const spinner = document.getElementById('batchSpinner');
        const btn = document.getElementById('processBatchBtn');
        
        if (loading) {
            text.classList.add('hidden');
            spinner.classList.remove('hidden');
            btn.disabled = true;
        } else {
            text.classList.remove('hidden');
            spinner.classList.add('hidden');
            btn.disabled = false;
        }
    }
    
    handleBatchResults(result) {
        result.results.forEach((imageResult, index) => {
            this.handleRecognitionResult(imageResult, `Batch Image ${index + 1}`);
        });
        
        this.updateStats({
            total_processing_time: result.total_processing_time,
            fish_detections: result.results.reduce((sum, r) => sum + (r.fish_detections?.length || 0), 0)
        });
    }
    
    handleRecognitionResult(result, title = null) {
        this.isRecognitionProcessing = false;
        
        // DEBUG: Log EVERYTHING
        console.log('=== HANDLE RECOGNITION RESULT DEBUG ===');
        console.log('Full result (stringified):', JSON.stringify(result, null, 2));
        console.log('Result keys:', Object.keys(result));
        console.log('identification_id:', result.identification_id);
        console.log('correction_url:', result.correction_url);
        console.log('correction_data:', result.correction_data);
        console.log('Has visualization_image:', !!result.visualization_image);
        if (result.visualization_image) {
            console.log('Visualization image data length:', result.visualization_image.length);
        }
        console.log('=== END HANDLE DEBUG ===');
        
        const container = document.getElementById('resultsContainer');
        
        // Remove placeholder if exists
        if (container.children.length === 1 && container.children[0].textContent.includes('No results yet')) {
            container.innerHTML = '';
        }
        
        const resultCard = this.createResultCard(result, title);
        container.insertBefore(resultCard, container.firstChild);
        
        // Keep only last 5 results
        while (container.children.length > 5) {
            container.removeChild(container.lastChild);
        }
        
        // Update overlay for camera mode
        if (this.currentMode === 'camera' && result.fish_detections) {
            console.log('=== DRAWING OVERLAY FOR CAMERA MODE ===');
            console.log('Current mode:', this.currentMode);
            console.log('Fish detections:', result.fish_detections.length);
            console.log('Fish detections data:', result.fish_detections);
            this.drawOverlay(result.fish_detections, result.faces);
            console.log('=== OVERLAY DRAWING COMPLETED ===');
        } else {
            console.log('=== NOT DRAWING OVERLAY ===');
            console.log('Current mode:', this.currentMode);
            console.log('Has fish detections:', !!result.fish_detections);
            console.log('Fish detections count:', result.fish_detections?.length || 0);
        }
    }
    
    createResultCard(result, title) {
        const div = document.createElement('div');
        div.className = 'result-card bg-gray-50 p-4 rounded-lg border';
        
        // Store identification ID if available
        if (result.identification_id) {
            div.setAttribute('data-identification-id', result.identification_id);
        }
        
        const fishCount = result.fish_detections?.length || 0;
        const faceCount = result.faces?.length || 0;
        const processingTime = result.total_processing_time || result.processing_time?.total || 0;
        const aggregate = result.aggregate_summary;
        const batchSummary = result.batch_summary;
        
        let fishDetails = '';
        if (result.fish_detections && result.fish_detections.length > 0) {
            fishDetails = result.fish_detections.map((fish, i) => {
                const classification = fish.classification?.[0];
                const llmVerification = fish.llm_verification;
                
                // DEBUG: Log the fish data structure
                console.log(`=== FISH ${i + 1} DATA STRUCTURE ===`);
                console.log('Full fish object:', fish);
                console.log('Classification:', classification);
                console.log('LLM Verification:', llmVerification);
                
                let detailsHtml = `<div class="text-xs bg-white p-2 rounded mt-2">
                    <strong>Fish ${i + 1}:</strong>`;
                
                if (classification) {
                    detailsHtml += ` <span class="species-name">${classification.name}</span><br>
                        <span class="text-gray-600">Accuracy: ${(classification.accuracy * 100).toFixed(1)}%</span>`;
                }
                
                // Add LLM verification if available
                if (llmVerification && !llmVerification.error) {
                    detailsHtml += `<br>
                        <div class="mt-1 pt-1 border-t border-gray-200">
                            <span class="text-purple-600 font-semibold">🤖 LLM Verification:</span><br>
                            <span class="text-gray-700 scientific-name">Scientific: ${llmVerification.scientific_name}</span><br>
                            <span class="text-gray-700">Indonesian: ${llmVerification.indonesian_name}</span><br>
                            <span class="text-gray-500 text-[10px]">LLM Time: ${llmVerification.processing_time?.toFixed(2) || 'N/A'}s</span>
                        </div>`;
                } else if (llmVerification && llmVerification.error) {
                    detailsHtml += `<br>
                        <div class="mt-1 pt-1 border-t border-gray-200">
                            <span class="text-red-600 text-[10px]">LLM Error: ${llmVerification.error}</span>
                        </div>`;
                }
                
                detailsHtml += `</div>`;
                
                return detailsHtml;
            }).join('');
        }
        
        let visualizationSection = '';
        if (result.visualization_image) {
            console.log('Adding visualization image to result card');
            visualizationSection = `
                <div class="mt-3">
                    <div class="text-xs font-semibold text-gray-700 mb-1">Visualization with Segmentation:</div>
                    <img src="${result.visualization_image}" alt="Visualization" class="w-full rounded border" style="max-height: 200px; object-fit: contain;">
                </div>
            `;
        } else {
            console.log('No visualization image to display');
        }

        let aggregateSection = '';
        if (aggregate && aggregate.top_species) {
            const top = aggregate.top_species;
            const ratio = (aggregate.majority_ratio * 100).toFixed(1);
            aggregateSection = `
                <div class="mt-3 bg-white rounded border border-emerald-200 p-3 text-xs">
                    <div class="flex items-center justify-between">
                        <span class="font-semibold text-emerald-600">Batch Consensus</span>
                        <span class="text-gray-600">${aggregate.frames_evaluated} frame</span>
                    </div>
                    <div class="mt-1 text-gray-700">
                        Spesies dominan: <strong>${top.name || 'Unknown'}</strong> (${top.count} suara, confidence terbaik ${(top.best_accuracy * 100).toFixed(1)}%).
                    </div>
                    <div class="text-gray-500 mt-1">Mayoritas ${ratio}% dari total ${aggregate.total_votes} deteksi.</div>
                </div>
            `;
        }

        let batchMetaSection = '';
        if (batchSummary) {
            batchMetaSection = `
                <div class="mt-2 text-[11px] text-gray-500">
                    Batch size: ${batchSummary.size} • Frame dipakai: ${batchSummary.frames_evaluated} • Total deteksi ikan: ${batchSummary.total_fish_detections}
                </div>
            `;
        }
        
        div.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <div class="font-semibold text-sm">${title || 'Recognition Result'}</div>
                <div class="text-xs text-gray-500">${new Date().toLocaleTimeString()}</div>
            </div>
            <div class="grid grid-cols-3 gap-2 text-xs">
                <div class="text-center">
                    <div class="font-semibold text-fish-blue">${fishCount}</div>
                    <div class="text-gray-600">Fish</div>
                </div>
                <div class="text-center">
                    <div class="font-semibold text-purple-500">${faceCount}</div>
                    <div class="text-gray-600">Faces</div>
                </div>
                <div class="text-center">
                    <div class="font-semibold text-green-500">${processingTime.toFixed(2)}s</div>
                    <div class="text-gray-600">Time</div>
                </div>
            </div>
            ${fishDetails}
            ${visualizationSection}
            ${aggregateSection}
            ${batchMetaSection}
        `;
        
        // DEBUG: Log identification_id
        console.log('=== CREATE RESULT CARD DEBUG ===');
        console.log('Result object:', result);
        console.log('Has identification_id:', !!result.identification_id);
        console.log('identification_id value:', result.identification_id);
        console.log('Has fish_detections:', !!result.fish_detections);
        console.log('Fish count:', result.fish_detections?.length || 0);
        
        // Add correction button if identification_id exists and we have fish detections
        if (result.identification_id && result.fish_detections && result.fish_detections.length > 0) {
            console.log('✅ Adding correction button');
            const fish = result.fish_detections[0];
            const classification = fish.classification?.[0];
            const llmVerification = fish.llm_verification;
            
            console.log('=== CORRECTION BUTTON DATA EXTRACTION ===');
            console.log('Fish object:', fish);
            console.log('Classification object:', classification);
            console.log('LLM Verification object:', llmVerification);
            console.log('Backend correction_data:', result.correction_data);
            console.log('Backend correction_url:', result.correction_url);
            
            // Get current data for pre-filling the correction form
            let currentData;
            
            // PRIORITY 1: Use correction_data from backend if available (most reliable)
            if (result.correction_data) {
                console.log('✅ Using correction_data from backend (most reliable)');
                currentData = {
                    scientific_name: result.correction_data.scientific_name || '',
                    indonesian_name: result.correction_data.indonesian_name || '',
                    english_name: result.correction_data.english_name || '',
                    kelompok: result.correction_data.kelompok || ''
                };
                console.log('Backend correction data:', currentData);
            } else {
                // PRIORITY 2: Construct from classification data (what's displayed in UI)
                console.log('⚠️ No correction_data from backend, constructing from classification');
                currentData = {
                    scientific_name: '',
                    indonesian_name: '',
                    english_name: '',
                    kelompok: ''
                };
                
                // Get data from classification (this is what user sees in the UI)
                if (classification) {
                    console.log('Using classification data (what user sees)');
                    console.log('Classification fields:', Object.keys(classification));
                    
                    // Get the name that's displayed in the UI
                    currentData.indonesian_name = classification.name || classification.indonesian_name || classification.species_name || '';
                    currentData.scientific_name = classification.scientific_name || classification.species || '';
                    currentData.english_name = classification.english_name || classification.common_name || '';
                    currentData.kelompok = classification.kelompok || classification.category || '';
                    
                    console.log('Classification data extracted:', currentData);
                }
                
                // Supplement with LLM verification if classification lacks data
                if (llmVerification && !llmVerification.error) {
                    console.log('📝 Supplementing with LLM verification data');
                    
                    if (!currentData.scientific_name && llmVerification.scientific_name) {
                        currentData.scientific_name = llmVerification.scientific_name;
                        console.log('Added scientific name from LLM:', currentData.scientific_name);
                    }
                    if (!currentData.english_name && llmVerification.english_name) {
                        currentData.english_name = llmVerification.english_name;
                        console.log('Added english name from LLM:', currentData.english_name);
                    }
                } else if (llmVerification && llmVerification.error) {
                    console.log('⚠️ LLM verification has error:', llmVerification.error);
                } else {
                    console.log('ℹ️ No LLM verification data available');
                }
            }
            
            console.log('📋 Final pre-fill data for correction modal:', currentData);
            
            const correctionBtn = document.createElement('button');
            correctionBtn.className = 'mt-3 w-full bg-orange-500 hover:bg-orange-600 text-white px-3 py-2 rounded text-sm font-semibold transition-colors flex items-center justify-center shadow-md';
            correctionBtn.innerHTML = '🔧 Koreksi Identifikasi Ikan';
            
            // Store correction URL in button data attribute if available
            if (result.correction_url) {
                correctionBtn.setAttribute('data-correction-url', result.correction_url);
            }
            
            correctionBtn.addEventListener('click', () => {
                console.log('🖱️ Correction button clicked!');
                console.log('Identification ID:', result.identification_id);
                console.log('Correction URL:', result.correction_url);
                console.log('Data to pass to modal:', currentData);
                this.showCorrectionModal(result.identification_id, currentData, result.correction_url);
            });
            
            div.appendChild(correctionBtn);
            console.log('✅ Correction button added to card');
        } else {
            console.log('❌ Correction button NOT added:', {
                hasId: !!result.identification_id,
                hasFish: !!result.fish_detections,
                fishCount: result.fish_detections?.length || 0
            });
        }
        
        return div;
    }
    
    drawOverlay(fishDetections, faces) {
        console.log('=== drawOverlay CALLED ===');
        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('overlayCanvas');
        const ctx = canvas.getContext('2d');
        
        console.log('Video element:', video);
        console.log('Canvas element:', canvas);
        console.log('Canvas context:', ctx);
        
        // Ensure video is loaded
        if (!video.videoWidth || !video.videoHeight || video.offsetWidth === 0 || video.offsetHeight === 0) {
            console.log('Video not ready for overlay, retrying in 100ms...');
            setTimeout(() => this.drawOverlay(fishDetections, faces), 100);
            return;
        }
        
        // Set canvas size to match video element display size
        canvas.width = video.offsetWidth;
        canvas.height = video.offsetHeight;
        canvas.style.width = video.offsetWidth + 'px';
        canvas.style.height = video.offsetHeight + 'px';
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scaling factors from video native resolution to display size
        const scaleX = video.offsetWidth / video.videoWidth;
        const scaleY = video.offsetHeight / video.videoHeight;
        
        console.log('Drawing overlay - Video dimensions:', video.videoWidth, 'x', video.videoHeight);
        console.log('Drawing overlay - Display dimensions:', video.offsetWidth, 'x', video.offsetHeight);
        console.log('Drawing overlay - Scale factors:', scaleX, 'x', scaleY);
        console.log('Drawing overlay for', fishDetections?.length || 0, 'fish detections');
        
        // Draw fish detections
        if (fishDetections && fishDetections.length > 0) {
            fishDetections.forEach((fish, i) => {
                console.log('Drawing fish', i + 1, ':', fish);
                
                const [x1, y1, x2, y2] = fish.bbox;
                console.log('Original bbox:', [x1, y1, x2, y2]);
                console.log('Scaled bbox:', [x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY]);
                
                // Draw segmentation polygon first (if available)
                const segmentation = fish.segmentation;
                if (segmentation && segmentation.has_segmentation && segmentation.polygon_data) {
                    console.log('Drawing segmentation polygon for fish', i + 1);
                    console.log('First few polygon points:', segmentation.polygon_data.slice(0, 5));
                    
                    ctx.beginPath();
                    ctx.strokeStyle = '#FBBF24'; // Yellow for segmentation
                    ctx.fillStyle = 'rgba(251, 191, 36, 0.2)'; // Semi-transparent yellow
                    ctx.lineWidth = 2;
                    
                    const polygonData = segmentation.polygon_data;
                    if (polygonData && polygonData.length > 2) {
                        const firstPoint = [polygonData[0][0] * scaleX, polygonData[0][1] * scaleY];
                        console.log('First polygon point scaled:', firstPoint);
                        ctx.moveTo(firstPoint[0], firstPoint[1]);
                        for (let j = 1; j < polygonData.length; j++) {
                            ctx.lineTo(polygonData[j][0] * scaleX, polygonData[j][1] * scaleY);
                        }
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                        console.log('Polygon drawn successfully');
                    }
                }
                
                // Draw bounding box
                ctx.strokeStyle = '#10B981';
                ctx.lineWidth = 3;
                const rectX = x1 * scaleX;
                const rectY = y1 * scaleY;
                const rectWidth = (x2 - x1) * scaleX;
                const rectHeight = (y2 - y1) * scaleY;
                console.log('Drawing bounding box at:', [rectX, rectY, rectWidth, rectHeight]);
                ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);
                
                // Draw label
                const classification = fish.classification?.[0];
                if (classification) {
                    let label = `${classification.name} (${(classification.accuracy * 100).toFixed(0)}%)`;
                    if (segmentation && segmentation.has_segmentation) {
                        label += ' [S]'; // Indicate segmentation
                    }
                    
                    ctx.fillStyle = '#10B981';
                    const labelWidth = ctx.measureText(label).width + 10;
                    ctx.fillRect(x1 * scaleX, (y1 - 25) * scaleY, labelWidth, 20);
                    ctx.fillStyle = 'white';
                    ctx.font = '14px Arial';
                    ctx.fillText(label, (x1 + 5) * scaleX, (y1 - 8) * scaleY);
                } else {
                    // Default label
                    let label = `Fish ${i + 1}`;
                    if (segmentation && segmentation.has_segmentation) {
                        label += ' [S]';
                    }
                    
                    ctx.fillStyle = '#10B981';
                    const labelWidth = ctx.measureText(label).width + 10;
                    ctx.fillRect(x1 * scaleX, (y1 - 25) * scaleY, labelWidth, 20);
                    ctx.fillStyle = 'white';
                    ctx.font = '14px Arial';
                    ctx.fillText(label, (x1 + 5) * scaleX, (y1 - 8) * scaleY);
                }
            });
        }
        
        // Draw face detections
        if (faces && faces.length > 0) {
            faces.forEach(face => {
                const [x1, y1, x2, y2] = face.bbox;
                
                ctx.strokeStyle = '#EF4444';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
                
                ctx.fillStyle = '#EF4444';
                ctx.fillRect(x1 * scaleX, (y1 - 20) * scaleY, 40, 15);
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.fillText('Face', (x1 + 2) * scaleX, (y1 - 8) * scaleY);
            });
        }
    }
    
    updateStats(result) {
        this.stats.processed++;
        if (result.success !== false) {
            this.stats.successful++;
        }
        if (result.fish_detections && Array.isArray(result.fish_detections)) {
            this.stats.fishDetected += result.fish_detections.length;
        }
        if (result.total_processing_time && !isNaN(result.total_processing_time)) {
            this.stats.totalTime += result.total_processing_time;
        }
        
        this.updateStatsDisplay();
    }
    
    updateStatsDisplay() {
        const processed = this.stats.processed || 0;
        const successful = this.stats.successful || 0;
        const totalTime = this.stats.totalTime || 0;
        const fishDetected = this.stats.fishDetected || 0;
        
        document.getElementById('processedCount').textContent = processed;
        document.getElementById('successRate').textContent = 
            processed > 0 ? `${((successful / processed) * 100).toFixed(1)}%` : '0.0%';
        document.getElementById('avgTime').textContent = 
            successful > 0 ? `${(totalTime / successful).toFixed(2)}s` : '0.00s';
        document.getElementById('fishCount').textContent = fishDetected;
    }
    
    updateSessionStats(stats) {
        console.log('Updating session stats:', stats);
        
        if (stats.frames_processed !== undefined && !isNaN(stats.frames_processed)) {
            document.getElementById('processedCount').textContent = stats.frames_processed;
        }
        if (stats.avg_processing_time !== undefined && !isNaN(stats.avg_processing_time)) {
            document.getElementById('avgTime').textContent = `${stats.avg_processing_time.toFixed(2)}s`;
        }
        if (stats.processing_rate !== undefined && !isNaN(stats.processing_rate)) {
            document.getElementById('successRate').textContent = `${(stats.processing_rate * 100).toFixed(1)}%`;
        }
        if (stats.total_fish_detected !== undefined && !isNaN(stats.total_fish_detected)) {
            document.getElementById('fishCount').textContent = stats.total_fish_detected;
        }
    }
    
    showQualityWarning(data) {
        this.showNotification(`Quality Warning: ${data.message}`, 'warning');
    }
    
    showError(message) {
        this.showNotification(`Error: ${message}`, 'error');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 max-w-sm ${
            type === 'error' ? 'bg-red-500' : 
            type === 'warning' ? 'bg-yellow-500' : 
            'bg-blue-500'
        } text-white`;
        
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('border-fish-blue');
    }
    
    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('border-fish-blue');
        
        const files = Array.from(event.dataTransfer.files).filter(file => file.type.startsWith('image/'));
        if (files.length > 0) {
            this.displayImagePreview(files[0]);
            
            // Set file to input for processing
            const dt = new DataTransfer();
            dt.items.add(files[0]);
            document.getElementById('imageInput').files = dt.files;
        }
    }
    
    showHelp() {
        document.getElementById('helpModal').classList.remove('hidden');
    }
    
    hideHelp() {
        document.getElementById('helpModal').classList.add('hidden');
    }
    
    showCorrectionModal(identificationId, currentData, correctionUrl = null) {
        console.log('=== SHOW CORRECTION MODAL ===');
        console.log('Identification ID:', identificationId);
        console.log('Current Data received:', currentData);
        console.log('Correction URL:', correctionUrl);
        
        this.currentCorrectionId = identificationId;
        this.currentCorrectionUrl = correctionUrl; // Store for later use in submit
        
        // Pre-fill form with current data
        const scientificInput = document.getElementById('correctScientificName');
        const indonesianInput = document.getElementById('correctIndonesianName');
        const englishInput = document.getElementById('correctEnglishName');
        const kelompokInput = document.getElementById('correctKelompok');
        const notesInput = document.getElementById('correctNotes');
        
        if (scientificInput) {
            scientificInput.value = currentData.scientific_name || '';
            console.log('Set scientific name:', scientificInput.value);
        }
        if (indonesianInput) {
            indonesianInput.value = currentData.indonesian_name || '';
            console.log('Set indonesian name:', indonesianInput.value);
        }
        if (englishInput) {
            englishInput.value = currentData.english_name || '';
            console.log('Set english name:', englishInput.value);
        }
        if (kelompokInput) {
            kelompokInput.value = currentData.kelompok || '';
            console.log('Set kelompok:', kelompokInput.value);
        }
        if (notesInput) {
            notesInput.value = '';
        }
        
        // Clear status
        const statusEl = document.getElementById('correctionStatus');
        if (statusEl) {
            statusEl.classList.add('hidden');
            statusEl.textContent = '';
        }
        
        // Show modal
        const modal = document.getElementById('correctionModal');
        if (modal) {
            modal.classList.remove('hidden');
            console.log('✅ Modal displayed');
        } else {
            console.error('❌ Modal element not found');
        }
    }
    
    hideCorrectionModal() {
        document.getElementById('correctionModal').classList.add('hidden');
        this.currentCorrectionId = null;
        document.getElementById('correctionForm').reset();
    }
    
    async submitCorrection(event) {
        event.preventDefault();
        
        if (!this.currentCorrectionId) {
            this.showCorrectionStatus('Mode test - tidak ada identification ID untuk disimpan', 'error');
            this.showNotification('Ini mode test saja. Gunakan tombol koreksi di result card untuk koreksi yang sebenarnya.', 'error');
            return;
        }
        
        const correctionData = {
            scientific_name: document.getElementById('correctScientificName').value.trim(),
            indonesian_name: document.getElementById('correctIndonesianName').value.trim(),
            english_name: document.getElementById('correctEnglishName').value.trim() || null,
            kelompok: document.getElementById('correctKelompok').value.trim() || null,
            notes: document.getElementById('correctNotes').value.trim() || null
        };
        
        if (!correctionData.scientific_name || !correctionData.indonesian_name) {
            this.showCorrectionStatus('Nama ilmiah dan nama Indonesia harus diisi', 'error');
            return;
        }
        
        this.setCorrectionLoading(true);
        
        try {
            // Use stored correction URL if available, otherwise construct it
            const url = this.currentCorrectionUrl || `${this.apiBase}/identifications/${this.currentCorrectionId}/correct/`;
            console.log('Submitting correction to URL:', url);
            console.log('Correction data:', correctionData);
            
            // Get CSRF token from cookies
            function getCookie(name) {
                let cookieValue = null;
                if (document.cookie && document.cookie !== '') {
                    const cookies = document.cookie.split(';');
                    for (let i = 0; i < cookies.length; i++) {
                        const cookie = cookies[i].trim();
                        if (cookie.substring(0, name.length + 1) === (name + '=')) {
                            cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                            break;
                        }
                    }
                }
                return cookieValue;
            }
            
            const csrftoken = getCookie('csrftoken');
            
            const headers = {
                'Content-Type': 'application/json',
            };
            
            if (csrftoken) {
                headers['X-CSRFToken'] = csrftoken;
            }
            
            console.log('Request headers:', headers);
            
            const response = await fetch(url, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(correctionData)
            });
            
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);
            
            // Clone response before reading to avoid "body stream already read" error
            const responseClone = response.clone();
            
            let result;
            let responseText;
            
            try {
                // Try to parse as JSON first
                result = await response.json();
                console.log('Response data:', result);
            } catch (jsonError) {
                console.error('Failed to parse JSON response:', jsonError);
                // If JSON parsing fails, try to get text from cloned response
                try {
                    responseText = await responseClone.text();
                    console.error('Response text:', responseText);
                } catch (textError) {
                    console.error('Failed to get text from response:', textError);
                }
                throw new Error('Invalid JSON response from server: ' + (responseText || 'Could not read response'));
            }
            
            if (!response.ok) {
                console.error('Response not ok:', result);
                const errorMsg = result.error || result.detail || result.message || JSON.stringify(result) || 'Failed to submit correction';
                throw new Error(errorMsg);
            }
            
            this.showCorrectionStatus('Koreksi berhasil disimpan!', 'success');
            this.showNotification('Identifikasi berhasil dikoreksi', 'success');
            
            // Update the result card in the UI
            this.updateResultCardAfterCorrection(this.currentCorrectionId, result);
            
            // Close modal after 1.5 seconds
            setTimeout(() => {
                this.hideCorrectionModal();
            }, 1500);
            
        } catch (error) {
            console.error('=== CORRECTION ERROR ===');
            console.error('Error object:', error);
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            this.showCorrectionStatus(`Error: ${error.message}`, 'error');
            this.showNotification('Gagal menyimpan koreksi: ' + error.message, 'error');
        } finally {
            this.setCorrectionLoading(false);
        }
    }
    
    setCorrectionLoading(loading) {
        const submitText = document.getElementById('correctionSubmitText');
        const spinner = document.getElementById('correctionSpinner');
        const submitBtn = document.querySelector('#correctionForm button[type=\"submit\"]');
        
        if (loading) {
            submitText.textContent = 'Menyimpan...';
            spinner.classList.remove('hidden');
            submitBtn.disabled = true;
        } else {
            submitText.textContent = 'Simpan Koreksi';
            spinner.classList.add('hidden');
            submitBtn.disabled = false;
        }
    }
    
    showCorrectionStatus(message, type = 'info') {
        const statusEl = document.getElementById('correctionStatus');
        statusEl.classList.remove('hidden', 'text-gray-600', 'text-green-600', 'text-red-600', 'text-blue-600');
        
        if (type === 'success') {
            statusEl.classList.add('text-green-600');
        } else if (type === 'error') {
            statusEl.classList.add('text-red-600');
        } else {
            statusEl.classList.add('text-blue-600');
        }
        
        statusEl.textContent = message;
    }
    
    updateResultCardAfterCorrection(identificationId, correctionResult) {
        // Find and update the result card in the UI
        const resultsContainer = document.getElementById('resultsContainer');
        const cards = resultsContainer.querySelectorAll('.result-card');
        
        cards.forEach(card => {
            const cardIdAttr = card.getAttribute('data-identification-id');
            if (cardIdAttr === identificationId) {
                // Update the species name display
                const speciesNameEl = card.querySelector('.species-name');
                if (speciesNameEl) {
                    speciesNameEl.textContent = correctionResult.current_indonesian_name;
                }
                
                const scientificNameEl = card.querySelector('.scientific-name');
                if (scientificNameEl) {
                    scientificNameEl.textContent = correctionResult.current_scientific_name;
                }
                
                // Add corrected badge
                const headerDiv = card.querySelector('.flex.items-center.justify-between');
                if (headerDiv && !card.querySelector('.corrected-badge')) {
                    const badge = document.createElement('span');
                    badge.className = 'corrected-badge text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded ml-2';
                    badge.textContent = '✓ Dikoreksi';
                    headerDiv.appendChild(badge);
                }
            }
        });
    }
    
    updateUI() {
        // Update quality threshold display
        document.getElementById('qualityValue').textContent = this.settings.qualityThreshold;
        
        // Set initial settings
        document.getElementById('includeFaces').checked = this.settings.includeFaces;
        document.getElementById('includeSegmentation').checked = this.settings.includeSegmentation;
        document.getElementById('includeVisualization').checked = this.settings.includeVisualization;
        document.getElementById('qualityThreshold').value = this.settings.qualityThreshold;
        document.getElementById('processingMode').value = this.settings.processingMode;
        document.getElementById('autoProcess').checked = this.settings.autoProcess;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fishApp = new FishRecognitionApp();
});
