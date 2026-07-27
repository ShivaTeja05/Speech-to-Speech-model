/**
 * Apollo Hospital Voice AI Assistant v2.0
 * Frontend JavaScript for audio recording, processing, and UI updates
 * 
 * v2.0 Features:
 * - Session management with session_id
 * - Context display showing patient info
 * - RAG/context indicators
 */

// ============================================================================
// STATE
// ============================================================================

const state = {
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    isProcessing: false,
    serverReady: false,
    sessionId: null
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
    statusIndicator: document.getElementById('statusIndicator'),
    languageSelect: document.getElementById('languageSelect'),
    micButton: document.getElementById('micButton'),
    recordingIndicator: document.getElementById('recordingIndicator'),
    uploadButton: document.getElementById('uploadButton'),
    audioUpload: document.getElementById('audioUpload'),
    fileName: document.getElementById('fileName'),
    pipelineSection: document.getElementById('pipelineSection'),
    totalLatency: document.getElementById('totalLatency'),
    escalationAlert: document.getElementById('escalationAlert'),
    escalationReason: document.getElementById('escalationReason'),
    conversationSection: document.getElementById('conversationSection'),
    conversationContainer: document.getElementById('conversationContainer'),
    audioPlayerSection: document.getElementById('audioPlayerSection'),
    responseAudio: document.getElementById('responseAudio'),
    signalsSection: document.getElementById('signalsSection'),
    signalIntent: document.getElementById('signalIntent'),
    signalUrgency: document.getElementById('signalUrgency'),
    signalTone: document.getElementById('signalTone'),
    signalLanguage: document.getElementById('signalLanguage'),
    // New elements for session/context (may not exist in old HTML)
    sessionInfo: document.getElementById('sessionInfo'),
    contextInfo: document.getElementById('contextInfo')
};

// ============================================================================
// INITIALIZATION
// ============================================================================

async function init() {
    // Check server health
    await checkServerHealth();
    
    // Set up event listeners
    setupEventListeners();
    
    // Check for microphone permissions
    await checkMicrophonePermission();
    
    // Initialize or restore session
    await initSession();
}

async function checkServerHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            state.serverReady = true;
            updateStatus('ready', 'Ready');
            
            // Show additional info
            console.log('Server status:', data);
            console.log('Redis connected:', data.redis_connected);
            console.log('RAG ready:', data.rag_ready);
        } else {
            updateStatus('loading', 'Loading models...');
            // Poll until ready
            setTimeout(checkServerHealth, 3000);
        }
    } catch (error) {
        updateStatus('error', 'Server unavailable');
        setTimeout(checkServerHealth, 5000);
    }
}

function updateStatus(status, text) {
    const indicator = elements.statusIndicator;
    if (indicator) {
        indicator.className = `status-indicator ${status}`;
        const statusText = indicator.querySelector('.status-text');
        if (statusText) {
            statusText.textContent = text;
        }
    }
}

async function checkMicrophonePermission() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());
    } catch (error) {
        console.warn('Microphone permission not granted:', error);
    }
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

async function initSession() {
    // Check if we have a session in sessionStorage
    const storedSessionId = sessionStorage.getItem('apollo_session_id');
    
    if (storedSessionId) {
        // Validate the session still exists on server
        try {
            const response = await fetch(`/api/session/${storedSessionId}`);
            if (response.ok) {
                state.sessionId = storedSessionId;
                console.log('Restored session:', state.sessionId);
                updateSessionDisplay();
                return;
            }
        } catch (error) {
            console.log('Session expired, creating new one');
        }
    }
    
    // Create new session
    await createNewSession();
}

async function createNewSession() {
    try {
        const response = await fetch('/api/session', { method: 'POST' });
        const data = await response.json();
        
        state.sessionId = data.session_id;
        sessionStorage.setItem('apollo_session_id', state.sessionId);
        
        console.log('Created new session:', state.sessionId);
        updateSessionDisplay();
    } catch (error) {
        console.error('Failed to create session:', error);
        // Generate a local session ID as fallback
        state.sessionId = 'local_' + Math.random().toString(36).substring(2, 10);
    }
}

function updateSessionDisplay() {
    // Update session info display if element exists
    if (elements.sessionInfo) {
        elements.sessionInfo.textContent = `Session: ${state.sessionId}`;
    }
}

function clearSession() {
    sessionStorage.removeItem('apollo_session_id');
    state.sessionId = null;
    createNewSession();
    
    // Clear conversation display
    if (elements.conversationContainer) {
        elements.conversationContainer.innerHTML = `
            <div class="empty-state">
                <p>Start speaking to begin the conversation</p>
            </div>
        `;
    }
    
    // Hide context
    if (elements.contextInfo) {
        elements.contextInfo.hidden = true;
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    // Microphone button - press and hold
    if (elements.micButton) {
        elements.micButton.addEventListener('mousedown', startRecording);
        elements.micButton.addEventListener('mouseup', stopRecording);
        elements.micButton.addEventListener('mouseleave', stopRecording);
        
        // Touch events for mobile
        elements.micButton.addEventListener('touchstart', (e) => {
            e.preventDefault();
            startRecording();
        });
        elements.micButton.addEventListener('touchend', (e) => {
            e.preventDefault();
            stopRecording();
        });
    }
    
    // Upload button
    if (elements.uploadButton) {
        elements.uploadButton.addEventListener('click', () => {
            elements.audioUpload.click();
        });
    }
    
    // File input change
    if (elements.audioUpload) {
        elements.audioUpload.addEventListener('change', handleFileUpload);
    }
    
    // New session button (if exists)
    const newSessionBtn = document.getElementById('newSessionBtn');
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', clearSession);
    }
}

// ============================================================================
// RECORDING
// ============================================================================

async function startRecording() {
    if (state.isRecording || state.isProcessing || !state.serverReady) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        state.audioChunks = [];
        
        state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audioChunks.push(event.data);
            }
        };
        
        state.mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());
            
            if (state.audioChunks.length > 0) {
                const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
                await processAudio(audioBlob);
            }
        };
        
        state.mediaRecorder.start(100);
        state.isRecording = true;
        
        // Update UI
        if (elements.micButton) {
            elements.micButton.classList.add('recording');
        }
        if (elements.recordingIndicator) {
            elements.recordingIndicator.classList.add('active');
        }
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Microphone access denied. Please allow microphone access.');
    }
}

function stopRecording() {
    if (!state.isRecording || !state.mediaRecorder) return;
    
    state.isRecording = false;
    state.mediaRecorder.stop();
    
    // Update UI
    if (elements.micButton) {
        elements.micButton.classList.remove('recording');
    }
    if (elements.recordingIndicator) {
        elements.recordingIndicator.classList.remove('active');
    }
}

// ============================================================================
// FILE UPLOAD
// ============================================================================

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (elements.fileName) {
        elements.fileName.textContent = file.name;
    }
    await processAudio(file);
    
    // Reset file input
    event.target.value = '';
}

// ============================================================================
// AUDIO PROCESSING
// ============================================================================

async function processAudio(audioBlob) {
    if (state.isProcessing) return;
    
    state.isProcessing = true;
    updateStatus('processing', 'Processing...');
    
    // Reset pipeline stages
    resetPipelineStages();
    hideEscalationAlert();
    
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        
        // Add language if selected
        const selectedLanguage = elements.languageSelect ? elements.languageSelect.value : '';
        if (selectedLanguage) {
            formData.append('language', selectedLanguage);
        }
        
        // Add session ID
        if (state.sessionId) {
            formData.append('session_id', state.sessionId);
        }
        
        // Show pipeline section
        if (elements.pipelineSection) {
            elements.pipelineSection.classList.add('active');
        }
        
        // Simulate pipeline progress
        animatePipelineProgress();
        
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update session ID if server created one
            if (data.session_id && data.session_id !== state.sessionId) {
                state.sessionId = data.session_id;
                sessionStorage.setItem('apollo_session_id', state.sessionId);
                updateSessionDisplay();
            }
            
            handleSuccessResponse(data);
        } else {
            showError(data.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('Error processing audio:', error);
        showError('Failed to process audio. Please try again.');
    } finally {
        state.isProcessing = false;
        updateStatus('ready', 'Ready');
    }
}

function animatePipelineProgress() {
    const stages = ['vad', 'stt', 'language_detection', 'layer1_signals', 'safety_gate', 'policy', 'llm', 'tts'];
    
    stages.forEach((stage, index) => {
        setTimeout(() => {
            const stageEl = document.querySelector(`[data-stage="${stage}"]`);
            if (stageEl) {
                stageEl.classList.add('processing');
            }
        }, index * 200);
    });
}

function resetPipelineStages() {
    document.querySelectorAll('.stage').forEach(stage => {
        stage.classList.remove('processing', 'complete', 'error');
        const latencyEl = stage.querySelector('.stage-latency');
        if (latencyEl) {
            latencyEl.textContent = '--';
        }
    });
    
    if (elements.totalLatency) {
        const latencyValue = elements.totalLatency.querySelector('.latency-value');
        if (latencyValue) {
            latencyValue.textContent = '--';
        }
    }
}

// ============================================================================
// RESPONSE HANDLING
// ============================================================================

function handleSuccessResponse(data) {
    // Update pipeline stages with actual latencies
    updatePipelineLatencies(data.pipeline);
    
    // Update total latency
    if (elements.totalLatency) {
        const latencyValue = elements.totalLatency.querySelector('.latency-value');
        if (latencyValue) {
            latencyValue.textContent = `${data.total_latency_ms.toFixed(0)}ms`;
        }
    }
    
    // Check for escalation
    if (data.escalation && data.escalation.should_escalate) {
        showEscalationAlert(data.escalation.reason);
    }
    
    // Add to conversation
    addToConversation('user', data.transcription, data.language_name);
    addToConversation('assistant', data.response, data.language_name);
    
    // Play audio response
    if (data.audio_response) {
        playAudioResponse(data.audio_response);
    }
    
    // Update signals
    if (data.pipeline && data.pipeline.layer1_signals) {
        updateSignals(data.pipeline.layer1_signals, data.language_name);
    }
    
    // Show context indicators
    showContextIndicators(data.context);
}

function updatePipelineLatencies(pipeline) {
    const stages = ['vad', 'stt', 'language_detection', 'layer1_signals', 'safety_gate', 'policy', 'llm', 'tts'];
    
    stages.forEach(stage => {
        const stageEl = document.querySelector(`[data-stage="${stage}"]`);
        if (stageEl && pipeline[stage]) {
            stageEl.classList.remove('processing');
            stageEl.classList.add('complete');
            
            const latencyEl = stageEl.querySelector('.stage-latency');
            if (latencyEl) {
                latencyEl.textContent = `${pipeline[stage].latency_ms.toFixed(0)}ms`;
            }
        }
    });
}

function addToConversation(role, text, language) {
    if (!elements.conversationContainer) return;
    
    // Remove empty state if present
    const emptyState = elements.conversationContainer.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const labelSpan = document.createElement('span');
    labelSpan.className = 'message-label';
    labelSpan.textContent = role === 'user' ? 'You' : 'Apollo AI';
    
    const langSpan = document.createElement('span');
    langSpan.className = 'message-language';
    langSpan.textContent = language;
    
    const textP = document.createElement('p');
    textP.className = 'message-text';
    textP.textContent = text;
    
    messageDiv.appendChild(labelSpan);
    messageDiv.appendChild(langSpan);
    messageDiv.appendChild(textP);
    
    elements.conversationContainer.appendChild(messageDiv);
    elements.conversationContainer.scrollTop = elements.conversationContainer.scrollHeight;
}

function playAudioResponse(base64Audio) {
    const audioBlob = base64ToBlob(base64Audio, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);
    
    if (elements.responseAudio) {
        elements.responseAudio.src = audioUrl;
        if (elements.audioPlayerSection) {
            elements.audioPlayerSection.hidden = false;
        }
        elements.responseAudio.play();
    }
}

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function updateSignals(signals, language) {
    if (elements.signalsSection) {
        elements.signalsSection.hidden = false;
    }
    
    if (elements.signalIntent) {
        elements.signalIntent.textContent = signals.intent;
        elements.signalIntent.className = `signal-value ${signals.intent}`;
    }
    
    if (elements.signalUrgency) {
        elements.signalUrgency.textContent = signals.urgency;
        elements.signalUrgency.className = `signal-value ${signals.urgency}`;
    }
    
    if (elements.signalTone) {
        elements.signalTone.textContent = signals.tone;
        elements.signalTone.className = `signal-value ${signals.tone}`;
    }
    
    if (elements.signalLanguage) {
        elements.signalLanguage.textContent = language;
    }
    
    // Show stress level if available
    const stressEl = document.getElementById('signalStress');
    if (stressEl && signals.stress_level) {
        stressEl.textContent = signals.stress_level;
        stressEl.className = `signal-value ${signals.stress_level}`;
    }
}

function showContextIndicators(context) {
    if (!context) return;
    
    // Show indicators for RAG and history usage
    const contextIndicators = document.getElementById('contextIndicators');
    if (contextIndicators) {
        let html = '';
        
        if (context.used_rag) {
            html += '<span class="context-badge rag">FAQ Matched</span>';
        }
        
        if (context.used_history) {
            html += '<span class="context-badge history">Context Used</span>';
        }
        
        contextIndicators.innerHTML = html;
        contextIndicators.hidden = !html;
    }
}

// ============================================================================
// ESCALATION HANDLING
// ============================================================================

function showEscalationAlert(reason) {
    if (elements.escalationAlert) {
        elements.escalationAlert.hidden = false;
        
        if (elements.escalationReason) {
            elements.escalationReason.textContent = reason;
        }
        
        // Add shake animation
        elements.escalationAlert.classList.add('shake');
        setTimeout(() => {
            elements.escalationAlert.classList.remove('shake');
        }, 500);
    }
}

function hideEscalationAlert() {
    if (elements.escalationAlert) {
        elements.escalationAlert.hidden = true;
    }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

function showError(message) {
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// ============================================================================
// START
// ============================================================================

document.addEventListener('DOMContentLoaded', init);
