class VoiceEmotionDetector {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.animationId = null;

        // DOM elements
        this.startButton = document.getElementById('startRecording');
        this.stopButton = document.getElementById('stopRecording');
        this.waveformCanvas = document.getElementById('waveform');
        this.ctx = this.waveformCanvas.getContext('2d');
        this.spinner = document.querySelector('.spinner');
        this.statusElement = document.querySelector('.status');
        this.errorElement = document.querySelector('.error');
        this.resultsElement = document.querySelector('.results');

        // Bind event listeners
        this.startButton.addEventListener('click', () => this.startRecording());
        this.stopButton.addEventListener('click', () => this.stopRecording());

        // Initialize
        this.setupCanvas();
        this.setupButtons();
    }

    setupCanvas() {
        // Set canvas size
        this.waveformCanvas.width = this.waveformCanvas.offsetWidth;
        this.waveformCanvas.height = this.waveformCanvas.offsetHeight;

        // Handle window resize
        window.addEventListener('resize', () => {
            this.waveformCanvas.width = this.waveformCanvas.offsetWidth;
            this.waveformCanvas.height = this.waveformCanvas.offsetHeight;
        });
    }

    setupButtons() {
        this.stopButton.disabled = true;
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.setupAudioContext(stream);
            this.setupMediaRecorder(stream);
            
            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateUI(true);
            this.drawWaveform();
            
            this.showStatus('Recording...');
        } catch (error) {
            this.showError('Error accessing microphone: ' + error.message);
        }
    }

    setupAudioContext(stream) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = this.audioContext.createMediaStreamSource(stream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 2048;
        source.connect(this.analyser);
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    }

    setupMediaRecorder(stream) {
        this.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm'
        });

        this.mediaRecorder.addEventListener('dataavailable', event => {
            this.audioChunks.push(event.data);
        });

        this.mediaRecorder.addEventListener('stop', () => this.processRecording());
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateUI(false);
            cancelAnimationFrame(this.animationId);
            this.showStatus('Processing audio...');
            this.spinner.style.display = 'block';
        }
    }

    async processRecording() {
        try {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio', audioBlob);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }

            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            this.showError('Error processing audio: ' + error.message);
        } finally {
            this.audioChunks = [];
            this.spinner.style.display = 'none';
        }
    }

    displayResults(result) {
        // Clear previous results
        this.resultsElement.innerHTML = '';
        this.showStatus('');

        // Create emotion result section
        const emotionResult = document.createElement('div');
        emotionResult.className = 'emotion-result';
        emotionResult.innerHTML = `
            <div class="emotion-label">Detected Emotion: ${result.emotion}</div>
        `;

        // Create probability bars
        const probabilityBars = document.createElement('div');
        probabilityBars.className = 'probability-bars';

        Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1])
            .forEach(([emotion, probability]) => {
                const percentage = (probability * 100).toFixed(1);
                const bar = document.createElement('div');
                bar.className = 'probability-bar';
                bar.innerHTML = `
                    <div class="probability-label">
                        <span>${emotion}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${percentage}%"></div>
                    </div>
                `;
                probabilityBars.appendChild(bar);
            });

        this.resultsElement.appendChild(emotionResult);
        this.resultsElement.appendChild(probabilityBars);
        this.resultsElement.style.display = 'block';
    }

    drawWaveform() {
        this.ctx.fillStyle = '#f8f9fa';
        this.ctx.fillRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);

        this.analyser.getByteTimeDomainData(this.dataArray);

        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#007bff';
        this.ctx.beginPath();

        const sliceWidth = this.waveformCanvas.width / this.analyser.frequencyBinCount;
        let x = 0;

        for (let i = 0; i < this.analyser.frequencyBinCount; i++) {
            const v = this.dataArray[i] / 128.0;
            const y = v * this.waveformCanvas.height / 2;

            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        this.ctx.lineTo(this.waveformCanvas.width, this.waveformCanvas.height / 2);
        this.ctx.stroke();

        if (this.isRecording) {
            this.animationId = requestAnimationFrame(() => this.drawWaveform());
        }
    }

    updateUI(isRecording) {
        this.startButton.disabled = isRecording;
        this.stopButton.disabled = !isRecording;
    }

    showStatus(message) {
        this.statusElement.textContent = message;
        this.errorElement.textContent = '';
    }

    showError(message) {
        this.errorElement.textContent = message;
        this.statusElement.textContent = '';
        this.spinner.style.display = 'none';
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VoiceEmotionDetector();
}); 
