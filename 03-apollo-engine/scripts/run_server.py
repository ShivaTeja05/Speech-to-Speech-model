"""
Apollo Voice Engine - Demo Web Server

A simple web interface to test the voice assistant.
Run with: python scripts/run_server.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import threading
import webbrowser

# Import safety classifier
from apollo_voice_engine.safety.classifier import SafetyClassifier, SafetyLevel

PORT = 8080
classifier = SafetyClassifier()

# HTML for the demo interface
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Apollo Voice Engine Demo</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .lang-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .lang-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
            background: rgba(255,255,255,0.1);
            color: white;
        }
        .lang-btn:hover { background: rgba(255,255,255,0.2); }
        .lang-btn.active { background: #00d9ff; color: #1a1a2e; }
        textarea {
            width: 100%;
            height: 100px;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(0,0,0,0.3);
            color: white;
            font-size: 16px;
            resize: none;
            margin-bottom: 16px;
        }
        textarea:focus { outline: none; border-color: #00d9ff; }
        .submit-btn {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            color: #1a1a2e;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .submit-btn:hover { transform: scale(1.02); }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 12px;
            display: none;
        }
        .result.safe { background: rgba(0,255,136,0.1); border: 1px solid #00ff88; }
        .result.caution { background: rgba(255,200,0,0.1); border: 1px solid #ffc800; }
        .result.emergency { background: rgba(255,50,50,0.1); border: 1px solid #ff3232; }
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .status-safe { background: #00ff88; color: #1a1a2e; }
        .status-caution { background: #ffc800; color: #1a1a2e; }
        .status-emergency { background: #ff3232; color: white; }
        .examples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .example {
            padding: 8px 14px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .example:hover { background: rgba(255,255,255,0.2); }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }
        .metric-value { font-size: 24px; font-weight: 700; color: #00d9ff; }
        .metric-label { font-size: 12px; color: #888; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Apollo Voice Engine</h1>
        <p class="subtitle">Unified Speech-to-Speech for Indian Languages</p>
        
        <div class="card">
            <h3 style="margin-bottom: 15px;">Select Language</h3>
            <div class="lang-buttons">
                <button class="lang-btn active" data-lang="hi">हिंदी (Hindi)</button>
                <button class="lang-btn" data-lang="ta">தமிழ் (Tamil)</button>
                <button class="lang-btn" data-lang="te">తెలుగు (Telugu)</button>
                <button class="lang-btn" data-lang="kn">ಕನ್ನಡ (Kannada)</button>
                <button class="lang-btn" data-lang="en">English</button>
            </div>
            
            <textarea id="queryInput" placeholder="Type your query here..."></textarea>
            <button class="submit-btn" onclick="analyzeQuery()">🔍 Analyze Safety & Generate Response</button>
            
            <div class="examples">
                <span style="color: #888; font-size: 13px;">Try:</span>
                <span class="example" onclick="setQuery('मुझे छाती में दर्द हो रहा है')">छाती में दर्द</span>
                <span class="example" onclick="setQuery('I need an ambulance!')">Ambulance</span>
                <span class="example" onclick="setQuery('கார்டியாலஜி எங்கே?')">Cardiology</span>
                <span class="example" onclick="setQuery('What time does pharmacy open?')">Pharmacy</span>
            </div>
        </div>
        
        <div id="result" class="result">
            <span id="statusBadge" class="status-badge"></span>
            <h3 id="resultTitle"></h3>
            <p id="resultText" style="margin-top: 10px; color: #ccc;"></p>
            <p id="keywords" style="margin-top: 10px; font-size: 13px; color: #888;"></p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="metricLatency">--</div>
                <div class="metric-label">Target Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metricCost">₹0.05</div>
                <div class="metric-label">Cost/min</div>
            </div>
            <div class="metric">
                <div class="metric-value">4</div>
                <div class="metric-label">Languages</div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedLang = 'hi';
        
        document.querySelectorAll('.lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedLang = btn.dataset.lang;
            });
        });
        
        function setQuery(text) {
            document.getElementById('queryInput').value = text;
        }
        
        function analyzeQuery() {
            const query = document.getElementById('queryInput').value;
            if (!query) return;
            
            // Simulate API call (in production, this would call the backend)
            const result = classifySafety(query);
            showResult(result);
        }
        
        function classifySafety(text) {
            const textLower = text.toLowerCase();
            
            const emergencyKeywords = [
                'emergency', 'ambulance', 'heart attack', 'chest pain', 'accident',
                'इमरजेंसी', 'एंबुलेंस', 'छाती में दर्द', 'दुर्घटना',
                'அவசரம்', 'நெஞ்சு வலி', 'அத்யவசர', 'ఛాతీ నొప్పి',
                'ತುರ್ತು', 'ಎದೆ ನೋವು'
            ];
            
            const cautionKeywords = [
                'pain', 'fever', 'vomiting', 'दर्द', 'बुखार', 'வலி', 'நோப్పి', 'ನೋವು'
            ];
            
            const matched = emergencyKeywords.filter(k => textLower.includes(k.toLowerCase()));
            if (matched.length > 0) {
                return {
                    level: 'emergency',
                    keywords: matched,
                    message: '🚨 Emergency detected! Transferring to human operator...',
                    transfer: true
                };
            }
            
            const cautionMatched = cautionKeywords.filter(k => textLower.includes(k.toLowerCase()));
            if (cautionMatched.length > 0) {
                return {
                    level: 'caution',
                    keywords: cautionMatched,
                    message: '⚠️ Medical concern detected. Proceeding with care.',
                    transfer: false
                };
            }
            
            return {
                level: 'safe',
                keywords: [],
                message: '✓ Safe query. Generating response...',
                transfer: false
            };
        }
        
        function showResult(result) {
            const resultDiv = document.getElementById('result');
            const badge = document.getElementById('statusBadge');
            const title = document.getElementById('resultTitle');
            const text = document.getElementById('resultText');
            const keywords = document.getElementById('keywords');
            const latency = document.getElementById('metricLatency');
            
            resultDiv.className = 'result ' + result.level;
            resultDiv.style.display = 'block';
            
            badge.className = 'status-badge status-' + result.level;
            badge.textContent = result.level.toUpperCase();
            
            title.textContent = result.message;
            
            if (result.transfer) {
                text.textContent = 'Connecting to healthcare professional...';
                latency.textContent = 'N/A';
            } else {
                text.textContent = 'Response would be generated by the model here.';
                latency.textContent = '<300ms';
            }
            
            if (result.keywords.length > 0) {
                keywords.textContent = 'Triggered: ' + result.keywords.join(', ');
            } else {
                keywords.textContent = '';
            }
        }
    </script>
</body>
</html>
"""

class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/classify':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            result = classifier.classify(data.get('text', ''))
            
            response = {
                'level': result.level.value,
                'keywords': result.triggered_keywords,
                'should_transfer': result.should_transfer,
                'message': result.message
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           🏥 Apollo Voice Engine - Demo Server                ║
╠═══════════════════════════════════════════════════════════════╣
║  Open in browser: http://localhost:8080                       ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    server = HTTPServer(('localhost', PORT), DemoHandler)
    
    # Open browser automatically
    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        server.shutdown()


if __name__ == "__main__":
    main()
