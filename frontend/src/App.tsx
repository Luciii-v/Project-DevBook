import { useState } from 'react';
import { Sparkles, History, TextCursorInput, Box, FileText, BookOpen } from 'lucide-react';
import { CodeEditor } from './components/Editor/CodeEditor';
import { ExplainerWorkspace } from './components/Workspace/ExplainerWorkspace';
import { IntroAnimation } from './components/Intro/IntroAnimation';
import { HistoryPanel } from './components/Dashboard/HistoryPanel';

function App() {
  const [mode, setMode] = useState<'line' | 'block' | 'file'>('file');
  const [showIntro, setShowIntro] = useState(true);
  const [showHistory, setShowHistory] = useState(false);
  
  const [aiInsight, setAiInsight] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Updated the welcome text to match the new brand
  const [codeContent, setCodeContent] = useState(`// Welcome to DevBook Workspace
// Paste your code here to begin understanding it dynamically.

function analyzeCode(input: string) {
  const result = parseAST(input);
  return generateInteractiveBreakdown(result);
}`);

  const handleRunCode = async () => {
    if (!codeContent.trim()) return;
    
    setIsAnalyzing(true);
    setAiInsight(""); // Clear previous insight so the loading state looks clean

    try {
      // DYNAMIC API URL SETUP FOR DEPLOYMENT
      const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${API_BASE_URL}/query?user_question=${encodeURIComponent(codeContent)}&mode=${mode}`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Backend Response:", data); 
      
      if (data.insight) {
        setAiInsight(data.insight);
      } else if (data.error) {
        setAiInsight(`Backend error: ${data.error}`);
      } else {
        setAiInsight("The Architect returned an empty blueprint.");
      }
      
    } catch (error) {
      console.error("Brain disconnected:", error);
      setAiInsight("The streets are silent. Check if your Python backend is running.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <>
      <div className="animated-bg-container">
        <div className="shape-1"></div>
        <div className="shape-2"></div>
        <div className="shape-3"></div>
      </div>

      {showIntro && <IntroAnimation onComplete={() => setShowIntro(false)} />}

      {!showIntro && (
        <div className="app-layout fade-in">
          <header className="header-container glass-panel">
            
            {/* --- BRAND NEW DEVBOOK LOGO --- */}
            <div className="header-logo" style={{ display: 'flex', alignItems: 'center', gap: '12px', userSelect: 'none', cursor: 'pointer', padding: '4px 8px' }}>
                {/* 1. Glowing Icon Container */}
                <div style={{ 
                    position: 'relative',
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    width: '40px', 
                    height: '40px', 
                    borderRadius: '12px', 
                    background: 'linear-gradient(135deg, #4f46e5 0%, #9333ea 100%)',
                    boxShadow: '0 0 20px rgba(147, 51, 234, 0.4)',
                    border: '1px solid rgba(255,255,255,0.1)'
                }}>
                    <BookOpen size={20} color="white" />
                    {/* Pulsing Sparkle */}
                    <div style={{ position: 'absolute', bottom: '-4px', right: '-4px', animation: 'pulse 2s infinite' }}>
                        <Sparkles size={14} color="#a5b4fc" />
                    </div>
                    <style>{`@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: .5; transform: scale(0.8); } }`}</style>
                </div>

                {/* 2. Gradient Typography */}
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <h1 style={{ 
                        fontSize: '22px', 
                        fontWeight: 900, 
                        letterSpacing: '-0.5px',
                        margin: 0,
                        lineHeight: '1.1',
                        background: 'linear-gradient(to right, #818cf8, #c084fc)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                    }}>
                        DevBook
                    </h1>
                    <span style={{ 
                        fontSize: '9px', 
                        fontWeight: 700, 
                        letterSpacing: '1.5px', 
                        color: '#a78bfa', 
                        textTransform: 'uppercase',
                        marginTop: '2px'
                    }}>
                        Interactive AI Architect
                    </span>
                </div>
            </div>

            <div className="mode-selector">
              <button
                className={`mode-btn ${mode === 'line' ? 'active' : ''}`}
                onClick={() => setMode('line')}
              >
                <TextCursorInput size={16} /> Explain Line
              </button>
              <button
                className={`mode-btn ${mode === 'block' ? 'active' : ''}`}
                onClick={() => setMode('block')}
              >
                <Box size={16} /> Explain Block
              </button>
              <button
                className={`mode-btn ${mode === 'file' ? 'active' : ''}`}
                onClick={() => setMode('file')}
              >
                <FileText size={16} /> Explain File
              </button>
            </div>

            <div className="header-actions">
              <button
                className={`mode-btn glass-panel glow-hover ${showHistory ? 'active' : ''}`}
                onClick={() => setShowHistory(!showHistory)}
              >
                <History size={16} /> History
              </button>
            </div>
          </header>

          <main className="main-workspace">
            <CodeEditor 
              code={codeContent} 
              onChange={(val) => setCodeContent(val || '')} 
              onRun={handleRunCode} 
            />
            <ExplainerWorkspace 
              mode={mode} 
              insight={aiInsight} 
              isLoading={isAnalyzing} 
            />
          </main>

          {showHistory && <HistoryPanel onClose={() => setShowHistory(false)} />}
        </div>
      )}
    </>
  );
}

export default App;