import { useState } from 'react';
import { UploadCloud, ShieldCheck, CheckCircle2, XCircle, AlertCircle } from 'lucide-react';
import './index.css';

function App() {
  const [masterFile, setMasterFile] = useState(null);
  const [masterPreview, setMasterPreview] = useState(null);
  
  const [testFile, setTestFile] = useState(null);
  const [testPreview, setTestPreview] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e, setFile, setPreview) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleVerify = async () => {
    if (!masterFile || !testFile) {
      setError("Please upload both signature documents.");
      return;
    }
    
    setError(null);
    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file_asli', masterFile);
    formData.append('file_uji', testFile);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/verify`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data.verifikasi);
    } catch (err) {
      console.error(err);
      setError("Connection failed. Ensure the FastAPI backend is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const isSuccess = result?.status === "ASLI (TERVERIFIKASI)";

  return (
    <div className="container">
      <header className="header">
        <h1 className="text-gradient">Legal Document AI</h1>
        <p>Forensic Signature Verification System</p>
      </header>

      <main className="glass-panel" style={{ padding: '2rem' }}>
        <div className="grid-2">
          {/* Master Signature Dropzone */}
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h3 style={{ marginBottom: '1rem', color: 'var(--accent-primary)' }}>
              Master Document (Genuine)
            </h3>
            <div className={`dropzone ${masterFile ? 'active' : ''}`}>
              <input 
                type="file" 
                accept="image/jpeg, image/png, image/jpg" 
                onChange={(e) => handleFileChange(e, setMasterFile, setMasterPreview)} 
              />
              {masterPreview ? (
                <img src={masterPreview} alt="Master Preview" className="image-preview" />
              ) : (
                <>
                  <UploadCloud className="dropzone-icon" />
                  <div className="dropzone-title">Drop Master Signature here</div>
                  <div className="dropzone-desc">Supports JPG, PNG</div>
                </>
              )}
            </div>
          </div>

          {/* Questioned Signature Dropzone */}
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h3 style={{ marginBottom: '1rem', color: 'var(--accent-secondary)' }}>
              Questioned Document (Test)
            </h3>
            <div className={`dropzone ${testFile ? 'active' : ''}`}>
              <input 
                type="file" 
                accept="image/jpeg, image/png, image/jpg" 
                onChange={(e) => handleFileChange(e, setTestFile, setTestPreview)} 
              />
              {testPreview ? (
                <img src={testPreview} alt="Test Preview" className="image-preview" />
              ) : (
                <>
                  <UploadCloud className="dropzone-icon" />
                  <div className="dropzone-title">Drop Test Signature here</div>
                  <div className="dropzone-desc">Supports JPG, PNG</div>
                </>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div style={{ color: 'var(--accent-error)', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem', padding: '1rem', background: 'rgba(239,68,68,0.1)', borderRadius: 'var(--radius-sm)' }}>
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        <button 
          className="btn-primary" 
          onClick={handleVerify} 
          disabled={isLoading || !masterFile || !testFile}
          style={{ width: '100%', padding: '1rem', fontSize: '1.2rem' }}
        >
          {isLoading ? (
            <div className="spinner" />
          ) : (
            <>
              <ShieldCheck size={24} />
              Verify Authenticity
            </>
          )}
        </button>

        {/* Results Section */}
        {result && (
          <div className="result-card glass-card">
            <div className={`status-badge ${isSuccess ? 'status-success' : 'status-fail'}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {isSuccess ? <CheckCircle2 size={24} /> : <XCircle size={24} />}
                {isSuccess ? 'VERIFIED [ASLI]' : 'REJECTED [FORGED/BEDA]'}
              </div>
            </div>
            
            <div className="score-container">
              <div className="score-value text-gradient">
                {(result.skor_kemiripan * 100).toFixed(2)}%
              </div>
              <div className="score-label">
                Similarity Score (System Threshold: {result.threshold_sistem})
              </div>
              
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ 
                    width: `${Math.min(100, Math.max(0, result.skor_kemiripan * 100))}%`,
                    background: isSuccess ? 'var(--accent-success)' : 'var(--accent-error)'
                  }} 
                />
              </div>
            </div>

            <div className="analysis-text">
              <strong>AI Analysis:</strong> {result.hasil_analisa}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
