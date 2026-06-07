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
      setError("Please upload both signature documents before proceeding.");
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
      setResult(data.verification);
    } catch (err) {
      console.error(err);
      setError("Connection failed. Please ensure the backend server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const isSuccess = result?.status === "AUTHENTIC (VERIFIED)";

  return (
    <div className="container">
      <header className="header">
        <h1 className="text-gradient">Legal Document AI</h1>
        <p>Forensic Signature Verification</p>
        <div className="divider" />
      </header>

      <main className="glass-panel" style={{ padding: '2rem' }}>
        <div className="grid-2">
          {/* Master Signature */}
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h3 className="card-label">Reference Specimen</h3>
            <div className={`dropzone ${masterFile ? 'active' : ''}`}>
              <input
                type="file"
                accept="image/jpeg, image/png, image/jpg"
                onChange={(e) => handleFileChange(e, setMasterFile, setMasterPreview)}
              />
              {masterPreview ? (
                <img src={masterPreview} alt="Reference Preview" className="image-preview" />
              ) : (
                <>
                  <UploadCloud className="dropzone-icon" />
                  <div className="dropzone-title">Upload Reference Signature</div>
                  <div className="dropzone-desc">The verified, authentic specimen</div>
                </>
              )}
            </div>
          </div>

          {/* Questioned Signature */}
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h3 className="card-label">Questioned Document</h3>
            <div className={`dropzone ${testFile ? 'active' : ''}`}>
              <input
                type="file"
                accept="image/jpeg, image/png, image/jpg"
                onChange={(e) => handleFileChange(e, setTestFile, setTestPreview)}
              />
              {testPreview ? (
                <img src={testPreview} alt="Questioned Preview" className="image-preview" />
              ) : (
                <>
                  <UploadCloud className="dropzone-icon" />
                  <div className="dropzone-title">Upload Questioned Signature</div>
                  <div className="dropzone-desc">The document under examination</div>
                </>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="error-box">
            <AlertCircle size={18} />
            {error}
          </div>
        )}

        <button
          className="btn-primary"
          onClick={handleVerify}
          disabled={isLoading || !masterFile || !testFile}
          style={{ width: '100%', padding: '1rem', fontSize: '0.95rem' }}
        >
          {isLoading ? (
            <div className="spinner" />
          ) : (
            <>
              <ShieldCheck size={20} />
              Verify Authenticity
            </>
          )}
        </button>

        {/* Results */}
        {result && (
          <div className="result-card glass-card">
            <div className={`status-badge ${isSuccess ? 'status-success' : 'status-fail'}`}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {isSuccess ? <CheckCircle2 size={20} /> : <XCircle size={20} />}
                {isSuccess ? 'Authentic' : 'Forgery Detected'}
              </div>
            </div>

            <div className="score-container">
              <div className="score-value text-gradient">
                {(result.similarity_score * 100).toFixed(2)}%
              </div>
              <div className="score-label">
                Similarity Score — Threshold: {result.system_threshold}
              </div>

              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, result.similarity_score * 100))}%`,
                    background: isSuccess
                      ? 'linear-gradient(90deg, #7a9e7e, #a3c4a7)'
                      : 'linear-gradient(90deg, #b85c5c, #d48a8a)'
                  }}
                />
              </div>
            </div>

            <div className="analysis-text">
              <strong>Analysis:</strong> {result.analysis}
            </div>
          </div>
        )}
      </main>

      <div className="footer">Powered by ResNet-18 Siamese Architecture</div>
    </div>
  );
}

export default App;
