import { useState, useEffect } from 'react';
import {
  ShieldCheck,
  AlertCircle,
} from 'lucide-react';
import { checkLiveness, checkReadiness, verifySignatures } from './services/api';
import TelemetryHeader from './components/TelemetryHeader';
import DropzoneCard from './components/DropzoneCard';
import VerdictBadge from './components/VerdictBadge';
import ScoreBar from './components/ScoreBar';
import './index.css';

function App() {
  const [masterFile, setMasterFile] = useState(null);
  const [masterPreview, setMasterPreview] = useState(null);

  const [testFile, setTestFile] = useState(null);
  const [testPreview, setTestPreview] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // System telemetry state
  const [apiStatus, setApiStatus] = useState({ online: false, ready: false, checking: true });

  useEffect(() => {
    async function verifySystemTelemetry() {
      const live = await checkLiveness();
      const ready = await checkReadiness();
      setApiStatus({
        online: live.online,
        ready: ready.ready,
        checking: false,
      });
    }
    verifySystemTelemetry();
    const interval = setInterval(verifySystemTelemetry, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleFileChange = (e, setFile, setPreview) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        setError(`File '${file.name}' exceeds the 5 MB maximum size limit.`);
        return;
      }
      setError(null);
      setFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const clearFile = (setFile, setPreview) => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const handleVerify = async () => {
    if (!masterFile || !testFile) {
      setError('Please upload both reference and questioned signature documents.');
      return;
    }

    setError(null);
    setIsLoading(true);
    setResult(null);

    try {
      const verification = await verifySignatures(masterFile, testFile);
      setResult(verification);
    } catch (err) {
      setError(err.message || 'Verification request failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <TelemetryHeader apiStatus={apiStatus} />

      <main className="glass-panel" style={{ padding: '2rem' }}>
        <div className="grid-2">
          {/* Master Signature */}
          <DropzoneCard
            title="Reference Specimen (Asli)"
            subtitle="Verified genuine specimen (Max 5MB)"
            file={masterFile}
            preview={masterPreview}
            onFileSelect={(e) => handleFileChange(e, setMasterFile, setMasterPreview)}
            onClear={() => clearFile(setMasterFile, setMasterPreview)}
          />

          {/* Questioned Signature */}
          <DropzoneCard
            title="Questioned Document (Uji)"
            subtitle="Document under examination (Max 5MB)"
            file={testFile}
            preview={testPreview}
            onFileSelect={(e) => handleFileChange(e, setTestFile, setTestPreview)}
            onClear={() => clearFile(setTestFile, setTestPreview)}
          />
        </div>

        {error && (
          <div className="error-box">
            <AlertCircle size={18} />
            <span>{error}</span>
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

        {/* Verification Results Panel */}
        {result && (
          <div className="result-card glass-card">
            <VerdictBadge verdict={result.verdict} />
            <ScoreBar result={result} />
          </div>
        )}
      </main>

      <footer className="footer">
        Powered by Siamese Network V2.1 — ResNet-18 Backbone & Projection Head
      </footer>
    </div>
  );
}

export default App;
