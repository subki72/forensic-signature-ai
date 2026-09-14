import { Clock, Cpu, Scale } from 'lucide-react';
import { getVerdictConfig } from '../utils/verdict';

/**
 * ScoreBar component showing numerical similarity, progress indicator, telemetry chips,
 * and legal forensic disclaimer.
 */
export function ScoreBar({ result }) {
  if (!result) return null;

  const config = getVerdictConfig(result.verdict);

  return (
    <div className="score-container">
      <div className="score-value text-gradient">
        {result.normalized_percentage.toFixed(2)}%
      </div>
      <div className="score-label">
        Similarity Score (Raw Cosine: {result.similarity_score.toFixed(4)}) — Threshold:{' '}
        {(((result.system_threshold + 1) / 2) * 100).toFixed(1)}%
      </div>

      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${Math.min(100, Math.max(0, result.normalized_percentage))}%`,
            background: config.gradient,
          }}
        />
      </div>

      {/* Metadata Badges */}
      <div className="meta-chips">
        <span className="meta-chip">
          <Clock size={12} />
          {result.latency_ms} ms
        </span>
        <span className="meta-chip">
          <Cpu size={12} />
          ResNet-18 Siamese
        </span>
        <span className="meta-chip">
          <Scale size={12} />
          {result.confidence_band}
        </span>
      </div>

      <div className="analysis-text" style={{ textAlign: 'left', marginTop: '1rem' }}>
        <strong>Forensic Analysis:</strong> {result.analysis}
      </div>

      {/* Legal Disclaimer Box */}
      <div className="disclaimer-banner">
        <div className="disclaimer-title">
          <Scale size={14} />
          Statutory Forensic Evidentiary Notice
        </div>
        {result.disclaimer}
      </div>
    </div>
  );
}

export default ScoreBar;
