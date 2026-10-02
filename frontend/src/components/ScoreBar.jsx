import { Clock, Cpu, Scale, ShieldAlert, ShieldCheck, Activity } from 'lucide-react';
import { getVerdictConfig } from '../utils/verdict';

/**
 * ScoreBar component showing similarity scores, progress indicator,
 * two-phase forensic telemetry panel, and legal disclaimer.
 */
export function ScoreBar({ result }) {
  if (!result) return null;

  const config = getVerdictConfig(result.verdict);

  // Determine stage label for rejected pairs
  const stageLabel =
    result.stage_rejected
      ? 'Stage 1 — Macro-Geometric Gate'
      : result.rejection_stage === 'micro'
      ? 'Stage 2 — Micro-Stroke Siamese'
      : 'Both Stages Passed';

  const microPercent =
    result.micro_score !== -1.0
      ? ((result.micro_score + 1.0) / 2.0 * 100).toFixed(1)
      : 'N/A (skipped)';

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
          {result.stage_rejected ? 'Stage 1 Only' : 'Stage 1 + Stage 2'}
        </span>
        <span className="meta-chip">
          <Scale size={12} />
          {result.confidence_band}
        </span>
      </div>

      {/* ── Two-Phase Pipeline Telemetry Panel ─────────────────────────────── */}
      <div
        className="two-phase-panel"
        style={{
          marginTop: '1.25rem',
          borderRadius: '12px',
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.10)',
          overflow: 'hidden',
        }}
      >
        {/* Panel Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '0.65rem 1rem',
            background: 'rgba(255,255,255,0.06)',
            borderBottom: '1px solid rgba(255,255,255,0.08)',
            fontSize: '0.78rem',
            fontWeight: 700,
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
            color: 'var(--color-text-secondary)',
          }}
        >
          <Activity size={13} />
          Two-Phase Forensic Pipeline — {stageLabel}
        </div>

        <div style={{ padding: '0.9rem 1rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>

          {/* Stage 1 */}
          <div style={{ gridColumn: '1 / -1' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '0.4rem' }}>
              {result.macro_score >= 0.25 ? (
                <ShieldCheck size={13} style={{ color: '#7a9e7e' }} />
              ) : (
                <ShieldAlert size={13} style={{ color: '#b85c5c' }} />
              )}
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>
                Stage 1 — Macro-Geometric Screening
              </span>
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  color: result.macro_score >= 0.25 ? '#7a9e7e' : '#b85c5c',
                }}
              >
                {(result.macro_score * 100).toFixed(1)}%
                {result.macro_score >= 0.25 ? ' ✓ Passed' : ' ✗ Rejected'}
              </span>
            </div>

            {/* Stage 1 sub-bar */}
            <div className="progress-bar" style={{ height: '5px', marginBottom: '0.5rem' }}>
              <div
                className="progress-fill"
                style={{
                  width: `${Math.min(100, result.macro_score * 100)}%`,
                  background: result.macro_score >= 0.25
                    ? 'linear-gradient(90deg, #7a9e7e, #a3c4a7)'
                    : 'linear-gradient(90deg, #b85c5c, #d48a8a)',
                }}
              />
            </div>

            {/* Sub-metrics grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.4rem', fontSize: '0.72rem' }}>
              <div style={{ textAlign: 'center', padding: '0.3rem', background: 'rgba(255,255,255,0.04)', borderRadius: '6px' }}>
                <div style={{ color: 'var(--color-text-secondary)', marginBottom: '2px' }}>Pixel IoU</div>
                <div style={{ fontWeight: 700 }}>{(result.pixel_iou * 100).toFixed(1)}%</div>
              </div>
              <div style={{ textAlign: 'center', padding: '0.3rem', background: 'rgba(255,255,255,0.04)', borderRadius: '6px' }}>
                <div style={{ color: 'var(--color-text-secondary)', marginBottom: '2px' }}>Hu Moments</div>
                <div style={{ fontWeight: 700 }}>{(result.hu_similarity * 100).toFixed(1)}%</div>
              </div>
              <div style={{ textAlign: 'center', padding: '0.3rem', background: 'rgba(255,255,255,0.04)', borderRadius: '6px' }}>
                <div style={{ color: 'var(--color-text-secondary)', marginBottom: '2px' }}>Pixel NCC</div>
                <div style={{ fontWeight: 700 }}>{(result.pixel_corr * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Stage 2 */}
          <div style={{ gridColumn: '1 / -1' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '0.4rem' }}>
              {result.stage_rejected ? (
                <ShieldAlert size={13} style={{ color: '#888' }} />
              ) : result.micro_score >= 0.70 ? (
                <ShieldCheck size={13} style={{ color: '#7a9e7e' }} />
              ) : (
                <ShieldAlert size={13} style={{ color: '#b85c5c' }} />
              )}
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-secondary)' }}>
                Stage 2 — Micro-Stroke Siamese ResNet
              </span>
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  color: result.stage_rejected
                    ? '#666'
                    : result.micro_score >= 0.70
                    ? '#7a9e7e'
                    : '#b85c5c',
                }}
              >
                {result.stage_rejected ? '— Skipped (fail-fast)' : `${microPercent}%`}
              </span>
            </div>

            {/* Stage 2 sub-bar */}
            <div className="progress-bar" style={{ height: '5px', marginBottom: '0.5rem' }}>
              <div
                className="progress-fill"
                style={{
                  width: result.stage_rejected
                    ? '0%'
                    : `${Math.min(100, Math.max(0, ((result.micro_score + 1) / 2) * 100))}%`,
                  background: result.stage_rejected
                    ? 'rgba(255,255,255,0.1)'
                    : result.micro_score >= 0.70
                    ? 'linear-gradient(90deg, #7a9e7e, #a3c4a7)'
                    : 'linear-gradient(90deg, #b85c5c, #d48a8a)',
                  opacity: result.stage_rejected ? 0.3 : 1,
                }}
              />
            </div>

            {result.stage_rejected && (
              <div
                style={{
                  fontSize: '0.72rem',
                  color: '#888',
                  fontStyle: 'italic',
                  padding: '0.25rem 0.4rem',
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: '6px',
                }}
              >
                Stage 1 fail-fast: Siamese inference was not executed (saves ~80ms per request).
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Forensic Analysis Text */}
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
