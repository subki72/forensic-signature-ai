import { getVerdictConfig } from '../utils/verdict';

/**
 * VerdictBadge component displaying biometric categorical verdict.
 */
export function VerdictBadge({ verdict }) {
  const config = getVerdictConfig(verdict);

  return (
    <div className={`status-badge ${config.badgeClass}`}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {config.icon}
        <span>{config.title}</span>
      </div>
    </div>
  );
}

export default VerdictBadge;
