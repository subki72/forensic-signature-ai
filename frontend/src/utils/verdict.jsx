import { CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';

/**
 * Helper to obtain styling and icons for 3-tier verdicts.
 */
export function getVerdictConfig(verdict) {
  switch (verdict) {
    case 'AUTHENTIC':
      return {
        badgeClass: 'status-success',
        icon: <CheckCircle2 size={20} />,
        title: 'Authentic (Verified)',
        gradient: 'linear-gradient(90deg, #7a9e7e, #a3c4a7)',
      };
    case 'INCONCLUSIVE':
      return {
        badgeClass: 'status-inconclusive',
        icon: <AlertTriangle size={20} />,
        title: 'Inconclusive (Review Required)',
        gradient: 'linear-gradient(90deg, #e09f3e, #f4c06f)',
      };
    case 'FORGERY':
    default:
      return {
        badgeClass: 'status-fail',
        icon: <XCircle size={20} />,
        title: 'Forgery Detected',
        gradient: 'linear-gradient(90deg, #b85c5c, #d48a8a)',
      };
  }
}
