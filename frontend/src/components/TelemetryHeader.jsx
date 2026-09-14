/**
 * TelemetryHeader component displaying system title and live API/Engine probes.
 */
export function TelemetryHeader({ apiStatus }) {
  return (
    <header className="header">
      <h1 className="text-gradient">Legal Document AI</h1>
      <p>Forensic Signature Verification</p>

      {/* Telemetry Status Bar */}
      <div className="telemetry-bar">
        <div className="telemetry-pill">
          <span
            className={`telemetry-dot ${
              apiStatus.online ? 'telemetry-online' : 'telemetry-offline'
            }`}
          />
          {apiStatus.checking
            ? 'PROBING BACKEND...'
            : apiStatus.online
            ? 'API: ONLINE'
            : 'API: UNREACHABLE'}
        </div>
        <div className="telemetry-pill">
          <span
            className={`telemetry-dot ${
              apiStatus.ready ? 'telemetry-online' : 'telemetry-degraded'
            }`}
          />
          {apiStatus.ready ? 'ENGINE: READY (V2.1)' : 'ENGINE: DEGRADED'}
        </div>
      </div>

      <div className="divider" />
    </header>
  );
}

export default TelemetryHeader;
