/**
 * LaneCard — displays one lane's current signal status and vehicle count.
 *
 * Props:
 *   lane_id          {number}  Zero-based lane index
 *   signal           {string}  "green" | "red"
 *   cumulative_count {number}  Vehicles accumulated since last green
 *   consecutive_greens {number} How many consecutive greens this lane has had
 */
export default function LaneCard({ lane_id, signal, cumulative_count, consecutive_greens }) {
  const isGreen = signal === 'green';

  const cardStyle = {
    background: isGreen ? '#0d2818' : '#1c1c24',
    border: `2px solid ${isGreen ? '#2a8a44' : '#30363d'}`,
    borderRadius: '10px',
    padding: '24px 20px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px',
    transition: 'border-color 0.3s, background 0.3s',
    minWidth: '140px',
  };

  const signalDotStyle = {
    width: '72px',
    height: '72px',
    borderRadius: '50%',
    background: isGreen ? '#2ecc71' : '#c0392b',
    boxShadow: isGreen
      ? '0 0 24px #2ecc71aa, 0 0 6px #2ecc71'
      : '0 0 16px #c0392b66, 0 0 4px #c0392b',
    transition: 'background 0.3s, box-shadow 0.3s',
    flexShrink: 0,
  };

  const laneLabelStyle = {
    fontFamily: 'monospace',
    fontSize: '11px',
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: isGreen ? '#56d364' : '#8b949e',
    fontWeight: 700,
  };

  const countStyle = {
    fontSize: '28px',
    fontWeight: 700,
    fontFamily: 'monospace',
    color: isGreen ? '#56d364' : '#e6edf3',
    lineHeight: 1,
  };

  const countLabelStyle = {
    fontSize: '11px',
    color: '#8b949e',
    fontFamily: 'monospace',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    marginTop: '-4px',
  };

  const badgeStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    background: '#21262d',
    border: '1px solid #30363d',
    borderRadius: '12px',
    padding: '2px 8px',
    fontSize: '11px',
    color: '#8b949e',
    fontFamily: 'monospace',
  };

  return (
    <div style={cardStyle} aria-label={`Lane ${lane_id} — ${signal}`}>
      <div style={laneLabelStyle}>Lane {lane_id}</div>
      <div style={signalDotStyle} role="img" aria-label={signal} />
      <div>
        <div style={countStyle}>{cumulative_count}</div>
        <div style={countLabelStyle}>vehicles</div>
      </div>
      {consecutive_greens > 0 && (
        <div style={badgeStyle}>
          <span>🟢</span>
          <span>×{consecutive_greens}</span>
        </div>
      )}
    </div>
  );
}
