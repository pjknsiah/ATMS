/**
 * SignalHistory — scrollable list of the last 20 signal decisions.
 *
 * Props:
 *   history  {Array}  signal_granted event objects, newest first
 */

function formatTime(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export default function SignalHistory({ history }) {
  if (!history || history.length === 0) {
    return (
      <div style={{ color: '#8b949e', fontFamily: 'monospace', fontSize: '13px', padding: '12px 0' }}>
        No decisions yet — pipeline is processing its first window…
      </div>
    );
  }

  const containerStyle = {
    maxHeight: '260px',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  };

  const rowStyle = (isFirst) => ({
    display: 'grid',
    gridTemplateColumns: '80px 80px 1fr',
    gap: '10px',
    alignItems: 'center',
    padding: '8px 12px',
    borderRadius: '5px',
    background: isFirst ? '#0d2818' : 'transparent',
    border: `1px solid ${isFirst ? '#2a8a44' : '#21262d'}`,
    fontFamily: 'monospace',
    fontSize: '12px',
    transition: 'background 0.2s',
  });

  const timeStyle = { color: '#8b949e' };
  const winnerStyle = { color: '#56d364', fontWeight: 700 };
  const countsStyle = { color: '#8b949e', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' };

  return (
    <div style={containerStyle}>
      {history.map((evt, i) => {
        const counts = evt.lanes
          ? evt.lanes.map(l => `${l.cumulative_count}`).join(' · ')
          : '—';
        return (
          <div key={i} style={rowStyle(i === 0)}>
            <span style={timeStyle}>{formatTime(evt.timestamp)}</span>
            <span style={winnerStyle}>Lane {evt.winner} 🟢</span>
            <span style={countsStyle}>[{counts}]</span>
          </div>
        );
      })}
    </div>
  );
}
