/**
 * App — root component for the ATMS dashboard.
 *
 * - Connects to ws://host/ws on mount
 * - Fetches /api/state for initial render
 * - On each signal_granted message: updates lane states and prepends to history
 * - Keeps the last 20 events in history for SignalHistory
 */
import { useEffect, useRef, useState } from 'react';
import LaneCard from './components/LaneCard';
import CountChart from './components/CountChart';
import SignalHistory from './components/SignalHistory';

const MAX_HISTORY = 20;

export default function App() {
  const [lanes, setLanes]     = useState([]);
  const [winner, setWinner]   = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus]   = useState('connecting'); // connecting | live | error
  const wsRef = useRef(null);

  useEffect(() => {
    // ── Initial state via REST ────────────────────────────────────────────
    fetch('/api/state')
      .then(r => r.json())
      .then(data => {
        if (data.lanes && data.lanes.length > 0) {
          setLanes(data.lanes);
          setWinner(data.winner);
        }
      })
      .catch(() => {/* pipeline may not have decided yet — that's fine */});

    // ── Live updates via WebSocket ────────────────────────────────────────
    const wsUrl = `ws://${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setStatus('live');

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.event === 'signal_granted') {
          setLanes(data.lanes);
          setWinner(data.winner);
          setHistory(prev => [data, ...prev].slice(0, MAX_HISTORY));
        }
      } catch (e) {
        console.error('WS parse error', e);
      }
    };

    ws.onerror  = () => setStatus('error');
    ws.onclose  = () => setStatus('connecting');

    return () => ws.close();
  }, []);

  // ── Styles ──────────────────────────────────────────────────────────────
  const appStyle = {
    minHeight: '100vh',
    background: '#0d1117',
    color: '#e6edf3',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
    padding: '0 0 48px',
  };

  const headerStyle = {
    borderBottom: '1px solid #21262d',
    padding: '0 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: '56px',
  };

  const logoStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontFamily: 'monospace',
    fontWeight: 700,
    fontSize: '14px',
    letterSpacing: '0.06em',
    color: '#e6edf3',
  };

  const logoMarkStyle = {
    width: '28px',
    height: '28px',
    borderRadius: '6px',
    background: '#2a8a44',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '16px',
  };

  const statusPillStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: 'monospace',
    fontSize: '11px',
    padding: '4px 10px',
    borderRadius: '20px',
    border: '1px solid',
    ...(status === 'live'
      ? { color: '#56d364', borderColor: '#2a8a44', background: '#0d2818' }
      : status === 'error'
      ? { color: '#f85149', borderColor: '#6e1a1a', background: '#1a0a0a' }
      : { color: '#8b949e', borderColor: '#30363d', background: '#161b22' }),
  };

  const statusDotStyle = {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: status === 'live' ? '#2ecc71' : status === 'error' ? '#e74c3c' : '#8b949e',
    animation: status === 'live' ? 'pulse 2s infinite' : 'none',
  };

  const mainStyle = {
    maxWidth: '960px',
    margin: '0 auto',
    padding: '32px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '32px',
  };

  const sectionStyle = {
    background: '#161b22',
    border: '1px solid #21262d',
    borderRadius: '8px',
    padding: '20px 24px',
  };

  const sectionHeadStyle = {
    fontFamily: 'monospace',
    fontSize: '10.5px',
    fontWeight: 700,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    color: '#8b949e',
    marginBottom: '16px',
  };

  const laneGridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
    gap: '12px',
  };

  return (
    <div style={appStyle}>
      {/* ── CSS for pulsing dot ── */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
      `}</style>

      {/* ── Header ── */}
      <header style={headerStyle}>
        <div style={logoStyle}>
          <div style={logoMarkStyle}>🚦</div>
          ATMS Dashboard
        </div>
        <div style={statusPillStyle}>
          <div style={statusDotStyle} />
          {status === 'live' ? 'Live' : status === 'error' ? 'Error' : 'Connecting'}
        </div>
      </header>

      {/* ── Main content ── */}
      <main style={mainStyle}>

        {/* Lane cards */}
        <section style={sectionStyle}>
          <div style={sectionHeadStyle}>Signal Status</div>
          <div style={laneGridStyle}>
            {lanes.length === 0
              ? <span style={{ color: '#8b949e', fontFamily: 'monospace', fontSize: '13px' }}>
                  Waiting for pipeline…
                </span>
              : lanes.map(lane => (
                  <LaneCard key={lane.lane_id} {...lane} />
                ))
            }
          </div>
        </section>

        {/* Bar chart */}
        <section style={sectionStyle}>
          <div style={sectionHeadStyle}>Vehicle Counts</div>
          <CountChart lanes={lanes} winner={winner} />
        </section>

        {/* Signal history */}
        <section style={sectionStyle}>
          <div style={sectionHeadStyle}>
            Signal History
            {history.length > 0 && (
              <span style={{ marginLeft: '8px', color: '#484f58', fontWeight: 400 }}>
                last {history.length}
              </span>
            )}
          </div>
          <SignalHistory history={history} />
        </section>

      </main>
    </div>
  );
}
