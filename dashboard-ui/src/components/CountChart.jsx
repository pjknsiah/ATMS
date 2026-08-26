/**
 * CountChart — Recharts bar chart of cumulative vehicle counts per lane.
 *
 * Props:
 *   lanes  {Array<{lane_id, cumulative_count, signal}>}
 *   winner {number|null}  lane_id of the current green lane
 */
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from 'recharts';

const GREEN = '#2a8a44';
const GRAY  = '#484f58';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#161b22',
      border: '1px solid #30363d',
      borderRadius: '6px',
      padding: '8px 12px',
      fontFamily: 'monospace',
      fontSize: '12px',
      color: '#e6edf3',
    }}>
      <div style={{ color: '#8b949e', marginBottom: '4px' }}>{label}</div>
      <div style={{ color: payload[0].value > 0 ? '#56d364' : '#8b949e' }}>
        {payload[0].value} vehicles
      </div>
    </div>
  );
};

export default function CountChart({ lanes, winner }) {
  if (!lanes || lanes.length === 0) {
    return (
      <div style={{ color: '#8b949e', fontFamily: 'monospace', fontSize: '13px', padding: '16px 0' }}>
        Waiting for first signal decision…
      </div>
    );
  }

  const data = lanes.map(l => ({
    name: `Lane ${l.lane_id}`,
    count: l.cumulative_count,
    isWinner: l.lane_id === winner,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
        <XAxis
          dataKey="name"
          tick={{ fill: '#8b949e', fontFamily: 'monospace', fontSize: 12 }}
          axisLine={{ stroke: '#30363d' }}
          tickLine={false}
        />
        <YAxis
          allowDecimals={false}
          tick={{ fill: '#8b949e', fontFamily: 'monospace', fontSize: 11 }}
          axisLine={{ stroke: '#30363d' }}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: '#ffffff08' }} />
        <Bar dataKey="count" radius={[3, 3, 0, 0]} minPointSize={4}>
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.isWinner ? GREEN : GRAY}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
