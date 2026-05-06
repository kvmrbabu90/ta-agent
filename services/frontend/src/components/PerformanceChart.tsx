import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ICPoint } from '@/api/types';
import { CHART_BLUE } from '@/utils/colors';

interface PerformanceChartProps {
  series: ICPoint[];
}

export function PerformanceChart({ series }: PerformanceChartProps) {
  if (!series.length) {
    return <div className="text-sm text-gray-500">Not enough settled predictions yet.</div>;
  }
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <LineChart data={series} margin={{ left: 8, right: 16, top: 16, bottom: 16 }}>
          <CartesianGrid stroke="#f3f4f6" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: '#6b7280' }}
            minTickGap={32}
          />
          <YAxis
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickFormatter={(v: number) => v.toFixed(2)}
            domain={['auto', 'auto']}
            width={48}
          />
          <ReferenceLine y={0} stroke="#d1d5db" strokeDasharray="3 3" />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(v: number | string, key: string) =>
              [typeof v === 'number' ? v.toFixed(4) : v, key]
            }
          />
          <Line
            type="monotone"
            dataKey="daily_ic"
            name="Daily IC"
            stroke={CHART_BLUE}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
