import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { OHLCVPoint } from '@/api/types';
import { CHART_BLUE, CHART_GRAY } from '@/utils/colors';

interface PriceChartProps {
  bars: OHLCVPoint[];
}

function rollingMean(values: number[], window: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    if (i + 1 >= window) out.push(sum / window);
    else out.push(null);
  }
  return out;
}

export function PriceChart({ bars }: PriceChartProps) {
  const data = useMemo(() => {
    if (!bars.length) return [] as Array<{ date: string; close: number; sma20: number | null; sma50: number | null }>;
    const closes = bars.map((b) => b.close);
    const sma20 = rollingMean(closes, 20);
    const sma50 = rollingMean(closes, 50);
    return bars.map((b, i) => ({
      date: b.bar_date,
      close: b.close,
      sma20: sma20[i],
      sma50: sma50[i],
    }));
  }, [bars]);

  if (!data.length) {
    return <div className="text-sm text-gray-500">No price data.</div>;
  }

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: 8, right: 16, top: 16, bottom: 16 }}>
          <CartesianGrid stroke="#f3f4f6" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: '#6b7280' }}
            minTickGap={32}
          />
          <YAxis
            tick={{ fontSize: 11, fill: '#6b7280' }}
            domain={['auto', 'auto']}
            width={64}
          />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(v: number | string) =>
              typeof v === 'number' ? v.toFixed(2) : v
            }
          />
          <Line
            type="monotone"
            dataKey="close"
            name="Close"
            stroke={CHART_BLUE}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="sma20"
            name="SMA 20"
            stroke={CHART_GRAY}
            strokeWidth={1}
            dot={false}
            strokeDasharray="3 3"
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="sma50"
            name="SMA 50"
            stroke="#4b5563"
            strokeWidth={1}
            dot={false}
            strokeDasharray="6 4"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
