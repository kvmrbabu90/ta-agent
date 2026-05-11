import { ResponsiveContainer, AreaChart, Area, YAxis } from 'recharts';

interface SparklineProps {
  data: { value: number }[];
  positive?: boolean;
  height?: number;
}

/** Tiny inline price chart, ~120px wide. No axes/labels — pure shape. */
export function Sparkline({ data, positive = true, height = 40 }: SparklineProps) {
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-[10px] text-gray-600"
        style={{ height }}
      >
        no data
      </div>
    );
  }
  const stroke = positive ? '#34d399' : '#f87171';
  const fill = positive ? 'url(#spark-grad-pos)' : 'url(#spark-grad-neg)';
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <defs>
          <linearGradient id="spark-grad-pos" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#34d399" stopOpacity={0.5} />
            <stop offset="100%" stopColor="#34d399" stopOpacity={0.0} />
          </linearGradient>
          <linearGradient id="spark-grad-neg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f87171" stopOpacity={0.5} />
            <stop offset="100%" stopColor="#f87171" stopOpacity={0.0} />
          </linearGradient>
        </defs>
        <YAxis hide domain={['dataMin', 'dataMax']} />
        <Area
          type="monotone"
          dataKey="value"
          stroke={stroke}
          strokeWidth={1.6}
          fill={fill}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
