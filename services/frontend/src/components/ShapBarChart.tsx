import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { FeatureContribution } from '@/api/types';
import { CHART_GREEN, CHART_RED } from '@/utils/colors';

interface ShapBarChartProps {
  contributions: FeatureContribution[];
}

export function ShapBarChart({ contributions }: ShapBarChartProps) {
  if (!contributions.length) {
    return <div className="text-sm text-gray-500">No SHAP attribution available.</div>;
  }

  // Recharts horizontal layout: features on Y, signed SHAP on X.
  const data = [...contributions]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .map((c) => ({
      feature: c.feature_name,
      shap: c.shap_value,
      featureValue: c.feature_value,
    }));

  const height = Math.max(220, data.length * 28 + 40);

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 8, right: 24, top: 8, bottom: 8 }}
        >
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: '#6b7280' }}
            tickFormatter={(v: number) => v.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 11, fill: '#374151' }}
            width={180}
          />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(v: number | string, key: string, item) => {
              if (key === 'shap' && typeof v === 'number') {
                const fv = (item?.payload as { featureValue: number | null } | undefined)?.featureValue;
                const fvStr = fv === null || fv === undefined ? '—' : fv.toFixed(4);
                return [`SHAP ${v.toFixed(4)} (feature value ${fvStr})`, key];
              }
              return [v as never, key];
            }}
          />
          <Bar dataKey="shap" isAnimationActive={false}>
            {data.map((d) => (
              <Cell
                key={d.feature}
                fill={d.shap >= 0 ? CHART_GREEN : CHART_RED}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
