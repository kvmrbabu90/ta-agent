import type { CalibrationBucket } from '@/api/types';
import { formatProba } from '@/utils/format';

interface CalibrationTableProps {
  buckets: CalibrationBucket[];
}

export function CalibrationTable({ buckets }: CalibrationTableProps) {
  if (!buckets.length) {
    return (
      <div className="text-sm text-gray-500">
        No calibration data — needs settled classifier predictions.
      </div>
    );
  }
  return (
    <div className="overflow-hidden rounded border border-gray-200 bg-white">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead className="bg-gray-50">
          <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
            <th className="px-3 py-2">Predicted top-quintile proba</th>
            <th className="px-3 py-2 text-right">Mean proba</th>
            <th className="px-3 py-2 text-right">Realized top-quintile rate</th>
            <th className="px-3 py-2 text-right">N</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {buckets.map((b) => (
            <tr key={b.proba_bucket}>
              <td className="px-3 py-2 font-mono text-xs text-gray-700">{b.proba_bucket}</td>
              <td className="px-3 py-2 text-right font-mono text-gray-700">
                {formatProba(b.mean_proba)}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-700">
                {formatProba(b.actual_top_quintile_rate)}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-500">
                {b.predicted_count}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
