import { ArrowLeft, CheckCircle2, XCircle } from 'lucide-react';
import { useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useExplain } from '@/hooks/useExplain';
import { useStockHistory } from '@/hooks/useStockHistory';
import { useStockOhlcv } from '@/hooks/useStockOhlcv';
import { ErrorMessage } from '@/components/ErrorMessage';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { PriceChart } from '@/components/PriceChart';
import { ShapBarChart } from '@/components/ShapBarChart';
import { signColor } from '@/utils/colors';
import { formatPercent, formatProba } from '@/utils/format';
import type { HistoryPoint } from '@/api/types';

export function StockDetailPage() {
  const params = useParams<{ universe: string; symbol: string }>();
  const universe = params.universe ?? '';
  const symbol = params.symbol ?? '';

  const ohlcvQ = useStockOhlcv(symbol);
  const historyQ = useStockHistory(universe, symbol, 180);
  const explainQ = useExplain(universe, symbol, undefined, 10);

  const latestPred = useMemo<HistoryPoint | null>(() => {
    const h = historyQ.data?.history ?? [];
    return h.length ? h[h.length - 1] : null;
  }, [historyQ.data]);

  return (
    <div className="space-y-4">
      <div>
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300"
        >
          <ArrowLeft className="h-3 w-3" />
          Back to dashboard
        </Link>
      </div>

      <header className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
        <div className="flex flex-wrap items-baseline gap-4">
          <h1 className="text-2xl font-semibold text-gray-100 font-mono">{symbol}</h1>
          <span className="text-sm text-gray-500">{universe}</span>
          {ohlcvQ.data?.bars.length ? (
            <span className="ml-auto text-sm text-gray-300">
              <span className="text-gray-500">Last close:</span>{' '}
              <span className="font-mono text-gray-100">
                {ohlcvQ.data.bars[ohlcvQ.data.bars.length - 1].close.toFixed(2)}
              </span>
            </span>
          ) : null}
        </div>
        {latestPred ? (
          <div className="mt-2 flex flex-wrap items-center gap-6 text-sm">
            <div>
              <span className="text-gray-500">Predicted (5d) </span>
              <span className={`font-mono ${signColor(latestPred.predicted_return_5d)}`}>
                {formatPercent(latestPred.predicted_return_5d)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Predicted quintile </span>
              <span className="font-mono text-gray-200">{latestPred.predicted_quintile ?? '—'}</span>
            </div>
            {latestPred.realized_return_5d !== null ? (
              <div>
                <span className="text-gray-500">Realized </span>
                <span className={`font-mono ${signColor(latestPred.realized_return_5d)}`}>
                  {formatPercent(latestPred.realized_return_5d)}
                </span>
              </div>
            ) : null}
            <div className="text-xs text-gray-500">As of {latestPred.as_of}</div>
          </div>
        ) : null}
      </header>

      <Section title="Price (last 12 months)" subtitle="Close + 20/50-day SMA overlays.">
        {ohlcvQ.isLoading ? (
          <LoadingSpinner label="Loading prices…" />
        ) : ohlcvQ.isError ? (
          <ErrorMessage error={ohlcvQ.error} onRetry={() => ohlcvQ.refetch()} />
        ) : (
          <PriceChart bars={ohlcvQ.data?.bars ?? []} />
        )}
      </Section>

      <Section
        title="SHAP attribution (most recent prediction)"
        subtitle="Top features contributing to the regression model's predicted 5-day return."
      >
        {explainQ.isLoading ? (
          <LoadingSpinner label="Computing SHAP…" />
        ) : explainQ.isError ? (
          <ErrorMessage
            error={explainQ.error}
            onRetry={() => explainQ.refetch()}
          />
        ) : (
          <ShapBarChart contributions={explainQ.data?.top_features ?? []} />
        )}
      </Section>

      <Section
        title="Prediction history"
        subtitle="Last 180 days. Hit = predicted direction matched realized direction."
      >
        {historyQ.isLoading ? (
          <LoadingSpinner label="Loading history…" />
        ) : historyQ.isError ? (
          <ErrorMessage error={historyQ.error} onRetry={() => historyQ.refetch()} />
        ) : (
          <HistoryTable history={historyQ.data?.history ?? []} />
        )}
      </Section>
    </div>
  );
}

function HistoryTable({ history }: { history: HistoryPoint[] }) {
  if (!history.length) {
    return <div className="text-sm text-gray-500">No predictions logged yet.</div>;
  }
  // Newest first.
  const rows = [...history].reverse();
  return (
    <div className="overflow-hidden rounded-lg border border-gray-800 bg-gray-950/40">
      <table className="min-w-full divide-y divide-gray-800 text-sm">
        <thead className="bg-gray-900/60">
          <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
            <th className="px-3 py-2">As of</th>
            <th className="px-3 py-2 text-right">Predicted</th>
            <th className="px-3 py-2 text-right">Realized</th>
            <th className="px-3 py-2 text-center">Hit</th>
            <th className="px-3 py-2 text-center">Pred. q</th>
            <th className="px-3 py-2 text-center">Real. q</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800/60">
          {rows.map((p) => {
            const realized = p.realized_return_5d;
            const hit =
              realized === null
                ? null
                : Math.sign(p.predicted_return_5d) === Math.sign(realized);
            return (
              <tr key={p.as_of} className="hover:bg-gray-900/40">
                <td className="px-3 py-2 font-mono text-xs text-gray-400">{p.as_of}</td>
                <td className={`px-3 py-2 text-right font-mono ${signColor(p.predicted_return_5d)}`}>
                  {formatPercent(p.predicted_return_5d)}
                </td>
                <td className={`px-3 py-2 text-right font-mono ${signColor(realized)}`}>
                  {realized === null ? <span className="text-gray-600">{formatProba(null)}</span> : formatPercent(realized)}
                </td>
                <td className="px-3 py-2 text-center">
                  {hit === null ? (
                    <span className="text-gray-600">—</span>
                  ) : hit ? (
                    <CheckCircle2 className="mx-auto h-4 w-4 text-emerald-400" />
                  ) : (
                    <XCircle className="mx-auto h-4 w-4 text-rose-400" />
                  )}
                </td>
                <td className="px-3 py-2 text-center font-mono text-gray-300">
                  {p.predicted_quintile ?? '—'}
                </td>
                <td className="px-3 py-2 text-center font-mono text-gray-300">
                  {p.realized_quintile ?? '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-2 rounded-lg border border-gray-800 bg-gray-900/60 p-4">
      <header>
        <h2 className="text-base font-semibold text-gray-100">{title}</h2>
        <p className="text-xs text-gray-500">{subtitle}</p>
      </header>
      {children}
    </section>
  );
}
