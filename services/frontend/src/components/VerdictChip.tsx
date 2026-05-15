import { AlertOctagon, CheckCircle2, HelpCircle } from 'lucide-react';
import type { NewsVerdict } from '@/api/types';

/** LLM news-verdict chip. AUDIT ONLY — does not change picks.
 *
 *  Long picks  (mean-reverting on losers):
 *    PANIC    — decline is sentiment-driven, no material bad news → keep
 *    RESET    — real structural bad news (earnings cut, exec dep, etc.) → avoid
 *
 *  Short picks (mean-reverting on winners):
 *    HYPE     — rally is sentiment-driven, no material good news → keep
 *    STRENGTH — real structural good news (beat+raise, M&A, etc.) → avoid
 *
 *  UNCLEAR (either side) — insufficient evidence to decide.
 */
export function VerdictChip({ verdict }: { verdict: NewsVerdict }) {
  const v = verdict.verdict;
  const conf = (verdict.confidence * 100).toFixed(0);

  const factors = (verdict.key_factors ?? []).slice(0, 3);
  const tooltip = [
    `Gemma 4 verdict: ${v} (${conf}% confidence)`,
    factors.length ? '' : '',
    ...factors.map((f) => `• ${f}`),
    '',
    `Sources read: ${verdict.n_sources ?? 0} SEC filings`,
    'Audit-only — does not change picks.',
  ].filter(Boolean).join('\n');

  // KEEP-side verdicts get green; AVOID-side get red; UNCLEAR gray.
  const config: Record<string, { Icon: typeof CheckCircle2; cls: string }> = {
    PANIC:    { Icon: CheckCircle2, cls: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' },
    HYPE:     { Icon: CheckCircle2, cls: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' },
    RESET:    { Icon: AlertOctagon, cls: 'border-rose-500/40    bg-rose-500/10    text-rose-300' },
    STRENGTH: { Icon: AlertOctagon, cls: 'border-rose-500/40    bg-rose-500/10    text-rose-300' },
    UNCLEAR:  { Icon: HelpCircle,   cls: 'border-gray-700       bg-gray-800/60    text-gray-400' },
  };
  const cfg = config[v];
  if (!cfg) return null;
  const { Icon } = cfg;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${cfg.cls}`}
      title={tooltip}
    >
      <Icon className="h-2.5 w-2.5" />
      {v}
    </span>
  );
}
