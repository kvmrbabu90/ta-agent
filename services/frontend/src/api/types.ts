// TypeScript mirror of the Pydantic v2 schemas in services/api/schemas.py.
// Dates come over the wire as ISO `YYYY-MM-DD` strings — kept as `string`
// here and converted at the call site if needed.

export interface UniverseInfo {
  name: string;
  n_members: number;
}

export interface MemberInfo {
  symbol: string;
  company_name: string | null;
  exchange: string | null;
}

export interface TopPick {
  rank: number;
  symbol: string;
  company_name: string | null;
  predicted_return_5d: number;
  predicted_quintile: number | null;
  top_quintile_proba: number | null;
  bottom_quintile_proba: number | null;
  model_version_regression: string | null;
  model_version_classification: string | null;
}

export type Direction = 'long' | 'short';

export interface TopPicksResponse {
  as_of: string;
  universe: string;
  direction: Direction;
  picks: TopPick[];
}

export interface HistoryPoint {
  as_of: string;
  predicted_return_5d: number;
  realized_return_5d: number | null;
  predicted_quintile: number | null;
  realized_quintile: number | null;
}

export interface StockHistoryResponse {
  universe: string;
  symbol: string;
  history: HistoryPoint[];
}

export interface OHLCVPoint {
  bar_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OHLCVResponse {
  symbol: string;
  bars: OHLCVPoint[];
}

export interface CalibrationBucket {
  proba_bucket: string;
  predicted_count: number;
  actual_top_quintile_rate: number | null;
  mean_proba: number | null;
}

export interface ICPoint {
  date: string;
  daily_ic: number;
  n_stocks: number;
}

export interface StrategyEquityPoint {
  bar_date: string;
  strategy_return: number;
  spy_return: number | null;
  cum_strategy_return: number;
  cum_spy_return: number | null;
}

export interface PerformanceResponse {
  universe: string;
  lookback_days: number;
  n_predictions: number;
  n_settled: number;
  mean_daily_ic: number | null;
  std_daily_ic: number | null;
  ic_t_stat: number | null;
  mean_daily_rank_ic: number | null;
  hit_rate: number | null;
  decile_spread_5d: number | null;
  directional_accuracy: number | null;
  n_directional_observations: number | null;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  spy_sharpe_ratio: number | null;
  spy_sortino_ratio: number | null;
  equity_curve: StrategyEquityPoint[];
  calibration: CalibrationBucket[];
  ic_timeseries: ICPoint[];
}

// --- /performance/model/{universe} ------------------------------------------

export interface ModelTargetInfo {
  target: string; // 'regression' | 'classification'
  model_id: string;
  train_start: string; // YYYY-MM-DD
  train_end: string;
  n_features: number;
  horizon_days: number;
  learning_rate: number | null;
  num_leaves: number | null;
  min_data_in_leaf: number | null;
  cv_mean_metrics: Record<string, number>;
  cv_std_metrics: Record<string, number>;
  cv_fold_count: number;
}

export interface ModelInfoResponse {
  universe: string;
  n_members: number;
  training_rows: number | null;
  training_symbols: number | null;
  training_date_range: [string, string] | null;
  targets: ModelTargetInfo[];
}

// --- /performance/walkforward/{universe} -------------------------------------

export interface WalkforwardEquityPoint {
  year: number;
  strategy_return_pct: number;
  strategy_aftertax_pct: number;
  strategy_equity: number;
  benchmark_return_pct: number;
  benchmark_equity_pretax: number;
  benchmark_equity_aftertax: number;
}

export interface WalkforwardSummary {
  starting_capital: number;
  strategy_final_pretax: number;
  strategy_final_aftertax: number;
  benchmark_final_pretax: number;
  benchmark_final_aftertax: number;
  outperformance_multiple: number;
  strategy_stcg_rate: number;
  benchmark_ltcg_rate: number;
}

export interface WalkforwardResponse {
  universe: string;
  benchmark_symbol: string;
  benchmark_label: string;
  currency: string; // 'USD' | 'INR'
  years: WalkforwardEquityPoint[];
  summary: WalkforwardSummary;
}

// --- /performance/strict-wf/{universe} ---------------------------------------

export interface StrictWfYearPoint {
  year: number;
  strategy_return_pct: number;
  // Populated only when the calendar year is complete in the WF data.
  // Mid-year rows return null — the UI renders an em-dash.
  strategy_return_after_tax_pct: number | null;
  benchmark_return_pct: number | null;
  excess_pct: number | null;
  sharpe: number | null;
  max_dd_pct: number | null;
  n_days: number;
}

export interface StrictWfSummary {
  starting_capital: number;
  strategy_cum_return_pct: number;
  // After-tax equivalents — populated once at least one calendar year
  // is complete in the WF. Null until then.
  strategy_cum_return_after_tax_pct: number | null;
  strategy_annualized_after_tax_pct: number | null;
  strategy_multiple_after_tax: number | null;
  benchmark_cum_return_pct: number;
  // Benchmark cum after LTCG (one-shot liquidation at the window end).
  // Null when the bench is flat/negative or the universe has no LTCG.
  benchmark_cum_return_after_tax_pct: number | null;
  strategy_annualized_pct: number;
  benchmark_annualized_pct: number;
  n_years: number;
  strategy_multiple: number;
}

export interface StrictWfMonthlyExcessCell {
  year: number;
  month: number; // 1..12
  strategy_pct: number | null;
  benchmark_pct: number | null;
  excess_pct: number | null;
}

export interface StrictWfEquityCurve {
  dates: string[];
  equity_pre_tax: number[];
  equity_post_tax: number[];
  benchmark_equity: number[];
  // Benchmark equity AFTER LTCG, sampled only at the last date. Sits
  // below benchmark_equity[last] when the benchmark gained; equal to it
  // otherwise. Rendered as a single dot on the chart.
  benchmark_post_ltcg_endpoint: number | null;
}

export interface StrictWfProgress {
  retrains_complete: number;
  retrains_total: number;
  last_retrain_date: string | null;
  last_retrain_at_utc: string | null;
  avg_retrain_minutes: number | null;
  eta_completion_utc: string | null;
  is_running: boolean;
}

export interface StrictWfResponse {
  universe: string;
  benchmark_symbol: string;
  benchmark_label: string;
  currency: string;
  progress: StrictWfProgress;
  years: StrictWfYearPoint[];
  summary: StrictWfSummary;
  equity_curve: StrictWfEquityCurve;
  monthly_excess: StrictWfMonthlyExcessCell[];
}

// --- Paper trading -----------------------------------------------------------

export interface PaperRunSummary {
  run_id: string;
  universe: string;
  starting_cash: number;
  position_size: number;
  n_long: number;
  n_short: number;
  short_threshold: number;
  started_at: string;
  first_trade_date: string | null;
  last_trade_date: string | null;
  final_equity: number | null;
  final_realized_pnl: number | null;
  notes: string | null;
  // v2 strategy fields
  holding_days: number | null;
  commission_model: string | null;
  stop_loss_enabled: boolean | null;
  support_lookback_days: number | null;
  stop_buffer_pct: number | null;
}

export interface PaperEquityPoint {
  trade_date: string;
  snapshot_kind: 'open_8am_ct' | 'close_5pm_ct';
  equity: number;
  cash: number;
  long_mv: number;
  short_mv: number;
  realized_pnl: number;
  unrealized_pnl: number;
}

export interface PaperPosition {
  symbol: string;
  side: 'long' | 'short';
  qty: number;
  entry_price: number;
  entry_date: string;
  last_price: number | null;
  unrealized_pnl: number;
}

export interface PaperTrade {
  trade_date: string;
  symbol: string;
  // v2 adds 'stop_close' for stop-loss exits; legacy short_* sides retained
  // for backward compat with older rows.
  side: 'long_open' | 'long_close' | 'stop_close' | 'short_open' | 'short_close';
  qty: number;
  fill_price: number;
  cash_delta: number;
  realized_pnl: number | null;
}

export interface PaperSnapshotResponse {
  run: PaperRunSummary;
  equity_curve: PaperEquityPoint[];
  positions: PaperPosition[];
}

export interface PaperTradesResponse {
  run_id: string;
  trades: PaperTrade[];
}

// --- System status -----------------------------------------------------------

export interface SystemStatusResponse {
  last_refresh_utc: string | null;
  latest_bar_date: string | null;
}

// --- News verdicts (LLM audit) ----------------------------------------------

// Long-side verdicts: PANIC = sentiment-driven decline (keep long),
//                     RESET = real bad news (avoid long).
// Short-side verdicts: HYPE = sentiment-driven rally (keep short),
//                      STRENGTH = real good news (avoid short).
// UNCLEAR = either side, insufficient evidence.
export type Verdict = 'PANIC' | 'RESET' | 'HYPE' | 'STRENGTH' | 'UNCLEAR';

export interface NewsVerdict {
  symbol: string;
  verdict: Verdict;
  confidence: number;
  key_factors: string[];
  evidence_sources: string[];
  n_sources: number | null;
  model_name: string | null;
  trail_5d: number | null;
  trail_20d: number | null;
  predicted_return: number | null;
}

export interface NewsVerdictsResponse {
  universe: string;
  as_of: string;
  verdicts: NewsVerdict[];
}

export interface FeatureContribution {
  rank: number;
  feature_name: string;
  feature_value: number | null;
  shap_value: number;
  contribution_direction: 'positive' | 'negative';
}

export interface ExplainResponse {
  universe: string;
  symbol: string;
  as_of: string;
  predicted_return_5d: number | null;
  top_features: FeatureContribution[];
}
