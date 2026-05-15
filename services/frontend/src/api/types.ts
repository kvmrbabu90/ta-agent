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
