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
  calibration: CalibrationBucket[];
  ic_timeseries: ICPoint[];
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
