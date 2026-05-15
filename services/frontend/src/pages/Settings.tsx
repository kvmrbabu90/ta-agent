import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, Copy, ExternalLink, KeyRound, XCircle } from 'lucide-react';
import { useState } from 'react';
import {
  exchangeKiteRequestToken,
  fetchKiteLoginUrl,
  fetchKiteStatus,
} from '@/api/admin';
import { ErrorMessage } from '@/components/ErrorMessage';
import { LoadingSpinner } from '@/components/LoadingSpinner';

export function SettingsPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-gray-100">Settings</h1>
        <p className="text-xs text-gray-500">
          Local-user convenience controls. The rest of the API is read-only.
        </p>
      </header>

      <KiteAuthSection />
    </div>
  );
}

function KiteAuthSection() {
  const queryClient = useQueryClient();
  const statusQ = useQuery({
    queryKey: ['kite-status'],
    queryFn: fetchKiteStatus,
    staleTime: 5_000,
  });
  const loginUrlQ = useQuery({
    queryKey: ['kite-login-url'],
    queryFn: fetchKiteLoginUrl,
    enabled: false, // user clicks to load
    retry: false,
  });

  const [requestToken, setRequestToken] = useState('');
  const exchangeM = useMutation({
    mutationFn: () => exchangeKiteRequestToken(requestToken.trim()),
    onSuccess: () => {
      setRequestToken('');
      queryClient.invalidateQueries({ queryKey: ['kite-status'] });
    },
  });

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      <header className="flex items-start gap-3">
        <KeyRound className="mt-0.5 h-5 w-5 text-sky-400 flex-shrink-0" />
        <div>
          <h2 className="text-base font-semibold text-gray-100">
            Kite (Zerodha) authentication
          </h2>
          <p className="text-xs text-gray-400 mt-0.5 max-w-2xl">
            Tokens expire daily around 6 AM IST. Click <strong className="text-gray-200">Get login URL</strong>,
            log in in a new tab, and paste the <code className="rounded bg-gray-800 px-1 font-mono text-[11px] text-gray-200">request_token</code> from the redirect URL below. The
            new token is persisted to <code className="rounded bg-gray-800 px-1 font-mono text-[11px] text-gray-200">data/processed/kite_session.json</code> and is picked up automatically by the next ingest / predict run — no API restart needed.
          </p>
        </div>
      </header>

      <KiteStatus statusQ={statusQ} />

      <div className="space-y-3">
        <Step n={1} label="Get login URL">
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded-md bg-sky-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-400 disabled:opacity-50"
              onClick={() => loginUrlQ.refetch()}
              disabled={loginUrlQ.isFetching}
            >
              {loginUrlQ.isFetching ? 'Loading…' : 'Get login URL'}
            </button>
            {loginUrlQ.error ? (
              <ErrorMessage error={loginUrlQ.error} />
            ) : null}
            {loginUrlQ.data ? (
              <div className="flex items-center gap-2">
                <a
                  className="inline-flex items-center gap-1 text-sky-400 hover:text-sky-300 hover:underline text-sm"
                  href={loginUrlQ.data.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open Zerodha login
                  <ExternalLink className="h-3 w-3" />
                </a>
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded-md border border-gray-700 bg-gray-800/60 px-2 py-1 text-xs text-gray-300 hover:bg-gray-800"
                  onClick={() => {
                    navigator.clipboard.writeText(loginUrlQ.data!.url).catch(() => {
                      /* clipboard blocked; ignore */
                    });
                  }}
                >
                  <Copy className="h-3 w-3" />
                  Copy
                </button>
              </div>
            ) : null}
          </div>
        </Step>

        <Step
          n={2}
          label="Paste the request_token from the redirect URL"
        >
          <p className="text-xs text-gray-500 mb-2">
            After successful login, the redirect URL contains{' '}
            <code className="rounded bg-gray-800 px-1 font-mono text-[11px] text-gray-200">
              ?status=success&request_token=...
            </code>
            . Copy just that value below.
          </p>
          <form
            className="flex gap-2 flex-wrap"
            onSubmit={(e) => {
              e.preventDefault();
              if (!requestToken.trim()) return;
              exchangeM.mutate();
            }}
          >
            <input
              type="text"
              value={requestToken}
              onChange={(e) => setRequestToken(e.target.value)}
              placeholder="request_token"
              className="font-mono text-sm rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5 min-w-[24ch] flex-1 text-gray-100 placeholder:text-gray-600 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500/40"
              autoComplete="off"
            />
            <button
              type="submit"
              className="rounded-md bg-sky-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-400 disabled:opacity-50"
              disabled={exchangeM.isPending || !requestToken.trim()}
            >
              {exchangeM.isPending ? 'Exchanging…' : 'Authenticate'}
            </button>
          </form>
          {exchangeM.error ? (
            <div className="mt-2">
              <ErrorMessage error={exchangeM.error} />
            </div>
          ) : null}
          {exchangeM.data ? (
            <div className="mt-2 flex items-center gap-2 text-sm text-emerald-400">
              <CheckCircle2 className="h-4 w-4" />
              Authenticated as <span className="font-mono">{exchangeM.data.user_id}</span>
              {exchangeM.data.user_name ? ` (${exchangeM.data.user_name})` : null}.
              Token persisted at {new Date(exchangeM.data.exchanged_at).toLocaleString()}.
            </div>
          ) : null}
        </Step>
      </div>
    </section>
  );
}

function KiteStatus({
  statusQ,
}: {
  statusQ: ReturnType<typeof useQuery<Awaited<ReturnType<typeof fetchKiteStatus>>>>;
}) {
  if (statusQ.isLoading) return <LoadingSpinner label="Loading Kite status…" />;
  if (statusQ.isError) {
    return <ErrorMessage error={statusQ.error} onRetry={() => statusQ.refetch()} />;
  }
  const s = statusQ.data!;
  const ok = s.has_token_env || s.has_token_file;
  return (
    <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3 text-sm">
      <div className="flex items-center gap-2">
        {ok ? (
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
        ) : (
          <XCircle className="h-4 w-4 text-rose-400" />
        )}
        <span className="font-medium text-gray-200">
          {ok ? 'Authenticated' : 'No active Kite token'}
        </span>
      </div>
      <dl className="mt-2 grid grid-cols-1 gap-1 sm:grid-cols-2 text-xs text-gray-400">
        <Row label="API key configured" value={s.configured_api_key ? 'yes' : 'no'} />
        <Row label="Token from env" value={s.has_token_env ? 'yes' : 'no'} />
        <Row label="Token from file" value={s.has_token_file ? 'yes' : 'no'} />
        {s.user_id ? <Row label="User" value={`${s.user_id}${s.user_name ? ` (${s.user_name})` : ''}`} /> : null}
        {s.exchanged_at ? (
          <Row label="Last exchange" value={new Date(s.exchanged_at).toLocaleString()} />
        ) : null}
      </dl>
      {s.file_error ? (
        <div className="mt-2 text-xs text-rose-400">file error: {s.file_error}</div>
      ) : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono text-gray-200">{value}</span>
    </div>
  );
}

function Step({
  n,
  label,
  children,
}: {
  n: number;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-gray-800 bg-gray-950/30 p-3">
      <div className="text-xs uppercase tracking-wide text-gray-500">Step {n}</div>
      <div className="text-sm font-medium text-gray-200 mt-0.5">{label}</div>
      <div className="mt-2">{children}</div>
    </div>
  );
}
