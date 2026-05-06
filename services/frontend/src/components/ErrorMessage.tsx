import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorMessageProps {
  error: unknown;
  onRetry?: () => void;
}

function describe(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return 'Something went wrong.';
}

export function ErrorMessage({ error, onRetry }: ErrorMessageProps) {
  return (
    <div className="flex items-start gap-3 rounded border border-red-200 bg-red-50 px-4 py-3">
      <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-red-600" />
      <div className="flex-1 text-sm">
        <div className="font-medium text-red-800">Request failed</div>
        <div className="text-red-700 break-all">{describe(error)}</div>
      </div>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="flex items-center gap-1 rounded border border-red-300 bg-white px-2 py-1 text-xs text-red-700 hover:bg-red-100"
        >
          <RefreshCw className="h-3 w-3" />
          Retry
        </button>
      ) : null}
    </div>
  );
}
