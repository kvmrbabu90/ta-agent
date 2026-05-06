import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  label?: string;
}

export function LoadingSpinner({ label = 'Loading…' }: LoadingSpinnerProps) {
  return (
    <div className="flex items-center gap-2 text-gray-500 text-sm py-6">
      <Loader2 className="h-4 w-4 animate-spin" />
      <span>{label}</span>
    </div>
  );
}
