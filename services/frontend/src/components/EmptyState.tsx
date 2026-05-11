import { Inbox } from 'lucide-react';
import type { ReactNode } from 'react';

interface EmptyStateProps {
  title: string;
  hint?: ReactNode;
}

export function EmptyState({ title, hint }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-gray-700 bg-gray-900/40 px-4 py-8 text-center">
      <Inbox className="h-6 w-6 text-gray-500" />
      <div className="text-sm font-medium text-gray-300">{title}</div>
      {hint ? <div className="max-w-md text-sm text-gray-500">{hint}</div> : null}
    </div>
  );
}
