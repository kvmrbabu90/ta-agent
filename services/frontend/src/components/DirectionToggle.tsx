import type { Direction } from '@/api/types';

interface DirectionToggleProps {
  value: Direction;
  onChange: (d: Direction) => void;
}

const buttonClass = (active: boolean) =>
  [
    'px-3 py-1 text-sm font-medium',
    active ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 hover:bg-gray-50',
  ].join(' ');

export function DirectionToggle({ value, onChange }: DirectionToggleProps) {
  return (
    <div className="inline-flex overflow-hidden rounded border border-gray-300">
      <button
        type="button"
        className={buttonClass(value === 'long')}
        onClick={() => onChange('long')}
      >
        Long
      </button>
      <button
        type="button"
        className={buttonClass(value === 'short')}
        onClick={() => onChange('short')}
      >
        Short
      </button>
    </div>
  );
}
