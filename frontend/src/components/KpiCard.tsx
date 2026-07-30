/*
  components/KpiCard.tsx

  Displays one KPI metric with its label, value, and unit.
  Optionally highlights the value in a warning colour if it signals degradation.

  Props:
    label     — human-readable name shown above the value
    value     — numeric value to display
    unit      — unit string shown after the value (e.g. "Mbps", "%")
    warn      — if true, colours the value amber (borderline)
    critical  — if true, colours the value red (degraded indicator)
    decimals  — number of decimal places to show (default 2)
*/

"use client";

interface Props {
  label: string;
  value: number;
  unit?: string;
  warn?: boolean;
  critical?: boolean;
  decimals?: number;
}

export default function KpiCard({
  label,
  value,
  unit = "",
  warn = false,
  critical = false,
  decimals = 2,
}: Props) {
  // Value colour: critical (red) → warn (amber) → normal (sky blue)
  const valueColor = critical
    ? "text-red-400"
    : warn
      ? "text-amber-400"
      : "text-sky-300";

  return (
    <div className="bg-[#1e293b] border border-slate-700/50 rounded-lg px-4 py-3 flex flex-col gap-1">
      {/* Label */}
      <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest truncate">
        {label}
      </p>

      {/* Value + unit */}
      <p className={`font-mono text-lg font-bold leading-tight ${valueColor}`}>
        {value.toFixed(decimals)}
        {unit && (
          <span className="text-xs font-normal text-slate-500 ml-1">
            {unit}
          </span>
        )}
      </p>
    </div>
  );
}
