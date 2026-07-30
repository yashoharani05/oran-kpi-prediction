/*
  components/KpiChart.tsx

  A Recharts bar chart showing the 6 most predictive KPI values
  against their healthy-range reference line.

  This helps the user see AT A GLANCE which metrics are in trouble.
  The chart uses the feature-importance order from the Random Forest:
    dl_mcs, dl_cqi, rx_errors_uplink_pct, ul_turbo_iters,
    prb_grant_ratio, tx_brate_downlink_mbps

  Props:
    kpiValues — the full KpiValues object from the dashboard state
*/

"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { KpiValues } from "@/types";

interface Props {
  kpiValues: KpiValues;
}

// Normalise each KPI to a 0–100 scale so they can be shown on the same axis.
// We define the "max" for each feature based on the dataset's realistic max.
const KPI_DEFS = [
  {
    key: "dl_mcs",
    label: "DL MCS",
    max: 16.3,
    // Low is bad: invert so a full bar = good
    invert: true,
    // Healthy-range reference line (as % of max)
    refPct: 59, // dataset Q25 = 6.57 → 6.57/16.3 ≈ 40% → 100-40 = 60 inverted
  },
  {
    key: "dl_cqi",
    label: "DL CQI",
    max: 15,
    invert: true,
    refPct: 42, // Q25 = 6.33 / 15 ≈ 42%
  },
  {
    key: "rx_errors_uplink_pct",
    label: "UL Errors",
    max: 100,
    invert: false, // High is bad — bar grows with error
    refPct: 57, // Q90 threshold
  },
  {
    key: "ul_turbo_iters",
    label: "Turbo Iters",
    max: 10,
    invert: false, // High is bad
    refPct: 68, // Q75 of active rows ≈ 6.85 / 10 = 68%
  },
  {
    key: "prb_grant_ratio",
    label: "PRB Ratio",
    max: 1.0,
    invert: true, // Low is bad (low granted/requested = congestion)
    refPct: 15, // Q25 ≈ 0.15 / 1.0 = 15%
  },
  {
    key: "tx_brate_downlink_mbps",
    label: "DL Throughput",
    max: 0.21,
    invert: true, // Low is bad
    refPct: 1, // Q10 ≈ 0.003 / 0.21 ≈ 1%
  },
];

// Colour the bar: green if healthy, amber if borderline, red if critical
function barColor(normalised: number, invert: boolean, refPct: number): string {
  // For invert=true: bar shows (100 - raw_pct). Below refPct is critical.
  const effectivePct = invert ? 100 - normalised : normalised;
  if (invert) {
    // Critical if raw_pct < refPct (inverted bar is tall, meaning low raw value)
    if (normalised > 100 - refPct + 15) return "#ef4444";
    if (normalised > 100 - refPct) return "#f59e0b";
    return "#22c55e";
  } else {
    if (effectivePct > refPct + 10) return "#ef4444";
    if (effectivePct > refPct - 10) return "#f59e0b";
    return "#22c55e";
  }
}

// Custom tooltip shown on hover
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e293b] border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono">
      <p className="text-slate-400 mb-1">{label}</p>
      <p className="text-white font-bold">{payload[0].value.toFixed(1)}%</p>
      <p className="text-slate-500 text-[10px] mt-0.5">of normalised range</p>
    </div>
  );
}

export default function KpiChart({ kpiValues }: Props) {
  // Build chart data by normalising each KPI to 0–100
  const chartData = KPI_DEFS.map((def) => {
    const raw = kpiValues[def.key as keyof KpiValues] as number;
    const pct = Math.min(100, Math.max(0, (raw / def.max) * 100));
    // For inverted features (low = bad), flip so the bar grows with health
    const value = def.invert ? 100 - pct : pct;
    return {
      label: def.label,
      value: value,
      color: barColor(value, def.invert, def.refPct),
      refLine: def.invert ? 100 - def.refPct : def.refPct,
    };
  });

  return (
    <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-5">
      <div className="mb-4">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          Key KPI Health
        </p>
        <p className="text-[10px] text-slate-600 mt-0.5">
          Bars show normalised health score (higher = better for inverted
          metrics). Coloured threshold lines show degradation boundaries.
        </p>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} barCategoryGap="35%">
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#334155"
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{
              fill: "#64748b",
              fontSize: 10,
              fontFamily: "JetBrains Mono",
            }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{
              fill: "#64748b",
              fontSize: 10,
              fontFamily: "JetBrains Mono",
            }}
            axisLine={false}
            tickLine={false}
            width={28}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: "rgba(51,65,85,0.3)" }}
          />

          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.color} />
            ))}
          </Bar>

          {/* Reference lines at the degradation threshold for each bar */}
          {chartData.map((entry, index) => (
            <ReferenceLine
              key={index}
              x={entry.label}
              y={entry.refLine}
              stroke="#f59e0b"
              strokeDasharray="4 2"
              strokeWidth={1.5}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
