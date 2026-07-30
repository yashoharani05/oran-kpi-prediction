/*
  components/ModelComparison.tsx

  Displays a side-by-side comparison table of Random Forest and XGBoost
  metrics fetched from GET /api/comparison.

  Also renders a small Recharts BarChart so the visual difference
  between the two models is instantly obvious.

  Props:
    models — array of ModelMetrics objects (from the comparison endpoint)
*/

"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ModelMetrics } from "@/lib/api";

interface Props {
  models: ModelMetrics[];
}

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score"] as const;
const METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1 Score"];

// Colours: RF = sky, XGBoost = violet
const MODEL_COLORS: Record<string, string> = {
  "Random Forest": "#38bdf8",
  XGBoost: "#a78bfa",
};

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e293b] border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono shadow-xl">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map((p: any) => (
        <p
          key={p.name}
          style={{ color: p.fill }}
          className="flex justify-between gap-4"
        >
          <span>{p.name}</span>
          <span className="font-bold">{(p.value * 100).toFixed(2)}%</span>
        </p>
      ))}
    </div>
  );
}

export default function ModelComparison({ models }: Props) {
  if (!models.length) return null;

  // Build chart data — one entry per metric
  const chartData = METRIC_KEYS.map((key, i) => {
    const entry: Record<string, any> = { metric: METRIC_LABELS[i] };
    models.forEach((m) => {
      entry[m.model] = m[key];
    });
    return entry;
  });

  return (
    <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-6 flex flex-col gap-6">
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          Model Comparison
        </p>
        <p className="text-[10px] text-slate-600 mt-0.5">
          Both models trained on the same 80/20 split — test set: 418 rows
        </p>
      </div>

      {/* Metric table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left text-slate-500 py-2 font-semibold uppercase tracking-widest">
                Metric
              </th>
              {models.map((m) => (
                <th
                  key={m.model}
                  className="text-right py-2 font-semibold"
                  style={{ color: MODEL_COLORS[m.model] ?? "#94a3b8" }}
                >
                  {m.model}
                </th>
              ))}
              <th className="text-right py-2 text-slate-500 font-semibold uppercase tracking-widest">
                Winner
              </th>
            </tr>
          </thead>
          <tbody>
            {METRIC_KEYS.map((key, i) => {
              const vals = models.map((m) => m[key]);
              const maxVal = Math.max(...vals);
              return (
                <tr key={key} className="border-b border-slate-800/60">
                  <td className="py-2.5 text-slate-400">{METRIC_LABELS[i]}</td>
                  {models.map((m) => (
                    <td
                      key={m.model}
                      className={`text-right py-2.5 ${
                        m[key] === maxVal
                          ? "text-white font-bold"
                          : "text-slate-500"
                      }`}
                    >
                      {(m[key] * 100).toFixed(2)}%
                    </td>
                  ))}
                  <td className="text-right py-2.5">
                    {/* Find winner name */}
                    {(() => {
                      const winner = models.find((m) => m[key] === maxVal);
                      const isTie = vals.every((v) => v === vals[0]);
                      return isTie ? (
                        <span className="text-slate-500">—</span>
                      ) : (
                        <span
                          className="px-1.5 py-0.5 rounded text-[10px] font-bold"
                          style={{
                            backgroundColor: `${MODEL_COLORS[winner?.model ?? ""] ?? "#94a3b8"}20`,
                            color:
                              MODEL_COLORS[winner?.model ?? ""] ?? "#94a3b8",
                          }}
                        >
                          {winner?.model === "Random Forest" ? "RF" : "XGB"}
                        </span>
                      );
                    })()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Confusion matrix summary */}
      <div>
        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
          Missed Degradation Events (False Negatives — lower is better)
        </p>
        <div className="grid grid-cols-2 gap-3">
          {models.map((m) => (
            <div
              key={m.model}
              className="rounded-lg border border-slate-700/50 bg-[#263348] px-4 py-3 flex flex-col gap-1"
            >
              <p
                className="text-[10px] font-semibold uppercase tracking-widest"
                style={{ color: MODEL_COLORS[m.model] ?? "#94a3b8" }}
              >
                {m.model}
              </p>
              <p className="font-mono text-2xl font-bold text-white">
                {m.fn}
                <span className="text-xs text-slate-500 font-normal ml-1">
                  missed
                </span>
              </p>
              <p className="text-[10px] text-slate-500">
                {m.tp} caught · {m.fp} false alarms
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Bar chart */}
      <div>
        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
          Visual Comparison
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={chartData} barCategoryGap="30%">
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#334155"
              vertical={false}
            />
            <XAxis
              dataKey="metric"
              tick={{
                fill: "#64748b",
                fontSize: 9,
                fontFamily: "JetBrains Mono",
              }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0.92, 1.0]}
              tick={{
                fill: "#64748b",
                fontSize: 9,
                fontFamily: "JetBrains Mono",
              }}
              axisLine={false}
              tickLine={false}
              width={32}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(51,65,85,0.3)" }}
            />
            <Legend
              wrapperStyle={{
                fontSize: "10px",
                fontFamily: "JetBrains Mono",
                color: "#64748b",
              }}
            />
            {models.map((m) => (
              <Bar
                key={m.model}
                dataKey={m.model}
                fill={MODEL_COLORS[m.model] ?? "#94a3b8"}
                radius={[3, 3, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
// Note: ModelDifferences is exported separately for use in the dashboard
export function ModelDifferences() {
  const rows = [
    {
      model: "Random Forest",
      color: "#38bdf8",
      badge: "bg-sky-500/20 text-sky-300 border-sky-500/30",
      type: "Ensemble (parallel trees)",
      input: "One row at a time",
      strength: "Fast, explainable, great baseline",
      limitation: "Ignores time-order between readings",
    },
    {
      model: "XGBoost",
      color: "#a78bfa",
      badge: "bg-violet-500/20 text-violet-300 border-violet-500/30",
      type: "Ensemble (sequential boosting)",
      input: "One row at a time",
      strength: "Corrects previous errors — highest accuracy here",
      limitation: "Ignores time-order between readings",
    },
    {
      model: "LSTM",
      color: "#34d399",
      badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
      type: "Recurrent neural network",
      input: "Window of last 20 readings",
      strength: "Learns temporal trends — suited for streaming data",
      limitation: "Needs much more data & longer training to shine",
    },
  ];

  return (
    <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-6 flex flex-col gap-5">
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
          How the Three Models Differ
        </p>
        <p className="text-[10px] text-slate-600 mt-0.5">
          Why LSTM scores lower on 8.7 minutes of data — and why that is
          expected.
        </p>
      </div>

      <div className="flex flex-col gap-3">
        {rows.map((r) => (
          <div
            key={r.model}
            className="bg-[#263348] rounded-lg p-4 flex flex-col gap-2"
          >
            <div className="flex items-center gap-2">
              <span
                className={`px-2 py-0.5 rounded text-[10px] font-bold border font-mono ${r.badge}`}
              >
                {r.model}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">
                {r.type}
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[10px]">
              <div>
                <p className="text-slate-600 mb-0.5 uppercase tracking-wider font-semibold">
                  Input
                </p>
                <p className="text-slate-300">{r.input}</p>
              </div>
              <div>
                <p className="text-slate-600 mb-0.5 uppercase tracking-wider font-semibold">
                  Strength
                </p>
                <p className="text-slate-300">{r.strength}</p>
              </div>
              <div>
                <p className="text-slate-600 mb-0.5 uppercase tracking-wider font-semibold">
                  Limitation here
                </p>
                <p className="text-slate-300">{r.limitation}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="border border-amber-500/20 bg-amber-950/20 rounded-lg px-4 py-3">
        <p className="text-[10px] font-semibold text-amber-400 mb-1">
          Why LSTM scores lower on this dataset
        </p>
        <p className="text-[10px] text-slate-400 leading-relaxed">
          The dataset spans only 8.7 minutes (2,090 rows). LSTM needs long
          sequences to learn meaningful temporal patterns — ideally hours or
          days of data. With 20-step windows on a short recording, it does not
          have enough context to outperform the tree models. In a real O-RAN
          deployment with continuous data, LSTM would likely improve
          significantly as the sequence history grows.
        </p>
      </div>
    </div>
  );
}
