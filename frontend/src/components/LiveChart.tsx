/*
  components/LiveChart.tsx

  A Recharts LineChart that shows the last N readings of two KPIs:
    - Downlink throughput (tx_brate_downlink_mbps)
    - Uplink error rate   (rx_errors_uplink_pct, on secondary axis)

  This chart is purely a display component — it receives history as a prop
  and re-renders whenever new data arrives from the stream.

  Props:
    history  — array of the last N StreamRow objects, newest last
    maxPoints — how many points to show (default 30)
*/

"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { StreamRow } from "@/lib/api";

interface Props {
  history: StreamRow[];
  maxPoints?: number;
}

// Tooltip styled to match the dark dashboard theme
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1e293b] border border-slate-700 rounded-lg px-3 py-2 text-xs font-mono shadow-xl">
      <p className="text-slate-500 mb-1.5">Row {label}</p>
      {payload.map((p: any) => (
        <p
          key={p.name}
          style={{ color: p.color }}
          className="flex justify-between gap-4"
        >
          <span>{p.name}</span>
          <span className="font-bold">{Number(p.value).toFixed(3)}</span>
        </p>
      ))}
    </div>
  );
}

export default function LiveChart({ history, maxPoints = 30 }: Props) {
  // Take only the last maxPoints entries
  const slice = history.slice(-maxPoints);

  // Build chart data — each entry is one tick on the x-axis
  const chartData = slice.map((row) => ({
    tick: row.row_index,
    throughput: Number(row.tx_brate_downlink_mbps.toFixed(4)),
    errorRate: Number(row.rx_errors_uplink_pct.toFixed(2)),
  }));

  return (
    <div className="bg-[#1e293b] border border-slate-700/50 rounded-xl p-5">
      <div className="mb-4 flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
            Live KPI History
          </p>
          <p className="text-[10px] text-slate-600 mt-0.5">
            Last {maxPoints} readings — updates every 2 s
          </p>
        </div>
        {/* Live indicator dot */}
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />
          <span className="text-[10px] text-slate-500 font-mono">LIVE</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <LineChart
          data={chartData}
          margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#334155"
            vertical={false}
          />

          <XAxis
            dataKey="tick"
            tick={{
              fill: "#475569",
              fontSize: 9,
              fontFamily: "JetBrains Mono",
            }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
            label={{
              value: "Row",
              position: "insideBottomRight",
              offset: -4,
              fill: "#475569",
              fontSize: 9,
            }}
          />

          {/* Left Y-axis — throughput in Mbps */}
          <YAxis
            yAxisId="left"
            tick={{
              fill: "#475569",
              fontSize: 9,
              fontFamily: "JetBrains Mono",
            }}
            axisLine={false}
            tickLine={false}
            width={38}
            tickFormatter={(v) => `${v.toFixed(2)}`}
            label={{
              value: "Mbps",
              angle: -90,
              position: "insideLeft",
              fill: "#38bdf8",
              fontSize: 9,
            }}
          />

          {/* Right Y-axis — error rate % */}
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={[0, 100]}
            tick={{
              fill: "#475569",
              fontSize: 9,
              fontFamily: "JetBrains Mono",
            }}
            axisLine={false}
            tickLine={false}
            width={30}
            tickFormatter={(v) => `${v}%`}
            label={{
              value: "Err%",
              angle: 90,
              position: "insideRight",
              fill: "#f87171",
              fontSize: 9,
            }}
          />

          <Tooltip content={<ChartTooltip />} />

          <Legend
            wrapperStyle={{
              fontSize: "10px",
              fontFamily: "JetBrains Mono",
              color: "#64748b",
            }}
          />

          {/* Downlink throughput — sky blue */}
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="throughput"
            name="DL Mbps"
            stroke="#38bdf8"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false} // disable animation for smooth live updates
          />

          {/* Uplink error rate — red */}
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="errorRate"
            name="UL Err%"
            stroke="#f87171"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
