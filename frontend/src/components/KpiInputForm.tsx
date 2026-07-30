/*
  components/KpiInputForm.tsx

  A form that lets the user enter all 18 KPI values before running a prediction.
  Groups inputs into three logical sections:
    - Downlink (base station → phone)
    - Uplink (phone → base station)
    - Resources & Signal

  Also provides two preset buttons:
    "Load Healthy Sample"  — fills typical normal-state values
    "Load Degraded Sample" — fills values that trigger a Degraded prediction

  Props:
    values     — current KpiValues state (controlled form)
    onChange   — called when any field changes
    onPreset   — called with a preset name to fill sample values
*/

"use client";

import type { KpiValues } from "@/types";

interface Props {
  values: KpiValues;
  onChange: (field: keyof KpiValues, value: number) => void;
  onPreset: (preset: "healthy" | "degraded") => void;
}

// Defines every input field: key, display label, unit, and step size
const FIELD_GROUPS = [
  {
    title: "Downlink KPIs (Base Station → Phone)",
    fields: [
      { key: "dl_mcs", label: "DL MCS", unit: "", step: 0.1 },
      { key: "dl_n_samples", label: "DL Samples", unit: "", step: 1 },
      { key: "dl_buffer_bytes", label: "DL Buffer", unit: "bytes", step: 1 },
      {
        key: "tx_brate_downlink_mbps",
        label: "DL Throughput",
        unit: "Mbps",
        step: 0.001,
      },
      { key: "tx_pkts_downlink", label: "DL Packets", unit: "", step: 1 },
      { key: "dl_cqi", label: "DL CQI", unit: "", step: 0.1 },
    ],
  },
  {
    title: "Uplink KPIs (Phone → Base Station)",
    fields: [
      { key: "ul_mcs", label: "UL MCS", unit: "", step: 0.1 },
      { key: "ul_n_samples", label: "UL Samples", unit: "", step: 1 },
      { key: "ul_buffer_bytes", label: "UL Buffer", unit: "bytes", step: 1 },
      {
        key: "rx_brate_uplink_mbps",
        label: "UL Throughput",
        unit: "Mbps",
        step: 0.001,
      },
      { key: "rx_pkts_uplink", label: "UL Packets", unit: "", step: 1 },
      {
        key: "rx_errors_uplink_pct",
        label: "UL Error Rate",
        unit: "%",
        step: 0.1,
      },
    ],
  },
  {
    title: "Signal Quality & Resources",
    fields: [
      { key: "ul_sinr", label: "UL SINR", unit: "dB", step: 0.1 },
      { key: "phr", label: "Power Headroom (PHR)", unit: "", step: 1 },
      { key: "sum_requested_prbs", label: "PRBs Requested", unit: "", step: 1 },
      { key: "sum_granted_prbs", label: "PRBs Granted", unit: "", step: 1 },
      {
        key: "ul_turbo_iters",
        label: "Turbo Decoder Iters",
        unit: "",
        step: 0.1,
      },
      {
        key: "prb_grant_ratio",
        label: "PRB Grant Ratio",
        unit: "",
        step: 0.001,
      },
    ],
  },
] as const;

export default function KpiInputForm({ values, onChange, onPreset }: Props) {
  return (
    <div className="flex flex-col gap-6">
      {/* Preset buttons — quick-fill for demo/testing */}
      <div className="flex gap-3 flex-wrap">
        <button
          onClick={() => onPreset("healthy")}
          className="
            px-4 py-2 text-xs font-semibold font-mono rounded-lg
            bg-green-950/50 border border-green-500/30 text-green-400
            hover:bg-green-900/50 hover:border-green-500/60
            transition-colors duration-150
          "
        >
          Load Healthy Sample
        </button>
        <button
          onClick={() => onPreset("degraded")}
          className="
            px-4 py-2 text-xs font-semibold font-mono rounded-lg
            bg-red-950/50 border border-red-500/30 text-red-400
            hover:bg-red-900/50 hover:border-red-500/60
            transition-colors duration-150
          "
        >
          Load Degraded Sample
        </button>
      </div>

      {/* Input groups */}
      {FIELD_GROUPS.map((group) => (
        <div key={group.title}>
          {/* Section header */}
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
            {group.title}
          </p>

          {/* Grid of inputs */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {group.fields.map((field) => (
              <div key={field.key} className="flex flex-col gap-1">
                {/* Label with unit */}
                <label
                  htmlFor={field.key}
                  className="text-[10px] text-slate-400 font-medium truncate"
                >
                  {field.label}
                  {field.unit && (
                    <span className="text-slate-600 ml-1">({field.unit})</span>
                  )}
                </label>

                {/* Number input */}
                <input
                  id={field.key}
                  type="number"
                  step={field.step}
                  value={values[field.key as keyof KpiValues]}
                  onChange={(e) =>
                    onChange(
                      field.key as keyof KpiValues,
                      parseFloat(e.target.value) || 0,
                    )
                  }
                  className="
                    w-full bg-[#0f172a] border border-slate-700 rounded-lg
                    px-3 py-2 text-sm font-mono text-sky-300
                    focus:outline-none focus:border-sky-500/60 focus:ring-1 focus:ring-sky-500/30
                    transition-colors duration-150
                  "
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
