/*
  components/Header.tsx

  Top navigation bar.
  Shows project name, backend health indicator, and a live clock.
*/

"use client";

import { useEffect, useState } from "react";
import { Activity, Wifi, WifiOff } from "lucide-react";

interface Props {
  modelLoaded: boolean | null; // null = unknown / still checking
}

export default function Header({ modelLoaded }: Props) {
  const [time, setTime] = useState("");

  // Update clock every second
  useEffect(() => {
    const tick = () => {
      setTime(new Date().toLocaleTimeString("en-GB", { hour12: false }));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="border-b border-slate-800 bg-[#0f172a]/95 backdrop-blur sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-4">
        {/* Logo mark + title */}
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-sky-500/20 border border-sky-500/40 flex items-center justify-center">
            <Activity className="w-3.5 h-3.5 text-sky-400" />
          </div>
          <div className="leading-tight">
            <p className="text-sm font-bold text-white tracking-tight">
              O-RAN KPI xApp
            </p>
            <p className="text-[9px] text-slate-600 text-white font-mono uppercase tracking-widest">
              Network Degradation Prediction
            </p>
          </div>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Backend status pill */}
        <div
          className={`
            flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-mono
            ${
              modelLoaded === true
                ? "border-green-500/30 bg-green-950/40 text-green-400"
                : modelLoaded === false
                  ? "border-red-500/30 bg-red-950/40 text-red-400"
                  : "border-slate-700 bg-slate-800/40 text-slate-500"
            }
          `}
        >
          {modelLoaded === true && <Wifi className="w-3 h-3" />}
          {modelLoaded === false && <WifiOff className="w-3 h-3" />}
          {modelLoaded === null && (
            <div className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" />
          )}
          <span>
            {modelLoaded === true
              ? "Model Ready"
              : modelLoaded === false
                ? "Model Offline"
                : "Checking..."}
          </span>
        </div>

        {/* Live clock */}
        <p className="text-xs font-mono text-slate-600 text-white tabular-nums w-16 text-right">
          {time}
        </p>
      </div>
    </header>
  );
}
