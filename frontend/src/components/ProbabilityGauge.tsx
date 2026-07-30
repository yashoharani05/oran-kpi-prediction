/*
  components/ProbabilityGauge.tsx

  The signature element of this dashboard: an SVG arc gauge that fills from
  green → amber → red as the degradation probability rises.

  The arc is a partial circle (220° of a 360° circle) drawn with a
  stroke-dasharray/stroke-dashoffset trick — the standard SVG technique
  for animating progress along a path.

  CIRCUMFERENCE OF ARC:
    Full circle circumference = 2π × r = 2π × 54 ≈ 339.3
    We show 220/360 of the circle = 339.3 × (220/360) ≈ 207.3
    The remaining 131.3 is the gap at the bottom.

  Props:
    probability  — 0.0 to 1.0 (from model.predict_proba)
    riskCode     — 0 (Normal) or 1 (Degraded)
*/

"use client";

interface Props {
  probability: number;
  riskCode: 0 | 1;
}

// Pick a stroke colour based on the probability level
function gaugeColor(probability: number): string {
  if (probability < 0.35) return "#22c55e"; // green  — low risk
  if (probability < 0.65) return "#f59e0b"; // amber  — moderate risk
  return "#ef4444"; // red    — high risk
}

export default function ProbabilityGauge({ probability, riskCode }: Props) {
  const radius = 54;
  const cx = 70;
  const cy = 70;

  // Total arc length for the 220° arc
  const fullArc = 2 * Math.PI * radius * (220 / 360);

  // How much of the arc to fill based on probability (0.0 → 1.0)
  const filledLength = fullArc * probability;
  const emptyLength = fullArc - filledLength;

  // We rotate the SVG so the gap sits at the bottom centre.
  // 220° arc → gap starts at 260° from top → rotate -250° from 3 o'clock
  const rotation = 160; // degrees, applied via transform

  const color = gaugeColor(probability);
  const pct = Math.round(probability * 100);

  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        width="140"
        height="140"
        viewBox="0 0 140 140"
        className="overflow-visible"
      >
        {/* Background track — full 220° arc in dark grey */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke="#334155"
          strokeWidth="10"
          strokeDasharray={`${fullArc} ${2 * Math.PI * radius - fullArc}`}
          strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cy})`}
        />

        {/* Foreground arc — fills according to probability */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${filledLength} ${emptyLength + (2 * Math.PI * radius - fullArc)}`}
          transform={`rotate(${rotation} ${cx} ${cy})`}
          style={{
            transition:
              "stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.4s",
          }}
        />

        {/* Centre text — probability percentage */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          className="font-mono"
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "22px",
            fontWeight: 700,
            fill: color,
            transition: "fill 0.4s",
          }}
        >
          {pct}%
        </text>

        {/* Sub-label */}
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          style={{
            fontFamily: "'Inter', sans-serif",
            fontSize: "9px",
            fontWeight: 500,
            fill: "#64748b",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          RISK PROBABILITY
        </text>
      </svg>
    </div>
  );
}
