/*
  components/StreamProgress.tsx

  A simple progress bar showing how far through the dataset the stream is.
  Also shows the current row index, total rows, and a loop indicator
  so the user knows the stream will restart from row 0 automatically.

  Props:
    rowIndex  — current row number (0-based)
    totalRows — total rows in the dataset
*/

"use client";

interface Props {
  rowIndex: number;
  totalRows: number;
}

export default function StreamProgress({ rowIndex, totalRows }: Props) {
  const pct = totalRows > 0 ? Math.round((rowIndex / totalRows) * 100) : 0;

  return (
    <div className="flex flex-col gap-1.5">
      {/* Label row */}
      <div className="flex items-center justify-between text-[10px] font-mono">
        <span className="text-slate-500">
          Dataset position: row{" "}
          <span className="text-slate-300">{rowIndex}</span>
          {" / "}
          <span className="text-slate-300">{totalRows}</span>
        </span>
        <span className="text-slate-600">{pct}% — loops at end</span>
      </div>

      {/* Progress bar track */}
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-sky-500/70 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
