"use client";

import { useState } from "react";
import { CONTROL_URL, useControl } from "@/lib/control";

type Column = { column: string; method: string; value: number; threshold: number; drifted: boolean };
type Drift = {
  status: "ok" | "drift" | "insufficient_data" | "no_report";
  generated_at?: string;
  n_reference?: number;
  n_current?: number;
  min_samples?: number;
  drift_share?: number;
  drift_share_threshold?: number;
  drifted_columns?: string[];
  columns?: Column[];
  recommended_action?: string;
};

const BADGE: Record<string, string> = {
  ok: "text-green-600",
  drift: "text-red-600",
  insufficient_data: "text-amber-600",
  no_report: "text-zinc-500",
};

export default function DriftPage() {
  const { data, error, loading, refresh } = useControl<Drift>("/control/drift");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setMessage(null);
    const res = await fetch(`${CONTROL_URL}/control/drift/run`, {
      method: "POST",
      headers: { "X-Admin-Token": token },
    });
    setMessage(res.ok ? "Drift report updated" : `Failed (${res.status})`);
    setBusy(false);
    refresh();
  }

  if (loading) return <p>Loading...</p>;
  if (error || !data) return <p className="text-red-500">Control API unreachable: {error}</p>;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <input
          type="password"
          aria-label="Admin token"
          placeholder="Admin token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          className="rounded border border-zinc-300 bg-transparent px-2 py-1 dark:border-zinc-700"
        />
        <button
          onClick={run}
          disabled={!token || busy}
          className="rounded-full bg-foreground px-4 py-1 text-background disabled:opacity-40"
        >
          {busy ? "Running..." : "Run drift check now"}
        </button>
        <a className="underline" href={`${CONTROL_URL}/control/drift/report`} target="_blank">
          Full Evidently report
        </a>
        {message && <span>{message}</span>}
      </div>

      <p className={`text-2xl font-semibold ${BADGE[data.status]}`}>
        {data.status === "drift" && "Drift detected"}
        {data.status === "ok" && "No significant drift"}
        {data.status === "insufficient_data" &&
          `Not enough traffic yet (${data.n_current} of ${data.min_samples} predictions)`}
        {data.status === "no_report" && "No drift report yet"}
      </p>

      {data.columns && (
        <>
          <p className="text-sm text-zinc-500">
            {data.drifted_columns?.length} of {data.columns.length} columns drifted (
            {((data.drift_share ?? 0) * 100).toFixed(0)}%, threshold{" "}
            {((data.drift_share_threshold ?? 0) * 100).toFixed(0)}%). {data.n_current} recent
            predictions vs {data.n_reference} reference. Generated {data.generated_at}.{" "}
            {data.recommended_action}.
          </p>
          <table className="w-full text-sm">
            <thead className="text-left text-zinc-500">
              <tr>
                <th>Column</th>
                <th>Test</th>
                <th>Score</th>
                <th>Threshold</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {data.columns.map((c) => (
                <tr key={c.column} className="border-t border-zinc-200 dark:border-zinc-800">
                  <td className="py-2">{c.column}</td>
                  <td>{c.method}</td>
                  <td>{c.value.toPrecision(3)}</td>
                  <td>{c.threshold}</td>
                  <td className={c.drifted ? "text-red-600" : "text-green-600"}>
                    {c.drifted ? "drifted" : "stable"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
