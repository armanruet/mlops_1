"use client";

import { GateCheck, ModelVersion, useControl } from "@/lib/control";

type Overview = {
  serving: { loaded: boolean; version?: string; exported_at?: string; test_macro_f1?: string } | null;
  champion: ModelVersion | null;
  candidate: ModelVersion | null;
  n_versions: number;
  latest_gates: { passed: boolean; checks: GateCheck[] } | null;
  latest_metrics: Record<string, number> | null;
  drift: { status: string; drift_share?: number; n_current?: number; generated_at?: string } | null;
};

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-zinc-300 p-4 dark:border-zinc-700">
      <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-zinc-500">{title}</h2>
      {children}
    </section>
  );
}

export default function OverviewPage() {
  const { data, error, loading } = useControl<Overview>("/control/overview");
  if (loading) return <p>Loading...</p>;
  if (error || !data)
    return <p className="text-red-500">Control API unreachable: {error}. Is it running on :8001?</p>;

  const served = data.serving;
  const champion = data.champion;
  const mismatch = served?.loaded && champion && String(served.version) !== champion.version;

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Card title="Serving">
        {served?.loaded ? (
          <>
            <p className="text-2xl font-semibold">v{served.version}</p>
            <p className="text-sm text-zinc-500">Exported {served.exported_at}</p>
          </>
        ) : (
          <p className="text-red-500">Inference API down or no model loaded</p>
        )}
        {mismatch && (
          <p className="mt-2 text-sm text-amber-600">
            Serving v{served?.version} but champion is v{champion?.version}. Reload needed.
          </p>
        )}
      </Card>

      <Card title="Champion (approved)">
        {champion ? (
          <>
            <p className="text-2xl font-semibold">v{champion.version}</p>
            <p className="text-sm text-zinc-500">
              macro-F1 {champion.metrics.test_macro_f1?.toFixed(4)} · approved{" "}
              {champion.tags.approved_at}
            </p>
          </>
        ) : (
          <p>None yet</p>
        )}
      </Card>

      <Card title="Awaiting approval">
        {data.candidate ? (
          <p>
            v{data.candidate.version} (macro-F1 {data.candidate.metrics.test_macro_f1?.toFixed(4)}) —
            approve on the Model registry page.
          </p>
        ) : (
          <p className="text-zinc-500">No candidate</p>
        )}
      </Card>

      <Card title="Registered versions">
        <p className="text-2xl font-semibold">{data.n_versions}</p>
      </Card>

      <div className="sm:col-span-2">
        <Card title="Data drift (latest report)">
          {!data.drift || data.drift.status === "no_report" ? (
            <p className="text-zinc-500">No report yet</p>
          ) : data.drift.status === "insufficient_data" ? (
            <p className="text-amber-600">Not enough traffic yet ({data.drift.n_current} predictions)</p>
          ) : (
            <p className={data.drift.status === "drift" ? "text-red-600" : "text-green-600"}>
              {data.drift.status === "drift" ? "Drift detected" : "No significant drift"} —{" "}
              {((data.drift.drift_share ?? 0) * 100).toFixed(0)}% of columns drifted
              <span className="text-zinc-500"> · {data.drift.generated_at}</span>
            </p>
          )}
        </Card>
      </div>

      <div className="sm:col-span-2">
        <Card title={`Latest quality gates — ${data.latest_gates?.passed ? "PASSED" : "FAILED"}`}>
          <table className="w-full text-sm">
            <thead className="text-left text-zinc-500">
              <tr>
                <th>Check</th>
                <th>Value</th>
                <th>Requirement</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {data.latest_gates?.checks.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>{c.value}</td>
                  <td>
                    {c.op} {c.threshold}
                  </td>
                  <td className={c.passed ? "text-green-600" : "text-red-600"}>
                    {c.passed ? "pass" : "fail"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    </div>
  );
}
