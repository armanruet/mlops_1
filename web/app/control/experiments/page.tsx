"use client";

import { useControl } from "@/lib/control";

type Run = {
  run_id: string;
  name: string;
  status: string;
  started_at: number;
  params: Record<string, string>;
  metrics: Record<string, number>;
};

export default function ExperimentsPage() {
  const { data, error, loading } = useControl<Run[]>("/control/experiments");
  if (loading) return <p>Loading...</p>;
  if (error || !data) return <p className="text-red-500">Control API unreachable: {error}</p>;
  if (data.length === 0) return <p>No runs yet. Run the pipeline with `dvc repro`.</p>;

  return (
    <table className="w-full text-sm">
      <thead className="text-left text-zinc-500">
        <tr>
          <th>Started</th>
          <th>Run</th>
          <th>Status</th>
          <th>ngram</th>
          <th>C</th>
          <th>Train F1</th>
          <th>Test acc</th>
          <th>Test F1</th>
        </tr>
      </thead>
      <tbody>
        {data.map((r) => (
          <tr key={r.run_id} className="border-t border-zinc-200 dark:border-zinc-800">
            <td className="py-2">{new Date(r.started_at).toLocaleString()}</td>
            <td className="font-mono">{r.run_id.slice(0, 8)}</td>
            <td>{r.status}</td>
            <td>{r.params.ngram_max}</td>
            <td>{r.params.C}</td>
            <td>{r.metrics.train_macro_f1?.toFixed(4)}</td>
            <td>{r.metrics.test_accuracy?.toFixed(4) ?? "—"}</td>
            <td>{r.metrics.test_macro_f1?.toFixed(4) ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
