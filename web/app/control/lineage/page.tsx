"use client";

import { useControl } from "@/lib/control";

type FileRef = { path: string; md5?: string; size?: number };
type Stage = {
  name: string;
  cmd: string;
  deps: FileRef[];
  outs: FileRef[];
  params: Record<string, Record<string, unknown>>;
};
type Lineage = {
  dataset: { source: string };
  data_stats: Record<string, { rows: number; duplicates_removed: number; mean_chars: number;
    class_balance: Record<string, number> }> | null;
  stages: Stage[];
};

const short = (h?: string) => (h ? h.slice(0, 8) : "—");

export default function LineagePage() {
  const { data, error, loading } = useControl<Lineage>("/control/lineage");
  if (loading) return <p>Loading...</p>;
  if (error || !data) return <p className="text-red-500">Control API unreachable: {error}</p>;

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h2 className="mb-1 font-medium">Dataset</h2>
        <a className="text-sm underline" href={data.dataset.source}>
          {data.dataset.source}
        </a>
        {data.data_stats && (
          <table className="mt-3 text-sm">
            <thead className="text-left text-zinc-500">
              <tr>
                <th className="pr-6">Split</th>
                <th className="pr-6">Rows</th>
                <th className="pr-6">Duplicates removed</th>
                <th className="pr-6">Class balance (0/1)</th>
                <th>Mean chars</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.data_stats).map(([split, s]) => (
                <tr key={split}>
                  <td>{split}</td>
                  <td>{s.rows}</td>
                  <td>{s.duplicates_removed}</td>
                  <td>{s.class_balance["0"]} / {s.class_balance["1"]}</td>
                  <td>{s.mean_chars}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="font-medium">Pipeline stages (from dvc.lock)</h2>
        {data.stages.map((s, i) => (
          <div key={s.name} className="rounded-lg border border-zinc-300 p-3 text-sm dark:border-zinc-700">
            <p className="font-medium">
              {i + 1}. {s.name} <span className="font-mono text-zinc-500">{s.cmd}</span>
            </p>
            <p className="mt-1 text-zinc-500">Inputs</p>
            <ul className="font-mono">
              {s.deps.map((d) => (
                <li key={d.path}>{d.path} · {short(d.md5)}</li>
              ))}
            </ul>
            <p className="mt-1 text-zinc-500">Outputs</p>
            <ul className="font-mono">
              {s.outs.map((o) => (
                <li key={o.path}>{o.path} · {short(o.md5)}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    </div>
  );
}
