"use client";

import { useState } from "react";
import { CONTROL_URL, ModelVersion, useControl } from "@/lib/control";

export default function RegistryPage() {
  const { data, error, loading, refresh } = useControl<ModelVersion[]>("/control/registry");
  const [token, setToken] = useState("");
  const [message, setMessage] = useState<string | null>(null);

  async function promote(version: string) {
    setMessage(null);
    const res = await fetch(`${CONTROL_URL}/control/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Token": token },
      body: JSON.stringify({ version }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) setMessage(`Failed: ${body.detail ?? res.status}`);
    else if (!body.ok) setMessage(`Blocked by quality gates for v${version}`);
    else setMessage(`v${version} is now champion${body.reloaded ? " and serving" : " (API reload failed)"}`);
    refresh();
  }

  if (loading) return <p>Loading...</p>;
  if (error || !data) return <p className="text-red-500">Control API unreachable: {error}</p>;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3 text-sm">
        <label htmlFor="token">Admin token</label>
        <input
          id="token"
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          className="rounded border border-zinc-300 bg-transparent px-2 py-1 dark:border-zinc-700"
        />
      </div>
      {message && <p className="text-sm">{message}</p>}

      <table className="w-full text-sm">
        <thead className="text-left text-zinc-500">
          <tr>
            <th>Version</th>
            <th>Aliases</th>
            <th>Status</th>
            <th>macro-F1</th>
            <th>Commit</th>
            <th>Notes</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {data.map((v) => (
            <tr key={v.version} className="border-t border-zinc-200 dark:border-zinc-800">
              <td className="py-2">v{v.version}</td>
              <td>{v.aliases.join(", ") || "—"}</td>
              <td>{v.tags.status}</td>
              <td>{v.metrics.test_macro_f1?.toFixed(4)}</td>
              <td className="font-mono">{v.tags.git_commit}</td>
              <td>{v.tags.rejection_reason ? `failed: ${v.tags.rejection_reason}` : v.tags.approved_at ?? ""}</td>
              <td>
                {v.aliases.includes("candidate") && (
                  <button
                    onClick={() => promote(v.version)}
                    disabled={!token}
                    className="rounded-full bg-foreground px-3 py-1 text-background disabled:opacity-40"
                  >
                    Approve
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
