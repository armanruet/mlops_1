"use client";

import { useState } from "react";
import {
  CONTROL_URL,
  DEFAULT_ADMIN_TOKEN,
  ModelVersion,
  useAdminToken,
  useControl,
} from "@/lib/control";

export default function RegistryPage() {
  const { data, error, loading, refresh } = useControl<ModelVersion[]>("/control/registry");
  const [token, setToken] = useAdminToken();
  const [showToken, setShowToken] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function promote(version: string) {
    setMessage(null);
    const activeToken = token.trim() || DEFAULT_ADMIN_TOKEN;
    const res = await fetch(`${CONTROL_URL}/control/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Admin-Token": activeToken },
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
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <label htmlFor="token" className="font-medium">Admin token:</label>
        <div className="flex items-center gap-1">
          <input
            id="token"
            type={showToken ? "text" : "password"}
            value={token}
            placeholder={DEFAULT_ADMIN_TOKEN}
            onChange={(e) => setToken(e.target.value)}
            className="w-48 rounded border border-zinc-300 bg-transparent px-2 py-1 font-mono text-sm dark:border-zinc-700"
          />
          <button
            type="button"
            onClick={() => setShowToken(!showToken)}
            className="rounded border border-zinc-300 px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
          >
            {showToken ? "Hide" : "Show"}
          </button>
        </div>
        {token === DEFAULT_ADMIN_TOKEN && (
          <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800 dark:bg-green-950 dark:text-green-300">
            ✓ Ready (dev default)
          </span>
        )}
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
