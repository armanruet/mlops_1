"use client";

import Link from "next/link";
import { useState } from "react";
import {
  CONTROL_URL,
  DEFAULT_ADMIN_TOKEN,
  FeedbackItem,
  FeedbackStats,
  useAdminToken,
  useControl,
} from "@/lib/control";

export default function FeedbackPage() {
  const { data: stats, refresh: refreshStats } = useControl<FeedbackStats>("/control/feedback/stats");
  const [tab, setTab] = useState<"pending" | "approved" | "rejected">("pending");
  const { data: items, error: itemsError, loading: itemsLoading, refresh: refreshItems } =
    useControl<FeedbackItem[]>(`/control/feedback?status=${tab}`);

  const [token, setToken] = useAdminToken();
  const [showToken, setShowToken] = useState(false);
  const [force, setForce] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  // Retraining state
  type RetrainResponse = {
    ok: boolean;
    run_id?: string;
    n_feedback?: number;
    reason?: string;
    candidate_registered?: boolean;
    gates?: { passed: boolean };
  };
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<RetrainResponse | null>(null);

  function refreshAll() {
    refreshStats();
    refreshItems();
  }

  async function handleReview(id: number, decision: "approve" | "reject", label?: string) {
    setActionMessage(null);
    const activeToken = token.trim() || DEFAULT_ADMIN_TOKEN;
    try {
      const res = await fetch(`${CONTROL_URL}/control/feedback/${id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Admin-Token": activeToken },
        body: JSON.stringify({ decision, label }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setActionMessage(`Failed to review item ${id}: ${body.detail ?? res.status}`);
      } else {
        setActionMessage(`Feedback #${id} marked as ${decision}${label ? ` (${label})` : ""}`);
        refreshAll();
      }
    } catch (e) {
      setActionMessage(e instanceof Error ? e.message : "Network error");
    }
  }

  async function triggerRetrain() {
    setActionMessage(null);
    setRetrainResult(null);
    const activeToken = token.trim() || DEFAULT_ADMIN_TOKEN;
    setRetraining(true);
    try {
      const res = await fetch(`${CONTROL_URL}/control/retrain`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Admin-Token": activeToken },
        body: JSON.stringify({ force }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setActionMessage(`Retraining failed: ${body.detail ?? res.status}`);
      } else {
        setRetrainResult(body);
        if (body.ok) {
          setActionMessage(
            `Retraining complete! Model registered as candidate with run ID ${body.run_id}`
          );
        } else {
          setActionMessage(`Retraining blocked: ${body.reason}`);
        }
        refreshAll();
      }
    } catch (e) {
      setActionMessage(e instanceof Error ? e.message : "Retraining error");
    } finally {
      setRetraining(false);
    }
  }

  const approvedCount = stats?.approved ?? 0;
  const threshold = stats?.threshold ?? 10;
  const readyToRetrain = approvedCount >= threshold;

  return (
    <div className="flex flex-col gap-8">
      {/* Admin Token Input */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-900/50">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label htmlFor="token" className="font-medium">
            Admin token:
          </label>
          <div className="flex items-center gap-1">
            <input
              id="token"
              type={showToken ? "text" : "password"}
              value={token}
              placeholder={DEFAULT_ADMIN_TOKEN}
              onChange={(e) => setToken(e.target.value)}
              className="w-48 rounded border border-zinc-300 bg-transparent px-3 py-1 font-mono text-sm dark:border-zinc-700"
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
          {token !== DEFAULT_ADMIN_TOKEN && (
            <button
              type="button"
              onClick={() => setToken(DEFAULT_ADMIN_TOKEN)}
              className="text-xs text-zinc-500 underline hover:text-zinc-800 dark:hover:text-zinc-200"
            >
              Reset to dev default
            </button>
          )}
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={refreshAll}
            className="rounded border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
          >
            ↻ Refresh data
          </button>
        </div>
      </div>

      {actionMessage && (
        <div className="rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
          {actionMessage}
        </div>
      )}

      {/* Retraining & Metrics Section */}
      <section className="grid gap-6 md:grid-cols-2">
        {/* Retraining Controls */}
        <div className="flex flex-col gap-4 rounded-lg border border-zinc-200 p-5 dark:border-zinc-800">
          <h2 className="text-lg font-semibold">Continuous Retraining</h2>
          <p className="text-sm text-zinc-500">
            Approved human reviews are merged with the baseline training set. Candidates must pass
            quality gates vs the current champion before promotion.
          </p>

          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between text-sm">
              <span>Approved samples:</span>
              <span className="font-mono font-medium">
                {approvedCount} / {threshold} required
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              <div
                className={`h-full transition-all ${
                  readyToRetrain ? "bg-green-500" : "bg-amber-500"
                }`}
                style={{ width: `${Math.min(100, (approvedCount / threshold) * 100)}%` }}
              />
            </div>
          </div>

          <div className="mt-2 flex items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
              <input
                type="checkbox"
                checked={force}
                onChange={(e) => setForce(e.target.checked)}
                className="rounded"
              />
              Force retrain (override threshold)
            </label>
          </div>

          <button
            onClick={triggerRetrain}
            disabled={retraining || (!readyToRetrain && !force)}
            className="rounded-full bg-foreground px-5 py-2 text-sm font-medium text-background disabled:opacity-40"
          >
            {retraining ? "Retraining in progress..." : "Trigger Model Retraining"}
          </button>

          {retrainResult?.ok && (
            <div className="mt-2 rounded border border-green-200 bg-green-50 p-3 text-xs text-green-900 dark:border-green-900 dark:bg-green-950 dark:text-green-200">
              <p className="font-semibold">Candidate Model Produced:</p>
              <p>MLflow Run ID: {retrainResult.run_id}</p>
              <p>Feedback samples included: {retrainResult.n_feedback}</p>
              <p>
                Gates Status:{" "}
                <span className={retrainResult.gates?.passed ? "font-bold text-green-600" : "font-bold text-red-600"}>
                  {retrainResult.gates?.passed ? "PASSED" : "FAILED"}
                </span>
              </p>
              <Link href="/control/registry" className="mt-2 inline-block font-semibold underline">
                Go to Model Registry to review and approve →
              </Link>
            </div>
          )}
        </div>

        {/* Feedback Statistics */}
        <div className="flex flex-col gap-4 rounded-lg border border-zinc-200 p-5 dark:border-zinc-800">
          <h2 className="text-lg font-semibold">Feedback Metrics</h2>
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded border border-zinc-200 p-3 text-center dark:border-zinc-800">
              <p className="text-2xl font-bold">{stats?.pending ?? 0}</p>
              <p className="text-xs text-zinc-500">Pending</p>
            </div>
            <div className="rounded border border-zinc-200 p-3 text-center dark:border-zinc-800">
              <p className="text-2xl font-bold text-green-600">{stats?.approved ?? 0}</p>
              <p className="text-xs text-zinc-500">Approved</p>
            </div>
            <div className="rounded border border-zinc-200 p-3 text-center dark:border-zinc-800">
              <p className="text-2xl font-bold text-red-500">{stats?.rejected ?? 0}</p>
              <p className="text-xs text-zinc-500">Rejected</p>
            </div>
          </div>

          <h3 className="text-sm font-medium">Model Error Rate on Reviewed Samples</h3>
          {stats?.error_rate_by_version && Object.keys(stats.error_rate_by_version).length > 0 ? (
            <div className="flex flex-col gap-2 text-xs">
              {Object.entries(stats.error_rate_by_version).map(([ver, data]) => (
                <div key={ver} className="flex items-center justify-between border-b border-zinc-100 py-1 dark:border-zinc-800">
                  <span className="font-mono">{ver}</span>
                  <span>{data.wrong} errors / {data.reviewed} reviews</span>
                  <span className="font-semibold">{(data.wrong_rate * 100).toFixed(1)}% error</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-500">No reviewed samples with error stats yet.</p>
          )}
        </div>
      </section>

      {/* Human-in-the-Loop Review Queue */}
      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-zinc-200 pb-2 dark:border-zinc-800">
          <h2 className="text-lg font-semibold">Human Review Queue</h2>
          <div className="flex gap-2">
            {(["pending", "approved", "rejected"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded px-3 py-1 text-xs capitalize ${
                  tab === t
                    ? "bg-foreground text-background"
                    : "border border-zinc-200 hover:bg-zinc-100 dark:border-zinc-800 dark:hover:bg-zinc-800"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {itemsLoading && <p className="text-sm text-zinc-500">Loading feedback queue...</p>}
        {itemsError && <p className="text-sm text-red-500">Error: {itemsError}</p>}

        {items && items.length === 0 && (
          <p className="text-sm text-zinc-500">No items found in {tab} queue.</p>
        )}

        {items && items.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-zinc-200 text-xs text-zinc-500 dark:border-zinc-800">
                <tr>
                  <th className="py-2">ID</th>
                  <th>Review Text</th>
                  <th>Served Model</th>
                  <th>Predicted</th>
                  <th>User Suggested</th>
                  <th>Status</th>
                  <th className="text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
                {items.map((item) => (
                  <tr key={item.id} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-900/50">
                    <td className="py-3 font-mono text-xs">#{item.id}</td>
                    <td className="max-w-xs truncate pr-4 text-xs" title={item.text}>
                      {item.text}
                    </td>
                    <td className="font-mono text-xs">{item.model_version}</td>
                    <td>
                      <span
                        className={`rounded px-2 py-0.5 text-xs font-medium ${
                          item.predicted_label === "positive"
                            ? "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300"
                            : "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
                        }`}
                      >
                        {item.predicted_label} ({(item.confidence * 100).toFixed(0)}%)
                      </span>
                    </td>
                    <td>
                      <span className="text-xs font-semibold">{item.user_label ?? "—"}</span>
                    </td>
                    <td className="text-xs capitalize">
                      {item.review_status === "approved"
                        ? `Approved (${item.reviewed_label})`
                        : item.review_status}
                    </td>
                    <td className="text-right">
                      {tab === "pending" && (
                        <div className="flex justify-end gap-1">
                          <button
                            onClick={() => handleReview(item.id, "approve", item.user_label ?? undefined)}
                            title="Approve user suggestion"
                            className="rounded bg-green-600 px-2 py-1 text-xs text-white hover:bg-green-700"
                          >
                            Approve
                          </button>
                          <button
                            onClick={() => handleReview(item.id, "approve", "positive")}
                            title="Force positive"
                            className="rounded border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700"
                          >
                            +Pos
                          </button>
                          <button
                            onClick={() => handleReview(item.id, "approve", "negative")}
                            title="Force negative"
                            className="rounded border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700"
                          >
                            -Neg
                          </button>
                          <button
                            onClick={() => handleReview(item.id, "reject")}
                            title="Reject review"
                            className="rounded bg-red-600 px-2 py-1 text-xs text-white hover:bg-red-700"
                          >
                            Reject
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
