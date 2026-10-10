"use client";

import Link from "next/link";
import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Prediction = {
  id?: number | null;
  label: "positive" | "negative";
  confidence: number;
  model_version: string;
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Feedback state
  const [feedbackSubmitted, setFeedbackSubmitted] = useState<string | null>(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  async function classify() {
    setLoading(true);
    setError(null);
    setResult(null);
    setFeedbackSubmitted(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(suggestedLabel: "positive" | "negative") {
    if (!result?.id) return;
    setFeedbackLoading(true);
    try {
      const res = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: result.id, label: suggestedLabel }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Error (${res.status})`);
      }
      setFeedbackSubmitted(`Feedback recorded as "${suggestedLabel}". Queued for human review!`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to submit feedback");
    } finally {
      setFeedbackLoading(false);
    }
  }

  const positive = result?.label === "positive";

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 px-6 py-16">
      <h1 className="text-3xl font-semibold">Movie review sentiment</h1>
      <p className="text-zinc-500">
        Paste a review and classify it as positive or negative.{" "}
        <Link href="/control" className="underline">
          MLOps Control Center
        </Link>
      </p>

      <textarea
        className="h-48 w-full rounded-lg border border-zinc-300 bg-transparent p-3 dark:border-zinc-700"
        placeholder="This movie was..."
        value={text}
        maxLength={10000}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        onClick={classify}
        disabled={loading || !text.trim()}
        className="self-start rounded-full bg-foreground px-6 py-2 text-background disabled:opacity-40"
      >
        {loading ? "Classifying..." : "Classify"}
      </button>

      {error && <p className="text-red-500">{error}</p>}

      {result && (
        <section className="flex flex-col gap-4 rounded-lg border border-zinc-300 p-5 dark:border-zinc-700">
          <div className="flex items-center justify-between">
            <p className={`text-2xl font-semibold ${positive ? "text-green-600" : "text-red-600"}`}>
              {result.label}
            </p>
            <span className="rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
              Model: {result.model_version}
            </span>
          </div>

          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </p>

          {/* Feedback & Review Queue Integration */}
          <div className="mt-2 border-t border-zinc-200 pt-3 dark:border-zinc-800">
            {feedbackSubmitted ? (
              <p className="text-sm font-medium text-green-600 dark:text-green-400">
                ✓ {feedbackSubmitted}
              </p>
            ) : result.id ? (
              <div className="flex flex-col gap-2">
                <p className="text-xs text-zinc-500">
                  Help improve the model: Was this classification correct?
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => sendFeedback(result.label)}
                    disabled={feedbackLoading}
                    className="rounded border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    👍 Correct
                  </button>
                  <button
                    onClick={() => sendFeedback(positive ? "negative" : "positive")}
                    disabled={feedbackLoading}
                    className="rounded border border-zinc-300 px-3 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    👎 Incorrect (mark as {positive ? "negative" : "positive"})
                  </button>
                </div>
              </div>
            ) : (
              <p className="text-xs text-zinc-400">Database logging inactive (no ID).</p>
            )}
          </div>
        </section>
      )}
    </main>
  );
}
