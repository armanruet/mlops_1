"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Prediction = { label: "positive" | "negative"; confidence: number; model_version: string };

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function classify() {
    setLoading(true);
    setError(null);
    setResult(null);
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

  const positive = result?.label === "positive";

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col gap-6 px-6 py-16">
      <h1 className="text-3xl font-semibold">Movie review sentiment</h1>
      <p className="text-zinc-500">Paste a review and classify it as positive or negative.</p>

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
        <section className="rounded-lg border border-zinc-300 p-4 dark:border-zinc-700">
          <p className={`text-2xl font-semibold ${positive ? "text-green-600" : "text-red-600"}`}>
            {result.label}
          </p>
          <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          <p className="text-sm text-zinc-500">Model: {result.model_version}</p>
        </section>
      )}
    </main>
  );
}
