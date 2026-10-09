"use client";

import { useEffect, useState } from "react";

export const CONTROL_URL = process.env.NEXT_PUBLIC_CONTROL_URL ?? "http://localhost:8001";

export function useControl<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetch(`${CONTROL_URL}${path}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Request failed (${r.status})`);
        return r.json();
      })
      .then((d) => !cancelled && (setData(d), setError(null)))
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [path, tick]);

  return { data, error, loading: !data && !error, refresh: () => setTick((t) => t + 1) };
}

export type GateCheck = {
  name: string;
  value: number;
  threshold: number;
  op: string;
  passed: boolean;
};

export type ModelVersion = {
  version: string;
  run_id: string;
  aliases: string[];
  tags: Record<string, string>;
  metrics: Record<string, number>;
};
