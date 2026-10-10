"use client";

import { useEffect, useState, useSyncExternalStore } from "react";

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

export type FeedbackItem = {
  id: number;
  prediction_id: number;
  text: string;
  predicted_label: string;
  confidence: number;
  model_version: string;
  user_label: string | null;
  reviewed_label: string | null;
  review_status: "pending" | "approved" | "rejected";
  created_at: string | null;
  reviewed_at: string | null;
};

export type FeedbackStats = {
  pending: number;
  approved: number;
  rejected: number;
  approved_since_last_retrain: number;
  error_rate_by_version: Record<string, { reviewed: number; wrong: number; wrong_rate: number }>;
  threshold?: number;
};

export const DEFAULT_ADMIN_TOKEN = "dev-admin-token";

function subscribe(callback: () => void) {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

function getSnapshot(): string {
  try {
    return localStorage.getItem("mlops_admin_token") ?? DEFAULT_ADMIN_TOKEN;
  } catch {
    return DEFAULT_ADMIN_TOKEN;
  }
}

function getServerSnapshot(): string {
  return DEFAULT_ADMIN_TOKEN;
}

export function useAdminToken() {
  const token = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const setToken = (val: string) => {
    try {
      localStorage.setItem("mlops_admin_token", val);
      window.dispatchEvent(new Event("storage"));
    } catch {
      // ignore
    }
  };

  return [token, setToken] as const;
}



