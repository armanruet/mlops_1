import Link from "next/link";

const NAV = [
  { href: "/control", label: "Overview" },
  { href: "/control/lineage", label: "Data & pipeline" },
  { href: "/control/experiments", label: "Experiments" },
  { href: "/control/registry", label: "Model registry" },
];

export default function ControlLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-8 flex flex-wrap items-center gap-6 border-b border-zinc-300 pb-4 dark:border-zinc-700">
        <h1 className="text-xl font-semibold">MLOps Control Center</h1>
        <nav className="flex flex-wrap gap-4 text-sm">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} className="hover:underline">
              {n.label}
            </Link>
          ))}
        </nav>
        <Link href="/" className="ml-auto text-sm text-zinc-500 hover:underline">
          ← Classifier
        </Link>
      </header>
      {children}
    </div>
  );
}
