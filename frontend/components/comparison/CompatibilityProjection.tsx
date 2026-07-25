'use client';

import { GaugeNeedle } from '@/components/charts/GaugeNeedle';
import type { AnyErrorCode, ApiResult } from '@/lib/api/errors';
import type { SetupResponse } from '@/lib/api/types';

// Compatibility projection (master §10.5). Div #62: there is NO circuit-aware "which driver is
// better for circuit X" endpoint — /compare is circuit-agnostic. So we reuse the F16 compatibility
// PROXY (#47): per-driver compatibility for the selected circuit = mean per-parameter reliability of
// its setup recommendation (GET /drivers/{id}/setup/{circuit}). The gauge shows the signed delta,
// labelled a proxy. A 404 for a driver → honest "—" (absence); a real 503 → a distinct code marker
// (failure ≠ absence).
function meanConfidence(setup: SetupResponse): number | null {
  const ps = setup.parameters;
  if (!ps.length) return null;
  return ps.reduce((sum, p) => sum + (p.confidence ?? 0), 0) / ps.length;
}

type Proxy = { value: number | null; code: AnyErrorCode | null };

function proxyOf(r: ApiResult<SetupResponse> | null): Proxy {
  if (!r) return { value: null, code: null };
  if (r.ok) return { value: meanConfidence(r.data), code: null };
  if (r.code === 'NOT_FOUND') return { value: null, code: null }; // honest absence → "—"
  return { value: null, code: r.code }; // real outage → code marker
}

function Marker({ label, proxy }: { label: string; proxy: Proxy }) {
  return (
    <div className="font-mono text-[11px]" data-testid="compat-driver">
      <span className="text-text-secondary">{label}: </span>
      {proxy.code ? (
        <span className="text-accent" data-metric-error data-code={proxy.code}>
          ⚠ {proxy.code}
        </span>
      ) : (
        <span className="text-text-primary">{proxy.value == null ? '—' : proxy.value.toFixed(2)}</span>
      )}
    </div>
  );
}

export function CompatibilityProjection({
  setup1,
  setup2,
  driver1,
  driver2,
}: {
  setup1: ApiResult<SetupResponse> | null;
  setup2: ApiResult<SetupResponse> | null;
  driver1: string;
  driver2: string;
}) {
  const p1 = proxyOf(setup1);
  const p2 = proxyOf(setup2);
  const delta = p1.value != null && p2.value != null ? p1.value - p2.value : null;

  return (
    <div data-testid="compatibility-projection">
      <p className="mb-1 font-mono text-[10px] uppercase text-text-secondary">
        Compatibility Δ (setup-reliability proxy — not a circuit-aware score)
      </p>
      <GaugeNeedle
        label="compatibility delta"
        value={delta}
        min={-1}
        max={1}
        rating="PROXY"
      />
      <div className="mt-2 flex gap-4">
        <Marker label={driver1} proxy={p1} />
        <Marker label={driver2} proxy={p2} />
      </div>
    </div>
  );
}
